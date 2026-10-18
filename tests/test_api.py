import numpy as np
import pytest

import api as api_module
from constraints import check_plan
from models import OptimizationResult
from serialization import problem_from_dict


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(api_module.app.config, "TESTING", True)
    return api_module.app.test_client()


@pytest.fixture
def scenario():
    return {
        "depots": [
            {"id": "B1", "stock": 20},
            {"id": "B2", "stock": 20},
        ],
        "stations": [
            {
                "id": "S1",
                "remaining": 5,
                "demand": 12,
                "safety": 3,
                "capacity": 20,
            },
            {
                "id": "S2",
                "remaining": 5,
                "demand": 17,
                "safety": 3,
                "capacity": 25,
            },
            {
                "id": "S3",
                "remaining": 0,
                "demand": 12,
                "safety": 3,
                "capacity": 20,
            },
        ],
        "costs": [[1, 4, 5], [5, 2, 1]],
    }


def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_validate_feasible(client, scenario):
    response = client.post("/api/validate", json=scenario)

    assert response.status_code == 200
    assert response.json["feasible"] is True
    assert response.json["minimum_delivery"] == [10, 15, 15]
    assert response.json["total_stock"] == 40
    assert response.json["total_required"] == 40
    assert response.json["violations"] == []


@pytest.mark.parametrize("method", ["highs", "highs-ds", "highs-ipm"])
def test_optimize(client, scenario, method):
    response = client.post(
        "/api/optimize",
        query_string={"method": method},
        json=scenario,
    )
    data = response.json
    depots, stations, _ = problem_from_dict(scenario)

    assert response.status_code == 200
    assert data["status"] == "optimal"
    assert data["total_cost"] == pytest.approx(75)
    assert check_plan(depots, stations, data["deliveries"]) == []
    assert np.asarray(data["deliveries"]).shape == (2, 3)
    assert data["depot_ids"] == ["B1", "B2"]
    assert data["station_ids"] == ["S1", "S2", "S3"]
    assert data["metrics"]["method"] == method
    assert data["summary"]["station_remaining"] == pytest.approx([3, 3, 3])


def test_validate_infeasible(client, scenario):
    scenario["depots"][0]["stock"] = 0
    response = client.post("/api/validate", json=scenario)

    assert response.status_code == 200
    assert response.json["feasible"] is False
    assert response.json["status"] == "infeasible"
    assert response.json["violations"][0]["code"] == "insufficient_stock"


def test_optimize_infeasible(client, scenario):
    scenario["depots"][0]["stock"] = 0
    response = client.post("/api/optimize", json=scenario)

    assert response.status_code == 200
    assert response.json["status"] == "infeasible"
    assert response.json["deliveries"] is None
    assert response.json["total_cost"] is None


@pytest.mark.parametrize("url", ["/api/validate", "/api/optimize"])
def test_invalid_data(client, scenario, url):
    scenario["costs"] = [[1]]
    response = client.post(url, json=scenario)

    assert response.status_code == 400
    assert response.json["status"] == "invalid_data"


@pytest.mark.parametrize("url", ["/api/validate", "/api/optimize"])
def test_invalid_json(client, url):
    response = client.post(
        url,
        data="{",
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json["status"] == "invalid_request"


@pytest.mark.parametrize("url", ["/api/validate", "/api/optimize"])
def test_wrong_content_type(client, url):
    response = client.post(
        url,
        data="{}",
        content_type="text/plain",
    )

    assert response.status_code == 415
    assert response.json["status"] == "invalid_request"


def test_unknown_method(client, scenario):
    response = client.post(
        "/api/optimize?method=unknown",
        json=scenario,
    )

    assert response.status_code == 400
    assert response.json["status"] == "invalid_data"


def test_wrong_http_method(client):
    response = client.get("/api/optimize")

    assert response.status_code == 405
    assert response.is_json
    assert "POST" in response.headers["Allow"]


@pytest.mark.parametrize(
    "status, http_status",
    [("limit", 200), ("error", 500)],
)
def test_solver_statuses(
    client, scenario, monkeypatch, status, http_status
):
    def stub(*args, **kwargs):
        return OptimizationResult(
            status=status,
            message="Тестовый статус",
        )

    monkeypatch.setattr(api_module, "optimize_plan", stub)
    response = client.post("/api/optimize", json=scenario)

    assert response.status_code == http_status
    assert response.json["status"] == status
    assert response.json["deliveries"] is None


def test_unexpected_error(client, scenario, monkeypatch):
    def failing_solver(*args, **kwargs):
        raise RuntimeError("Внутренняя подробность")

    monkeypatch.setattr(api_module, "optimize_plan", failing_solver)
    monkeypatch.setitem(
        api_module.app.config,
        "PROPAGATE_EXCEPTIONS",
        False,
    )

    response = client.post("/api/optimize", json=scenario)

    assert response.status_code == 500
    assert response.json == {
        "status": "error",
        "message": "Внутренняя ошибка сервера.",
    }