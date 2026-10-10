import json
from copy import deepcopy

import numpy as np
import pytest

from optimizer import optimize_plan
from serialization import load_problem, problem_from_dict, result_to_dict, save_result


@pytest.fixture
def scenario():
    return {
        "depots": [
            {"id": "B1", "stock": 20},
            {"id": "B2", "stock": 20},
        ],
        "stations": [
            {"id": "S1", "remaining": 5, "demand": 12, "safety": 3, "capacity": 20},
            {"id": "S2", "remaining": 5, "demand": 17, "safety": 3, "capacity": 25},
            {"id": "S3", "remaining": 0, "demand": 12, "safety": 3, "capacity": 20},
        ],
        "costs": [[1, 4, 5], [5, 2, 1]],
    }


def test_load_problem(tmp_path, scenario):
    path = tmp_path / "example.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")

    depots, stations, costs = load_problem(path)

    assert [depot.id for depot in depots] == ["B1", "B2"]
    assert [station.id for station in stations] == ["S1", "S2", "S3"]
    np.testing.assert_array_equal(costs, scenario["costs"])
    assert optimize_plan(depots, stations, costs).total_cost == pytest.approx(75)


@pytest.mark.parametrize(
    "path, value",
    [
        (("depots",), {}),
        (("depots",), []),
        (("depots", 0, "stock"), "20"),
        (("depots", 0, "stock"), True),
        (("depots", 0, "stock"), -1),
        (("depots", 1, "id"), "B1"),
        (("stations", 0, "remaining"), 100),
        (("costs",), [[1, 2], [3, 4]]),
        (("costs", 0, 0), True),
        (("costs", 0, 0), "1"),
        (("costs", 0, 0), -1),
        (("costs", 0, 0), float("inf")),
    ],
)
def test_invalid_values(scenario, path, value):
    data = deepcopy(scenario)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        problem_from_dict(data)


@pytest.mark.parametrize(
    "section, field",
    [(None, "costs"), ("depots", "stock"), ("stations", "capacity")],
)
def test_missing_fields(scenario, section, field):
    target = scenario if section is None else scenario[section][0]
    del target[field]

    with pytest.raises(ValueError, match="отсутствуют поля"):
        problem_from_dict(scenario)


def test_unknown_field(scenario):
    scenario["stations"][0]["capaciti"] = 20

    with pytest.raises(ValueError, match="неизвестные поля"):
        problem_from_dict(scenario)


@pytest.mark.parametrize(
    "text",
    ['{"depots":', "[]", "null", '{"costs": NaN}', '{"costs": Infinity}'],
)
def test_invalid_json(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        load_problem(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_problem(tmp_path / "missing.json")


def test_save_optimal_result(tmp_path, scenario):
    depots, stations, costs = problem_from_dict(scenario)
    result = optimize_plan(depots, stations, costs)

    path = tmp_path / "result.json"
    save_result(path, result, depots, stations)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["status"] == "optimal"
    assert data["depot_ids"] == ["B1", "B2"]
    assert data["station_ids"] == ["S1", "S2", "S3"]
    assert data["total_cost"] == pytest.approx(75)
    np.testing.assert_allclose(data["deliveries"], result.deliveries)
    assert data["summary"]["station_remaining"] == pytest.approx([3, 3, 3])
    assert data["metrics"]["method"] == "highs"
    assert data["metrics"]["solver_seconds"] >= 0


def test_serialize_infeasible_result(scenario):
    for depot in scenario["depots"]:
        depot["stock"] = 0

    depots, stations, costs = problem_from_dict(scenario)
    result = optimize_plan(depots, stations, costs)
    data = result_to_dict(result, depots, stations)

    assert data["status"] == "infeasible"
    assert data["deliveries"] is None
    assert data["total_cost"] is None
    assert data["summary"] is None
    json.dumps(data, allow_nan=False)