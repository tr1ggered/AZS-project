import numpy as np
import pytest

from constraints import check_plan
from models import Depot, Station
from optimizer import optimize_plan


def test_known_optimum():
    depots = [Depot("B1", 20), Depot("B2", 20)]
    stations = [
        Station("S1", 5, 12, 3, 20),
        Station("S2", 5, 17, 3, 25),
        Station("S3", 0, 12, 3, 20),
    ]

    result = optimize_plan(depots, stations, [[1, 4, 5], [5, 2, 1]])

    assert result.status == "optimal"
    assert result.total_cost == pytest.approx(75)
    np.testing.assert_allclose(result.deliveries, [[10, 10, 0], [0, 5, 15]])
    assert check_plan(depots, stations, result.deliveries) == []
    summary = result.summary
    assert summary is not None
    np.testing.assert_allclose(summary.depot_shipped, [20, 20])
    np.testing.assert_allclose(summary.depot_remaining, [0, 0])
    np.testing.assert_allclose(summary.station_received, [10, 15, 15])
    np.testing.assert_allclose(summary.station_after_delivery, [15, 20, 15])
    np.testing.assert_allclose(summary.station_remaining, [3, 3, 3])
    np.testing.assert_allclose(summary.route_costs, [[10, 40, 0], [0, 10, 15]])
    assert summary.route_costs.sum() == pytest.approx(result.total_cost)


def test_insufficient_stock():
    result = optimize_plan(
        [Depot("B1", 4)],
        [Station("S1", 0, 5, 0, 10)],
        [[2]],
    )

    assert result.status == "infeasible"
    assert result.deliveries is None
    assert result.total_cost is None
    assert result.summary is None


def test_fractional_volume():
    result = optimize_plan(
        [Depot("B1", 2)],
        [Station("S1", 0.5, 1, 0.25, 2)],
        [[2]],
    )

    assert result.status == "optimal"
    assert result.deliveries[0, 0] == pytest.approx(0.75)
    assert result.total_cost == pytest.approx(1.5)


@pytest.mark.parametrize(
    "costs",
    [
        [[-1]],
        [[float("nan")]],
        [[float("inf")]],
        [[1, 2]],
        [["тариф"]],
        [[1 + 2j]],
    ],
)
def test_invalid_costs(costs):
    with pytest.raises(ValueError):
        optimize_plan(
            [Depot("B1", 10)],
            [Station("S1", 0, 1, 0, 10)],
            costs,
        )


def test_summary_with_unused_stock():
    result = optimize_plan(
        [Depot("B1", 100), Depot("B2", 50)],
        [
            Station("S1", 3, 5, 1, 10),
            Station("S2", 4, 1, 1, 8),
        ],
        [[2, 2], [8, 8]],
    )

    assert result.status == "optimal"
    summary = result.summary
    assert summary is not None
    np.testing.assert_allclose(summary.depot_shipped, [3, 0])
    np.testing.assert_allclose(summary.depot_remaining, [97, 50])
    np.testing.assert_allclose(summary.station_received, [3, 0])
    np.testing.assert_allclose(summary.station_after_delivery, [6, 4])
    np.testing.assert_allclose(summary.station_remaining, [1, 3])
    np.testing.assert_allclose(summary.route_costs, [[6, 0], [0, 0]])
    assert result.total_cost == pytest.approx(6)


def test_summary_without_deliveries():
    result = optimize_plan(
        [Depot("B1", 0)],
        [Station("S1", 10, 2, 3, 20)],
        [[7]],
    )

    assert result.status == "optimal"
    summary = result.summary
    assert summary is not None
    np.testing.assert_allclose(result.deliveries, [[0]])
    np.testing.assert_allclose(summary.depot_shipped, [0])
    np.testing.assert_allclose(summary.depot_remaining, [0])
    np.testing.assert_allclose(summary.station_received, [0])
    np.testing.assert_allclose(summary.station_after_delivery, [10])
    np.testing.assert_allclose(summary.station_remaining, [8])
    np.testing.assert_allclose(summary.route_costs, [[0]])
    assert result.total_cost == pytest.approx(0)


@pytest.mark.parametrize("method", ["highs", "highs-ds", "highs-ipm"])
def test_method_and_metrics(method):
    result = optimize_plan(
        [Depot("B1", 2)],
        [Station("S1", 0.5, 1, 0.25, 2)],
        [[2]],
        method=method,
    )

    assert result.status == "optimal"
    assert result.total_cost == pytest.approx(1.5)
    assert result.metrics is not None
    assert result.metrics.method == method
    assert np.isfinite(result.metrics.solver_seconds)
    assert result.metrics.solver_seconds >= 0
    assert result.metrics.iterations >= 0
    assert result.metrics.crossover_iterations >= 0