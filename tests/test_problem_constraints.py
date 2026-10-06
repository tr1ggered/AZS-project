import numpy as np
import pytest

from constraints import build_constraints, check_feasibility, check_plan
from models import Depot, Station


@pytest.fixture
def example():
    return (
        [Depot("B1", 20), Depot("B2", 20)],
        [
            Station("S1", 5, 12, 3, 20),
            Station("S2", 5, 17, 3, 25),
            Station("S3", 0, 12, 3, 20),
        ],
    )


def test_feasible_example(example):
    report = check_feasibility(*example)
    assert report.feasible
    assert report.station_ids == ("S1", "S2", "S3")
    np.testing.assert_array_equal(report.minimum_delivery, [10, 15, 15])
    np.testing.assert_array_equal(report.available_capacity, [15, 20, 20])
    assert report.total_stock == report.total_required == 40


def test_total_stock_shortage(example):
    _, stations = example
    report = check_feasibility([Depot("B1", 39)], stations)
    assert not report.feasible
    assert [v.code for v in report.violations] == ["insufficient_stock"]
    assert report.violations[0].amount == 1


def test_capacity_shortage_before_consumption():
    report = check_feasibility(
        [Depot("B1", 100)], [Station("S1", 8, 10, 1, 10)]
    )
    assert not report.feasible
    assert [v.code for v in report.violations] == ["insufficient_capacity"]
    assert report.violations[0].amount == 1
    assert "S1" in report.violations[0].message


def test_both_shortages_reported():
    report = check_feasibility(
        [Depot("B1", 0)], [Station("S1", 0, 3, 0, 2)]
    )
    assert [v.code for v in report.violations] == [
        "insufficient_capacity", "insufficient_stock"
    ]
    assert [v.amount for v in report.violations] == [1, 3]


def test_surplus_at_one_station_cannot_supply_another():
    report = check_feasibility(
        [Depot("B1", 0)],
        [Station("S1", 10, 0, 0, 10), Station("S2", 0, 5, 0, 10)],
    )
    np.testing.assert_array_equal(report.minimum_delivery, [0, 5])
    assert report.total_required == 5
    assert not report.feasible


@pytest.mark.parametrize(
    "stock, station, minimum, room",
    [
        (0, Station("S1", 10, 5, 3, 10), 0, 0),
        (0, Station("S1", 0, 0, 0, 0), 0, 0),
        (2.5, Station("S1", 0.5, 2, 1, 3), 2.5, 2.5),
    ],
)
def test_no_delivery_and_exact_fractional_limits(stock, station, minimum, room):
    report = check_feasibility([Depot("B1", stock)], [station])
    assert report.feasible
    assert report.minimum_delivery[0] == minimum
    assert report.available_capacity[0] == room


def test_feasibility_tolerance():
    report = check_feasibility(
        [Depot("B1", 10)], [Station("S1", 0, 10 + 1e-8, 0, 10)]
    )
    assert report.feasible  # Допуск не изменяет исходные значения.
    assert report.minimum_delivery[0] > 10


def test_matrix_coefficients_and_order(example):
    system = build_constraints(*example)
    np.testing.assert_array_equal(system.A_ub, [
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [-1, 0, 0, -1, 0, 0],
        [0, -1, 0, 0, -1, 0],
        [0, 0, -1, 0, 0, -1],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
    ])
    np.testing.assert_array_equal(system.b_ub, [20, 20, -10, -15, -15, 15, 20, 20])
    assert system.bounds == ((0.0, None),) * 6
    assert system.variable_ids == (
        ("B1", "S1"), ("B1", "S2"), ("B1", "S3"),
        ("B2", "S1"), ("B2", "S2"), ("B2", "S3"),
    )
    assert system.row_labels == (
        "depot_stock:B1", "depot_stock:B2",
        "station_safety:S1", "station_safety:S2", "station_safety:S3",
        "station_capacity:S1", "station_capacity:S2", "station_capacity:S3",
    )


def test_surplus_preserves_original_inequality():
    system = build_constraints(
        [Depot("B1", 0)], [Station("S1", 10, 2, 3, 20)]
    )
    np.testing.assert_array_equal(system.b_ub, [0, 5, 10])
    assert np.all(system.A_ub @ np.array([0.0]) <= system.b_ub)


def test_infeasible_problem_still_has_matrices():
    system = build_constraints(
        [Depot("B1", 1)], [Station("S1", 0, 2, 0, 3)]
    )
    np.testing.assert_array_equal(system.b_ub, [1, -2, 3])


@pytest.mark.parametrize("m,n", [(1, 1), (1, 3), (3, 1), (2, 3)])
def test_matrix_agrees_with_plan_checker(m, n):
    depots = [Depot(f"B{i}", n + 1) for i in range(m)]
    stations = [Station(f"S{j}", 0, 1, 0, m + 1) for j in range(n)]
    system = build_constraints(depots, stations)
    rng = np.random.default_rng(42)
    plans = [np.ones((m, n)), np.zeros((m, n))]
    plans.extend(rng.integers(-1, 4, size=(40, m, n)).astype(float))
    for plan in plans:
        vector = plan.ravel(order="C")
        # Целые значения исключают неоднозначность около численного допуска.
        accepted = bool(np.all(system.A_ub @ vector <= system.b_ub))
        accepted = accepted and all(
            value >= lower and (upper is None or value <= upper)
            for value, (lower, upper) in zip(vector, system.bounds)
        )
        assert accepted == (check_plan(depots, stations, plan) == [])


@pytest.mark.parametrize("function", [check_feasibility, build_constraints])
@pytest.mark.parametrize(
    "depots,stations",
    [
        ([], [Station("S1", 0, 0, 0, 10)]),
        ([Depot("B1", 10)], []),
        ([Depot("B1", -1)], [Station("S1", 0, 0, 0, 10)]),
        ([Depot("B1", float("inf"))], [Station("S1", 0, 0, 0, 10)]),
        ([Depot("B1", 10)], [Station("S1", 0, float("nan"), 0, 10)]),
        ([Depot("B1", 10)], [Station("S1", 11, 0, 0, 10)]),
        ([Depot("B1", 10), Depot("B1", 10)], [Station("S1", 0, 0, 0, 10)]),
        ([Depot("B1", 10)], [Station("S1", 0, 0, 0, 10)] * 2),
        ([Depot("B1", 10)], [Station("", 0, 0, 0, 10)]),
        ([Depot("B1", True)], [Station("S1", 0, 0, 0, 10)]),
        ([Depot("B1", 10**400)], [Station("S1", 0, 0, 0, 10)]),
    ],
)
def test_new_functions_validate_inputs(function, depots, stations):
    with pytest.raises(ValueError):
        function(depots, stations)


@pytest.mark.parametrize("function", [check_feasibility, build_constraints])
def test_limit_overflow_is_input_error(function):
    with pytest.raises(ValueError, match="Слишком большие"):
        function([Depot("B1", 1)], [Station("S1", 0, 1e308, 1e308, 1e308)])


def test_total_overflow_is_input_error():
    with pytest.raises(ValueError, match="Слишком большие"):
        check_feasibility(
            [Depot("B1", 1e308), Depot("B2", 1e308)],
            [Station("S1", 0, 0, 0, 1)],
        )


def test_numpy_integer_inputs_do_not_overflow():
    value = np.int64(6_000_000_000_000_000_000)
    depots = [Depot("B1", value)]
    stations = [Station("S1", value, value, value, value)]
    report = check_feasibility(depots, stations)
    assert not report.feasible
    assert report.minimum_delivery[0] == float(value)
    system = build_constraints(depots, stations)
    assert system.b_ub[1] == -float(value)
