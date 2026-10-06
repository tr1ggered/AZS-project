import pytest

from constraints import check_plan
from models import Depot, Station


@pytest.mark.parametrize(
    "stock, remaining, demand, safety, capacity, delivery, code, amount",
    [
        (20, 5, 12, 3, 20, 10, None, 0),          # Допустимый план.
        (5, 0, 0, 0, 20, 6, "depot_stock", 1),
        (20, 2, 5, 3, 20, 4, "station_safety", 2),
        (20, 8, 10, 0, 10, 3, "station_capacity", 1),
        (20, 5, 0, 0, 20, -1, "negative_delivery", 1),
        (10, 0, 8, 2, 10, 10, None, 0),          # Точные границы.
        (0, 10, 5, 3, 20, 0, None, 0),           # Пополнение не нужно.
        (2.5, 0.5, 2, 1, 5, 2.5, None, 0),      # Дробные объёмы.
        (10, 0, 0, 0, 20, 10 + 1e-8, None, 0),  # Численная погрешность.
    ],
)
def test_constraints(
    stock, remaining, demand, safety, capacity, delivery, code, amount
):
    depots = [Depot("B1", stock)]
    stations = [Station("S1", remaining, demand, safety, capacity)]

    result = check_plan(depots, stations, [[delivery]])

    if code is None:
        assert result == []
    else:
        assert len(result) == 1
        assert result[0].code == code
        assert result[0].amount == pytest.approx(amount)


@pytest.mark.parametrize(
    "deliveries",
    [
        [[1, 2]],       # Неверный размер.
        [[float("nan")]],
        [[float("inf")]],
        [["топливо"]],
        [[1 + 2j]],
    ],
)
def test_invalid_matrix(deliveries):
    with pytest.raises(ValueError):
        check_plan(
            [Depot("B1", 20)],
            [Station("S1", 0, 0, 0, 20)],
            deliveries,
        )


def test_invalid_initial_stock():
    with pytest.raises(ValueError):
        check_plan(
            [Depot("B1", 20)],
            [Station("S1", 21, 0, 0, 20)],
            [[0]],
        )


def test_multiple_depots_and_stations():
    depots = [Depot("B1", 20), Depot("B2", 20)]
    stations = [
        Station("S1", 5, 12, 3, 20),
        Station("S2", 5, 17, 3, 25),
        Station("S3", 0, 12, 3, 20),
    ]

    assert check_plan(depots, stations, [[10, 10, 0], [0, 5, 15]]) == []

    # Общего топлива по-прежнему хватает, но S3 недополучает 1 м³.
    result = check_plan(depots, stations, [[10, 10, 0], [0, 6, 14]])
    assert [item.code for item in result] == ["station_safety"]
    assert result[0].amount == pytest.approx(1)
