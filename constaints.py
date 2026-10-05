from math import isfinite
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from models import Depot, Station, Violation

ATOL = 1e-7
RTOL = 1e-7


def greater(left: float, right: float) -> bool:
    """Превышение с учётом погрешности вычислений."""
    tolerance = ATOL + RTOL * max(abs(left), abs(right))
    return left - right > tolerance


def validate_data(
    depots: list[Depot],
    stations: list[Station],
) -> None:
    """Ошибки исходных данных вызывают ValueError."""
    for objects, fields in (
        (depots, ("stock",)),
        (stations, ("remaining", "demand", "safety", "capacity")),
    ):
        if not objects:
            raise ValueError("Нужны хотя бы одна нефтебаза и одна АЗС.")

        ids = set()
        for obj in objects:
            if not isinstance(obj.id, str) or not obj.id.strip():
                raise ValueError("Идентификатор должен быть непустой строкой.")
            if obj.id in ids:
                raise ValueError(f"Повторяющийся идентификатор: {obj.id}")
            ids.add(obj.id)

            for field in fields:
                value = getattr(obj, field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        f"{obj.id}: {field} должно быть конечным "
                        "неотрицательным числом."
                    )

    for station in stations:
        if station.remaining > station.capacity:
            raise ValueError(
                f"{station.id}: текущий остаток превышает вместимость."
            )


def check_plan(
    depots: list[Depot],
    stations: list[Station],
    deliveries: ArrayLike,
) -> list[Violation]:
    """Проверить заданный план. Пустой список означает допустимость."""
    validate_data(depots, stations)

    try:
        raw = np.asarray(deliveries)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Ожидаются действительные числа.")
        x = raw.astype(float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Поставки должны быть числовой матрицей.") from error

    expected_shape = (len(depots), len(stations))
    if x.shape != expected_shape:
        raise ValueError(
            f"Размер матрицы должен быть {expected_shape}, получен {x.shape}."
        )
    if not np.isfinite(x).all():
        raise ValueError("Матрица содержит NaN или бесконечность.")

    try:
        with np.errstate(over="raise", invalid="raise"):
            shipped = x.sum(axis=1)   # По строкам: отгрузки нефтебаз.
            received = x.sum(axis=0)  # По столбцам: поступления на АЗС.
            after_delivery = received + np.array(
                [station.remaining for station in stations]
            )
            after_consumption = after_delivery - np.array(
                [station.demand for station in stations]
            )
    except FloatingPointError as error:
        raise ValueError("Слишком большие объёмы для расчёта.") from error

    violations = []

    # x_ij >= 0
    for i, depot in enumerate(depots):
        for j, station in enumerate(stations):
            value = float(x[i, j])
            if greater(0.0, value):
                violations.append(Violation(
                    "negative_delivery",
                    f"{depot.id} → {station.id}: отрицательная поставка.",
                    -value,
                ))

    # sum_j x_ij <= a_i
    for i, depot in enumerate(depots):
        total = float(shipped[i])
        if greater(total, depot.stock):
            violations.append(Violation(
                "depot_stock",
                f"{depot.id}: отгрузка {total:g} м³ "
                f"при запасе {depot.stock:g} м³.",
                total - depot.stock,
            ))

    for j, station in enumerate(stations):
        before = float(after_delivery[j])
        after = float(after_consumption[j])

        # r_j + sum_i x_ij - d_j >= s_j
        if greater(station.safety, after):
            violations.append(Violation(
                "station_safety",
                f"{station.id}: конечный остаток {after:g} м³ "
                f"при страховом запасе {station.safety:g} м³.",
                station.safety - after,
            ))

        # r_j + sum_i x_ij <= V_j: проверяем ДО потребления.
        if greater(before, station.capacity):
            violations.append(Violation(
                "station_capacity",
                f"{station.id}: после поставки {before:g} м³ "
                f"при вместимости {station.capacity:g} м³.",
                before - station.capacity,
            ))

    return violations