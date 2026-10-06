from math import isfinite
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from models import Depot, FeasibilityReport, LinearConstraints, Station, Violation

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
                try:
                    finite = isfinite(value) if isinstance(value, Real) else False
                except OverflowError:
                    finite = False
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not finite
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
                [station.remaining for station in stations], dtype=float
            )
            after_consumption = after_delivery - np.array(
                [station.demand for station in stations], dtype=float
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


def _station_limits(stations: list[Station]) -> tuple[np.ndarray, np.ndarray]:
    """Границы поступлений; вызывается после validate_data."""
    remaining = np.array([s.remaining for s in stations], dtype=float)
    demand = np.array([s.demand for s in stations], dtype=float)
    safety = np.array([s.safety for s in stations], dtype=float)
    capacity = np.array([s.capacity for s in stations], dtype=float)
    try:
        with np.errstate(over="raise", invalid="raise"):
            required = demand + safety - remaining
            available = capacity - remaining
    except FloatingPointError as error:
        raise ValueError("Слишком большие объёмы для расчёта границ.") from error
    return required, available


def check_feasibility(
    depots: list[Depot],
    stations: list[Station],
) -> FeasibilityReport:
    """Проверить выполнимость без заданного плана, с численным допуском.

    Условия достаточны только при доступности всех направлений и отсутствии
    дополнительных ограничений транспорта. План поставок здесь не строится.
    """
    validate_data(depots, stations)
    required, available = _station_limits(stations)
    minimum = np.maximum(required, 0.0)
    try:
        with np.errstate(over="raise", invalid="raise"):
            total_stock = float(np.sum([d.stock for d in depots], dtype=float))
            total_required = float(minimum.sum())
    except FloatingPointError as error:
        raise ValueError("Слишком большие объёмы для суммирования.") from error

    violations = []
    for j, station in enumerate(stations):
        need, room = float(minimum[j]), float(available[j])
        if greater(need, room):
            violations.append(Violation(
                "insufficient_capacity",
                f"{station.id}: требуется поставить минимум {need:g} м³, "
                f"свободно только {room:g} м³.",
                need - room,
            ))

    if greater(total_required, total_stock):
        violations.append(Violation(
            "insufficient_stock",
            f"Требуется {total_required:g} м³, "
            f"общий запас нефтебаз — {total_stock:g} м³.",
            total_required - total_stock,
        ))

    return FeasibilityReport(
        station_ids=tuple(s.id for s in stations),
        minimum_delivery=minimum,
        available_capacity=available,
        total_stock=total_stock,
        total_required=total_required,
        violations=tuple(violations),
    )


def build_constraints(
    depots: list[Depot],
    stations: list[Station],
) -> LinearConstraints:
    """Сформировать A_ub @ x <= b_ub и границы x >= 0.

    Вектор x разворачивается по строкам: x11, ..., x1n, x21, ..., xmn.
    Блоки строк: запасы нефтебаз, минимум на АЗС, вместимость на АЗС.
    Матрицы строятся и для корректно заданной, но невыполнимой задачи.
    """
    validate_data(depots, stations)
    required, available = _station_limits(stations)
    m, n = len(depots), len(stations)
    A_ub = np.zeros((m + 2 * n, m * n), dtype=float)

    for i in range(m):
        A_ub[i, i * n:(i + 1) * n] = 1.0
    for j in range(n):
        A_ub[m + j, j::n] = -1.0
        A_ub[m + n + j, j::n] = 1.0

    # Сохраняем исходное r-d-s, даже если минимальная поставка равна нулю.
    b_ub = np.concatenate((
        np.array([d.stock for d in depots], dtype=float),
        -required,
        available,
    ))
    return LinearConstraints(
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=tuple((0.0, None) for _ in range(m * n)),
        variable_ids=tuple((d.id, s.id) for d in depots for s in stations),
        row_labels=tuple(
            [f"depot_stock:{d.id}" for d in depots]
            + [f"station_safety:{s.id}" for s in stations]
            + [f"station_capacity:{s.id}" for s in stations]
        ),
    )
