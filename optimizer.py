from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

from constraints import build_constraints, check_feasibility, check_plan
from models import (
    Depot,
    OptimizationResult,
    PlanSummary,
    SolverMetrics,
    Station,
)


def validate_costs(
    costs: ArrayLike,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Проверить матрицу тарифов и вернуть массив действительных чисел."""
    try:
        raw = np.asarray(costs)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Ожидаются действительные числа.")
        matrix = raw.astype(float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Тарифы должны быть числовой матрицей.") from error

    if matrix.shape != shape:
        raise ValueError(
            f"Размер матрицы тарифов должен быть {shape}, "
            f"получен {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Тарифы содержат NaN или бесконечность.")
    if np.any(matrix < 0):
        raise ValueError("Тарифы не могут быть отрицательными.")

    return matrix


def optimize_plan(
    depots: list[Depot],
    stations: list[Station],
    costs: ArrayLike,
    method: str = "highs",
) -> OptimizationResult:
    """Найти план с минимальной стоимостью и проверить ограничения."""
    if method not in ("highs", "highs-ds", "highs-ipm"):
        raise ValueError("Допустимые методы: highs, highs-ds, highs-ipm.")

    report = check_feasibility(depots, stations)
    matrix = validate_costs(costs, (len(depots), len(stations)))

    if not report.feasible:
        return OptimizationResult(
            status="infeasible",
            message=" ".join(v.message for v in report.violations),
        )

    system = build_constraints(depots, stations)
    started = perf_counter()
    result = linprog(
        c=matrix.ravel(order="C"),
        A_ub=system.A_ub,
        b_ub=system.b_ub,
        bounds=system.bounds,
        method=method,
    )
    solver_seconds = perf_counter() - started

    metrics = SolverMetrics(
        method=method,
        solver_seconds=solver_seconds,
        iterations=int(result.nit),
        crossover_iterations=int(result.crossover_nit),
    )

    if result.status == 2:
        return OptimizationResult(
            status="infeasible",
            metrics=metrics,
            message="Решатель сообщил об отсутствии допустимого плана.",
        )
    if result.status == 1:
        return OptimizationResult(
            status="limit",
            metrics=metrics,
            message="Расчёт остановлен по лимиту. Оптимум не подтверждён.",
        )
    if result.status != 0 or not result.success:
        return OptimizationResult(
            status="error",
            metrics=metrics,
            message=f"Ошибка решателя: {result.message}",
        )

    deliveries = result.x.reshape(matrix.shape, order="C")

    # Независимая проверка найденного плана по исходным формулам.
    try:
        violations = check_plan(depots, stations, deliveries)
    except ValueError as error:
        return OptimizationResult(
            status="error",
            metrics=metrics,
            message=f"Ошибка проверки результата: {error}",
        )

    if violations:
        return OptimizationResult(
            status="error",
            metrics=metrics,
            message="Найденный план нарушает ограничения: "
                    + " ".join(v.message for v in violations),
        )

    # Пересчитываем стоимость и показатели по проверенному плану.
    try:
        with np.errstate(over="raise", invalid="raise"):
            route_costs = matrix * deliveries
            total_cost = float(route_costs.sum())

            depot_shipped = deliveries.sum(axis=1)
            depot_remaining = np.array(
                [depot.stock for depot in depots], dtype=float
            ) - depot_shipped

            station_received = deliveries.sum(axis=0)
            station_after_delivery = np.array(
                [station.remaining for station in stations], dtype=float
            ) + station_received
            station_remaining = station_after_delivery - np.array(
                [station.demand for station in stations], dtype=float
            )
    except FloatingPointError:
        return OptimizationResult(
            status="error",
            metrics=metrics,
            message="Численное переполнение при формировании сводки.",
        )

    if not np.isclose(total_cost, result.fun, rtol=1e-7, atol=1e-7):
        return OptimizationResult(
            status="error",
            metrics=metrics,
            message="Пересчитанная стоимость не совпала с ответом решателя.",
        )

    summary = PlanSummary(
        depot_shipped=depot_shipped,
        depot_remaining=depot_remaining,
        station_received=station_received,
        station_after_delivery=station_after_delivery,
        station_remaining=station_remaining,
        route_costs=route_costs,
    )

    return OptimizationResult(
        status="optimal",
        message="Оптимальный план найден и проверен.",
        deliveries=deliveries,
        total_cost=total_cost,
        summary=summary,
        metrics=metrics,
    )