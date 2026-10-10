import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from constraints import validate_data
from models import Depot, OptimizationResult, Station
from optimizer import validate_costs


def _object(value: object, fields: tuple[str, ...], label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: ожидается объект JSON.")

    expected = set(fields)
    missing = expected - value.keys()
    extra = value.keys() - expected

    if missing:
        raise ValueError(f"{label}: отсутствуют поля {', '.join(sorted(missing))}.")
    if extra:
        names = ", ".join(sorted(str(name) for name in extra))
        raise ValueError(f"{label}: неизвестные поля {names}.")

    return value


def _array(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label}: ожидается массив JSON.")
    return value


def problem_from_dict(
    data: object,
) -> tuple[list[Depot], list[Station], NDArray[np.float64]]:
    """Преобразовать данные JSON и проверить их структуру и значения."""
    data = _object(data, ("depots", "stations", "costs"), "Сценарий")

    depots = [
        Depot(**_object(item, ("id", "stock"), f"depots[{i}]"))
        for i, item in enumerate(_array(data["depots"], "depots"))
    ]
    stations = [
        Station(
            **_object(
                item,
                ("id", "remaining", "demand", "safety", "capacity"),
                f"stations[{j}]",
            )
        )
        for j, item in enumerate(_array(data["stations"], "stations"))
    ]

    validate_data(depots, stations)

    rows = _array(data["costs"], "costs")
    for i, row in enumerate(rows):
        for value in _array(row, f"costs[{i}]"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"costs[{i}]: тарифы должны быть числами.")

    costs = validate_costs(rows, (len(depots), len(stations)))
    return depots, stations, costs


def _reject_constant(value: str) -> None:
    raise ValueError(f"Недопустимое значение JSON: {value}.")


def load_problem(
    path: str | Path,
) -> tuple[list[Depot], list[Station], NDArray[np.float64]]:
    """Загрузить сценарий из файла UTF-8."""
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file, parse_constant=_reject_constant)
    return problem_from_dict(data)


def result_to_dict(
    result: OptimizationResult,
    depots: list[Depot],
    stations: list[Station],
) -> dict:
    """Подготовить результат для JSON, сохранив порядок строк и столбцов."""
    summary = None
    if result.summary is not None:
        summary = {
            name: values.tolist()
            for name, values in asdict(result.summary).items()
        }

    return {
        "status": result.status,
        "message": result.message,
        "depot_ids": [depot.id for depot in depots],
        "station_ids": [station.id for station in stations],
        "deliveries": (
            result.deliveries.tolist()
            if result.deliveries is not None else None
        ),
        "total_cost": result.total_cost,
        "summary": summary,
        "metrics": asdict(result.metrics) if result.metrics is not None else None,
    }


def save_result(
    path: str | Path,
    result: OptimizationResult,
    depots: list[Depot],
    stations: list[Station],
) -> None:
    """Сохранить результат без округления вычисленных значений."""
    text = json.dumps(
        result_to_dict(result, depots, stations),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    Path(path).write_text(text + "\n", encoding="utf-8")