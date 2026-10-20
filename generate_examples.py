import json
from copy import deepcopy
from pathlib import Path

from constraints import check_feasibility
from serialization import problem_from_dict


def build_examples(base: dict) -> dict[str, dict]:
    names = (
        "balanced.json",
        "shortage.json",
        "surplus.json",
        "capacity_conflict.json",
        "no_delivery.json",
    )
    cases = {name: deepcopy(base) for name in names}

    stocks_by_case = {
        "balanced.json": [100, 100, 100, 80, 52],
        "shortage.json": [80, 70, 60, 50, 40],
        "surplus.json": [200, 200, 200, 200, 200],
    }

    for name, stocks in stocks_by_case.items():
        for depot, stock in zip(cases[name]["depots"], stocks, strict=True):
            depot["stock"] = stock

    station = next(
        item
        for item in cases["capacity_conflict.json"]["stations"]
        if item["id"] == "S1"
    )
    station["capacity"] = 30

    for station in cases["no_delivery.json"]["stations"]:
        station["remaining"] = max(
            station["remaining"],
            station["demand"] + station["safety"],
        )

    return cases


def main() -> None:
    directory = Path(__file__).resolve().parent / "data"
    base = json.loads((directory / "example.json").read_text(encoding="utf-8"))
    problem_from_dict(base)

    for name, scenario in build_examples(base).items():
        depots, stations, _ = problem_from_dict(scenario)
        report = check_feasibility(depots, stations)
        path = directory / name

        if path.exists():
            print(f"{name}: уже существует, пропущен.")
            continue

        with path.open("x", encoding="utf-8") as file:
            json.dump(
                scenario, file,
                ensure_ascii=False, indent=2, allow_nan=False,
            )
            file.write("\n")

        status = "выполнима" if report.feasible else "невыполнима"
        print(
            f"{name}: {status}; запас {report.total_stock:g} м³; "
            f"минимальные поставки {report.total_required:g} м³."
        )


if __name__ == "__main__":
    main()