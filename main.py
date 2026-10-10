import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from optimizer import optimize_plan
from serialization import load_problem, save_result


def compare_methods(depots, stations, costs, runs: int) -> None:
    methods = ("highs", "highs-ds", "highs-ipm")
    samples = {method: [] for method in methods}

    # Прогрев не входит в измеряемые запуски.
    for method in methods:
        optimize_plan(depots, stations, costs, method=method)

    for run in range(runs):
        # Меняем порядок, чтобы один режим не был всегда первым.
        offset = run % len(methods)
        order = methods[offset:] + methods[:offset]
        for method in order:
            started = perf_counter()
            result = optimize_plan(depots, stations, costs, method=method)
            total_seconds = perf_counter() - started
            samples[method].append((result, total_seconds))

    print(f"Сравнение режимов: {runs} запусков после прогрева.")
    print(
        f"{'Режим':<11} {'Статус':<11} {'Успех':<7} {'Цена, руб.':>11} "
        f"{'Решатель, мс':>13} {'Весь вызов, мс':>15} "
        f"{'Итер.':>7} {'Cross.':>7}"
    )

    for method in methods:
        records = samples[method]
        successful = [
            result for result, _ in records
            if result.status == "optimal" and result.metrics is not None
        ]
        total_ms = median(seconds for _, seconds in records) * 1000

        if not successful:
            status = records[-1][0].status
            print(
                f"{method:<11} {status:<11} {f'0/{runs}':<7} {'—':>11} "
                f"{'—':>13} {total_ms:>15.3f} {'—':>7} {'—':>7}"
            )
            continue

        cost = median(result.total_cost for result in successful)
        solver_ms = median(
            result.metrics.solver_seconds for result in successful
        ) * 1000
        iterations = median(result.metrics.iterations for result in successful)
        crossover = median(
            result.metrics.crossover_iterations for result in successful
        )
        status = "optimal" if len(successful) == runs else "mixed"
        success = f"{len(successful)}/{runs}"

        print(
            f"{method:<11} {status:<11} {success:<7} {cost:>11.6g} "
            f"{solver_ms:>13.3f} {total_ms:>15.3f} "
            f"{iterations:>7g} {crossover:>7g}"
        )

    print("В таблице показаны медианы. Столбец «Успех» — проверенные оптимумы.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оптимизация поставок топлива")
    parser.add_argument(
        "--method",
        choices=("highs", "highs-ds", "highs-ipm"),
        default="highs",
        help="Режим решателя",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Сравнить все три режима на одинаковых данных",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Количество измеряемых запусков каждого режима",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "example.json",
        help="JSON-файл с исходными данными",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Сохранить результат одиночного расчёта в JSON",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs должен быть положительным числом")
    if args.compare and args.output is not None:
        parser.error("--output используется только для одиночного расчёта")
    if args.output is not None and args.output.resolve() == args.input.resolve():
        parser.error("--input и --output должны быть разными файлами")

    try:
        depots, stations, costs = load_problem(args.input)

        if args.compare:
            compare_methods(depots, stations, costs, args.runs)
            return

        result = optimize_plan(depots, stations, costs, method=args.method)
    except (OSError, ValueError) as error:
        print(f"Ошибка входных данных: {error}")
        return

    if args.output is not None:
        try:
            save_result(args.output, result, depots, stations)
        except (OSError, ValueError) as error:
            print(f"Ошибка сохранения результата: {error}")
        else:
            print(f"Результат сохранён: {args.output}")

    print(result.message)
    if result.metrics is not None:
        metrics = result.metrics
        print(f"Режим решателя: {metrics.method}")
        print(f"Время вызова решателя: {metrics.solver_seconds * 1000:.3f} мс")
        print(
            f"Итерации: {metrics.iterations}; "
            f"crossover: {metrics.crossover_iterations}"
        )
    if result.status != "optimal":
        print(f"Статус: {result.status}")
        return

    summary = result.summary
    if summary is None:
        print("Ошибка: у оптимального результата отсутствует сводка.")
        return

    print("План поставок, м³:")
    print(result.deliveries)
    print(f"Общая стоимость: {result.total_cost:g} руб.")

    print("\nНефтебазы:")
    for i, depot in enumerate(depots):
        print(
            f"{depot.id}: запас {depot.stock:g} м³; "
            f"отгружено {summary.depot_shipped[i]:g} м³; "
            f"осталось {summary.depot_remaining[i]:g} м³."
        )

    print("\nАЗС:")
    for j, station in enumerate(stations):
        print(
            f"{station.id}: начальный остаток {station.remaining:g} м³; "
            f"поступило {summary.station_received[j]:g} м³; "
            f"после поставки {summary.station_after_delivery[j]:g} м³; "
            f"после потребления {summary.station_remaining[j]:g} м³."
        )

    print("\nИспользуемые направления:")
    for i, depot in enumerate(depots):
        for j, station in enumerate(stations):
            volume = result.deliveries[i, j]
            if volume <= 0:
                continue

            print(
                f"{depot.id} → {station.id}: "
                f"{volume:g} м³; "
                f"{summary.route_costs[i, j]:g} руб."
            )
            
if __name__ == "__main__":
    main()