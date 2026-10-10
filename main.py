import argparse
from statistics import median
from time import perf_counter
from models import Depot, Station
from optimizer import optimize_plan


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
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs должен быть положительным числом")

    depots = [
        Depot(id="B1", stock=120),
        Depot(id="B2", stock=100),
        Depot(id="B3", stock=110),
        Depot(id="B4", stock=90),
        Depot(id="B5", stock=80),
    ]

    stations = [
        Station(id="S1", remaining=10, demand=30, safety=5, capacity=60),
        Station(id="S2", remaining=15, demand=45, safety=8, capacity=80),
        Station(id="S3", remaining=8, demand=25, safety=5, capacity=50),
        Station(id="S4", remaining=20, demand=50, safety=10, capacity=90),
        Station(id="S5", remaining=12, demand=35, safety=6, capacity=65),
        Station(id="S6", remaining=25, demand=40, safety=8, capacity=75),
        Station(id="S7", remaining=5, demand=30, safety=5, capacity=55),
        Station(id="S8", remaining=18, demand=55, safety=10, capacity=95),
        Station(id="S9", remaining=10, demand=28, safety=6, capacity=55),
        Station(id="S10", remaining=22, demand=42, safety=8, capacity=80),
        Station(id="S11", remaining=15, demand=38, safety=7, capacity=70),
        Station(id="S12", remaining=6, demand=22, safety=5, capacity=45),
        Station(id="S13", remaining=30, demand=20, safety=5, capacity=70),
        Station(id="S14", remaining=12, demand=48, safety=9, capacity=90),
        Station(id="S15", remaining=8, demand=32, safety=6, capacity=60),
    ]

    # Строки: B1–B5. Столбцы: S1–S15. Тарифы в руб./м³.
    costs = [
        [220, 250, 300, 500, 550, 600, 700, 750, 800, 850, 900, 950, 650, 720, 780],
        [380, 320, 260, 230, 280, 350, 550, 600, 650, 700, 750, 800, 520, 590, 660],
        [650, 600, 550, 450, 380, 300, 240, 280, 330, 480, 520, 600, 380, 450, 520],
        [850, 800, 750, 680, 600, 520, 400, 330, 260, 220, 270, 350, 500, 360, 420],
        [920, 880, 830, 780, 720, 650, 580, 520, 460, 400, 330, 250, 300, 220, 240],
    ]

    try:
        if args.compare:
            compare_methods(depots, stations, costs, args.runs)
            return

        result = optimize_plan(depots, stations, costs, method=args.method)
    except ValueError as error:
        print(f"Ошибка входных данных: {error}")
        return

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