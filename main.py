from constaints import check_plan
from models import Depot, Station


def main() -> None:
    depots = [
        Depot(id="B1", stock=20),
        Depot(id="B2", stock=20),
    ]
    stations = [
        Station(id="S1", remaining=5, demand=12, safety=3, capacity=20),
        Station(id="S2", remaining=5, demand=17, safety=3, capacity=25),
        Station(id="S3", remaining=0, demand=12, safety=3, capacity=20),
    ]

    # Строки соответствуют depots, столбцы — stations.
    deliveries = [
        [10, 10, 0],
        [0, 5, 15],
    ]

    try:
        violations = check_plan(depots, stations, deliveries)
    except ValueError as error:
        print(f"Ошибка входных данных: {error}")
        return

    if not violations:
        print("План допустим: все ограничения выполнены.")
    else:
        for violation in violations:
            print(
                f"[{violation.code}] {violation.message} "
                f"Величина нарушения: {violation.amount:g} м³."
            )


if __name__ == "__main__":
    main()