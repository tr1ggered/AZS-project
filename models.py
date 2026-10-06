from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Depot:
    id: str
    stock: float  # a_i: доступный запас, м³


@dataclass(frozen=True)
class Station:
    id: str
    remaining: float  # r_j: текущий остаток
    demand: float     # d_j: потребление за период
    safety: float     # s_j: страховой запас
    capacity: float   # V_j: вместимость резервуара


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    amount: float  # Величина нарушения, м³


@dataclass(frozen=True)
class FeasibilityReport:
    station_ids: tuple[str, ...]
    minimum_delivery: NDArray[np.float64]  # q_j = max(0, d_j + s_j - r_j)
    available_capacity: NDArray[np.float64]  # u_j = V_j - r_j
    total_stock: float
    total_required: float
    violations: tuple[Violation, ...]

    @property
    def feasible(self) -> bool:
        """Условия выполнимости пройдены с учётом численного допуска."""
        return not self.violations


@dataclass(frozen=True)
class LinearConstraints:
    A_ub: NDArray[np.float64]
    b_ub: NDArray[np.float64]
    bounds: tuple[tuple[float, float | None], ...]
    variable_ids: tuple[tuple[str, str], ...]  # (нефтебаза, АЗС) для столбцов
    row_labels: tuple[str, ...]
