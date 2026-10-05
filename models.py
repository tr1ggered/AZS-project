from dataclasses import dataclass


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