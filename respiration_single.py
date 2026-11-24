"""博士论文 §2.2 的单腔呼吸力学，采用 Vicario 等的线性串联弹性。"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SingleCompartment:
    airway_resistance_cmh2o_s_l: float = 1.7459
    lung_compliance_l_cmh2o: float = .2
    chest_wall_compliance_l_cmh2o: float = .2445
    lung_unstressed_l: float = 1.25
    chest_wall_unstressed_l: float = 3.4725

    def __post_init__(self):
        if (not all(isfinite(value) and value > 0 for value in (
                self.airway_resistance_cmh2o_s_l, self.lung_compliance_l_cmh2o,
                self.chest_wall_compliance_l_cmh2o))
                or not all(isfinite(value) for value in (self.lung_unstressed_l, self.chest_wall_unstressed_l))):
            raise ValueError("气道阻力与顺应性须为有限正值，无应力容积须有限")

    def equilibrium_volume(self, airway_cmh2o=0.0, muscle_cmh2o=0.0):
        lung_elastance = 1.0 / self.lung_compliance_l_cmh2o
        chest_elastance = 1.0 / self.chest_wall_compliance_l_cmh2o
        return (airway_cmh2o - muscle_cmh2o + self.lung_unstressed_l * lung_elastance
                + self.chest_wall_unstressed_l * chest_elastance) / (lung_elastance + chest_elastance)

    def evaluate(self, volume_l, airway_cmh2o, muscle_cmh2o):
        """返回容积导数/气流（L/s）、肺泡压和胸膜压（cmH₂O）。"""
        p_pl = muscle_cmh2o + (volume_l - self.chest_wall_unstressed_l) / self.chest_wall_compliance_l_cmh2o
        p_alveolar = p_pl + (volume_l - self.lung_unstressed_l) / self.lung_compliance_l_cmh2o
        flow = (airway_cmh2o - p_alveolar) / self.airway_resistance_cmh2o_s_l
        return flow, p_alveolar, p_pl
