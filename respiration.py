"""Albanese 等（2016）原文式 (4)–(5)、(A30)–(A41) 的呼吸力学。"""

from dataclasses import dataclass
from math import exp

import numpy as np


CMH2O_TO_MMHG = 0.73555924


@dataclass(frozen=True)
class MechanicsParameters:
    """表 3 的阻力、顺应性、无应力气量和呼气末胸膜压。"""

    resistance_cmh2o_s_l: tuple = (1.021, 0.3369, 0.3063, 0.0817)
    compliance_l_cmh2o: tuple = (0.00127, 0.00238, 0.0131, 0.2)
    unstressed_volume_l: tuple = (0.0344, 0.00663, 0.0187, 1.263)
    chest_wall_compliance_l_cmh2o: float = 0.2445
    pleural_end_expiratory_cmh2o: float = -5.0


ALBANESE_2016 = MechanicsParameters()


def quiet_muscle_pressure(time_s, breaths_per_minute=12.0, minimum_cmh2o=-5.0):
    """表 3 的自主呼吸肌压输入。"""
    period = 60.0 / breaths_per_minute
    t = time_s % period
    expiration = period / 1.6
    inspiration = period - expiration
    if t <= inspiration:
        return minimum_cmh2o * t * (period - t) / (inspiration * expiration)
    tau = expiration / 5.0
    end = exp(-expiration / tau)
    return minimum_cmh2o * (exp(-(t - inspiration) / tau) - end) / (1.0 - end)


def end_expiratory_state(parameters=ALBANESE_2016):
    """零气道压和零肌压下的呼气末气量。"""
    p_pl = parameters.pleural_end_expiratory_cmh2o
    volumes = np.asarray(parameters.unstressed_volume_l) - np.array([0, p_pl, p_pl, p_pl]) * (
        parameters.compliance_l_cmh2o
    )
    return np.r_[volumes, 0.0]


def evaluate(state, airway_opening_cmh2o, muscle_cmh2o, parameters=ALBANESE_2016):
    """状态为四腔气量和胸腔气量变化，压力量均为相对大气压。"""
    state = np.asarray(state, dtype=float)
    p_pl = (parameters.pleural_end_expiratory_cmh2o
            + state[4] / parameters.chest_wall_compliance_l_cmh2o + muscle_cmh2o)
    pressure = (state[:4] - parameters.unstressed_volume_l) / parameters.compliance_l_cmh2o
    pressure[1:] += p_pl
    flow = -np.diff(np.r_[airway_opening_cmh2o, pressure]) / parameters.resistance_cmh2o_s_l
    derivative = np.r_[flow[:-1] - flow[1:], flow[-1], flow[1]]
    return derivative, pressure, p_pl, flow


def solve_drive(duration_s, *, airway_at, muscle_at, parameters=ALBANESE_2016):
    """求解气道与胸壁力学，并返回可供肺循环取样的连续呼吸输入。"""
    from scipy.integrate import solve_ivp

    def rhs(time_s, state):
        return evaluate(state, airway_at(time_s), muscle_at(time_s), parameters)[0]

    solution = solve_ivp(rhs, (0.0, duration_s), end_expiratory_state(parameters),
                         method="Radau", rtol=1e-9, atol=1e-11, dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)

    def drive_at(time_s):
        state = solution.sol(time_s)
        _, pressure, p_pl, _ = evaluate(state, airway_at(time_s), muscle_at(time_s), parameters)
        return p_pl * CMH2O_TO_MMHG, pressure[-1] * CMH2O_TO_MMHG, state[3]

    return drive_at
