"""四腔泵、心包及室间隔。顺应性单位 mL/mmHg，压力相对大气。"""

from dataclasses import dataclass
from math import copysign, expm1, log1p, pi, sin

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class HeartParameters:
    """按右、左顺序保存两侧参数，原文表 I 的弹性参数为基础值。"""

    atrial_compliance: tuple = (31.25, 19.23)
    atrial_unstressed_ml: tuple = (25.0, 25.0)
    inlet_resistance: tuple = (0.0025, 0.0025)
    ventricular_viscous_gain: tuple = (0.00014, 0.000375)
    ventricular_elastance: tuple = (1.412, 2.392)
    ventricular_unstressed_ml: tuple = (35.904, 14.758)
    ventricular_exponent: tuple = (0.011, 0.014)
    ventricular_pressure_scale: tuple = (1.5, 1.5)
    septal_elastance: float = 32.4
    period_s: float = 0.85


def activation(time_s, period_s, phase=None):
    """Ursino 1998 式 (19)、(22)，用恒定心周期驱动机械试验。"""
    systole_s = 0.5 - 0.075 / period_s
    phase_s = time_s % period_s if phase is None else (phase % 1.0) * period_s
    return sin(pi * phase_s / systole_s) ** 2 if phase_s < systole_s else 0.0


def pericardial_pressure(chambers_ml, pleural_mmhg):
    """Karamolegkos 2021 式 (1) 和表 I。40 mL 为心包液量。"""
    enclosed_ml = sum(chambers_ml) + 40.0
    return pleural_mmhg + 0.5 * expm1(0.005 * (enclosed_ml - 200.0))


def wall_pressure(volume_ml, activity, elastance, unstressed_ml, exponent, pressure_scale):
    return (activity * elastance * (volume_ml - unstressed_ml)
            + (1.0 - activity) * pressure_scale * expm1(exponent * volume_ml))


def septal_volume(pressure_difference, activity, septal_elastance):
    """博士论文式 (2.3)–(2.5)，先求两种状态的容积，再按激活插值。"""
    systolic_ml = pressure_difference / septal_elastance
    diastolic_ml = copysign(log1p(abs(pressure_difference) / 1.11), pressure_difference) / .175
    return activity * systolic_ml + (1.0 - activity) * diastolic_ml


def ventricular_pressures(rv_ml, lv_ml, activity, params, pericardial_mmhg=0.0,
                          downstream=(float("inf"), float("inf"))):
    """用腔内左室减右室压力求室间隔容积。返回等容腔压、室间隔容积和容积残差。"""
    e_r, e_l = params.ventricular_elastance
    vu_r, vu_l = params.ventricular_unstressed_ml
    ke_r, ke_l = params.ventricular_exponent
    p0_r, p0_l = params.ventricular_pressure_scale

    def isometric(septal_ml):
        p_r = wall_pressure(rv_ml + septal_ml, activity, e_r, vu_r, ke_r, p0_r)
        p_l = wall_pressure(lv_ml - septal_ml, activity, e_l, vu_l, ke_l, p0_l)
        return p_r + pericardial_mmhg, p_l + pericardial_mmhg

    def residual(septal_ml):
        p_r, p_l = isometric(septal_ml)
        # Ursino 理想出口瓣开放时，黏性压降使腔内压等于下游动脉压力。
        gradient = min(p_l, downstream[1]) - min(p_r, downstream[0])
        return septal_ml - septal_volume(gradient, activity, params.septal_elastance)

    septal_ml = brentq(residual, -rv_ml, lv_ml, xtol=1e-11)
    return np.array(isometric(septal_ml)), septal_ml, residual(septal_ml)


def evaluate(time_s, chambers_ml, pleural_mmhg, pulmonary_mmhg, aortic_mmhg, parameters, phase=None):
    """输出 RA、LA、RV、LV 压力及 TV、MV、PV、AV 流量。"""
    p_peri = pericardial_pressure(chambers_ml, pleural_mmhg)
    p_atria = p_peri + ((chambers_ml[:2] - parameters.atrial_unstressed_ml)
                       / parameters.atrial_compliance)
    p_artery = np.array([pulmonary_mmhg, aortic_mmhg])
    p_iso, septal_ml, residual = ventricular_pressures(
        *chambers_ml[2:], activation(time_s, parameters.period_s, phase), parameters, p_peri, p_artery
    )
    gradient = np.maximum(p_iso - p_artery, 0.0)
    resistance = p_iso * parameters.ventricular_viscous_gain
    if np.any((gradient > 0.0) & (resistance <= 0.0)):
        raise ValueError("心室射血时阻力非正")
    q_out = np.divide(gradient, resistance, out=np.zeros(2), where=gradient > 0.0)
    p_ventricle = p_iso - resistance * q_out
    q_in = np.maximum(p_atria - p_ventricle, 0.0) / parameters.inlet_resistance
    return np.r_[p_atria, p_ventricle], np.r_[q_in, q_out], np.array([p_peri, septal_ml, residual])
