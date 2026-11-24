"""Albanese 2016 表 4–5 及附录 (A42)–(A78) 的气体交换方程。"""

from dataclasses import dataclass
from math import sqrt

import numpy as np


DRY_PRESSURE_MMHG = 760.0 - 47.0
BTPS_PER_STPD = 1.2103
# 1 mmol/L 血的气体在标准温压下占 22.414 mL/L。
STPD_ML_PER_ML_PER_MMOL_L = 0.022414
INSPIRED = (0.210379, 0.000421)
TISSUE_ML = np.array([2673., 262., 31200., 284., 1300.])
METABOLISM_ML_S = np.column_stack((
    -np.array([108.419, 14.683, 51.6, 24., 47.502]),
    np.array([91.0720, 12.3337, 43.344, 20.16, 39.9017]),
)) / 60.0
ARTERIAL_TO_TISSUE_S = 18.0
VENOUS_TO_LUNG_S = 10.0
K = np.array([14.99, 194.4])
ALPHA = np.array([.03198, .05591])
BETA = np.array([.008275, .03255])
HILL = np.array([.3836, 1.819])
SATURATION = np.array([9., 86.11]) * STPD_ML_PER_ML_PER_MMOL_L


@dataclass(frozen=True)
class LungInputs:
    dead_space_l: float
    alveolar_l: float
    airway_flow_l_s: float
    alveolar_flow_l_s: float
    pulmonary_inflow_ml_s: float
    exchange_volume_ml: float
    gas_bed_outflow_ml_s: float
    shunt_outflow_ml_s: float
    shunt_fraction: float
    inspired: tuple = INSPIRED


def dissociation(partial_pressure):
    """Spencer 二元解离关系和雅可比，输出 STPD mL 气体/mL 血。"""
    p = np.asarray(partial_pressure, dtype=float)
    if p.shape != (2,) or np.any(p <= 0) or not np.isfinite(p).all():
        raise ValueError("氧气与二氧化碳分压须为有限正值")
    exponent = 1. / HILL
    other = p[::-1]
    x = p * (1. + BETA * other) / (K * (1. + ALPHA * other))
    power = x ** exponent
    fraction = power / (1. + power)
    log_slope = SATURATION * exponent * fraction * (1. - fraction)
    cross = log_slope * (BETA / (1. + BETA * other) - ALPHA / (1. + ALPHA * other))
    jacobian = np.array([[log_slope[0] / p[0], cross[0]],
                         [cross[1], log_slope[1] / p[1]]])
    return SATURATION * fraction, jacobian


def partial_pressures(concentration):
    """解析反解双气体分压，保留混合血的 Bohr/Haldane 交叉作用。"""
    c_o2, c_co2 = concentration
    max_o2, max_co2 = SATURATION
    if not (0. < c_o2 < max_o2 and 0. < c_co2 < max_co2):
        raise ValueError("血气浓度须位于解离关系的有效范围")
    a = K[0] * (c_o2 / (max_o2 - c_o2)) ** HILL[0]
    b = K[1] * (c_co2 / (max_co2 - c_co2)) ** HILL[1]
    alpha_o2, alpha_co2 = ALPHA
    beta_o2, beta_co2 = BETA
    quadratic = beta_o2 + beta_co2 * a * alpha_o2
    linear = 1. + beta_co2 * a - b * (beta_o2 + alpha_co2 * a * alpha_o2)
    constant_abs = b * (1. + alpha_co2 * a)
    discriminant = sqrt(linear * linear + 4. * quadratic * constant_abs)
    p_co2 = (2. * constant_abs / (discriminant + linear) if linear >= 0.
             else (discriminant - linear) / (2. * quadratic))
    p_o2 = a * (1. + alpha_o2 * p_co2) / (1. + beta_o2 * p_co2)
    return p_o2, p_co2


def lung_exchange(fractions, delayed_venous, inputs, *, last_arterial=None):
    """四个气体分数的导数和血液混合结果。"""
    f_dead, f_alveolar = np.asarray(fractions).reshape((2, 2))
    capillary, jacobian = dissociation(f_alveolar * DRY_PRESSURE_MMHG)
    if inputs.airway_flow_l_s > 0.0:
        d_dead = inputs.airway_flow_l_s / inputs.dead_space_l * (inputs.inspired - f_dead)
        wash = inputs.alveolar_flow_l_s * (f_dead - f_alveolar)
    else:
        d_dead = inputs.alveolar_flow_l_s / inputs.dead_space_l * (f_dead - f_alveolar)
        wash = np.zeros(2)
    perfusion = inputs.pulmonary_inflow_ml_s * (1 - inputs.shunt_fraction)
    flux = perfusion * (capillary - delayed_venous)
    gas_capacity = 1000.0 * inputs.alveolar_l * np.eye(2)
    blood_capacity = BTPS_PER_STPD * inputs.exchange_volume_ml * DRY_PRESSURE_MMHG * jacobian
    d_alveolar = np.linalg.solve(gas_capacity + blood_capacity, 1000.0 * wash - BTPS_PER_STPD * flux)
    q_capillary, q_shunt = inputs.gas_bed_outflow_ml_s, inputs.shunt_outflow_ml_s
    if last_arterial is not None:
        # 逆流支路从混合节点取血。无正向来流时延用最近一次动脉血气读数。
        q_capillary, q_shunt = max(q_capillary, 0.), max(q_shunt, 0.)
    mixture_flow = q_capillary + q_shunt
    if mixture_flow > 0.:
        arterial = (q_capillary * capillary + q_shunt * delayed_venous) / mixture_flow
    elif last_arterial is not None:
        arterial = last_arterial
    else:
        raise ValueError("进入肺静脉的气体混合血流须为正")
    return np.r_[d_dead, d_alveolar], arterial


def tissue_exchange(tissue, veins, central_venous, delayed_arterial, *,
                    peripheral_volume_ml, venous_volume_ml, central_volume_ml,
                    peripheral_inflow_ml_s, peripheral_outflow_ml_s, venous_flow_ml_s):
    """五个组织床、五个静脉床和胸内混合静脉池的浓度导数。"""
    d_tissue = (peripheral_inflow_ml_s[:, None] * (delayed_arterial - tissue)
                + METABOLISM_ML_S) / (TISSUE_ML + peripheral_volume_ml)[:, None]
    d_veins = peripheral_outflow_ml_s[:, None] * (tissue - veins) / venous_volume_ml[:, None]
    d_central = (venous_flow_ml_s[:, None] * (veins - central_venous)).sum(axis=0) / central_volume_ml
    return d_tissue, d_veins, d_central
