"""按博士论文 §2.2 的内脏/内脏外拓扑实现体循环。"""

from dataclasses import dataclass
from math import exp, expm1

import numpy as np

# 顺序为内脏、内脏外、骨骼肌、心脏、脑。
VENOUS_C = np.array([42.777, 14.0, 10.997, 2.499, 7.497])
VENOUS_VU = np.array([1435.4, 640.73, 503.26, 98.21, 294.64])
PERIPHERAL_R = np.array([2.49, 1.655, 2.106, 19.71, 6.6667])
VENOUS_R = np.array([.038, .04, .05, .224, .075])
BED_C = np.array([1.1532, 1.0788, .8184, .1488, .5208])
BED_VU = np.array([274.4, 134.64, 105.8, 24.0, 72.13])
TOTAL_BLOOD_ML = 5300.0


def parallel_sum(values):
    """保留内脏床，汇总内脏外、骨骼肌、心脏和脑。"""
    return (float(values[0]), float(sum(values[1:])))


def parallel_resistance(values):
    return tuple(1.0 / value for value in parallel_sum(1.0 / values))


@dataclass(frozen=True)
class TwoBedSystemic:
    # 论文明确连接方式。未给合并参数表时以 Albanese 表 1 的并联值重建。
    peripheral_compliance: tuple = parallel_sum(BED_C)
    peripheral_unstressed_ml: tuple = parallel_sum(BED_VU)
    peripheral_resistance: tuple = parallel_resistance(PERIPHERAL_R)
    venous_compliance: tuple = parallel_sum(VENOUS_C)
    venous_unstressed_ml: tuple = parallel_sum(VENOUS_VU)
    venous_resistance: tuple = parallel_resistance(VENOUS_R)


NOMINAL = TwoBedSystemic()


def thoracic_vein(volume_ml, pleural_mmhg):
    if volume_ml <= 0.0:
        raise ValueError("胸内静脉血量须为正")
    p_tm = (.3855 + .15 * (volume_ml - 130.0)
            if volume_ml >= 130.0 else -5.0 + .4 * exp(volume_ml / 50.0))
    p_tm -= 2.0 / expm1(volume_ml / 8.0)
    resistance = .025 + .001 * (350.0 / volume_ml) ** 2
    return p_tm + pleural_mmhg, resistance


def pressures(state, pleural_mmhg, parameters=NOMINAL):
    p_sa = state[0] / .28
    p_peripheral = ((state[1] - sum(parameters.peripheral_unstressed_ml))
                    / sum(parameters.peripheral_compliance))
    p_venous = (state[2:4] - parameters.venous_unstressed_ml) / parameters.venous_compliance
    p_tv, r_tv = thoracic_vein(state[4], pleural_mmhg)
    return p_sa, p_peripheral, p_venous, p_tv, r_tv


def bed_flows(p_peripheral, p_venous, p_tv, q_sa, parameters=NOMINAL):
    """床端流量及质量守恒推得的床端入口流量。"""
    q_peripheral = np.maximum(p_peripheral - p_venous, 0.0) / parameters.peripheral_resistance
    q_venous = np.maximum(p_venous - p_tv, 0.0) / parameters.venous_resistance
    compliance_fraction = np.asarray(parameters.peripheral_compliance) / sum(parameters.peripheral_compliance)
    q_inflow = q_peripheral + compliance_fraction * (q_sa - q_peripheral.sum())
    return q_inflow, q_peripheral, q_venous


def evaluate(state, pressure, right_atrium_mmhg, left_outflow_ml_s, parameters=NOMINAL):
    """六个状态包括主动脉、外围、两静脉、胸内静脉血量以及主动脉流量。"""
    p_sa, p_peripheral, p_venous, p_tv, r_tv = pressure
    q_sa = state[5]
    _, q_peripheral, q_venous = bed_flows(p_peripheral, p_venous, p_tv, q_sa, parameters)
    q_tv = (p_tv - right_atrium_mmhg) / r_tv
    return np.r_[left_outflow_ml_s - q_sa, q_sa - q_peripheral.sum(),
                 q_peripheral - q_venous, q_venous.sum() - q_tv,
                 (p_sa - p_peripheral - .06 * q_sa) / .00022], q_tv


def tissue_inputs(p_peripheral, grouped_venous_volume, q_in, q_out, q_venous, parameters=NOMINAL):
    """试算时将两床血流按名义电导投影到 Albanese 的五个组织床。"""
    def distribute(values, weights):
        weights = weights[1:] / sum(weights[1:])
        return np.r_[values[0], values[1] * weights]

    peripheral_volume = (distribute(parameters.peripheral_unstressed_ml, BED_VU)
                         + distribute(parameters.peripheral_compliance, BED_C) * p_peripheral)
    venous_volume = (distribute(parameters.venous_unstressed_ml, VENOUS_VU)
                     + distribute(grouped_venous_volume - parameters.venous_unstressed_ml, VENOUS_C))
    return (peripheral_volume, venous_volume, distribute(q_in, 1. / PERIPHERAL_R),
            distribute(q_out, 1. / PERIPHERAL_R), distribute(q_venous, 1. / VENOUS_R))
