"""论文附录的调节效应方程。所有输入延迟由调用方定义。"""

from math import log1p

import numpy as np


def septal_derivative(delta_elastance, delayed_sympathetic_spikes_s):
    """Karamolegkos 2021 (A1)–(A3) 与表 VII。传出信号延迟为 2 s。"""
    drive = 6.44 * log1p(max(delayed_sympathetic_spikes_s - 2.66, 0.0))
    return (drive - delta_elastance) / 8.0


def respiratory_derivative(state, delayed_peripheral_spikes_s, delayed_arterial_co2_mmhg):
    """Albanese (A81)–(A84)，采用表 6 的下标及项目已有的 CGS 幅值解释。"""
    stimulus = np.array([delayed_peripheral_spikes_s - 3.7, delayed_arterial_co2_mmhg - 40.0])
    target = np.r_[np.array([1310., 850.]) / 980.665 * stimulus,
                   np.array([.8735, .9]) * stimulus]
    return (target - np.asarray(state)) / np.array([83., 105., 147.78, 400.])


def respiratory_setpoints(state):
    """以正幅值表示吸气肌压强度，返回有符号肌压和每分钟呼吸次数。"""
    return -5.0 - sum(state[:2]), 12.0 + sum(state[2:])
