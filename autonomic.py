"""Albanese 心血管反射的文献重建，状态单位为 mmHg、spikes/s 与效应器单位。"""

from math import log, log1p

import numpy as np
from scipy.special import expit

import reflex


# 顺序为外周阻力交感、静脉交感、心脏交感、迷走。
BARO_WEIGHT = np.array([-1.1375, -1.0806, -1.75])
CHEMO_WEIGHT = np.array([1.716, 1.716, 1.])
STRETCH_WEIGHT = np.array([-.3997, -.2907, 0.])
CNS_BASE = np.array([13.32, 13.32, 3.6])
CNS_CO2_GAIN = np.array([1.5, 0., 1.])
CNS_HYPOXIA_GAIN = np.array([6., 6., 53.])
CNS_O2_THRESHOLD = np.array([30., 30., 45.])
CNS_O2_WIDTH = np.array([2., 2., 6.])

# 效应器顺序为右室、左室、室间隔、两床阻力、两床静脉容量、心周期交感及迷走项。
EFFECTOR_TAU = np.array([8., 8., 8., 6., 6., 20., 20., 2., 1.5])
CONTROL_STATE_SIZE = 18


def afferent_baroreceptor(arterial_mmhg, filter_state_mmhg):
    # 等价实现 (1 + tau_z*s)/(1 + tau_p*s)，不对动脉压做数值微分。
    pressure = filter_state_mmhg + (6.37 / 2.076) * (arterial_mmhg - filter_state_mmhg)
    return 2.52 + (47.78 - 2.52) * expit((pressure - 92.) / 11.76)


def chemoreceptor_target(po2, pco2):
    arterial_o2_response = .835 + (12.3 - .835) * expit((45. - po2) / 29.27)
    # 中段按高、低氧端点连续插值，保留文献描述的低氧时 CO₂ 阈值降低。
    k = 3. - 1.6 * min(max((80. - po2) / 40., 0.), 1.)
    return arterial_o2_response * max(k * log(pco2 / 40.) + 1.4, 0.)


def central_targets(po2, pco2):
    hypoxia = CNS_HYPOXIA_GAIN * expit((CNS_O2_THRESHOLD - po2) / CNS_O2_WIDTH)
    hypercapnia = CNS_CO2_GAIN * (pco2 - 40.)
    return hypoxia, hypercapnia


def neural_output(state, arterial_mmhg):
    baroreceptor = afferent_baroreceptor(arterial_mmhg, state[0])
    chemoreceptor, stretch = state[1:3]
    offset = CNS_BASE - state[3:6] - state[6:9]
    stimulation = (BARO_WEIGHT * baroreceptor + CHEMO_WEIGHT * chemoreceptor
                   + STRETCH_WEIGHT * stretch - offset)
    sympathetic = np.minimum(2.1 + (16.11 - 2.1) * np.exp(.0675 * stimulation), 60.)
    vagal = (3.2 + (6.3 - 3.2) * expit((baroreceptor - 25.) / 7.06)
             + .2 * chemoreceptor - .103 * stretch + .68)
    return np.r_[sympathetic, vagal], baroreceptor


def sympathetic_effect(firing):
    return log1p(max(firing - 2.66, 0.))


def effector_targets(delayed_2s, delayed_5s, delayed_vagal):
    arteries = sympathetic_effect(delayed_2s[0])
    veins = sympathetic_effect(delayed_5s[1])
    heart = sympathetic_effect(delayed_2s[2])
    return np.array([.282 * heart, .475 * heart, 6.44 * heart, .695 * arteries,
                     .53 * arteries, -265.4 * veins, -132.5 * veins,
                     -.13 * heart, .09 * delayed_vagal])


def derivative(state, arterial_mmhg, arterial_gases, last_tidal_l, delayed_2s,
               delayed_5s, delayed_vagal):
    po2, pco2 = arterial_gases
    hypoxia, hypercapnia = central_targets(po2, pco2)
    effects = (effector_targets(delayed_2s, delayed_5s, delayed_vagal) - state[9:]) / EFFECTOR_TAU
    effects[2] = reflex.septal_derivative(state[11], delayed_2s[2])
    return np.r_[(arterial_mmhg - state[0]) / 2.076,
                 (chemoreceptor_target(po2, pco2) - state[1]) / 2.,
                 (11.76 * last_tidal_l - state[2]) / 2.,
                 (hypoxia - state[3:6]) / 30., (hypercapnia - state[6:9]) / 20., effects]


def initial_state(arterial, arterial_gases, tidal_l):
    hypoxia, hypercapnia = central_targets(*arterial_gases)
    state = np.r_[arterial, chemoreceptor_target(*arterial_gases), 11.76 * tidal_l,
                  hypoxia, hypercapnia, np.zeros(9)]
    efferent, _ = neural_output(state, arterial)
    state[9:] = effector_targets(efferent, efferent, efferent[-1])
    return state


def heart_period(state, uninnervated_period=.58):
    period = uninnervated_period + sum(state[-2:])
    if period <= .15:
        raise ValueError("自主调节后的心周期超出文献激活公式的定义域")
    return period
