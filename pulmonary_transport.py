"""肺循环至主动脉的双向血气输运，肺泡与毛细血管维持瞬时平衡。"""

import numpy as np

from blood_gas import BTPS_PER_STPD, DRY_PRESSURE_MMHG, dissociation


# 肺动脉、肺小动脉、肺毛细血管、肺静脉、左房、左室、近端主动脉。
MECHANICAL_POOLS = np.array([10, 11, 12, 13, 1, 3, 4])
INVENTORY_POOLS = np.array([0, 1, 3, 4, 5, 6])
INVENTORY_SIZE = 2 * len(INVENTORY_POOLS)
AIR_SCALE = 1000. / BTPS_PER_STPD


def evaluate(fractions, blood_inventory, volumes, capillary_dvolume, alveolar_volume, airflow,
             flows, rv_outflow, venous_inlet, systemic_return, dead_space, inspired):
    """气体库存用 STPD mL，血液体积和血流用 mL 与 mL/s。"""
    f_dead, f_alveolar = fractions.reshape((2, 2))
    inspired = np.asarray(inspired)
    capillary, jacobian = dissociation(f_alveolar * DRY_PRESSURE_MMHG)
    stored_volumes = volumes[INVENTORY_POOLS]
    if np.any(volumes <= 0.):
        raise ValueError("肺循环、左心或主动脉的血气库存容积非正")
    concentrations = np.empty((7, 2))
    concentrations[2] = capillary
    concentrations[INVENTORY_POOLS] = blood_inventory / stored_volumes[:, None]

    # 同一条边的气体通量在供体和受体只计算一次。
    chain = flows[:6, None]
    flux = chain * np.where(chain >= 0., concentrations[:-1], concentrations[1:])
    shunt_flow = flows[7]
    shunt = shunt_flow * concentrations[1 if shunt_flow >= 0. else 3]
    arterial_flow = flows[6]
    systemic_out = arterial_flow * (concentrations[6] if arterial_flow >= 0. else systemic_return)
    inlet = rv_outflow * venous_inlet
    blood_change = np.vstack((inlet, flux)) - np.vstack((flux, systemic_out))
    blood_change[1] -= shunt
    blood_change[3] += shunt

    if airflow >= 0.:
        d_dead = (airflow / dead_space) * (inspired - f_dead)
        wash = airflow * (f_dead - f_alveolar)
        mouth = AIR_SCALE * airflow * inspired
    else:
        d_dead = (airflow / dead_space) * (f_dead - f_alveolar)
        wash = np.zeros(2)
        mouth = AIR_SCALE * airflow * f_dead

    # 合并肺泡气体与毛细血管血气的库存，消去膜交换内通量。
    capacity = AIR_SCALE * alveolar_volume * np.eye(2) + volumes[2] * DRY_PRESSURE_MMHG * jacobian
    d_alveolar = np.linalg.solve(
        capacity, AIR_SCALE * wash + blood_change[2] - capillary_dvolume * capillary
    )
    inventory_change = blood_change[INVENTORY_POOLS]
    d_alveolar_capillary_amount = (capacity @ d_alveolar
                                   + AIR_SCALE * airflow * f_alveolar + capillary_dvolume * capillary)
    residual = (AIR_SCALE * dead_space * d_dead + d_alveolar_capillary_amount
                + inventory_change.sum(axis=0) - (mouth + inlet - systemic_out))
    return np.r_[d_dead, d_alveolar, inventory_change.ravel()], residual
