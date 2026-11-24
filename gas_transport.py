"""把已复现的机械循环与 Albanese 肺泡/组织交换及纯运输时延连接。"""

import argparse
from dataclasses import dataclass
from math import floor
from pathlib import Path

import numpy as np

import blood_gas as gas
import pulmonary
import pulmonary_transport as transport
import respiration
import systemic
from mechanical import COLUMNS, MechanicalModel, fixed_rk4


FIELDS = COLUMNS.split(",")
QUANTITY = {name: FIELDS.index(name) for name in ("p_pleural", "p_pal", "p_pc", "p_pv")}
GAS_FIELDS = ("fd_o2", "fd_co2", "fa_o2", "fa_co2", "ca_o2_stpd", "ca_co2_stpd",
              "cv_o2_stpd", "cv_co2_stpd", "q_pa", "q_pal", "q_pc", "q_shunt",
              "pulmonary_o2_balance", "pulmonary_co2_balance")
DEAD_SPACE_L = respiration.end_expiratory_state()[:3].sum()
GAS_STATE_SIZE = 26 + transport.INVENTORY_SIZE
ARTERIAL_OUTPUT = slice(len(FIELDS) + 4, len(FIELDS) + 6)
VENOUS_OUTPUT = slice(len(FIELDS) + 6, len(FIELDS) + 8)
VALVE_FIELDS = tuple(FIELDS.index(name) for name in
                     ("q_pulmonary_valve", "q_pulmonary_venous", "q_mv", "q_aortic_valve"))


@dataclass(frozen=True)
class GasTransport:
    mechanical: MechanicalModel = MechanicalModel()
    inspired: tuple = gas.INSPIRED

    def __post_init__(self):
        o2, co2 = self.inspired
        if not (0. < o2 < 1. and 0. <= co2 < 1. and o2 + co2 < 1.):
            raise ValueError("吸入氧和二氧化碳分数超出定义域")

    def initial_state(self, airway_cmh2o=0.):
        """初始气体分数及浓度采用独立起点，时延历史从其平接。"""
        empty = np.array([100., 40.]) / gas.DRY_PRESSURE_MMHG
        venous = gas.dissociation(np.array([40., 46.]))[0]
        arterial = gas.dissociation(np.array([100., 40.]))[0]
        mechanics = self.mechanical.initial_state(airway_cmh2o)
        volumes = mechanics[transport.MECHANICAL_POOLS[transport.INVENTORY_POOLS]]
        inventories = volumes[:, None] * np.vstack(([venous] * 2, [arterial] * 4))
        return np.r_[mechanics, empty, empty, np.tile(venous, 11), inventories.ravel()]

    def evaluate(self, time_s, state, airway_cmh2o, muscle_cmh2o, arterial_history, venous_history, *,
                 phase=None):
        mechanics, gases = state[:16], state[16:]
        d_mechanics, output = self.mechanical.evaluate(time_s, mechanics, airway_cmh2o, muscle_cmh2o, phase=phase)
        pleural = output[QUANTITY["p_pleural"]]
        p_pal, p_pc, p_pv = (output[QUANTITY[name]] for name in ("p_pal", "p_pc", "p_pv"))
        airflow = d_mechanics[-1]
        p_bed, p_veins, p_tv = systemic.pressures(mechanics[4:10], pleural, self.mechanical.body)[1:4]
        q_in, q_out, q_venous = systemic.bed_flows(p_bed, p_veins, p_tv, mechanics[9], self.mechanical.body)
        v_tissue, v_veins, q_in, q_out, q_venous = systemic.tissue_inputs(
            p_bed, mechanics[6:8], q_in, q_out, q_venous, self.mechanical.body
        )
        va = mechanics[15]
        r_pal, r_pc = pulmonary.resistances(va, self.mechanical.lung)
        shunt = self.mechanical.lung.shunt_fraction
        q_pal = (p_pal - p_pc) / r_pal
        q_shunt = shunt / (1 - shunt) * q_pal
        q_pc = (p_pc - p_pv) / r_pc
        q_rv, q_pv, q_mv, q_ao = output[list(VALVE_FIELDS)]
        flows = np.array([mechanics[14], q_pal, q_pc, q_pv, q_mv, q_ao, mechanics[9], q_shunt])
        inventories = gases[26:].reshape((-1, 2))
        d_lung, balance = transport.evaluate(
            gases[:4], inventories, mechanics[transport.MECHANICAL_POOLS], d_mechanics[12],
            va, airflow, flows, q_rv, venous_history, arterial_history, DEAD_SPACE_L, self.inspired,
        )
        arterial = self.arterial_concentration(state)
        tissue = gases[4:14].reshape((5, 2))
        veins = gases[14:24].reshape((5, 2))
        central_venous = gases[24:26]
        d_tissue, d_veins, d_central = gas.tissue_exchange(
            tissue, veins, central_venous, arterial_history,
            peripheral_volume_ml=v_tissue, venous_volume_ml=v_veins, central_volume_ml=mechanics[8],
            peripheral_inflow_ml_s=q_in, peripheral_outflow_ml_s=q_out, venous_flow_ml_s=q_venous,
        )
        derivative = np.r_[d_mechanics, d_lung[:4], d_tissue.ravel(), d_veins.ravel(), d_central, d_lung[4:]]
        return derivative, np.r_[output, gases[:4], arterial, central_venous,
                                 mechanics[14], q_pal, q_pc, q_shunt, balance]

    @staticmethod
    def initial_venous_concentration(state):
        return state[40:42]

    @staticmethod
    def arterial_concentration(state):
        return state[-2:] / state[4]

    def simulate(self, duration_s, airway_at, muscle_at, *, step_s=.0005, sample_every=10):
        history = np.empty((round(duration_s / step_s) + 1, 4))
        state = self.initial_state(airway_at(0.))
        initial_arterial = gas.dissociation(np.array([100., 40.]))[0]
        initial = np.r_[initial_arterial, self.initial_venous_concentration(state)]

        def sample(time_s, delay_s, fields):
            if time_s <= delay_s:
                return initial[fields]
            coordinate = (time_s - delay_s) / step_s
            index = floor(coordinate)
            fraction = coordinate - index
            return (1.0 - fraction) * history[index, fields] + fraction * history[index + 1, fields]

        def evaluate(time_s, x):
            return self.evaluate(time_s, x, airway_at(time_s), muscle_at(time_s),
                                 sample(time_s, gas.ARTERIAL_TO_TISSUE_S, slice(0, 2)),
                                 sample(time_s, gas.VENOUS_TO_LUNG_S, slice(2, 4)))

        def commit(index, output):
            history[index] = np.r_[output[ARTERIAL_OUTPUT], output[VENOUS_OUTPUT]]

        return fixed_rk4(state, evaluate, duration_s, step_s, sample_every, commit)


def main():
    parser = argparse.ArgumentParser(description="复核机械循环、血气交换和固定运输时延")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = GasTransport().simulate(args.duration_s, lambda _: 0., respiration.quiet_muscle_pressure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, result, delimiter=",", header=",".join(FIELDS + list(GAS_FIELDS)), comments="")
    print(f"写入 {len(result)} 个采样点，{args.output}")


if __name__ == "__main__":
    main()
