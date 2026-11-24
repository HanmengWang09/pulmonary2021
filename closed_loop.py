"""Karamolegkos 2021 的闭环研究重建，保留未核实的论文接口说明。"""

import argparse
from dataclasses import replace
from math import floor, isfinite
from pathlib import Path

import numpy as np

import autonomic
import blood_gas
from breathing import AdaptiveBreathing
import figure6_airway
from gas_transport import ARTERIAL_OUTPUT, FIELDS, GAS_FIELDS, GAS_STATE_SIZE, GasTransport, VENOUS_OUTPUT
from mechanical import fixed_rk4
import reflex


GAS_END = 16 + GAS_STATE_SIZE
CONTROL_END = GAS_END + autonomic.CONTROL_STATE_SIZE
CONTROL_FIELDS = (
    "pa_o2_mmhg", "pa_co2_mmhg", "f_baroreceptor", "f_peripheral_chemo", "f_stretch",
    "f_arteries_symp", "f_veins_symp", "f_cardiac_symp", "f_vagus",
    "heart_cycle", "heart_period_s", "e_rv", "e_lv", "e_septum", "r_sp", "r_ep",
    "vu_sv_ml", "vu_ev_ml", "respiratory_rate", "muscle_pressure_cmh2o", "last_tidal_l",
    "airway_pressure_cmh2o",
)
OUTPUT_FIELDS = FIELDS + list(GAS_FIELDS) + list(CONTROL_FIELDS)
P_AORTA = FIELDS.index("p_ao")


class DelayHistory:
    """仅存储 RK4 接受步的观测值。中间阶段按实际时间插值。"""

    def __init__(self, initial, steps, step_s):
        self.initial = initial.copy()
        self.history = np.empty((steps, initial.size))
        self.step_s = step_s
        self.latest = -1

    def sample(self, time_s, delay_s, columns):
        coordinate = (time_s - delay_s) / self.step_s
        if coordinate < 0. or self.latest < 0:
            return self.initial[columns]
        index = floor(coordinate)
        fraction = coordinate - index
        if index > self.latest or (index == self.latest and fraction > 1e-9):
            raise ValueError("时延必须落在已接受的历史时间范围内")
        if index == self.latest:
            return self.history[index, columns]
        return ((1. - fraction) * self.history[index, columns]
                + fraction * self.history[index + 1, columns])

    def commit(self, index, row):
        self.history[index] = row
        self.latest = index

def controlled_model(base, control, period):
    heart = base.mechanical.cardiac
    body = base.mechanical.body
    elastance = tuple(x + dx for x, dx in zip(heart.ventricular_elastance, control[9:11]))
    resistance = tuple(x + dx for x, dx in zip(body.peripheral_resistance, control[12:14]))
    unstressed = tuple(x + dx for x, dx in zip(body.venous_unstressed_ml, control[14:16]))
    values = (*elastance, heart.septal_elastance + control[11], *resistance)
    if min(values) <= 0.:
        raise ValueError("受调节的弹性或阻力须保持正值")
    return replace(base, mechanical=replace(
        base.mechanical, cardiac=replace(heart, ventricular_elastance=elastance,
                                         septal_elastance=values[2], period_s=period),
        body=replace(body, peripheral_resistance=resistance, venous_unstressed_ml=unstressed)
    ))


def simulate(duration_s, *, step_s=.0005, sample_every=20, airway_at=lambda _: 0.,
             tidal_threshold_l=None, tidal_threshold_at=None, gas_model=GasTransport(),
             uninnervated_period_s=.58, passive_rate=None, support_at=None):
    if not isfinite(duration_s) or not isfinite(step_s) or not 0. < step_s < .2:
        raise ValueError("时长须有限，步长须小于最短反馈时延 0.2 s")
    steps = round(duration_s / step_s)
    if steps < 1 or not np.isclose(steps * step_s, duration_s, atol=1e-10, rtol=0.):
        raise ValueError("时长须为正的步长整数倍")
    if sample_every < 1:
        raise ValueError("采样间隔须至少为一个积分步")
    if passive_rate is not None and (tidal_threshold_l is not None or tidal_threshold_at is not None):
        raise ValueError("潮气量终止阈值仅用于自主呼吸")
    state = gas_model.initial_state(airway_at(0.))
    ventilation = AdaptiveBreathing(.53, tidal_threshold_l, threshold_at=tidal_threshold_at)
    p_arterial = state[4] / .28
    resting_gases = (100., 40.)
    control = autonomic.initial_state(p_arterial, resting_gases, ventilation.tidal_l)
    arterial = blood_gas.dissociation(resting_gases)[0]
    firing, _ = autonomic.neural_output(control, p_arterial)
    venous = gas_model.initial_venous_concentration(state)
    delay = DelayHistory(np.r_[arterial, venous, firing, control[1], resting_gases[1]], steps, step_s)
    state = np.r_[state, control, np.zeros(5)]  # 四个通气效应器与连续心搏相位

    def accepted(time_s, xs):
        if passive_rate is None:
            muscle, rate = reflex.respiratory_setpoints(xs[CONTROL_END:CONTROL_END + 4])
        else:
            muscle, rate = 0., passive_rate
        gases = None
        if tidal_threshold_at is not None:
            gases = blood_gas.partial_pressures(gas_model.arterial_concentration(xs[:GAS_END]))
        ventilation.observe(time_s, xs[15], muscle, rate, arterial_gases=gases)

    def evaluate(time_s, xs):
        c = xs[GAS_END:CONTROL_END]
        respiratory = xs[CONTROL_END:CONTROL_END + 4]
        phase = xs[-1]
        period = autonomic.heart_period(c, uninnervated_period_s)
        model = controlled_model(gas_model, c, period)
        muscle = ventilation.muscle_pressure(time_s)
        airway = airway_at(time_s)
        if support_at is not None:
            airway += support_at(time_s, time_s - ventilation.start_s)
        d_plant, output = model.evaluate(
            time_s, xs[:GAS_END], airway, muscle,
            delay.sample(time_s, blood_gas.ARTERIAL_TO_TISSUE_S, slice(0, 2)),
            delay.sample(time_s, blood_gas.VENOUS_TO_LUNG_S, slice(2, 4)),
            phase=phase,
        )
        gases = blood_gas.partial_pressures(output[ARTERIAL_OUTPUT])
        firing, baro = autonomic.neural_output(c, output[P_AORTA])
        d_cardio = autonomic.derivative(
            c, output[P_AORTA], gases, ventilation.tidal_l,
            delay.sample(time_s, 2., slice(4, 7)),
            delay.sample(time_s, 5., slice(4, 7)), delay.sample(time_s, .2, 7),
        )
        d_respiratory = reflex.respiratory_derivative(
            respiratory, delay.sample(time_s, 7., 8), delay.sample(time_s, 8., 9)
        )
        heart = model.mechanical.cardiac
        body = model.mechanical.body
        feedback = np.r_[gases, baro, c[1:3], firing, phase, period,
                         heart.ventricular_elastance, heart.septal_elastance,
                         body.peripheral_resistance, body.venous_unstressed_ml,
                         ventilation.rate, muscle, ventilation.tidal_l, airway]
        return np.r_[d_plant, d_cardio, d_respiratory, 1. / period], np.r_[output, feedback]

    def commit(index, output):
        signals = output[len(FIELDS) + len(GAS_FIELDS):]
        delay.commit(index, np.r_[output[ARTERIAL_OUTPUT], output[VENOUS_OUTPUT],
                                 signals[5:9], signals[3], signals[1]])

    rows = fixed_rk4(state, evaluate, duration_s, step_s, sample_every, commit, accepted=accepted)
    return rows, ventilation.events_s


def simulate_figure6(duration_s, *, step_s=.0005, sample_every=20):
    """图 6 参数和数字化气道压力。气道压力作为被动通气边界。"""
    gas_model = GasTransport()
    mechanical = gas_model.mechanical
    heart = replace(mechanical.cardiac, ventricular_elastance=(1.412, 1.9),
                    ventricular_exponent=(.011, .016), ventricular_pressure_scale=(1.5, .8))
    body = replace(mechanical.body, peripheral_resistance=(3., 1.))
    mechanical = replace(mechanical, cardiac=heart, body=body, respiratory=figure6_airway.ARDS)
    return simulate(duration_s, step_s=step_s, sample_every=sample_every,
                    airway_at=figure6_airway.airway_pressure, gas_model=replace(gas_model, mechanical=mechanical),
                    uninnervated_period_s=.35, passive_rate=15.)


def simulate_figure4(stage_duration_s=1800., *, step_s=.0005, sample_every=20, tidal_threshold_l=None):
    """期刊图 4 压力级别，支持压力按神经吸气起点触发，限时 1.3 s。"""
    if not isfinite(stage_duration_s) or stage_duration_s <= 0.:
        raise ValueError("每个压力支持阶段须为有限正时长")
    gas_model = GasTransport()
    mechanical = gas_model.mechanical
    heart = replace(mechanical.cardiac, ventricular_elastance=(.8, 3.05),
                    ventricular_exponent=(.007, .008))
    gas_model = replace(gas_model, mechanical=replace(mechanical, cardiac=heart))

    def peep(time_s):
        return 0. if time_s < stage_duration_s else (10. if time_s < 2. * stage_duration_s else 20.)

    def support(time_s, elapsed_s):
        return 3. if time_s >= stage_duration_s and 0. <= elapsed_s < 1.3 else 0.

    return simulate(3. * stage_duration_s, step_s=step_s, sample_every=sample_every,
                    airway_at=peep, support_at=support, tidal_threshold_l=tidal_threshold_l,
                    gas_model=gas_model)


def main():
    parser = argparse.ArgumentParser(description="运行论文闭环重建的静息、图 4 或图 6 工况")
    parser.add_argument("--case", choices=("resting", "figure4", "figure6"), default="resting")
    parser.add_argument("--duration-s", type=float, default=60.)
    parser.add_argument("--stage-duration-s", type=float, default=1800., help="图 4 每个压力阶段的时长")
    parser.add_argument("--step-s", type=float, default=.0005)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--hb-threshold-l", type=float, help="显式给定的试验阈值")
    parser.add_argument("--inspired-o2", type=float, default=blood_gas.INSPIRED[0],
                        help="静息试算的吸入氧体积分数")
    parser.add_argument("--inspired-co2", type=float, default=blood_gas.INSPIRED[1],
                        help="静息试算的吸入二氧化碳体积分数")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    numerics = dict(step_s=args.step_s, sample_every=args.sample_every)
    inspired = (args.inspired_o2, args.inspired_co2)
    if args.case != "resting" and inspired != blood_gas.INSPIRED:
        parser.error("吸入气体组成参数仅用于静息闭环试算")
    if args.case == "resting":
        rows, events = simulate(args.duration_s, tidal_threshold_l=args.hb_threshold_l,
                                gas_model=GasTransport(inspired=inspired), **numerics)
    elif args.case == "figure4":
        rows, events = simulate_figure4(args.stage_duration_s, tidal_threshold_l=args.hb_threshold_l, **numerics)
    else:
        if args.hb_threshold_l is not None:
            parser.error("潮气量终止阈值仅用于自主呼吸")
        rows, events = simulate_figure6(args.duration_s, **numerics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, rows, delimiter=",", header=",".join(OUTPUT_FIELDS), comments="")
    print(f"写入 {len(rows)} 个采样点，吸气提前终止事件 {len(events)} 个，{args.output}")


if __name__ == "__main__":
    main()
