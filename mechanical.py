"""原论文机械闭环的分阶段复现，以独立初值和给定的呼吸驱动运行。"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from breathing import HeringBreuerDrive
import heart
import pulmonary
import respiration
import systemic
from respiration_single import SingleCompartment


COLUMNS = (
    "time_s,p_ra,p_la,p_rv,p_lv,p_ao,p_pa,p_pal,p_pc,p_pv,"
    "v_ra,v_la,v_rv,v_lv,q_tv,q_mv,q_pulmonary_valve,q_aortic_valve,"
    "q_pulmonary_venous,p_pericardial,v_septal,septal_volume_residual_ml,blood_volume,"
    "p_pleural,p_alveolar,v_alveolar,q_systemic_venous"
)


@dataclass(frozen=True)
class MechanicalModel:
    cardiac: heart.HeartParameters = heart.HeartParameters()
    lung: pulmonary.Parameters = pulmonary.NOMINAL
    body: systemic.TwoBedSystemic = systemic.NOMINAL
    respiratory: SingleCompartment = SingleCompartment()

    def initial_state(self, airway_cmh2o=0.):
        """Albanese 2016 血量。各池初值为机械核的数值起点。"""
        chambers = [75.0, 80.0, 120.0, 125.0]
        veins = np.asarray(self.body.venous_unstressed_ml) + np.asarray(self.body.venous_compliance) * 8.0
        v_bed = sum(self.body.peripheral_unstressed_ml) + sum(self.body.peripheral_compliance) * 84.0
        circulation = np.r_[.28 * 88.0, v_bed,
                            veins, 170.0, 80.0]
        alveolar_l = self.respiratory.equilibrium_volume(airway_cmh2o)
        _, p_a, p_pl = self.respiratory.evaluate(alveolar_l, airway_cmh2o, 0.)
        cm = respiration.CMH2O_TO_MMHG
        lung = pulmonary.steady_state(pulmonary.Boundary(80.0, 2.0, p_pl * cm, p_a * cm, alveolar_l), self.lung)
        circulation[3] += systemic.TOTAL_BLOOD_ML - sum(chambers) - sum(circulation[:5]) - sum(lung[:4])
        return np.r_[chambers, circulation, lung, alveolar_l]

    def evaluate(self, time_s, state, airway_cmh2o, muscle_cmh2o, *, phase=None):
        chambers, body, lung, volume_l = state[:4], state[4:10], state[10:15], state[15]
        d_airway, alveolar_cmh2o, pleural_cmh2o = self.respiratory.evaluate(volume_l, airway_cmh2o, muscle_cmh2o)
        pleural = pleural_cmh2o * respiration.CMH2O_TO_MMHG
        alveolar = alveolar_cmh2o * respiration.CMH2O_TO_MMHG
        p_pa = pleural + (lung[0] - self.lung.unstressed_volume_ml[0]) / self.lung.compliance_ml_mmhg[0]
        p_body = systemic.pressures(body, pleural, self.body)
        p_heart, q_heart, extra = heart.evaluate(time_s, chambers, pleural, p_pa, p_body[0], self.cardiac, phase)
        q_tv, q_mv, q_pa, q_ao = q_heart
        d_body, q_systemic_venous = systemic.evaluate(body, p_body, p_heart[0], q_ao, self.body)
        d_lung, p_lung, q_lung = pulmonary.evaluate(
            lung, pulmonary.Boundary(q_pa, p_heart[1], pleural, alveolar, volume_l), self.lung
        )
        d_heart = np.array([q_systemic_venous - q_tv, q_lung[-1] - q_mv,
                            q_tv - q_pa, q_mv - q_ao])
        output = np.r_[time_s, p_heart, p_body[0], p_lung, chambers, q_heart, q_lung[-1], extra,
                       self.blood_volume(state), pleural, alveolar, volume_l, q_systemic_venous]
        return np.r_[d_heart, d_body, d_lung, d_airway], output

    @staticmethod
    def blood_volume(state):
        return state[:9].sum() + state[10:14].sum()

    def simulate(self, duration_s, airway_at, muscle_at, *, step_s=.0005, sample_every=10):
        def evaluate(time_s, x):
            return self.evaluate(time_s, x, airway_at(time_s), muscle_at(time_s))

        def observe(time_s, x):
            muscle_at.observe(time_s, x[15])

        accepted = None
        if isinstance(muscle_at, HeringBreuerDrive):
            muscle_at.reset()
            accepted = observe
        return fixed_rk4(self.initial_state(airway_at(0.)), evaluate, duration_s, step_s,
                         sample_every, accepted=accepted)


def fixed_rk4(state, evaluate, duration_s, step_s, sample_every, commit=None, *, accepted=None):
    count = round(duration_s / step_s)
    if count < 1 or not np.isclose(count * step_s, duration_s, atol=1e-10, rtol=0.0):
        raise ValueError("运行时长须为正的步长整数倍")
    rows = []
    for index in range(count):
        t = index * step_s
        if accepted is not None:
            accepted(t, state)
        k1, output = evaluate(t, state)
        if commit is not None:
            commit(index, output)
        if index % sample_every == 0:
            rows.append(output)
        mid = t + step_s / 2.0
        k2 = evaluate(mid, state + step_s * k1 / 2.0)[0]
        k3 = evaluate(mid, state + step_s * k2 / 2.0)[0]
        k4 = evaluate(t + step_s, state + step_s * k3)[0]
        state += step_s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(state).all():
            raise ValueError(f"积分失败，时间 {t:g} s")
    rows.append(evaluate(duration_s, state)[1])
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser(description="运行机械闭环的独立核查例")
    parser.add_argument("--duration-s", type=float, default=5.1, help="运行时长")
    parser.add_argument("--output", type=Path, required=True, help="结果 CSV")
    parser.add_argument("--drive", choices=("quiet", "passive_step"), default="quiet", help="呼吸驱动")
    parser.add_argument("--hb-threshold-l", type=float, help="用于核查切换的显式潮气量阈值，单位 L")
    args = parser.parse_args()
    if args.drive == "quiet":
        muscle_at = respiration.quiet_muscle_pressure
        if args.hb_threshold_l is not None:
            muscle_at = HeringBreuerDrive(lambda _: args.hb_threshold_l)
        result = MechanicalModel().simulate(args.duration_s, lambda _: 0.0, muscle_at)
        if isinstance(muscle_at, HeringBreuerDrive):
            print(f"吸气提前终止时刻（s）{muscle_at.events_s}")
    else:
        if args.hb_threshold_l is not None:
            parser.error("潮气量触发参数仅用于自主呼吸试算")
        result = MechanicalModel().simulate(args.duration_s, lambda t: 5.0 if t >= 1 else 0.0, lambda _: 0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, result, delimiter=",", header=COLUMNS, comments="")
    print(f"写入 {len(result)} 个采样点，{args.output}")


if __name__ == "__main__":
    main()
