"""用 2021 图 6 读出的气道压力驱动论文的被动单腔模型。"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from respiration_single import SingleCompartment


# 2021 图 6 左上面板的灰色气道压力曲线，按右轴从图读取。
FIGURE_TIME_S = (0., .29, .42, .51, .61, .73, .87, 1.07, 1.28, 1.29, 1.78,
                 1.84, 1.91, 2.02, 2.1, 2.23, 2.45, 2.75, 4.)
FIGURE_PRESSURE_CMH2O = (5., 5., 11.9, 14.8, 17.6, 20.1, 21.8, 23.5, 24.2, 20.7, 20.7,
                         15.1, 11.3, 8.3, 7.0, 5.8, 5.1, 5., 5.)
ARDS = replace(SingleCompartment(), lung_compliance_l_cmh2o=.06,
               chest_wall_compliance_l_cmh2o=.11, lung_unstressed_l=.4, chest_wall_unstressed_l=1.)


def airway_pressure(time_s):
    return float(np.interp(time_s % 4., FIGURE_TIME_S, FIGURE_PRESSURE_CMH2O))


def replay():
    def derivative(time_s, state):
        return [ARDS.evaluate(state[0], airway_pressure(time_s), 0.)[0]]

    time_s = np.linspace(0., 4., 8001)
    solution = solve_ivp(derivative, (0., 4.), [ARDS.equilibrium_volume(5.)],
                         t_eval=time_s, max_step=.0005, atol=1e-10, rtol=1e-9)
    if not solution.success:
        raise RuntimeError(solution.message)
    volume = solution.y[0]
    pressures = np.array([airway_pressure(t) for t in time_s])
    flow, alveolar, pleural = ARDS.evaluate(volume, pressures, 0.)
    return np.column_stack([time_s, pressures, alveolar, pleural, volume, flow])


def main():
    parser = argparse.ArgumentParser(description="重放图 6 气道压力并核查呼吸力学")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = replay()
    np.savetxt(args.output, rows, delimiter=",", comments="",
               header="time_s,p_airway_cmh2o,p_alveolar_cmh2o,p_pleural_cmh2o,v_alveolar_l,q_airway_l_s")
    print(f"潮气量 {np.ptp(rows[:, 4]):.4f} L，呼气末胸膜压 {rows[-1, 3]:.4f} cmH2O")


if __name__ == "__main__":
    main()
