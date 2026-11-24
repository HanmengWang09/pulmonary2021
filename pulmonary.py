"""Karamolegkos 等（2021）图 3 的肺循环模型。"""

import argparse
import csv
from dataclasses import dataclass, replace
from math import isfinite, pi, sin
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Parameters:
    """表 II。压力用 mmHg，血液容积用 mL，时间用 s。"""

    compliance_ml_mmhg: tuple = (0.76, 1.93, 3.87, 25.37)
    unstressed_volume_ml: tuple = (0.0, 43.30, 64.94, 105.6)
    r_pa: float = 0.023
    r_pv: float = 0.0056
    r_periphery: float = 0.0894
    l_pa: float = 0.00018
    frc_l: float = 2.25
    shunt_fraction: float = 0.017


@dataclass(frozen=True)
class Boundary:
    right_ventricle_outflow_ml_s: float
    left_atrium_pressure_mmhg: float
    pleural_pressure_mmhg: float
    alveolar_pressure_mmhg: float
    alveolar_volume_l: float


NOMINAL = Parameters()
PAPER_STEP_S = 0.0005


def resistances(alveolar_volume_l, parameters=NOMINAL):
    """附录 (A24)–(A25)。"""
    if not isfinite(alveolar_volume_l) or alveolar_volume_l <= 0:
        raise ValueError("肺泡容积必须为有限正数，单位为 L")
    r_pc = parameters.r_periphery / (2.0 * (1.0 - parameters.shunt_fraction))
    r_pal = r_pc * (alveolar_volume_l / parameters.frc_l) ** 2
    return r_pal, r_pc


def steady_state(boundary, parameters=NOMINAL):
    """冻结边界条件所得的解析稳态，用作独立运行的初值。"""
    r_pal, r_pc = resistances(boundary.alveolar_volume_l, parameters)
    q_pa = boundary.right_ventricle_outflow_ml_s
    q_pal = (1.0 - parameters.shunt_fraction) * q_pa
    p_pv = boundary.left_atrium_pressure_mmhg + parameters.r_pv * q_pa
    p_pc = p_pv + r_pc * q_pal
    p_pal = p_pc + r_pal * q_pal
    p_pa = p_pal + parameters.r_pa * q_pa
    p_pl, p_a = boundary.pleural_pressure_mmhg, boundary.alveolar_pressure_mmhg
    p_tm = np.array([p_pa - p_pl, p_pal - p_pl, p_pc - p_a, p_pv - p_pl])
    volumes = np.asarray(parameters.unstressed_volume_ml) + p_tm * parameters.compliance_ml_mmhg
    return np.r_[volumes, q_pa]


def evaluate(state, boundary, parameters=NOMINAL):
    """计算状态导数、血管腔内压力和有符号血流。"""
    state = np.asarray(state, dtype=float)
    q_pa = state[4]
    p_pl, p_a = boundary.pleural_pressure_mmhg, boundary.alveolar_pressure_mmhg
    external_pressure = np.array([p_pl, p_pl, p_a, p_pl])
    pressure = external_pressure + (state[:4] - parameters.unstressed_volume_ml) / parameters.compliance_ml_mmhg
    p_pa, p_pal, p_pc, p_pv = pressure
    r_pal, r_pc = resistances(boundary.alveolar_volume_l, parameters)
    q_pal = (p_pal - p_pc) / r_pal
    q_pc = (p_pc - p_pv) / r_pc
    q_ps = parameters.shunt_fraction / (1.0 - parameters.shunt_fraction) * q_pal
    q_pv = (p_pv - boundary.left_atrium_pressure_mmhg) / parameters.r_pv
    q_rv = boundary.right_ventricle_outflow_ml_s
    derivative = np.array([
        q_rv - q_pa,
        q_pa - q_pal - q_ps,
        q_pal - q_pc,
        q_ps + q_pc - q_pv,
        (p_pa - p_pal - parameters.r_pa * q_pa) / parameters.l_pa,
    ])
    return derivative, pressure, np.array([q_pa, q_pal, q_pc, q_ps, q_pv])


def simulate(boundary_at, duration_s, *, step_s=PAPER_STEP_S, initial=None, parameters=NOMINAL):
    """固定步长 RK4。逐步给出时间、状态、压力、流量与净入血量积分。"""
    if not isfinite(step_s) or not isfinite(duration_s) or not 0 < step_s <= duration_s:
        raise ValueError("时长须大于或等于步长，二者均须为有限正数")
    steps = round(duration_s / step_s)
    if abs(steps * step_s - duration_s) > step_s * 1e-8:
        raise ValueError("仿真时长必须为积分步长的整数倍")
    boundary = boundary_at(0.0)
    state = steady_state(boundary, parameters) if initial is None else np.array(initial, dtype=float)
    if state.shape != (5,) or not np.isfinite(state).all():
        raise ValueError("初始状态须包含四个有限血量和一个有限流量")
    net_inflow_ml = 0.0
    for index in range(steps + 1):
        time_s = index * step_s
        first, pressure, flow = evaluate(state, boundary, parameters)
        yield time_s, state.copy(), pressure, flow, net_inflow_ml
        if index == steps:
            break
        middle = boundary_at(time_s + 0.5 * step_s)
        end = boundary_at((index + 1) * step_s)
        second, _, flow_2 = evaluate(state + 0.5 * step_s * first, middle, parameters)
        third, _, flow_3 = evaluate(state + 0.5 * step_s * second, middle, parameters)
        fourth, _, flow_4 = evaluate(state + step_s * third, end, parameters)
        state = state + step_s / 6 * (first + 2 * second + 2 * third + fourth)
        net_inflow_ml += step_s / 6 * (
            boundary.right_ventricle_outflow_ml_s - flow[-1]
            + 2 * (middle.right_ventricle_outflow_ml_s - flow_2[-1])
            + 2 * (middle.right_ventricle_outflow_ml_s - flow_3[-1])
            + end.right_ventricle_outflow_ml_s - flow_4[-1]
        )
        if not np.isfinite(state).all():
            raise FloatingPointError(f"积分产生非有限状态，时间 {(index + 1) * step_s:g} s")
        boundary = end


def example_boundary(time_s):
    """独立运行的示例输入。振幅和波形仅用于演示。"""
    respiratory_phase = 2 * pi * time_s / 4.0
    return Boundary(
        85.0 + 60.0 * sin(2 * pi * time_s / 0.85),
        4.35,
        -3.68 - 0.8 * sin(respiratory_phase),
        -0.35 * sin(respiratory_phase),
        2.25 + 0.14 * (1 - np.cos(respiratory_phase)),
    )


def main():
    parser = argparse.ArgumentParser(description="运行论文肺循环模型的独立边界示例")
    parser.add_argument("--output", required=True, type=Path, help="结果 CSV 路径")
    parser.add_argument("--duration-s", type=float, default=8.0, help="仿真时长，单位为 s")
    parser.add_argument("--drive", choices=("mechanics", "prescribed"), default="mechanics",
                        help="采用呼吸力学计算外压或直接给定示例波形")
    args = parser.parse_args()
    if not isfinite(args.duration_s) or args.duration_s < PAPER_STEP_S:
        parser.error("仿真时长须至少为一个论文积分步长")
    boundary_at = example_boundary
    if args.drive == "mechanics":
        from respiration import ALBANESE_2016, quiet_muscle_pressure, solve_drive

        # 用 2021 年论文的名义肺泡容积初始化独立示例。
        unstressed = NOMINAL.frc_l + (
            ALBANESE_2016.compliance_l_cmh2o[-1] * ALBANESE_2016.pleural_end_expiratory_cmh2o
        )
        mechanics = replace(
            ALBANESE_2016, unstressed_volume_l=(*ALBANESE_2016.unstressed_volume_l[:3], unstressed)
        )
        drive_at = solve_drive(args.duration_s, airway_at=lambda _: 0.0,
                               muscle_at=quiet_muscle_pressure, parameters=mechanics)

        def boundary_at(time_s):
            p_pl, p_a, v_a = drive_at(time_s)
            return Boundary(85.0 + 60.0 * sin(2 * pi * time_s / 0.85), 4.35, p_pl, p_a, v_a)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        output = csv.writer(stream)
        output.writerow([
            "time_s", "v_pa_ml", "v_pal_ml", "v_pc_ml", "v_pv_ml",
            "p_pa_mmhg", "p_pal_mmhg", "p_pc_mmhg", "p_pv_mmhg",
            "q_pa_ml_s", "q_pal_ml_s", "q_pc_ml_s", "q_ps_ml_s", "q_pv_ml_s", "net_inflow_ml",
        ])
        for time_s, state, pressure, flow, net_inflow_ml in simulate(boundary_at, args.duration_s):
            output.writerow([time_s, *state[:4], *pressure, *flow, net_inflow_ml])
    print(f"结果已写入 {args.output}")


if __name__ == "__main__":
    main()
