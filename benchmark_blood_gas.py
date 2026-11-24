"""按明确的时间窗汇总肺血流逆向量和血气守恒。"""

import argparse
import json
from pathlib import Path

import numpy as np


FLOW_FIELDS = ("q_pulmonary_venous", "q_pa", "q_pal", "q_pc", "q_shunt")
SIGNAL_FIELDS = ("pa_o2_mmhg", "pa_co2_mmhg", "f_peripheral_chemo",
                 "respiratory_rate", "muscle_pressure_cmh2o", "p_pa", "p_la")


def reverse_metrics(time, flow):
    """按采样点间线性插值拆分零交叉，分别计算正向、逆向积分。"""
    flips = flow[1:] * flow[:-1] < 0.
    crossings = time[:-1][flips] - flow[:-1][flips] * np.diff(time)[flips] / np.diff(flow)[flips]
    t = np.sort(np.r_[time, crossings])
    q = np.interp(t, time, flow)
    forward = np.trapezoid(np.maximum(q, 0.), t)
    reverse = np.trapezoid(np.maximum(-q, 0.), t)
    reverse_time = np.diff(t)[q[:-1] + q[1:] < 0.].sum()
    return dict(min_ml_s=float(flow.min()), forward_ml=float(forward), reverse_ml=float(reverse),
                reverse_duration_s=float(reverse_time),
                reverse_fraction=float(reverse / forward) if forward > 0. else None,
                min_time_s=float(time[flow.argmin()]))


def summarize(rows, start, end):
    t = rows["time_s"]
    if not t[0] <= start < end <= t[-1]:
        raise ValueError("分析窗口必须在输出时间范围内")
    times = np.r_[start, t[(t > start) & (t < end)], end]
    columns = rows.dtype.names

    def field(name):
        return np.interp(times, t, rows[name])

    result = {
        "window_s": [start, end],
        "flow": {name: reverse_metrics(times, field(name)) for name in FLOW_FIELDS if name in columns},
        "signals": {name: dict(mean=float(np.trapezoid(field(name), times) / (end - start)),
                               min=float(field(name).min()), max=float(field(name).max()))
                    for name in SIGNAL_FIELDS},
    }
    index = int(np.argmin(field("q_pulmonary_venous")))
    minimum = {
        name: float(field(name)[index]) for name in ("p_pv", "p_la", "p_pleural", "p_pericardial", "q_mv")
    }
    result["pulmonary_venous_minimum"] = dict(
        minimum,
        pressure_gradient_mmhg=minimum["p_pv"] - minimum["p_la"],
        venous_transmural_mmhg=minimum["p_pv"] - minimum["p_pleural"],
        atrial_transmural_mmhg=minimum["p_la"] - minimum["p_pericardial"],
        pericardial_transmural_mmhg=minimum["p_pericardial"] - minimum["p_pleural"],
    )
    result["balance_max_abs"] = {name: float(np.abs(field(name)).max()) for name in
                                 ("pulmonary_o2_balance", "pulmonary_co2_balance") if name in columns}
    return result


def main():
    parser = argparse.ArgumentParser(description="汇总肺血流方向与闭环血气输出")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = np.genfromtxt(args.csv, delimiter=",", names=True, encoding="utf-8")
    summary = summarize(rows, args.start_s, args.end_s)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
