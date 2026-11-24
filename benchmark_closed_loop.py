"""从原步长闭环输出选取完整心搏，按论文口径计算压力和容积。"""

import argparse
import json
from pathlib import Path

import numpy as np
from blood_gas import DRY_PRESSURE_MMHG
from respiration import CMH2O_TO_MMHG


def beats(rows):
    time = rows["time_s"]
    phase = rows["heart_cycle"]
    crossings = np.interp(np.arange(np.ceil(phase[0]), np.floor(phase[-1]) + 1.), phase, time)
    return list(zip(crossings[:-1], crossings[1:]))


def beat_metrics(rows, left, right):
    sample_time = rows["time_s"]
    time = np.r_[left, sample_time[(sample_time > left) & (sample_time < right)], right]

    def field(name):
        return np.interp(time, sample_time, rows[name])

    def mean(values):
        return np.trapezoid(values, time) / (right - left)

    pressure = {}
    for name in ("ao", "pa", "la", "ra", "lv", "rv"):
        p = field("p_" + name)
        pressure[name] = {"mean": mean(p), "peak": p.max(), "min": p.min()}
    ventricular_volume = {}
    for chamber, valve in (("lv", "aortic"), ("rv", "pulmonary")):
        v = field("v_" + chamber)
        ventricular_volume[chamber] = {
            "edv_ml": v.max(),
            "esv_ml": v.min(),
            "sv_ml": np.trapezoid(field("q_" + valve + "_valve"), time),
        }
        pressure[chamber]["edp"] = field("p_" + chamber)[v.argmax()]
    return {
        "beat_s": [left, right], "pressure_mmhg": pressure, "ventricular_volume": ventricular_volume,
        "heart_rate_bpm": 60. / (right - left),
        "cardiac_output_l_min": .06 * mean(field("q_aortic_valve")),
    }


def resting_summary(rows):
    time = rows["time_s"]
    muscle = rows["muscle_pressure_cmh2o"]
    minimum = (muscle[1:-1] > muscle[:-2]) & (muscle[1:-1] >= muscle[2:])
    starts = time[1:-1][minimum]
    if starts.size < 2:
        raise ValueError("需至少两个完整呼吸起点")
    completed_beats = [(left, right) for left, right in beats(rows) if right <= starts[-1]]
    if not completed_beats:
        raise ValueError("最后一次呼气末前没有可比较的完整心搏")
    result = beat_metrics(rows, *completed_beats[-1])
    breathing = rows[(time >= starts[-2]) & (time <= starts[-1])]
    tidal = np.ptp(breathing["v_alveolar"])
    rate = 60. / (starts[-1] - starts[-2])
    result["respiration"] = {
        "cycle_s": [starts[-2], starts[-1]], "tidal_l": tidal, "rate_bpm": rate,
        "minute_volume_l_min": tidal * rate,
        "alveolar_o2_mean_mmhg": DRY_PRESSURE_MMHG * np.mean(breathing["fa_o2"]),
        "alveolar_co2_mean_mmhg": DRY_PRESSURE_MMHG * np.mean(breathing["fa_co2"]),
    }
    return result


def figure6_summary(rows):
    time = rows["time_s"]
    end = 4. * np.floor(time[-1] / 4.)
    start = end - 4.
    if start < time[0]:
        raise ValueError("需至少一个完整的 4 s 气道压力重放周期")
    window = (time >= start) & (time <= end)
    intervals = [(left, right) for left, right in beats(rows) if start <= left and right <= end]
    matched = [beat_metrics(rows, left, right) for left, right in intervals]
    return {"breath_s": [start, end], "tidal_l": np.ptp(rows["v_alveolar"][window]),
            "end_exp_pleural_cmh2o": np.interp(end, time, rows["p_pleural"]) / CMH2O_TO_MMHG,
            "beats": matched}


def main():
    parser = argparse.ArgumentParser(description="汇总静息末次完整呼吸或图 6 最后一周期")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--case", choices=("resting", "figure6"), default="resting")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = np.genfromtxt(args.csv, delimiter=",", names=True, encoding="utf-8")
    result = resting_summary(rows) if args.case == "resting" else figure6_summary(rows)
    result["blood_volume_max_deviation_ml"] = np.max(np.abs(rows["blood_volume"] - rows["blood_volume"][0]))
    result["septal_volume_max_residual_ml"] = np.max(np.abs(rows["septal_volume_residual_ml"]))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
