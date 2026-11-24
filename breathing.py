"""博士论文 §2.2 的吸气终止事件。潮气量阈值函数由调用方提供。"""

from math import exp, floor, isfinite

from respiration import quiet_muscle_pressure


def triggered_expiration(elapsed, period, cutoff, rate, minimum):
    initial_pressure = quiet_muscle_pressure(cutoff, rate, minimum)
    tau = (period - cutoff) / 5.
    end = exp(-5.)
    return initial_pressure * (exp(-(elapsed - cutoff) / tau) - end) / (1. - end)


class HeringBreuerDrive:
    def __init__(self, threshold_at, breaths_per_minute=12., minimum_cmh2o=-5.):
        if not isfinite(breaths_per_minute) or breaths_per_minute <= 0:
            raise ValueError("呼吸频率须为有限正值")
        if not isfinite(minimum_cmh2o) or minimum_cmh2o > 0:
            raise ValueError("呼吸肌最小压力须为有限非正值")
        self.threshold_at = threshold_at
        self.rate = breaths_per_minute
        self.minimum = minimum_cmh2o
        self.period = 60. / breaths_per_minute
        self.inspiratory_time = self.period * (1. - 1. / 1.6)
        self.reset()

    def reset(self):
        """为一次新的独立仿真清除呼吸周期历史。"""
        self.cycle = -1
        self.initial_volume = 0.
        self.cutoff = None
        self.events_s = []

    def observe(self, time_s, alveolar_l):
        """只在已接受的积分节点观察容积并记录切换。"""
        cycle = floor(time_s / self.period)
        elapsed = time_s - cycle * self.period
        if cycle != self.cycle:
            self.cycle = cycle
            self.initial_volume = alveolar_l
            self.cutoff = None
        if self.cutoff is None and elapsed < self.inspiratory_time:
            threshold = self.threshold_at(time_s)
            if not isfinite(threshold) or threshold <= 0.:
                raise ValueError("潮气量终止阈值须为有限正值，单位为 L")
            if alveolar_l - self.initial_volume >= threshold:
                self.cutoff = elapsed
                self.events_s.append(time_s)

    def __call__(self, time_s):
        cycle = floor(time_s / self.period)
        elapsed = time_s - cycle * self.period
        if cycle != self.cycle or self.cutoff is None or elapsed < self.cutoff:
            return quiet_muscle_pressure(time_s, self.rate, self.minimum)
        return triggered_expiration(elapsed, self.period, self.cutoff, self.rate, self.minimum)


class AdaptiveBreathing:
    """在呼吸起点锁存化学反馈，牵张使用上一完整呼吸的潮气量。"""

    def __init__(self, initial_tidal_l, threshold_l=None, *, threshold_at=None):
        if threshold_l is not None and (not isfinite(threshold_l) or threshold_l <= 0):
            raise ValueError("显式潮气量阈值须为有限正值")
        if threshold_at is not None and (threshold_l is not None or not callable(threshold_at)):
            raise ValueError("血气依赖阈值须为函数，且与固定阈值互斥")
        self.threshold_l = threshold_l
        self.threshold_at = threshold_at
        self.tidal_l = initial_tidal_l
        self.events_s = []
        self.start_s = 0.
        self.next_s = 0.
        self.cutoff = None
        self.high_l = self.low_l = self.reference_l = 0.

    def observe(self, time_s, volume_l, minimum_cmh2o, rate, *, arterial_gases=None):
        if time_s >= self.next_s:
            if self.next_s > 0.:
                self.tidal_l = self.high_l - self.low_l
            if not isfinite(rate) or rate <= 0. or not isfinite(minimum_cmh2o) or minimum_cmh2o > 0.:
                raise ValueError(f"化学调节的呼吸设定超出定义域，时间 {time_s:g} s")
            self.start_s = time_s
            self.rate = rate
            self.minimum = minimum_cmh2o
            self.period = 60. / rate
            self.next_s = self.start_s + self.period
            self.reference_l = self.high_l = self.low_l = volume_l
            self.cutoff = None
        self.high_l = max(self.high_l, volume_l)
        self.low_l = min(self.low_l, volume_l)
        elapsed = time_s - self.start_s
        if self.cutoff is not None or elapsed >= self.period * (1. - 1. / 1.6):
            return
        threshold = self.threshold_l
        if self.threshold_at is not None:
            if arterial_gases is None:
                raise ValueError("血气依赖阈值需要动脉氧和二氧化碳分压")
            threshold = self.threshold_at(*arterial_gases)
            if not isfinite(threshold) or threshold <= 0.:
                raise ValueError("血气依赖阈值须返回有限正值，单位 L")
        if threshold is not None and volume_l - self.reference_l >= threshold:
            self.cutoff = elapsed
            self.events_s.append(time_s)

    def muscle_pressure(self, time_s):
        elapsed = time_s - self.start_s
        if elapsed >= self.period:
            return 0.
        if self.cutoff is not None and elapsed >= self.cutoff:
            return triggered_expiration(elapsed, self.period, self.cutoff, self.rate, self.minimum)
        return quiet_muscle_pressure(elapsed, self.rate, self.minimum)
