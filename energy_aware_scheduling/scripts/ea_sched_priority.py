#!/usr/bin/env python3
import math
import datetime

class SimTime:
    """
    A simple stand‐in for datetime: holds simulation seconds and exposes
    both .timestamp() and .hour
    """
    def __init__(self, seconds):
        self._seconds = seconds

    def timestamp(self):
        return self._seconds

    @property
    def hour(self):
        # Wrap to [0..23]
        return int(self._seconds // 3600) % 24

class Job:
    def __init__(self, predicted_power, predicted_runtime):
        """
        predicted_power: in Watts
        predicted_runtime: in seconds
        """
        self.predicted_power = predicted_power
        self.predicted_runtime = predicted_runtime

class EnergyScheduler:
    def __init__(self, re_map, P_min, P_max,
                 power_time_boost_start, power_time_boost_end,
                 power_alpha, power_beta, power_gamma,
                 power_weight):
        # Renewable‐energy map: {timestamp_seconds → availability in [0,1]}
        self.re_map = re_map
        self.re_min_time = min(re_map.keys()) if re_map else 0
        self.re_max_time = max(re_map.keys()) if re_map else 0

        # Power normalization
        self.P_min = P_min
        self.P_max = P_max

        # Time‐of‐day boost window
        self.power_time_boost_start = power_time_boost_start
        self.power_time_boost_end   = power_time_boost_end

        # Exponents & weights
        self.power_alpha  = power_alpha
        self.power_beta   = power_beta
        self.power_gamma  = power_gamma
        self.power_weight = power_weight

        # This will be set to a SimTime() before calling _power_priority()
        self.time = None

    def _get_re_availability(self, current_time):
        if self.re_map is None:
            return 0.0
        if current_time <= self.re_min_time:
            return self.re_map[self.re_min_time]
        if current_time >= self.re_max_time:
            return self.re_map[self.re_max_time]
        return self.re_map.get(current_time, 0.0)

    def _power_priority(self, job):
        # current_time
        if hasattr(self.time, "timestamp"):
            current_time = int(round(self.time.timestamp()))
        else:
            current_time = int(round(self.time))

        # RE availability lookup
        re_avail = self._get_re_availability(current_time)

        # normalize predicted power
        P_job = job.predicted_power
        normalized_P = (P_job - self.P_min) / (self.P_max - self.P_min)
        normalized_P = max(0.0, min(normalized_P, 1.0))

        # determine hour
        if hasattr(self.time, "hour"):
            current_hour = self.time.hour
        else:
            current_hour = datetime.datetime.fromtimestamp(self.time).hour

        # time‐based boost
        if self.power_time_boost_start <= current_hour < self.power_time_boost_end:
            time_boost = normalized_P
        else:
            time_boost = 1 - normalized_P

        # RE‐based boost
        if re_avail > 0:
            re_boost = normalized_P * (re_avail ** self.power_alpha)
        else:
            re_boost = 1 - normalized_P

        # runtime blend
        runtime_hours = job.predicted_runtime / 3600.0
        runtime_blend = min((runtime_hours / 8.0) ** self.power_beta, 1.0)

        # final factor & weighted priority
        power_factor = ((runtime_blend + self.power_gamma) * time_boost +
                        (2 - runtime_blend - self.power_gamma) * re_boost) / 2
        return self.power_weight * power_factor

def generate_solar_re_map():
    """
    Generates a simple solar availability map for 0–23h (at each hour).
    Availability = sin curve between 6h (0) → 12h (1) → 18h (0), else 0.
    """
    re_map = {}
    for h in range(24):
        ts = h * 3600
        if 6 <= h <= 18:
            angle = math.pi * (h - 6) / 12.0
            avail = math.sin(angle)
        else:
            avail = 0.0
        re_map[ts] = round(avail, 3)
    return re_map

if __name__ == "__main__":
    # Build toy "scheduler"
    re_map = generate_solar_re_map()
    scheduler = EnergyScheduler(
        re_map=re_map,
        P_min=0,
        P_max=1000,
        power_time_boost_start=8,
        power_time_boost_end=16,
        power_alpha=1.0,
        power_beta=1.0,
        power_gamma=0.5,
        power_weight=10.0
    )

    # Five examples: (power W, runtime h, hour of day)
    examples = [
        (400, 2,  7),   # early morning, low RE
        (600, 4, 10),   # late morning, rising RE
        (800, 1, 12),   # noon, peak RE
        (500, 8, 16),   # late afternoon, falling RE
        (700, 3, 20),   # night, zero RE
    ]

    # 3) Run & print
    for i, (p, rt_h, hr) in enumerate(examples, 1):
        job = Job(predicted_power=p, predicted_runtime=rt_h * 3600)
        scheduler.time = SimTime(hr * 3600)
        priority = scheduler._power_priority(job)
        re_avail  = scheduler._get_re_availability(int(round(scheduler.time.timestamp())))
        print(f"Example {i}: time={hr:02d}:00, power={p} W, runtime={rt_h} h → "
              f"RE_avail={re_avail:.3f}, priority={priority:.3f}")
