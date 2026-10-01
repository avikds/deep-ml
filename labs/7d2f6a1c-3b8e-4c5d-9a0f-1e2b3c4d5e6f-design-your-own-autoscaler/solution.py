import numpy as np


class Autoscaler:
    def __init__(self, min_replicas, max_replicas, replica_capacity, cold_start_s):
        self.min_replicas = int(min_replicas)
        self.max_replicas = int(max_replicas)
        self.replica_capacity = float(replica_capacity)
        self.cold_start_s = float(cold_start_s)

        self.ewma = None
        self.prev_load = None
        self.trend = 0.0
        self.low_load_steps = 0

    def step(self, t, offered_load, queue, ready, pending):
        """Return the total number of replicas (ready + pending) to target."""
        load = max(0.0, float(offered_load))
        queue = max(0.0, float(queue))
        ready = int(ready)
        pending = int(pending)

        # Reset state if a new trace starts.
        if self.prev_load is not None and t <= 0:
            self.ewma = None
            self.prev_load = None
            self.trend = 0.0
            self.low_load_steps = 0

        if self.ewma is None:
            self.ewma = load
            self.prev_load = load
        else:
            delta = load - self.prev_load
            self.trend = 0.75 * self.trend + 0.25 * delta
            self.ewma = 0.80 * self.ewma + 0.20 * load
            self.prev_load = load

        # Forecast over the cold-start horizon.
        positive_trend = max(0.0, self.trend)
        forecast = self.ewma + positive_trend * self.cold_start_s

        # Never forecast below current demand; keep modest headroom.
        predicted_load = max(load, forecast)
        target_capacity = predicted_load * 1.10

        target = int(np.ceil(target_capacity / self.replica_capacity))

        # Aggressively clear an existing queue: enough capacity for
        # current work plus the backlog to drain within about one second.
        if queue > 0.0:
            queue_target = int(
                np.ceil((load + queue) / self.replica_capacity)
            )
            target = max(target, queue_target)

        # Rapidly react to a sharp upward move.
        if ready > 0 and load > ready * self.replica_capacity * 0.90:
            target = max(
                target,
                int(np.ceil(load / self.replica_capacity)) + 1
            )

        # Conservative scale-down hysteresis.
        if queue <= 0.0 and ready > self.min_replicas:
            if load < ready * self.replica_capacity * 0.65:
                self.low_load_steps += 1
            else:
                self.low_load_steps = 0

            if self.low_load_steps < 30:
                target = max(target, ready)
        else:
            self.low_load_steps = 0

        # Never ask for fewer replicas than are already booting.
        target = max(target, pending, self.min_replicas)
        target = min(target, self.max_replicas)

        return int(target)