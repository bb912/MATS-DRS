from collections import deque
from math import sqrt


class StatsTracker:
    def __init__(self):
        self.num = 0
        self.mean = 0
        self.sum_squares = 0

    def add_point(self, point):
        self.num += 1
        next_mean = self.mean + (point - self.mean) / self.num
        self.sum_squares += (point - self.mean) * (point - next_mean)
        self.mean = next_mean

    def remove_point(self, point):
        assert self.num > 0

        if self.num == 1:
            self.num = 0
            self.mean = 0
            self.sum_squares = 0
        else:
            mean_without = (self.num * self.mean - point) / (self.num - 1)
            self.sum_squares -= (point - self.mean) * (point - mean_without)
            self.mean = mean_without
            self.num -= 1

    def average(self):
        return self.mean

    def variance(self):
        if self.num > 1:
            return self.sum_squares / (self.num - 1)
        else:
            return 0

    def stddev(self):
        return sqrt(self.variance())


class MovingStatsTracker(StatsTracker):
    def __init__(self, last_n):
        super().__init__()
        self.last_n = last_n
        self.last_queue = deque(maxlen=last_n)

    def add_point(self, point):
        super().add_point(point)
        self.last_queue.append(point)
        if len(self.last_queue) == self.last_n:
            super().remove_point(self.last_queue.popleft())
