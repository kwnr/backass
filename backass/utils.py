import numpy as np
import time
from typing import TypedDict, List

class ExponentialMovingAverage():
    def __init__(self, w) -> None:
        """
        s_0 = x_0 \n
        s_t = w*x_{t-1} + (1-w)*s_{t-1} \n
        """
        self.prev = None
        self.weight = w
    
    def update(self, val):
        if self.prev == None:
            self.prev = val
        ret = self.weight * val + (1 - self.weight) * self.prev
        self.prev = ret
        return ret


class LowPassFilter(object):
    def __init__(self, cut_off_freqency, ts):
    	# cut_off_freqency: 차단 주파수
        # ts: 주기
        # https://velog.io/@7cmdehdrb/LowPassFilter
        
        self.ts = ts
        self.cut_off_freqency = cut_off_freqency
        self.tau = self.get_tau()

        self.prev_data = None
        
    def get_tau(self):
        return 1 / (2 * np.pi * self.cut_off_freqency)

    def filter(self, data):
        if self.prev_data is None:
            self.prev_data = data
        val = (self.ts * data + self.tau * self.prev_data) / (self.tau + self.ts)
        self.prev_data = val
        return val


class MovingAverage(object):
    def __init__(self, window_size) -> None:
        self.window_size = window_size
        self.ma = []
    
    def filter(self, val):
        if len(self.ma) < self.window_size:
            self.ma.append(val)
        else:
            self.ma.pop(0)
            self.ma.append(val)
        return sum(self.ma) / len(self.ma)


class NoiseFilter:
    def __init__(self) -> None:
        self.prev = np.zeros(16)
        self.threshold = 2
        self.is_prev_initialized = False

    def update(self, value):
        if not self.is_prev_initialized:
            self.prev = value
            self.is_prev_initialized = True
        to_filter = abs((value - self.prev)) > self.threshold
        res = self.prev * to_filter + value * np.logical_not(to_filter)
        """ if any(to_filter):
            print(f"from index {np.where(to_filter)}")
            print(f"{value[to_filter]} is ignored")
            print(f"{self.prev[to_filter]} used instead")
            print(f"result is : {res}") """
        self.prev = res
        return res


class JointPosition(object):
    def __init__(self) -> None:
        self.t = 0.
        self.pos = None
        self.vel = np.zeros(16, dtype=float)
        self.acc = np.zeros(16, dtype=float)
        
        self.t_prev = 0.
        self.pos_prev = np.zeros(16, dtype=float)
        self.vel_prev = np.zeros(16, dtype=float)
        self.acc_prev = np.zeros(16, dtype=float)
        
        self.t_pprev = 0.
        self.pos_pprev = np.zeros(16, dtype=float)
        self.vel_pprev = np.zeros(16, dtype=float)
        self.acc_pprev = np.zeros(16, dtype=float)        
        
    def append(self, val: np.ndarray):
        self.t = time.perf_counter()
        self.pos = val
        self.vel = (self.pos - self.pos_prev) / (self.t - self.t_prev)
        self.acc = (self.vel - self.vel_prev) / (self.t - self.t_prev)
        
        self.t_pprev = self.t_prev
        self.pos_pprev = self.pos_prev
        self.vel_pprev = self.vel_prev
        self.acc_pprev = self.acc_prev
        
        self.t_prev = self.t
        self.pos_prev = self.pos
        self.vel_prev = self.vel
        self.acc_prev = self.acc

    
class SlaveState(TypedDict):
    frame: int
    joint_pos: np.ndarray


class CtrlState(TypedDict):
    frame: int
    cmd: np.ndarray
    err: np.ndarray
    err_i: np.ndarray
    err_d: np.ndarray
    err_norm: float


class PumpState(TypedDict):
    frame: int
    act_rpm: int
    des_rpm: int
    des_cur: float
    temp: int

from collections.abc import Set
from interass.msg import UI2Robot
class CtrlCommand(TypedDict):
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_acc: np.ndarray
    ref_pos: np.ndarray
    joint_power: List
    smooth_factor: np.ndarray
    cmd_override: np.ndarray
    max_rpm: int
    des_rpm: int


class PumpCommand(TypedDict):
    power: bool
    mode: int
    tgt_rpm: int
    err_norm: float
    max_rpm: int
    min_rpm: int
    max_err: int


class DXLCommand(TypedDict):
    pos_cmd: np.ndarray
    
    
