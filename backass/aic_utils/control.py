from multiprocessing import Process, Pipe
import numpy as np
import time
import rclpy
from rclpy.node import Node

from Phidget22.Devices import VoltageOutput
from Phidget22 import ErrorCode
from Phidget22 import PhidgetException

import traceback
import logging

class Control:
    def __init__(self, pipeset):
        print("Initializing Control Process...")
        self.conn = pipeset[0]
        self.conn_opp = pipeset[1]
        self.conn_slow = pipeset[2]
        self.conn_slow_opp = pipeset[3]
        self.conn_pause = pipeset[4]
        self.conn_pause_opp = pipeset[5]

        self.ctrl_cmd: np.ndarray = np.zeros(16)  # generated in this process
        self._joint_pos: np.ndarray = np.zeros(16)  # received from state over main
        self._ref_pos: np.ndarray = np.zeros(16)  # received from master over main

        self.joint_power: np.ndarray = np.ones(16)

        self.phidget_1002 = [VoltageOutput.VoltageOutput() for i in range(16)]

        self.log_size = 64
        self.log_i_size = 1000
        self._err: np.ndarray = np.zeros(16)
        self._err_i: np.ndarray = np.zeros(16)
        self._err_d: np.ndarray = np.zeros(16)

        self._err_log: np.ndarray = np.zeros((self.log_size, 16))
        self._err_i_log: np.ndarray = np.zeros((self.log_i_size, 16))
        self._err_d_log: np.ndarray = np.zeros((self.log_size, 16))
        self._timestamp: np.ndarray = np.arange(self.log_size).reshape(self.log_size, 1)
        
        self.smooth_factor = np.ones(16)
        self.is_clamp = np.array([False] * 16)
        
        self.cmd_override = np.array([np.nan]*16, dtype=float)

        self.err_norm = 0.
        
        self.phidget_retry_counter = 1
                
        self.joint_min = np.array([-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
                                   -1000, -1000, -1000, -1000, -1000, -1000, 34.16, -56])
        self.joint_max = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                                   1000, 1000, 1000, 1000, 1000, 1000, 34.7, -46])
        
        # 34.1, -56
        # 34.2, -46

        self.kp = np.array([.25*2, .25*2, .5, .5, .5, .5, .5, .5*2,] * 2)
        self.ki = np.array([.0125/2, .0125/2, .0125/4, .0125/2, .0125*2, .0125/2, .0125/2, .0125]*2)
        self.kd = np.array([1/2, 1/2, 2, 1, 2, 1, 1, 1*2] * 2)
        self.integrator = Integrator()
        self.integrator_der = Integrator()
        
        self.lpm8_flow = np.polynomial.Polynomial([0, 0.822176, -0.02439, 0.227971, -0.03879, 0.00184])
        self.lpm4_flow = np.polynomial.Polynomial([0, 0.86869, 0.16426, -0.01125])
        
        self.k_fd = 0

    def __del__(self):
        self.close_phidget()
        print(f"class {__name__} deleted")

    def init_phidget(self):
        try:
            for ch in range(4):
                self.phidget_1002[ch].setDeviceSerialNumber(525285)  # L1~L4
                self.phidget_1002[ch].setChannel(ch)
                self.phidget_1002[ch].openWaitForAttachment(5000)
                self.phidget_1002[ch+4].setDeviceSerialNumber(525266)  # L5~L8
                self.phidget_1002[ch+4].setChannel(ch)
                self.phidget_1002[ch+4].openWaitForAttachment(5000)
                self.phidget_1002[ch+8].setDeviceSerialNumber(525068)  # R1~R4
                self.phidget_1002[ch+8].setChannel(ch)
                self.phidget_1002[ch+8].openWaitForAttachment(5000)  # 연결을 5000ms 까지 대기함
                self.phidget_1002[ch+12].setDeviceSerialNumber(525324)  # R5~R8
                self.phidget_1002[ch+12].setChannel(ch)
                self.phidget_1002[ch+12].openWaitForAttachment(5000)
                
        except VoltageOutput.PhidgetException as e:
            print("PHIDGET_ERR ", e)
            print("Fail to initiate Phidget board... closing control...")
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            
    def close_phidget(self):
        for ch in range(4):
            self.phidget_1002[ch].close()
            self.phidget_1002[ch+4].close()
            self.phidget_1002[ch+8].close()
            self.phidget_1002[ch+12].close()

    def receiver(self):
        items = self.conn.recv()
        self.joint_power = np.array(items["joint_power"])
        self.joint_pos = np.array(items["joint_pos"])
        self.ref_pos = np.array(items["ref_pos"])
        self.smooth_factor = np.array(items["smooth_factor"])
        self.cmd_override = np.array(items["cmd_override"], dtype=float)
        self.max_rpm = np.array(items["max_rpm"])
        self.des_rpm = np.array(items["des_rpm"])

    def send_result(self, value):
        self.conn.send(value)

    def err_calc(self):
        self.err = self.ref_pos - self.joint_pos
        self.err = (self.err * self.smooth_factor) * self.joint_power * np.isnan(self.cmd_override)
        self.err_norm = np.linalg.norm(self.err * np.array(
            [1.73, 0.97, 0.97, 1.73, 0.19, 0.19, 0, 0,
             1.73, 0.97, 0.97, 1.73, 0.19, 0.19, 0, 0]) * self.joint_power, 1)
        
        return self.err
    
    def err_i_calc(self, ns, err):
        err_clamped = err * ~self.is_clamp
        self.integrator.value *= (self.joint_power * np.isnan(self.cmd_override))
        err_i = self.integrator.update(ns, err_clamped)
        return err_i
    
    def err_d_calc(self, ns, err, cutoff_freq=0.01):
        es = err - self.integrator_der.value
        err_d = (es * cutoff_freq) * self.joint_power * np.isnan(self.cmd_override)
        self.integrator_der.update(ns, err_d)
        return err_d
    
    def pid_alt(self, err, err_i, err_d):
        k = .5
        gamma = 5
        kp = 0.7
        ki = 0.1  # kp**2 > 2ki
        
        K = k * np.eye(16)
        KP = kp * np.eye(16)
        KI = ki * np.eye(16)
        
        res = (K + 1 / (gamma ** 2) * np.eye(16)) @ (err_d + KP @ err + KI @ err_i)
                
        return res
    

    def allocate_cmd(self, cmd):
        cmd = np.clip(cmd, -9.9, 9.9)
        try:
            for ch in range(4):
                self.phidget_1002[ch].setVoltage_async(cmd[ch], self.ph_async_cb)
                self.phidget_1002[ch+4].setVoltage_async(cmd[ch + 4], self.ph_async_cb)
                self.phidget_1002[ch+8].setVoltage_async(cmd[ch + 8], self.ph_async_cb)
                self.phidget_1002[ch+12].setVoltage(cmd[ch + 12])
                
        except PhidgetException.PhidgetException as e:
            print(e)
            t_0 = time.time()
            print(traceback.format_exc())
            print(f"retry to reconnect phidget. attempt {self.phidget_retry_counter}")
            while time.time() - t_0 < 1:
                pass
            self.phidget_retry_counter += 1
            self.close_phidget()
            self.init_phidget()
            
    def set_phidget_enabled(self, idx: list):
        for i, phidget in enumerate(self.phidget_1002):
            if i in idx:
                phidget.setVoltage(0)
                phidget.setEnabled(True)
            
    def set_phidget_disabled(self, idx: list):
        for i, phidget in enumerate(self.phidget_1002):
            if i in idx:
                phidget.setVoltage(0)
                phidget.setEnabled(False)
                
    def joint_flow_estimator(self, v: np.ndarray):
        flow_4 = self.lpm4_flow(v)
        flow_8 = self.lpm8_flow(v)
        res = abs(np.where([True, False, False, True, False, False, False, False] * 2,
                        flow_8, flow_4))
        return res
                
    @staticmethod
    def mhpu_flow_estimator(rpm):
        return rpm * 2.7 / 1000
    
                         
    @staticmethod      
    def flow_distributer(vd: np.ndarray, Q_j, Q_h):
        k_fd = np.divide(Q_h, Q_j.sum()+1.35, out=np.zeros(1), where=Q_j.sum()!=0)
        v_fd = np.multiply(vd, k_fd, out=vd, where=k_fd<1)
        return v_fd
        
    
    @staticmethod
    def joint_saturator(v):
        pump_4lpm_sat_v = 5.4
        pump_8lpm_sat_v = 3.52
        
        v_p_max = np.where([True, False, False, True, False, False, False, False] * 2, pump_8lpm_sat_v, pump_4lpm_sat_v)
        v_h_max = np.array([3.27, 4.06, 4.06, 3.27, 1.5, 1.5, 1.5, 1.5] * 2) * 1.4  # saturated by hose 
        v = np.clip(v, -v_p_max, v_p_max)
        v = np.clip(v, -v_h_max, v_h_max)
        return v
        
    def sliding_mode_controller(self, err, err_d):
        lbd = 0.15
        K_S = self.kp / 100 * 2
        s = K_S * np.sign(lbd * err + err_d)
        return s                
    
    def min_max_saturation(self, cmd):
        is_greater = self.joint_pos >= self.joint_max
        is_lesser = self.joint_pos <= self.joint_min
        for i in range(16):
            if is_greater[i] and cmd[i] > 0:
                cmd[i] = 0
            if is_lesser[i] and cmd[i] < 0:
                cmd[i] = 0
        return cmd
        
    def run(self):
        # rate = rospy.Rate(1000)
        frame = 0
        f = open("log_kfd.txt", 'w')
        try:
            self.init_phidget()
            self.set_phidget_enabled(list(range(16)))
            clamp_sat_limit = 5.5
            actuator_sat_limit = 9.9
            while True:
                if self.conn.poll():
                    self.receiver()
                t = time.monotonic_ns()
                err = self.err_calc()
                err = err * [1, 1, 1, 1, -1, 1, 1, 1,
                             1, 1, 1, 1, -1, 1, 1, 1,]
                err_i = self.err_i_calc(t, err)
                err_d = self.err_d_calc(t, err)
                vd = err * self.kp + err_i * self.ki + err_d * self.kd + self.sliding_mode_controller(err, err_d)
                vd_sat = self.joint_saturator(vd)
                self.is_clamp = (vd_sat != vd) & (np.sign(vd) == np.sign(err))
                Q_j_est = self.joint_flow_estimator(vd_sat)
                Q_p_est = self.mhpu_flow_estimator(self.des_rpm)
                v_fc = self.flow_distributer(vd_sat, Q_j_est, Q_p_est)
                v_dc = self.dead_zone_compensate(v_fc, 1.2, 0.01)
                cmd = v_dc
                cmd[~np.isnan(self.cmd_override)] = self.cmd_override[~np.isnan(self.cmd_override)]
                cmd = self.min_max_saturation(cmd)
                cmd = np.clip(cmd, -actuator_sat_limit, actuator_sat_limit)  # saturate command in range of phidget's range
                self.allocate_cmd(cmd)
                frame += 1
                if not self.conn_opp.poll():
                    self.send_result(
                        {
                            "frame": frame,
                            "cmd": cmd,
                            "err": err,
                            "err_i": err_i,
                            "err_d": err_d,
                            "err_norm": self.err_norm
                        }
                    )
                f.write(f"{self.k_fd}\n")
                    
                    
        except KeyboardInterrupt:
            self.close_phidget()
            print(f"Inturrupted by user. Process {__name__} closed.")
            return


    @property
    def joint_pos(self):
        return self._joint_pos

    @joint_pos.setter
    def joint_pos(self, value):
        if np.shape(value) == (16,):
            self._joint_pos = value
        else:
            raise ValueError(f"incorrect shape {np.shape(value)}. Shape should be (16,)")

    @property
    def ref_pos(self):
        return self._ref_pos

    @ref_pos.setter
    def ref_pos(self, value):
        if np.shape(value) == (16,):
            self._ref_pos = value
        else:
            raise ValueError(f"incorrect shape {np.shape(value)}. Shape should be (16,)")

    @property
    def err(self):
        return self._err

    @err.setter
    def err(self, value):
        if np.shape(value) == (16,):
            self._err = value
        else:
            raise ValueError(f"incorrect shape {np.shape(value)} for err. Shape should be (16,)")

    @property
    def err_i(self):
        return self._err_i

    @err_i.setter
    def err_i(self, value):
        if np.shape(value) == (16,):
            self._err_i = value
        else:
            raise ValueError(f"incorrect shape {np.shape(value)}. Shape should be (16,)")

    @property
    def err_d(self):
        return self._err_d

    @err_d.setter
    def err_d(self, value):
        if np.shape(value) == (16,):
            self._err_d = value
        else:
            raise ValueError(f"incorrect shape {np.shape(value)}. Shape should be (16,)")

    @property
    def err_log(self):
        return self._err_log

    @err_log.setter
    def err_log(self, value):
        if np.shape(value) == (self.log_size, 16):
            self._err_log = value
        else:
            raise ValueError(f"incorrect shape {np.shape(value)}. Shape should be (self.log_size, 16)")

    @property
    def err_i_log(self):
        return self._err_i_log

    @err_i_log.setter
    def err_i_log(self, value):
        if np.shape(value) == (self.log_i_size, 16):
            self._err_i_log = value
        else:
            raise ValueError(f"incorrect shape {np.shape(value)}. Shape should be (self.log_size, 16)")

    @property
    def err_d_log(self):
        return self._err_d_log

    @err_d_log.setter
    def err_d_log(self, value):
        if np.shape(value) == (self.log_size, 16):
            self._err_d_log = value
        else:
            raise ValueError(f"incorrect shape {np.shape(value)}. Shape should be (self.log_size, 16)")

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        if np.shape(value) == (self.log_size, 1):
            self._timestamp = value
        else:
            raise ValueError(f"incorrect shape {np.shape(value)}. Shape should be ({self.log_size, 1})")

    @staticmethod
    def ph_async_cb(ch, res, details):
        if res != ErrorCode.ErrorCode.EPHIDGET_OK:
            print(f"Async Failure, errno{res}: {details} at {ch}")

    @staticmethod
    def np_logging(log, value):
        log[:-1, :] = log[1:, :]
        log[-1:, :] = value[0]
        return log

    @staticmethod
    def saturation(value: np.ndarray, _min, _max) -> np.ndarray:
        value = np.clip(value, _min, _max)
        return value

    @staticmethod
    def dead_zone_compensate(value: np.ndarray, compensation: float, tol: float) -> np.ndarray:
        res =  ((value + np.sign(value) * compensation) * (abs(value) > tol) + 
                (value * (compensation + tol)/tol) * (abs(value) <= tol))
        return res
    
class Integrator():
    
    def __init__(self) -> None:
        self.t_prev = None
        self.data_prev = None
        self.value = np.zeros(16)
    
    def update(self, t, data):
        if self.t_prev is not None and self.data_prev is not None:
            self.value += (t - self.t_prev) * (data + self.data_prev) / 2 / 1e9
        self.t_prev = t
        self.data_prev = data
        return self.value
    

if __name__ == "__main__":
    rclpy.init()
    node = Node("t")
    this_conn, that_conn = Pipe()
    ctrl = Control(that_conn, this_conn)
    ctrl.run()