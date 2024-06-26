from multiprocessing import Pipe
import numpy as np
import time

from Phidget22.Devices import VoltageOutput
from Phidget22 import ErrorCode
from Phidget22 import PhidgetException
from Phidget22 import Phidget

import copy
import traceback

import rclpy
from rclpy.node import Node
import rclpy.time
import rclpy.timer

from trajectory_msgs.msg import JointTrajectoryPoint


class Control:
    def __init__(self, pipeset, parent_node: Node):
        self.node = parent_node
        self.node.get_logger().info("Initializing Control Process...")
        self.node.get_logger().set_level(10)
        self.conn = pipeset[0]
        self.conn_opp = pipeset[1]
        self.conn_slow = pipeset[2]
        self.conn_slow_opp = pipeset[3]
        self.conn_pause = pipeset[4]
        self.conn_pause_opp = pipeset[5]

        self.ctrl_cmd: np.ndarray = np.zeros(16)  # generated in this process
        self._joint_pos: np.ndarray = np.zeros(16)  # received from state over main
        self._ref_pos: np.ndarray = np.zeros(16)  # received from master over main

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

        self.joint_power = np.ones(16)
        self.joint_pos = np.zeros(16, float)
        self.joint_vel = np.zeros(16, float)
        self.joint_acc = np.zeros(16, float)

        self.ref_pos = np.ones(16)
        self.smooth_factor = np.ones(16)
        self.cmd_override = np.array([np.nan] * 16, dtype=float)
        self.max_rpm = 0
        self.des_rpm = 0

        self.err_norm = 0.0
        self.traj_point = JointTrajectoryPoint()

        self.phidget_retry_counter = 1

        self.joint_min = np.array(
            [
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                34.1,
                -56.,
            ]
        )
        self.joint_max = np.array(
            [
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                34.2,
                -46.,
            ]
        )

        # 34.1, -56
        # 34.2, -46

        self.kp = np.array(
            [
                0.25 * 2,
                0.25 * 2,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5 * 2,
            ]
            * 2
        )
        self.ki = np.array(
            [
                0.0125 / 2,
                0.0125 / 2,
                0.0125 / 4,
                0.0125 / 2,
                0.0125 * 4,
                0.0125 / 2,
                0.0125 / 2,
                0.0125,
            ]
            * 2
        )
        self.kd = np.array([2, 1 / 2, 2, 1, 2,
                            0.5, 1, 1 * 2] * 2)

        self.kp_l = np.array(
            [
                0.5,
                0.5,
                0.25,
                0.5,
                0.05,
                0.5,
                0.55,
                0.18,
            ]
            +
            [
                0.5,
                0.5,
                0.25,
                0.5,
                0.1,
                0.5,
                0.55,
                0.18,
            ]
        )
        self.ki_l = np.array(
            [
                0.0125,
                0.0125 / 2,
                0.0125 / 4,
                0.0125 / 2,
                0.0125 / 2,
                0.0125 / 2,
                0.0125,
                0.0125,
            ]
            +            
            [
                0.0125,
                0.0125 / 2,
                0.0125 / 4,
                0.0125 / 2,
                0.0125 * 4,
                0.0125 / 2,
                0.0125,
                0.0125,
            ]
        )
        self.kd_l = np.array([1, 2, 2, 1,5, 0.5, 1
                              , 1 ] * 2)
        self.integrator = Integrator()
        self.integrator_der = Integrator()
        self.st_w = Integrator()

        self.lpm4_volt2flow = np.polynomial.Polynomial([0, 7.98165981e-01, 2.13200807e-01,  -2.22038022e-02,  6.18481728e-04])
        self.lpm8_volt2flow = np.polynomial.Polynomial([0, 1.31690797,  -7.82755895e-01, 6.23129162e-01,  -1.30625241e-01, 1.16425220e-02,  -3.91794337e-04])
        self.lpm4_flow2volt = np.polynomial.Polynomial([0, 1.06211344e+00,  -9.80945796e-02, 8.65724068e-03,  -2.38136350e-04])
        self.lpm8_flow2volt = np.polynomial.Polynomial([0, 1.10423005e+00,  -1.81689101e-01, 1.95820118e-02,  -1.12389653e-03, 3.25762157e-05,  -3.70024441e-07])

        self.k_fd = 0

        self.q1_diff = FiniteDiff()
        self.q2_diff = FiniteDiff()
        self.e_diff = FiniteDiff()
        self.v_diff = FiniteDiff()
        self.q_ddiff = FiniteDDiff()
        self.con_q_integrator = Integrator()
        self.moving_average_filter = MovingAverage(window_size=1, max_change=0.05, threshold=1, reset_threshold=1)
        self.c1_prev = 0.0
        self.c2_prev = 0.0

        self.v = 0.0

        self.node.get_logger().info("Control Process Initialized")

    def __del__(self):
        self.close_phidget()
        print(f"class {__name__} deleted")

    def init_phidget(self):
        try:
            for ch in range(4):
                self.phidget_1002[ch].setDeviceSerialNumber(525285)  # L1~L4
                self.phidget_1002[ch].setChannel(ch)
                self.phidget_1002[ch].openWaitForAttachment(5000)

                self.phidget_1002[ch + 4].setDeviceSerialNumber(525266)  # L5~L8
                self.phidget_1002[ch + 4].setChannel(ch)
                self.phidget_1002[ch + 4].openWaitForAttachment(5000)

                self.phidget_1002[ch + 8].setDeviceSerialNumber(525068)  # R1~R4
                self.phidget_1002[ch + 8].setChannel(ch)
                self.phidget_1002[ch + 8].openWaitForAttachment(5000)

                self.phidget_1002[ch + 12].setDeviceSerialNumber(525324)  # R5~R8
                self.phidget_1002[ch + 12].setChannel(ch)
                self.phidget_1002[ch + 12].openWaitForAttachment(5000)

        except VoltageOutput.PhidgetException as e:
            self.node.get_logger().fatal(f"PHIDGET_ERR {e}")
            self.node.get_logger().fatal(
                "Fail to initiate Phidget board... closing control..."
            )
            exit(1002)

    def close_phidget(self):
        for ch in range(4):
            self.phidget_1002[ch].close()
            self.phidget_1002[ch + 4].close()
            self.phidget_1002[ch + 8].close()
            self.phidget_1002[ch + 12].close()
        Phidget.Phidget.finalize(0)

    def receiver(self):
        items: dict = self.conn.recv()
        self.joint_power = np.array(items["joint_power"])
        self.joint_pos = np.array(items["joint_pos"])
        self.joint_vel = np.array(items["joint_vel"])
        self.joint_acc = np.array(items["joint_acc"])
        self.ref_pos = np.array(items["ref_pos"])
        self.smooth_factor = np.array(items["smooth_factor"])
        self.cmd_override = np.array(items["cmd_override"], dtype=float)
        self.max_rpm = np.array(items["max_rpm"])
        self.des_rpm = np.array(items["des_rpm"])
        if 'traj_point' in items.keys():
            self.traj_point: JointTrajectoryPoint = items["traj_point"]

    def send_result(self, value):
        self.conn.send(value)

    def err_calc(self):
        self.err = self.ref_pos - self.joint_pos
        self.err = (
            # (self.err * self.smooth_factor)
            self.err
            * self.joint_power
            * np.isnan(self.cmd_override)
        )
        self.err_norm = np.linalg.norm(
            self.err
            * np.array([1.73, 0.97, 0.97, 1.73, 0.19, 0.19, 0, 0] * 2)
            * self.joint_power,
            1,
        )

        return self.err

    def err_i_calc(self, ns, err):
        err_clamped = err * ~self.is_clamp
        err_i = self.integrator.update(ns, err_clamped)
        self.integrator.value *= self.joint_power * np.isnan(self.cmd_override)  # type: ignore
        return err_i

    def err_d_calc(self, ns, err, cutoff_freq=0.01):
        es = err - self.integrator_der.value
        err_d = (es * cutoff_freq) * self.joint_power * np.isnan(self.cmd_override)
        self.integrator_der.update(ns, err_d)
        return err_d

    def pid_alt(self, err, err_i, err_d):
        k = 0.5
        gamma = 5
        kp = 0.7
        ki = 0.1  # kp**2 > 2ki

        K = k * np.eye(16)
        KP = kp * np.eye(16)
        KI = ki * np.eye(16)

        res = (K + 1 / (gamma**2) * np.eye(16)) @ (err_d + KP @ err + KI @ err_i)

        return res

    def allocate_cmd(self, cmd):
        cmd = np.clip(cmd, -9.9, 9.9)
        try:
            for ch in range(4):
                self.phidget_1002[ch].setVoltage_async(cmd[ch], self.ph_async_cb)
                self.phidget_1002[ch + 4].setVoltage_async(
                    cmd[ch + 4], self.ph_async_cb
                )
                self.phidget_1002[ch + 8].setVoltage_async(
                    cmd[ch + 8], self.ph_async_cb
                )
                self.phidget_1002[ch + 12].setVoltage(cmd[ch + 12])

        except PhidgetException.PhidgetException as e:
            self.node.get_logger().error(e)
            t_0 = time.time()
            self.node.get_logger().error(traceback.format_exc())
            self.node.get_logger().error(
                f"retry to reconnect phidget. attempt {self.phidget_retry_counter}"
            )
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
        flow_4 = self.lpm4_volt2flow(np.abs(v))
        flow_8 = self.lpm8_volt2flow(np.abs(v))
        res = abs(
            np.where(
                [True, False, False, True, False, False, False, False] * 2,
                flow_8,
                flow_4,
            )
        )
        return res

    @staticmethod
    def mhpu_flow_estimator(rpm):
        return rpm * 2.7 / 1000

    def flow_distributer(self, vd: np.ndarray, Q_j, Q_h):
        k_fd = np.divide(Q_h, np.sum(Q_j[8:13]+0.5), out=np.zeros(1), where=np.sum(Q_j[8:13]) != 0)
        # print("Q_j.sum()",Q_j.sum())
        if k_fd < 1:
            Q_fd = Q_j*k_fd
            # print("Q_fd[13]", Q_fd[13])
            volt_flow_4 = self.lpm4_flow2volt(Q_fd)*np.sign(vd)
            volt_flow_8 = self.lpm8_flow2volt(Q_fd)*np.sign(vd)
            v_fd = np.where(
                [True, False, False, True, False, False, False, False] * 2,
                volt_flow_8,
                volt_flow_4,
            )
        else:
            v_fd = vd
        # print("Q_j.sum()",Q_j.sum())
        # print("k_fd", k_fd)
        # v_fd = np.multiply(vd, k_fd, out=vd, where=k_fd < 1)
        return v_fd

    @staticmethod
    def joint_saturator(v):
        #pump MaxQ -> 2.7*2800/1000 = 7.56lpm
        pump_4lpm_sat_v = 5.8
        pump_8lpm_sat_v = 3.8

        v_p_max = np.where(
            [True, False, False, True, False, False, False, False] * 2,
            pump_8lpm_sat_v,
            pump_4lpm_sat_v,
        ) # saturated by pump
        v_h_max = (
            np.array([3.5, 5.0, 5.0, 3.5, 1.7, 1.7, 1.7, 1.7] * 2)
        )  # saturated by hose
           # 1~4 hose max Q : 6.3LPM(추정)
           # 5~8 hose max Q : 1.46LPM

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

    def saturation_s(self, s):
        phi = 0.04
        if s >= phi:
            s = 1
        elif -phi < s and s < phi:
            s = s / phi
        else:
            s = -1
        return s
    
    def super_twisting(self,t,s):
        U = 0.001
        lamda = np.sqrt(U)
        W = 1.1*U
        w_dot = -W*np.sign(s)
        print("w_dot:",w_dot)
        w = self.st_w.update(t*1e9,w_dot)
        print("w:",w)
        s = lamda*np.sqrt(np.abs(s))*np.sign(s)-w
        return s

    def smc_1_dof(self, t, pos, vel, ref):
        t = t / 1e9

        m1 = 4.472 / 9.81  # kgf
        m2 = 0.92 / 9.81  # kgf
        l1 = 0.45  # m
        l2 = 0.26  # m
        g = 9.81

        l1_base = 0.335
        l1_rot = 0.08
        l2_base = 0.173
        l2_rot = 0.025

        q1_dl_0 = np.deg2rad(76.17)
        q2_dl_0 = np.deg2rad(109.8)

        q1 = np.deg2rad(-pos[11])
        q2 = np.deg2rad(pos[13])
        des_q1 = -np.deg2rad(20)
        #des_q1 = -np.deg2rad(20*np.sin(2*np.pi*t/10))
        #des_q2 = np.deg2rad(20*np.sin(2*np.pi*t/8))
        # des_q1 = -np.deg2rad(-20)
        des_q2 = np.deg2rad(20)
        e1 = des_q1 - q1
        e2 = des_q2 - q2


        q1_dot = np.deg2rad(-vel[11])
        q2_dot = np.deg2rad(vel[13])

        q = np.array([[q1], [q2]])
        q_dot = np.array([[q1_dot], [q2_dot]])
        m = np.array([[m1], [m2]])
        l = np.array([[l1], [l2]])
        e = np.array([[e1], [e2]])

        M, C, G = self.CalDynamics(q, q_dot, m, l)
        D = np.array([[0.04], [0.07]])

        g_x = np.linalg.inv(M)
        f_x = g_x @ (-C * np.array([[q1_dot], [q2_dot]]) - G)

        c = np.array([[0.2], [0.9]])
        e_dot = self.e_diff.update(t, e)
        # print("e_dot:",e_dot)
        
        # des_q_ddot = self.q_ddiff.update(t, np.array([[des_q1], [des_q2]]))
        des_q_ddot = np.clip(self.q_ddiff.update(t, np.array([[des_q1], [des_q2]])),np.deg2rad(-5),np.deg2rad(5))
        # des_q_ddot = np.array([[0],[0]])
        s = c * e + e_dot

        # u = M @ (c * e_dot + des_q_ddot - f_x + D * np.sign(s))
        #s[0] = self.saturation_s(s[0])
        #s[0] = self.super_twisting(t,s[0])
        s[0] = self.saturation_s(s[0])
        u = M @ (c * e_dot + des_q_ddot - f_x + D*s)
        # print("D*s",D*s)
        # u = M @ (c * e_dot + des_q_ddot - f_x + D * s)


        con_q_ddot = np.linalg.solve(M, (u - C * q_dot - G))
        # con_q_ddot = np.clip(con_q_ddot,np.deg2rad(50),np.deg2rad(50))
        # con_q_dot = np.clip(con_q_ddot/100,np.deg2rad(-0.5),np.deg2rad(0.5))
        con_q_dot = np.clip(
            self.con_q_integrator.update(t*1e9, con_q_ddot),
            np.deg2rad(-0.5),
            np.deg2rad(0.5))
        
        # if abs(e[0]) < np.deg2rad(0.03):
        #     con_q_dot[0] = 0
        # if abs(e[1]) < np.deg2rad(0.03):
        #     con_q_dot[1] = 0


        dc1_dq = (
            l1_base
            * l1_rot
            * np.sin(q1_dl_0 - q1)
            / (
                np.sqrt(
                    l1_base**2 - 2 * l1_base * l1_rot * np.cos(q1_dl_0 + q1) + l1_rot**2
                )
            )
        )
        dc2_dq = (
            l2_base
            * l2_rot
            * np.sin(q2_dl_0 - q2)
            / (
                np.sqrt(
                    l2_base**2 - 2 * l2_base * l2_rot * np.cos(q2_dl_0 + q2) + l2_rot**2
                )
            )
        )

        dc1_dt = dc1_dq * -con_q_dot[0]
        dc2_dt = dc2_dq * con_q_dot[1]


        A1_push = 0.1696
        A1_pull = 0.1272
        A2_push = 0.0433
        A2_pull = 0.0353

        Q1 = (
            A1_push * dc1_dt * (dc1_dt >= 0) + A1_pull * dc1_dt * (dc1_dt < 0)
        ) * 60000
        Q2 = (
            A2_push * dc2_dt * (dc2_dt >= 0) + A2_pull * dc2_dt * (dc2_dt < 0)
        ) * 60000

        lpm4_vol = np.polynomial.Polynomial([0, 0.7206])
        lpm8_vol = np.polynomial.Polynomial([0, 0.7361, -0.0408, 0.001])
        v1 = lpm8_vol(abs(Q1)) * np.sign(Q1)
        v2 = lpm4_vol(abs(Q2)) * np.sign(Q2)

        return v1, v2

    @staticmethod
    def CalDynamics(q, q_dot, m, l):
        g = 9.81
        q1 = q[0]
        q2 = q[1]
        q1_dot = q_dot[0]
        q2_dot = q_dot[1]
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]

        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(q2)
        M12 = m2 * l2**2 + m2 * l1 * l2 * np.cos(q2)
        M21 = M12
        M22 = m2 * l2**2
        M = np.array([[M11[0], M12[0]], [M21[0], M22[0]]])
        C1 = -m2 * l1 * l2 * np.sin(q2) * q2_dot - m2 * l1 * l2 * np.sin(q2) * (
            q1_dot + q2_dot
        )
        C2 = m2 * l1 * l2 * np.sin(q2) * q1_dot
        C = np.array([[C1[0]], [C2[0]]])
        G1 = (m1 + m2) * g * l1 * np.cos(q1) + m2 * g * l2 * np.cos(q1 + q2)
        G2 = m2 * g * l2 * np.cos(q1 + q2)
        G = np.array([[G1[0]], [G2[0]]])
        return M, C, G
    
    def qtoc(self,q):
        c = np.zeros(16)
        l_base = np.array([0.275, 0.157, 0.065, 0.334,0, 0.183, 0.109, 0.027]*2) #m
        l_rot = np.array([0.068, 0.111, 0.258, 0.080,0, 0.025,0.043, 0.155]*2) #m
        q_0 =np.array([36.51, 101.17, 33.31, 76.17,0, 79.0,29.25, 96.0]*2) #deg
        for i in range(16):
            ## 5 Axis
            if i == 4 or i == 12:
                c[i] = q[i]
            ## 7 Axis
            elif i == 6 or i == 14:
                c[i] = 2*l_rot[i]*(np.sin(np.deg2rad(q[i]+q_0[i])))+l_base[i]
            else:
            ## 1,2,3,4,6,8 Axis
                c[i] = np.sqrt(l_base[i]**2+l_rot[i]**2-2*l_base[i]*l_rot[i]*np.cos(np.deg2rad(q[i]+q_0[i])))
        return c ## m
    
    def cal_gainScaleFactor(self):
        k = np.zeros(16)
        c_5 = np.zeros(16)
        c_minus5 = np.zeros(16)
        l_base = np.array([0.275, 0.157, 0.065, 0.334,0, 0.183, 0.109, 0.027]*2) #m
        l_rot = np.array([0.068, 0.111, 0.258, 0.080,0, 0.025,0.043, 0.155]*2) #m
        q_0 =np.array([36.51, 101.17, 33.31, 76.17,0, 79.0,29.25, 96.0]*2) #deg
        for i in range(16):
            ## 5 Axis
            if i == 4 or i == 12:
                k[i] = 1
            ## 7 Axis
            elif i == 6 or i == 14:
                c_5[i] = 2*l_rot[i]*(np.sin(np.deg2rad(10+q_0[i])))+l_base[i]
                c_minus5[i] = 2*l_rot[i]*(np.sin(np.deg2rad(0+q_0[i])))+l_base[i]
                k[i] = 10/(c_5[i]-c_minus5[i])
            else:
            ## 1,2,3,4,6,8 Axis
                c_5[i] = np.sqrt(l_base[i]**2+l_rot[i]**2-2*l_base[i]*l_rot[i]*np.cos(np.deg2rad(5+q_0[i])))
                c_minus5[i] = np.sqrt(l_base[i]**2+l_rot[i]**2-2*l_base[i]*l_rot[i]*np.cos(np.deg2rad(-5+q_0[i])))
                k[i] = 10/(c_5[i]-c_minus5[i])
        return k
    
    def cal_v_tr_max(self,joint_pos,v_tr):
        ## 5축은 쓰레기값
        l_base = np.array([0.275, 0.157, 0.065, 0.334,1, 0.183, 0.109, 0.027]*2) #m
        l_rot = np.array([0.068, 0.111, 0.258, 0.080,1, 0.025,0.043, 0.155]*2) #m
        q_0 =np.array([36.51, 101.17, 33.31, 76.17,1, 79.0,29.25, 96.0]*2) #deg
        A_push = np.array([1257, 707, 707, 1257, 0, 962, 962, 962]*2)*1e-6 #m^2
        A_pull = np.array([942.5, 530.1, 530.1, 942.5, 0, 785.4, 785.4, 785.4]*2)*1e-6 #m^2
        lpm4_vol = np.polynomial.Polynomial([0, 0.9])
        lpm8_vol = np.polynomial.Polynomial([0, 0.7361, -0.0408, 0.001])
        dc_dq = np.zeros(16)
        ref_v = np.zeros(16)
        
        dc_dq = (l_base* l_rot* np.sin(np.deg2rad(q_0 + joint_pos))/(np.sqrt(l_base**2 - 2 * l_base * l_rot * np.cos(np.deg2rad(q_0 + joint_pos)) + l_rot**2)))
        tr_c_dot = v_tr*dc_dq
        tr_Q = tr_c_dot*A_push*(tr_c_dot>=0)*60000+tr_c_dot*A_pull*(tr_c_dot<0)*60000

        for i in range(16):
            if i == 0 or i == 3 or i == 8 or i == 11:
                ref_v[i] = lpm8_vol(abs(tr_Q[i])) * np.sign(tr_Q[i])
            elif i ==4 or i == 12:
                ref_v[i] = 5
            else:
                ref_v[i] = lpm4_vol(abs(tr_Q[i])) * np.sign(tr_Q[i])
        return ref_v
    
    def cal_v_dot_tr_max(self,joint_pos,a_tr):
        ## 5축은 쓰레기값
        l_base = np.array([0.275, 0.157, 0.065, 0.334,1, 0.183, 0.109, 0.027]*2) #m
        l_rot = np.array([0.068, 0.111, 0.258, 0.080,1, 0.025,0.043, 0.155]*2) #m
        q_0 =np.array([36.51, 101.17, 33.31, 76.17,1, 79.0,29.25, 96.0]*2) #deg
        A_push = np.array([1257, 707, 707, 1257, 0, 962, 962, 962]*2)*1e-6 #m^2
        A_pull = np.array([942.5, 530.1, 530.1, 942.5, 0, 785.4, 785.4, 785.4]*2)*1e-6 #m^2
        lpm4_vol = np.polynomial.Polynomial([0, 0.7206])
        lpm8_vol = np.polynomial.Polynomial([0, 0.7361, -0.0408, 0.001])
        ddc_ddq = np.zeros(16)
        tr_v_dot = np.zeros(16)
        
        ddc_ddq = (l_base*l_rot*(l_base**2+l_rot**2)*np.cos(np.deg2rad(joint_pos+q_0))-l_base*l_rot*np.sin(np.deg2rad(joint_pos+q_0))**2-2*l_rot*l_base*np.cos(np.deg2rad(joint_pos+q_0))**2)/np.sign((l_base**2-2*l_base*l_rot+np.cos(np.deg2rad(joint_pos+q_0))+l_rot**2))*np.abs((l_base**2-2*l_base*l_rot+np.cos(np.deg2rad(joint_pos+q_0))+l_rot**2))**(3/2)
        tr_c_ddot = a_tr*ddc_ddq
        tr_Q_dot = tr_c_ddot*A_push*(tr_c_ddot>=0)*60000+tr_c_ddot*A_pull*(tr_c_ddot<0)*60000

        for i in range(16):
            if i == 0 or i == 3 or i == 8 or i == 11:
                tr_v_dot[i] = lpm8_vol(abs(tr_Q_dot[i])) * np.sign(tr_Q_dot[i])
            elif i ==4 or i == 12:
                tr_v_dot[i] = 5
            else:
                tr_v_dot[i] = lpm4_vol(abs(tr_Q_dot[i])) * np.sign(tr_Q_dot[i])
        return tr_v_dot
    
    def tr_joint_saturator(self,v,v_t_max,v_dot_t_max,v_diff,t):
        # if self.t_pre is None:
        #     self.t_pre = copy.deepcopy(t)
        #     self.v_pre = copy.deepcopy(v)
        for i in range(16):
        #     if np.abs(v_diff[i])>np.abs(v_dot_t_max[i]):
        #         v[i] = self.v_pre[i] + v_dot_t_max[i]*(t-self.t_pre)
            if v_t_max[i] != 0:
                if v_t_max[i]>=0:
                    v[i] = np.clip(v[i],-np.abs(v[i]),np.abs(v_t_max[i]))
                else:
                    v[i] = np.clip(v[i],-np.abs(v_t_max[i]),np.abs(v[i]))
        # self.t_pre = copy.deepcopy(t)
        # self.v_pre = copy.deepcopy(v)
        return v
    
    def joint_acc_saturator(self,v,v_diff,t):
        v_dot_max = [5, 5, 5, 5, 5, 5, 5, 5, 0.05, 0.05, 0.05, 0.05, 5, 0.05, 0.05, 0.05]
        if self.t_pre is None:
            self.t_pre = copy.deepcopy(t)
            self.v_pre = copy.deepcopy(v)
        for i in range(16):
            if np.abs(v_diff[i])>np.abs(v_dot_max[i]):
                v[i] = self.v_pre[i] + v_dot_max[i]*(t-self.t_pre)*np.sign(v_diff[i])
                # print("Saurated",i)
        self.t_pre = copy.deepcopy(t)
        self.v_pre = copy.deepcopy(v)
        return v
        

        
            
    def run(self):
        frame = 0
        try:
            self.init_phidget()
            self.set_phidget_enabled(list(range(16)))
            t = time.monotonic_ns()
            self.integrator.initialize(t, np.zeros(16))
            self.integrator_der.initialize(t, np.zeros(16))
            # clamp_sat_limit = 5.5
            actuator_sat_limit = 9.9
            self.t_pre = None
            self.v_pre = None
            # a_tr = np.zeros(16)
            # v_tr = np.zeros(16)
            while True:
                if self.conn.poll():
                    self.receiver()
                t = time.monotonic_ns()
                """if len(self.traj_point.accelerations)!=0:
                    for i in range(6):
                        a_tr[i+8]=self.traj_point.accelerations[i]
                        v_tr[i+8]=self.traj_point.velocities[i]"""

                err = self.err_calc()
                err_l = self.qtoc(self.ref_pos) - self.qtoc(self.joint_pos)
                err = err_l*self.smooth_factor
                err[4] *= -1
                err[12] *= -1

                err_i = self.err_i_calc(t, err)
                err_d = self.err_d_calc(t, err)
                vd = (
                    err * self.kp_l + err_i * self.ki_l + err_d * self.kd_l
                )
                k = self.cal_gainScaleFactor()
                vd = vd*k
                # for i in range(16):
                #     pullGainFactor = np.array([0.75, 0.75, 0.75, 0.75, 1, 0.816, 0.816, 0.816]*2)
                #     if vd[i] < 0:
                #         vd[i] = vd[i]*pullGainFactor[i]
                
                v_tr_max = np.zeros(16)
                # v_tr_max = self.cal_v_tr_max(self.joint_pos,v_tr)
                # v_tr_max = v_tr_max
                # v_tr_max[9] = v_tr_max[9]*2
                # v_tr_max[13] = v_tr_max[13]*2
                # # # # print("v_t_max[11]",v_t_max[11])
                # v_dot_tr_max = self.cal_v_dot_tr_max(self.joint_pos,a_tr)
                # t = t/1e93
                #v_diff = self.v_diff.update(t,vd)
                # vd_sat = self.tr_joint_saturator(vd,v_tr_max,v_dot_tr_max,v_diff,t)
                #vd = self.joint_acc_saturator(vd,v_diff,t)
                vd_sat = self.joint_saturator(vd)
                vd_sat = np.array(vd_sat)

                self.is_clamp = (vd_sat != vd) & (np.sign(vd) == np.sign(err))
                if np.abs(err[12]) < 0.15:
                    self.is_clamp[12]=True
                if np.abs(err[4]) < 0.15:
                    self.is_clamp[4]=True
                # Q_j_est = self.joint_flow_estimator(vd_sat)
                # Q_p_est = self.mhpu_flow_estimator(self.des_rpm)
                # v_fc = self.flow_distributer(vd_sat, Q_j_est, Q_p_est)
                v_fc = self.moving_average_filter.update(vd_sat) 
                v_dc = self.dead_zone_compensate(v_fc, 1.2, 0.01)
                cmd = v_dc
                cmd[~np.isnan(self.cmd_override)] = self.cmd_override[~np.isnan(self.cmd_override)]
                cmd = self.min_max_saturation(cmd)
                cmd = np.clip(
                    cmd, -actuator_sat_limit, actuator_sat_limit
                )  # saturate command in range of phidget's range
                # print("smooth_factor", self.smooth_factor[13])
                cmd = np.where(self.joint_power.astype(bool), cmd, 0.0)
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
                            "err_norm": self.err_norm,
                            "v_tr_max" : v_tr_max,
                            ## 데이터 추가
                        }
                    )

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
            raise ValueError(
                f"incorrect shape {np.shape(value)}. Shape should be (16,)"
            )

    @property
    def ref_pos(self):
        return self._ref_pos

    @ref_pos.setter
    def ref_pos(self, value):
        if np.shape(value) == (16,):
            self._ref_pos = value
        else:
            raise ValueError(
                f"incorrect shape {np.shape(value)}. Shape should be (16,)"
            )

    @property
    def err(self):
        return self._err

    @err.setter
    def err(self, value):
        if np.shape(value) == (16,):
            self._err = value
        else:
            raise ValueError(
                f"incorrect shape {np.shape(value)} for err. Shape should be (16,)"
            )

    @property
    def err_i(self):
        return self._err_i

    @err_i.setter
    def err_i(self, value):
        if np.shape(value) == (16,):
            self._err_i = value
        else:
            raise ValueError(
                f"incorrect shape {np.shape(value)}. Shape should be (16,)"
            )

    @property
    def err_d(self):
        return self._err_d

    @err_d.setter
    def err_d(self, value):
        if np.shape(value) == (16,):
            self._err_d = value
        else:
            raise ValueError(
                f"incorrect shape {np.shape(value)}. Shape should be (16,)"
            )

    @property
    def err_log(self):
        return self._err_log

    @err_log.setter
    def err_log(self, value):
        if np.shape(value) == (self.log_size, 16):
            self._err_log = value
        else:
            raise ValueError(
                f"incorrect shape {np.shape(value)}. Shape should be (self.log_size, 16)"
            )

    @property
    def err_i_log(self):
        return self._err_i_log

    @err_i_log.setter
    def err_i_log(self, value):
        if np.shape(value) == (self.log_i_size, 16):
            self._err_i_log = value
        else:
            raise ValueError(
                f"incorrect shape {np.shape(value)}. Shape should be (self.log_size, 16)"
            )

    @property
    def err_d_log(self):
        return self._err_d_log

    @err_d_log.setter
    def err_d_log(self, value):
        if np.shape(value) == (self.log_size, 16):
            self._err_d_log = value
        else:
            raise ValueError(
                f"incorrect shape {np.shape(value)}. Shape should be (self.log_size, 16)"
            )

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        if np.shape(value) == (self.log_size, 1):
            self._timestamp = value
        else:
            raise ValueError(
                f"incorrect shape {np.shape(value)}. Shape should be ({self.log_size, 1})"
            )

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
    def dead_zone_compensate(
        value: np.ndarray, compensation: float, tol: float
    ) -> np.ndarray:
        res = (value + np.sign(value) * compensation) * (abs(value) > tol) + (
            value * (compensation + tol) / tol
        ) * (abs(value) <= tol)
        return res


from typing import Optional


class Integrator:
    def __init__(self) -> None:
        self.t_prev = None
        self.data_prev = None
        self.value: Optional[np.ndarray] = None

    def initialize(self, t, data):
        self.t_prev = t
        self.data_prev = data
        self.value = np.zeros_like(data)

    def update(self, t, data):
        if self.value is None:
            self.value = np.zeros_like(data)
        elif self.t_prev is not None and self.data_prev is not None:
            self.value = (
                self.value + (t - self.t_prev) * (data + self.data_prev) / 2/ 1e9
            )
        self.t_prev = t
        self.data_prev = data
        return self.value


class FiniteDiff:
    def __init__(self):
        self.data_prev = None
        self.t_prev = time.time()

    def update(self, t, data):
        if self.data_prev is None:
            self.t_prev = copy.deepcopy(t)
            diff = np.zeros_like(data)
        else:
            diff = (data - self.data_prev) / (t - self.t_prev)
        self.t_prev = copy.deepcopy(t)
        self.data_prev = copy.deepcopy(data)
        return diff


class FiniteDDiff:
    def __init__(self) -> None:
        self.data_prev = None
        self.data_pprev = None
        self.t_prev = time.time()
        self.t_pprev = time.time()

    def update(self, t, data):
        if self.data_prev is None:
            ddiff = np.zeros_like(data)
            self.t_prev=copy.deepcopy(t)
        elif self.data_pprev is None:
            ddiff = np.zeros_like(data)
            self.t_pprev=copy.deepcopy(t)
        else:
            ddiff = (data - 2 * self.data_prev + self.data_pprev) / np.average(
                np.diff([t, self.t_prev, self.t_pprev])
            ) ** 2
        self.data_pprev = copy.deepcopy(self.data_prev)
        self.data_prev = copy.deepcopy(data)
        self.t_pprev = copy.deepcopy(self.t_prev)
        self.t_prev = copy.deepcopy(t)
        return ddiff

class MovingAverage:
    def __init__(self, window_size, max_change, threshold, reset_threshold):
        """
        Initialize the moving average filter with the specified parameters.

        Parameters:
        window_size (int): The size of the moving window.
        max_change (float): The maximum allowed change per update.
        threshold (float): The threshold for applying the moving average.
        reset_threshold (float): The threshold for resetting the moving average.
        """
        self.window_size = window_size
        self.windows = np.zeros((16, window_size))
        self.indices = np.zeros(16, dtype=int)
        self.counts = np.zeros(16, dtype=int)
        self.last_values = np.full(16, None)
        self.exceed_counts = np.zeros(16, dtype=int)
        self.max_change = max_change
        self.threshold = threshold
        self.reset_threshold = reset_threshold
        self.last_smooth_values = np.zeros(16)  # Initialize the last smoothed values
        self.view_change = None

    def update(self, values):
        """
        Update the moving average with new values if the change exceeds the threshold.
        Reset the moving average if the change exceeds the reset threshold.

        Parameters:
        values (np.ndarray): The new values to add to the windows.

        Returns:
        np.ndarray: The current moving averages.
        """
        smoothed_values = np.zeros(16)
        
        for i in range(16):
            value = values[i]
            
            if abs(value - self.last_smooth_values[i]) < self.threshold:
                self.exceed_counts[i] += 1
            else:
                self.exceed_counts[i] = 0

            if self.exceed_counts[i] >= self.reset_threshold:
                self.reset(i)

            # Adjust value within allowed change limits
            if self.last_values[i] is not None:
                change = value - self.last_values[i]
                if change > self.max_change:
                    value = self.last_values[i] + self.max_change
                elif change < -self.max_change:
                    value = self.last_values[i] - self.max_change

            self.windows[i, self.indices[i]] = value
            self.indices[i] = (self.indices[i] + 1) % self.window_size
            self.counts[i] = min(self.counts[i] + 1, self.window_size)

            self.last_values[i] = value
            self.last_smooth_values[i] = np.mean(self.windows[i, :self.counts[i]])
            smoothed_values[i] = self.last_smooth_values[i]
        
        return smoothed_values

    def reset(self, index):
        """Reset the moving average filter for a specific index."""
        self.windows[index].fill(0)
        self.indices[index] = 0
        self.counts[index] = 0
        self.last_values[index] = None
        self.exceed_counts[index] = 0
        self.last_smooth_values[index] = 0.0



if __name__ == "__main__":
    rclpy.init()
    node = Node("t")
    this_conn, that_conn = Pipe()
    ctrl = Control([that_conn, this_conn] * 3, node)
    ctrl.run()
