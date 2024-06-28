#!/usr/bin/env python3
from multiprocessing import Process, Pipe
from threading import Thread
import os
import numpy as np
import signal
import copy
import time

from Phidget22.Devices.TemperatureSensor import TemperatureSensor
from Phidget22.PhidgetException import PhidgetException
from Phidget22.Phidget import Phidget

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data, qos_profile_services_default
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import MotionPlanResponse
from ass_msgs.msg import (
    Robot2UI,
    Hold,
    Preset,
    UIAction,
    PoseIteration,
    ArmMasterCommInt,
    TrajectoryExecution,
    TrajectoryEnabled,
    TrajectoryFeedback,
    DXLCommand,
    ManualVolt,
    ARMstrongTrajectory,
    ARMstrongTrajectoryEnabled,
    ARMstrongTrajectoryExecution,
    ARMstrongTrajectoryFeedback
)
import control
import pump
import state
import dxl_control

from datetime import datetime
import traceback


class Robot(Node):
    def __init__(self):
        super().__init__("ass_core")  # type: ignore
        self.get_logger().set_level(10)
        self.get_logger().info(
            f"logging level set to {self.get_logger().get_effective_level()}")

        Phidget.resetLibrary()

        # fast pipe sends data everytime when available
        self.sl_fast_pipe, self.sl_fast_pipe_opp = Pipe()
        self.ctrl_fast_pipe, self.ctrl_fast_pipe_opp = Pipe()
        self.pump_fast_pipe, self.pump_fast_pipe_opp = Pipe()
        self.dxl_fast_pipe, self.dxl_fast_pipe_opp = Pipe()

        # slow pipe sends data when data change
        self.sl_slow_pipe, self.sl_slow_pipe_opp = Pipe()
        self.ctrl_slow_pipe, self.ctrl_slow_pipe_opp = Pipe()
        self.pump_slow_pipe, self.pump_slow_pipe_opp = Pipe()
        self.dxl_slow_pipe, self.dxl_slow_pipe_opp = Pipe()

        # pause pipe sends "PAUSE" when to pause, "RESUME when to resume
        self.sl_pause_pipe, self.sl_pause_pipe_opp = Pipe()
        self.ctrl_pause_pipe, self.ctrl_pause_pipe_opp = Pipe()
        self.pump_pause_pipe, self.pump_pause_pipe_opp = Pipe()
        self.dxl_pause_pipe, self.dxl_pause_pipe_opp = Pipe()

        pipeset_sl = [
            self.sl_fast_pipe_opp,
            self.sl_fast_pipe,
            self.sl_slow_pipe_opp,
            self.sl_slow_pipe,
            self.sl_pause_pipe_opp,
            self.sl_pause_pipe,
        ]

        pipeset_ctrl = [
            self.ctrl_fast_pipe_opp,
            self.ctrl_fast_pipe,
            self.ctrl_slow_pipe_opp,
            self.ctrl_slow_pipe,
            self.ctrl_pause_pipe_opp,
            self.ctrl_pause_pipe,
        ]

        pipeset_pump = [
            self.pump_fast_pipe_opp,
            self.pump_fast_pipe,
            self.pump_slow_pipe_opp,
            self.pump_slow_pipe,
            self.pump_pause_pipe_opp,
            self.pump_pause_pipe,
        ]

        pipeset_dxl = [
            self.dxl_fast_pipe_opp,
            self.dxl_fast_pipe,
            self.dxl_slow_pipe_opp,
            self.dxl_slow_pipe,
            self.dxl_pause_pipe_opp,
            self.dxl_fast_pipe,
        ]

        self.sl_handle = state.SensorRead(pipeset_sl, self)
        self.ctrl_handle = control.Control(pipeset_ctrl, self)
        self.pump_handle = pump.Pump(pipeset_pump, self)
        self.dxl_handle = dxl_control.DXLControl(pipeset_dxl, self)

        self.sl_proc_handle = Process(target=self.sl_handle.run, daemon=True)
        self.ctrl_proc_handle = Process(target=self.ctrl_handle.run, daemon=True)
        self.pump_proc_handle = Process(target=self.pump_handle.run, daemon=True)
        self.dxl_proc_handle = Process(target=self.dxl_handle.run, daemon=True)

        signal.signal(signal.SIGINT, self.sigint_handler)

        self.mask = np.array([False] * 16).reshape(2, 8)
        self.ms_state = {
            "joint_pos": np.zeros(16, dtype=float),
            "track_cmd": np.array([500.0, 500.0]),
            "lift_cmd": np.zeros(2),
            "frame": 0,
        }
        self.sl_state = {
            "joint_pos": np.zeros(16, dtype=float),
            "joint_vel": np.zeros(16, dtype=float),
            "joint_acc": np.zeros(16, dtype=float),
            "frame": 0,
        }
        self.ctrl_state = {
            "cmd": np.zeros(16, dtype=float),
            "err": np.zeros(16, dtype=float),
            "err_i": np.zeros(16, dtype=float),
            "err_d": np.zeros(16, dtype=float),
            "err_norm": 0.0,
            "frame": 0,
            "v_tr_max": np.zeros(16, dtype=float),
        }
        self.ms_state["joint_pos"][3] = 20
        self.ms_state["joint_pos"][10] = 20
        self.ref_pos = np.zeros(16, dtype=float)
        self.pump_state = {
            "act_rpm": 0,
            "des_rpm": 0,
            "des_cur": 0.0,
            "elmo_temp": 0.0,
            "phidget_temp": 0.0,
            "frame": 0,
        }

        self.cmd_override = np.array([np.nan] * 16, dtype=float)

        self.ms_joint_pos = np.zeros(16)
        self.flag_is_hold = np.array([False] * 16)

        self.pub = self.create_publisher(
            Robot2UI, "robot_to_ui", qos_profile_system_default
        )
        self.create_subscription(
            ArmMasterCommInt,
            "cmd_input",
            callback=self.cb_ms_state,
            qos_profile=qos_profile_sensor_data,
        )

        self.moveit_pub = self.create_publisher(
            JointState, "joint_states", qos_profile_system_default
        )
        self.hold_sub = self.create_subscription(
            Hold,
            "hold",
            callback=self.cb_hold_sub,
            qos_profile=qos_profile_system_default,
        )

        self.is_on_hold = np.array([False] * 16)
        self.is_on_hold_prev = self.is_on_hold

        self.joint_on_smooth = np.array([False] * 16)
        self.joint_when_on = np.zeros(16)

        self.preset_sub = self.create_subscription(
            Preset,
            "preset",
            callback=self.cb_preset_sub,
            qos_profile=qos_profile_system_default,
        )
        self.preset_pos = np.zeros(16)
        self.is_preset_mode = False

        self.is_ref_init = False

        self.ui_action_sub = self.create_subscription(
            UIAction,
            "ui_action",
            callback=self.cb_ui_action_sub,
            qos_profile=qos_profile_system_default,
        )
        self.ui_state = UIAction(
            pump_pwr=False,  # pump_pwr
            pump_mode=0,  # pump_mode
            pump_tgt_speed=0,  # pump_tgt_speed
            joint_power=[False] * 16,  # joint_power
            joint_to_smooth=[False] * 16,  # joint_to_smooth
            pump_max_rpm=2800,  # pump_max_rpm
            pump_min_rpm=500,  # pump_min_rpm
            pump_max_err=50,  # pump_max_err
            ctrl_mode=0,  # ctrl_mode
        )
        self.pose_iter_sub = self.create_subscription(
            PoseIteration,
            "pose_iter",
            callback=self.cb_pose_override,
            qos_profile=qos_profile_system_default,
        )
        self.trajetory_execution_sub = self.create_subscription(
            TrajectoryExecution,
            "traj_exec",
            callback=self.cb_traj_exec,
            qos_profile=qos_profile_services_default
        )
        self.planned_trajectory_sub = self.create_subscription(
            MotionPlanResponse,
            "planned_trajectory",
            callback=self.cb_planned_trajectory,
            qos_profile=qos_profile_services_default
        )
        self.trajectory_enabled_sub = self.create_subscription(
            TrajectoryEnabled,
            "traj_enabled",
            qos_profile=qos_profile_services_default,
            callback=self.cb_trajectory_enabled,
        )
        self.trajectory_feedback_pub = self.create_publisher(
            TrajectoryFeedback,
            "traj_feedback",
            qos_profile=qos_profile_services_default,
        )
        self.dxl_command_sub = self.create_subscription(
            DXLCommand,
            "dxl_command",
            qos_profile=qos_profile_system_default,
            callback=self.cb_dxl_command
        )
        self.manual_volt_sub = self.create_subscription(
            ManualVolt,
            "manual_volt",
            qos_profile=qos_profile_system_default,
            callback=self.cb_manual_volt
        )
        self.astr_traj_sub = self.create_subscription(
            ARMstrongTrajectory,
            "astr_traj",
            qos_profile=qos_profile_system_default,
            callback=self.cb_astr_traj
        )
        self.astr_traj_enabled_sub = self.create_subscription(
            ARMstrongTrajectoryEnabled,
            "astr_traj_enabled",
            qos_profile=qos_profile_system_default,
            callback=self.cb_astr_traj_enabled
        )
        self.astr_traj_exec_sub = self.create_subscription(
            ARMstrongTrajectoryExecution,
            "astr_traj_execution",
            qos_profile=qos_profile_system_default,
            callback=self.cb_astr_traj_execution
        )
        self.astr_traj_fb_pub = self.create_publisher(
            ARMstrongTrajectoryFeedback,
            "astr_traj_feedback",
            qos_profile=qos_profile_system_default
        )

        self.pose_override_enabled = False
        self.pose_override_dxl = 0

        self.traj_enabled = False
        self.traj_point = JointTrajectoryPoint()

        self.t_traj_exec_started = None
        self.trajectory: list = []
        self.trajectory_dxl = []
        self.trajectory_idx = 0

        self.dxl_enabled = False
        self.dxl_manual_value = [0, 0, 0, 0]

        self.enable_logging = True
        self.freq = np.zeros(3, dtype=float)

        self.publisher_timer = self.create_timer(0.01, self.cb_publisher)
        self.publisher_timer.reset()

        self.th_spin = Thread(target=rclpy.spin, kwargs={"node": self}, daemon=True)
        self.th_spin.start()

        self.phidget_temp = TemperatureSensor()
        try:
            self.phidget_temp.setChannel(0)
            self.phidget_temp.openWaitForAttachment(1000)
            self.get_logger().info("phidget temperature connected")
        except PhidgetException as e:
            self.get_logger().warn(f"{e}")
            self.phidget_temp.close()
            self.phidget_temp.resetLibrary()
            self.phidget_temp = None

        if self.enable_logging:
            header = (
                ["timestamp"]
                + [f"L{i} pos" for i in range(1, 9)]
                + [f"R{i} pos" for i in range(1, 9)]
                + [f"L{i} ref" for i in range(1, 9)]
                + [f"R{i} ref" for i in range(1, 9)]
                + [f"L{i} cmd" for i in range(1, 9)]
                + [f"R{i} cmd" for i in range(1, 9)]
                + ["FB rpm", "des rpm", "des cur", "elmo_temp"]
                + [f"L{i} v_t_max" for i in range(1, 9)]
                + [f"R{i} v_t_max" for i in range(1, 9)]
                # 행추가
            )
            if self.phidget_temp is not None:
                header.append("phidget_temp")
            self.log_dir = os.curdir + "/log/values_log/"
            self.log_name = (
                self.log_dir + f"log{datetime.now().strftime('%y%m%d%H%M%S')}.csv"
            )
            self.log_file = open(self.log_name, "a")
            self.log_file.write(",".join(header) + "\n")
            self.log_timer = self.create_timer(0.01, self.log_data)
            self.log_timer.reset()

    def log_data(self):

        logdata = [
            time.time(),
            *self.sl_state["joint_pos"],
            *self.ref_pos.tolist(),
            *self.ctrl_state["cmd"].tolist(),
        ]
        logdata.extend(
            [
                self.pump_state["act_rpm"],
                self.pump_state["des_rpm"],
                self.pump_state["des_cur"],
                self.pump_state["elmo_temp"],
                # 데이터 추가
            ],
        )
        logdata.extend(
            self.ctrl_state["v_tr_max"],
        )
        if self.phidget_temp is not None:
            temp = self.phidget_temp.getTemperature()
            logdata.append(temp)
        self.log_file.write(",".join(map("{:.4f}".format, logdata)) + "\n")

    def cb_publisher(self):
        t = self.get_clock().now()
        self.pub.publish(
            Robot2UI(
                header=Header(stamp=t.to_msg()),
                pump_act_rpm=self.pump_state["act_rpm"],
                pump_des_rpm=self.pump_state["des_rpm"],
                pump_des_cur=self.pump_state["des_cur"],
                pump_temp=self.pump_state["elmo_temp"],
                arm_state=self.sl_state["joint_pos"],
                ref_state=self.ref_pos,
                track_left_state=int(self.ms_state["track_cmd"][0]),
                track_right_state=int(self.ms_state["track_cmd"][1]),
                cmd_voltage=self.ctrl_state["cmd"],
                err=self.ctrl_state["err"],
                err_i=self.ctrl_state["err_i"],
                err_d=self.ctrl_state["err_d"],
                err_norm=self.ctrl_state["err_norm"],
                freq=self.freq,
            )
        )
        joint_state = JointState()
        joint_state.name = [
            "joint1_left", "joint2_left", "joint3_left", "joint4_left", "joint5_left", "joint6_left",
            "joint7r_1_left", "joint7r_left", "joint7r_2_left",
            "joint7l_1_left", "joint7l_left", "joint7l_2_left", "joint8_left",
            "joint1_right", "joint2_right", "joint3_right", "joint4_right", "joint5_right", "joint6_right",
            "joint7r_1_right", "joint7r_right", "joint7r_2_right",
            "joint7l_1_right", "joint7l_right", "joint7l_2_right", "joint8_right"]

        joint_pos = np.deg2rad(self.sl_state["joint_pos"])
        joint_pos = np.insert(joint_pos, 6, [joint_pos[6]] * 5)
        joint_pos = np.insert(joint_pos, -2, [joint_pos[-2]] * 5)
        joint_vel = np.deg2rad(self.sl_state["joint_vel"])
        joint_vel = np.insert(joint_vel, 6, [joint_vel[6]] * 5)
        joint_vel = np.insert(joint_vel, -2, [joint_vel[-2]] * 5)
        joint_pos = joint_pos * ([1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1] * 2)
        joint_state.position = joint_pos.tolist()
        joint_state.velocity = joint_vel.tolist()
        self.moveit_pub.publish(joint_state)

    def cb_trajectory_enabled(self, data: TrajectoryEnabled):
        self.traj_enabled = data.enabled
        self.t_traj_exec_started = None
        self.get_logger().info(f"trajectory by moveit mode: {self.traj_enabled}")

    def cb_traj_exec(self, data):
        self.get_logger().info("starting execution....")
        self.trajectory_idx = 0
        self.t_traj_exec_started = time.time_ns()

    def cb_planned_trajectory(self, data: MotionPlanResponse):
        self.t_traj_exec_started = None
        self.trajectory = data.trajectory.joint_trajectory.points  # type: ignore
        self.trajectory_idx = 0
        # self.hpu_profile = self.asdf(self.trajectory)
        self.get_logger().info("planned trajectory received")

    def cb_astr_traj(self, data: ARMstrongTrajectory):
        self.t_traj_exec_started = None
        self.trajectory = list(data.points)
        self.trajectory_dxl = data.trigger
        self.trajectory_idx = 0
        self.get_logger().info("planned trajectory received")

    def cb_astr_traj_enabled(self, data: ARMstrongTrajectoryEnabled):
        self.traj_enabled = data.enabled
        self.t_traj_exec_started = None
        self.get_logger().info(f"trajectory mode: {self.traj_enabled}")

    def cb_astr_traj_execution(self, data: ARMstrongTrajectoryExecution):
        self.get_logger().info("starting execution....")
        self.t_traj_exec_started = time.time_ns()
        self.trajectory_idx = 0

    def calc_Q(self, traj_pos, traj_vel):
        # 5축은 쓰레기값
        l_base = np.array([0.275, 0.157, 0.065, 0.334, 1, 0.183, 0.109, 0.027]*2)  # m
        l_rot = np.array([0.068, 0.111, 0.258, 0.080, 1, 0.025, 0.043, 0.155]*2)  # m
        # theta_0_deg = np.array([36.51, 101.17, 33.31, 76.17, 1, 79.0, 29.25, 96.0]*2)  # deg
        theta_0 = np.array([0.6372, 1.7657, 0.5814, 1.3294, 0.0175, 1.3788, 0.5105, 1.6755])  # rad

        A_push = np.array([1257, 707, 707, 1257, 0, 962, 962, 962]*2)*1e-6  # m^2
        A_pull = np.array([942.5, 530.1, 530.1, 942.5, 0, 785.4, 785.4, 785.4]*2)*1e-6  # m^2
        dc_dq = np.zeros(16)

        l_1 = l_base**2 + l_rot**2
        l_2 = l_base*l_rot

        dc_dq = (l_2*np.sin(theta_0 + traj_pos)/(np.sqrt(l_1-2*l_2*np.cos(theta_0 + traj_pos))))
        dc_dt = dc_dq*traj_vel
        tr_Q = dc_dt*A_push*(traj_vel >= 0)*60000+dc_dt*A_pull*(traj_vel < 0)*60000

        return tr_Q

    def start_proc(self):
        # self.ms_proc_handle.start()
        self.sl_proc_handle.start()
        self.ctrl_proc_handle.start()
        self.pump_proc_handle.start()
        self.dxl_proc_handle.start()
        self.get_logger().info(
            f"proc::state started with pid {self.sl_proc_handle.pid}")
        self.get_logger().info(
            f"proc::control started with pid {self.ctrl_proc_handle.pid}")
        self.get_logger().info(
            f"proc::pump started with pid {self.pump_proc_handle.pid}")

    def stop_proc(self):
        self.sl_proc_handle.terminate()
        self.ctrl_proc_handle.terminate()
        self.pump_proc_handle.terminate()

    def cb_hold_sub(self, data: Hold):
        res = data.enabled
        self.is_on_hold = np.array(res)

    def cb_ui_action_sub(self, data: UIAction):
        self.ui_state = data
        if any(self.ui_state.joint_to_smooth):
            joint_to_smooth = np.array(self.ui_state.joint_to_smooth)
            self.joint_when_on = (
                np.broadcast_to(time.time(), 16) * joint_to_smooth
                + self.joint_when_on * ~joint_to_smooth
            )

    def cb_preset_sub(self, data: Preset):
        self.is_preset_mode = data.enabled
        self.preset_pos = np.array(data.preset_pos)
        self.get_logger().info(f"{self.is_preset_mode, self.preset_pos}")

    def cb_ms_state(self, data: ArmMasterCommInt):
        ms_joint_state = np.array(
            [
                data.l1, data.l2, data.l3, data.l4, data.l5, data.l6,
                self.ms_state["joint_pos"][6], self.ms_state["joint_pos"][7],  # L7~8
                data.r1, data.r2, data.r3, data.r4, data.r5, data.r6,
                self.ms_state["joint_pos"][14], self.ms_state["joint_pos"][15],
            ]
        )  # R7~8
        if data.l7 != 0:  # if master l7 lever is not zero
            ms_joint_state[6] = self.sl_state["joint_pos"][6]  # ref pos is slave's pos
            self.cmd_override[6] = data.l7 * 3  # overrides cmd value to 3
            self.flag_is_hold[6] = False  # this axis is not held
        elif not self.flag_is_hold[6]:  # if master l7 lever is zero and is not held
            ms_joint_state[6] = self.sl_state["joint_pos"][6]  # ref pos is slave's pos
            self.cmd_override[6] = np.nan  # not overrides cmd value
            self.flag_is_hold[6] = True  # this axis is held

        if data.l8 != 0:
            ms_joint_state[7] = self.sl_state["joint_pos"][7]
            self.cmd_override[7] = data.l8 * 3
            self.flag_is_hold[7] = False
        elif not self.flag_is_hold[7]:
            ms_joint_state[7] = self.sl_state["joint_pos"][7]
            self.cmd_override[7] = np.nan
            self.flag_is_hold[7] = True

        if data.r7 != 0:
            ms_joint_state[14] = self.sl_state["joint_pos"][14]
            self.cmd_override[14] = 3 * data.r7
            self.flag_is_hold[14] = False
        elif not self.flag_is_hold[14]:
            ms_joint_state[14] = self.sl_state["joint_pos"][14]
            self.cmd_override[14] = np.nan
            self.flag_is_hold[14] = True

        if data.r8 != 0:
            ms_joint_state[15] = self.sl_state["joint_pos"][15]
            self.cmd_override[15] = 3 * data.r8
            self.flag_is_hold[15] = False
        elif not self.flag_is_hold[15]:
            ms_joint_state[15] = self.sl_state["joint_pos"][15]
            self.cmd_override[15] = np.nan
            self.flag_is_hold[15] = True

        self.ms_state["joint_pos"] = ms_joint_state
        # FIXME
        self.ms_state["lift_cmd"] = np.array([data.lifter, 2 if data.pump == -1 else data.pump])
        self.ms_state["track_cmd"] = np.array([data.lever_0, data.lever_1])

    def cb_dxl_command(self, data: DXLCommand):
        self.dxl_enabled = data.enabled
        self.dxl_manual_value[0] = data.left_lever
        self.dxl_manual_value[1] = data.right_lever
        self.dxl_manual_value[2] = data.lift_toggle
        self.dxl_manual_value[3] = data.mode_toggle

    def cb_manual_volt(self, data: ManualVolt):
        value = np.where(list(data.override_enabled), data.volt_override, np.nan)
        self.cmd_override = value

    def init_ref(self):
        pos_now = copy.deepcopy(self.sl_state["joint_pos"])
        if np.array(pos_now, dtype=bool).any():
            self.ms_state["joint_pos"] = pos_now
            self.is_ref_init = True

    def cb_pose_override(self, data: PoseIteration):
        self.pose_override_enabled = data.enabled
        self.pose_override_pose = data.poses[:16]
        # FIXME
        self.pose_override_dxl = data.trigger

    def sigint_handler(self, signum, frame):
        raise KeyboardInterrupt

    def run(self):
        self.start_proc()
        time_frame = np.array([self.get_clock().now().nanoseconds, 0, 0, 0])
        ms_joint_pos = np.zeros(16)
        while True:
            try:
                # rclpy.spin_once(self)
                t_ros = self.get_clock().now()
                # if self.ms_conn_p.poll():
                #    self.ms_state = self.ms_conn_p.recv()
                if self.sl_fast_pipe.poll():
                    self.sl_state = self.sl_fast_pipe.recv()
                    if not self.is_ref_init:
                        self.init_ref()
                if self.ctrl_fast_pipe.poll():
                    self.ctrl_state = self.ctrl_fast_pipe.recv()
                if self.pump_fast_pipe.poll():
                    self.pump_state = self.pump_fast_pipe.recv()
                is_on_hold = self.is_on_hold
                ui_state = self.ui_state
                ms_state = copy.deepcopy(self.ms_state)

                if self.is_preset_mode:
                    ms_state["joint_pos"] = self.preset_pos

                if self.pose_override_enabled:
                    ms_state["joint_pos"] = self.pose_override_pose
                    is_on_hold = np.where(
                        np.isnan(self.pose_override_pose), True, is_on_hold
                    )
                    ms_state["lift_cmd"][1] = 1 if self.pose_override_dxl != -1 else 0
                    ms_state["track_cmd"][1] = (self.pose_override_dxl * 5 + 500
                                                if self.pose_override_dxl != -1
                                                else -1)

                if self.traj_enabled and self.t_traj_exec_started is not None:
                    nsec = (self.trajectory[self.trajectory_idx]
                            .time_from_start.sec * 1e9
                            + self.trajectory[self.trajectory_idx]
                            .time_from_start.nanosec)
                    time_passed = time.time_ns() - self.t_traj_exec_started
                    if time_passed > nsec:  # increase index if time passed
                        self.trajectory_idx += 1
                        self.get_logger().info(f"{self.trajectory_idx}/{len(self.trajectory)}")
                        self.trajectory_feedback_pub.publish(
                            TrajectoryFeedback(index=self.trajectory_idx, finished=False))

                    if self.trajectory_idx >= len(self.trajectory):  # end trajectory mode
                        self.t_traj_exec_started = None
                        self.trajectory_feedback_pub.publish(
                            TrajectoryFeedback(finished=True))
                        continue
                    elif self.trajectory_idx == len(self.trajectory) - 1:  # finalize trajectory mode
                        interpolated = np.rad2deg(
                            self.trajectory[self.trajectory_idx].positions)
                    else:  # trajectory mode normal condition
                        position = np.rad2deg(self.trajectory[self.trajectory_idx].positions)
                        position_next = np.rad2deg(self.trajectory[self.trajectory_idx + 1].positions)
                        nsec_next = (self.trajectory[self.trajectory_idx + 1].time_from_start.sec * 1e9
                                     + self.trajectory[self.trajectory_idx + 1].time_from_start.nanosec)
                        interpolated = (position
                                        + (time_passed - nsec) * (position_next - position) / (nsec_next - nsec))

                    if interpolated.shape[0] == 16:
                        position = np.deg2rad(interpolated)
                        is_on_hold = np.where(np.isnan(position), True, is_on_hold)
                        ms_state["joint_pos"] = np.array(position)
                        ms_state["track_cmd"][0] = -1
                        ms_state["track_cmd"][1] = self.trajectory_dxl[self.trajectory_idx]
                        ms_state["lift_cmd"][0] = 0
                        ms_state["lift_cmd"][1] = 1
                    else:
                        position = [np.nan] * 8
                        position.extend(interpolated)
                        position.extend([np.nan] * 2)
                        is_on_hold = np.where(np.isnan(position), True, is_on_hold)
                        ms_state["joint_pos"] = np.array(position)
                elif self.traj_enabled:
                    is_on_hold = np.ones_like(is_on_hold, dtype=bool)

                if self.dxl_enabled:
                    ms_state["track_cmd"][0] = self.dxl_manual_value[0]
                    ms_state["track_cmd"][1] = self.dxl_manual_value[1]
                    ms_state["lift_cmd"][0] = self.dxl_manual_value[2]
                    ms_state["lift_cmd"][1] = self.dxl_manual_value[3]

                ms_joint_pos = (
                    np.where(~is_on_hold, ms_state["joint_pos"], 0)
                    + np.where(
                        (self.is_on_hold_prev & is_on_hold) ^ is_on_hold,
                        self.sl_state["joint_pos"],
                        0,
                    )
                    + np.where(
                        (~self.is_on_hold_prev & is_on_hold) ^ is_on_hold,
                        ms_joint_pos,
                        0,
                    )
                )
                # print(ms_joint_pos)
                # ms_joint_pos[13] = 50*np.sin(2*np.pi*t_ros.nanoseconds/1e9/6)
                ms_state["joint_pos"] = ms_joint_pos
                self.ref_pos = ms_state["joint_pos"]

                self.is_on_hold_prev = is_on_hold

                smooth_factor = np.clip(
                    (t_ros.nanoseconds / 1e9 - self.joint_when_on)/2, 0, 1
                )
                ctrl_cmd = {
                    "joint_pos": self.sl_state["joint_pos"],
                    "joint_vel": self.sl_state["joint_vel"],
                    "joint_acc": self.sl_state["joint_acc"],
                    "ref_pos": ms_state["joint_pos"],
                    "joint_power": list(ui_state.joint_power),
                    "smooth_factor": smooth_factor,
                    "cmd_override": self.cmd_override,
                    "max_rpm": ui_state.pump_max_rpm,
                    "des_rpm": self.pump_state["des_rpm"],
                }
                if self.traj_enabled:
                    ctrl_cmd["traj_point"] = self.traj_point
                pump_cmd = {
                    "power": ui_state.pump_pwr,
                    "mode": ui_state.pump_mode,
                    "tgt_rpm": ui_state.pump_tgt_speed,
                    "err_norm": self.ctrl_state["err_norm"],
                    "max_rpm": ui_state.pump_max_rpm,
                    "min_rpm": ui_state.pump_min_rpm,
                    "max_err": ui_state.pump_max_err,
                }
                dxl_pos_cmd = np.array(
                    [
                        ms_state["track_cmd"][0],
                        ms_state["track_cmd"][1],
                        ms_state["lift_cmd"][0],
                        ms_state["lift_cmd"][1],
                        ]
                )
                dxl_cmd = {"pos_cmd": dxl_pos_cmd}

                if not self.ctrl_fast_pipe_opp.poll():
                    self.ctrl_fast_pipe.send(ctrl_cmd)
                if not self.pump_fast_pipe_opp.poll():
                    self.pump_fast_pipe.send(pump_cmd)
                if not self.dxl_fast_pipe_opp.poll():
                    self.dxl_fast_pipe.send(dxl_cmd)

                time_frame = np.vstack(
                    (time_frame,
                     [t_ros.nanoseconds,
                      self.sl_state["frame"],
                      self.ctrl_state["frame"],
                      self.pump_state["frame"]]))[-5000:]
                self.freq = (
                    (time_frame[-1][1:] - time_frame[0][1:])
                    / (time_frame[-1][0] - time_frame[0][0])
                    * 1e9
                )
                # self.ms_state = copy.deepcopy(ms_state)

                sl_proc_exitcode = self.sl_proc_handle.exitcode
                ctrl_proc_exitcode = self.ctrl_proc_handle.exitcode
                pump_proc_exitcode = self.pump_proc_handle.exitcode
                if sl_proc_exitcode:
                    raise UserWarning(
                        f"process state exits with {self.sl_proc_handle.exitcode}"
                    )

                if ctrl_proc_exitcode:
                    raise UserWarning(
                        f"process control exits with {self.sl_proc_handle.exitcode}"
                    )

                if pump_proc_exitcode:
                    raise UserWarning(
                        f"process pump exits with {self.sl_proc_handle.exitcode}"
                    )

            except KeyboardInterrupt:
                self.get_logger().warning("interrupting...")
                break
            except UserWarning as e:
                self.get_logger().error(traceback.format_exc())
                self.get_logger().error(f"{e}")
                self.log_timer.destroy()
                self.stop_proc()
                self.destroy_node()
                exit(1)

        self.stop_proc()
        if self.enable_logging:
            self.log_timer.destroy()
            self.log_file.close()
            print(f"log{datetime.now().strftime('%y%m%d%H%M%S')}.csv saved")
        self.destroy_node()
        exit(0)


if __name__ == "__main__":
    rclpy.init(domain_id=1)
    robot = Robot()
    robot.run()
