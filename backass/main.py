#!/usr/bin/env python3
from multiprocessing import Process, Pipe
import os
import numpy as np
import signal
import copy
import itertools
import time

from typing import List, Union, TypedDict
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data

from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from interass.msg import UI2Robot, Robot2UI, Hold, Preset, UIAction, PoseIteration, ArmMasterCommInt
from aic_utils import control, pump, sensor_read, dxl_control

from datetime import datetime
import traceback


class Robot:
    def __init__(self):
        rclpy.init()
        node = Node("aic_main")
        
        self.sl_fast_pipe, self.sl_fast_pipe_opp = Pipe()  # fast pipe sends data everytime when available
        self.ctrl_fast_pipe, self.ctrl_fast_pipe_opp = Pipe()
        self.pump_fast_pipe, self.pump_fast_pipe_opp = Pipe()
        self.dxl_fast_pipe, self.dxl_fast_pipe_opp = Pipe()
        
        self.sl_slow_pipe, self.sl_slow_pipe_opp = Pipe()  # slow pipe sends data when data changed 
        self.ctrl_slow_pipe, self.ctrl_slow_pipe_opp = Pipe()
        self.pump_slow_pipe, self.pump_slow_pipe_opp = Pipe()
        self.dxl_slow_pipe, self.dxl_slow_pipe_opp = Pipe()
        
        self.sl_pause_pipe, self.sl_pause_pipe_opp = Pipe()  # pause pipe sends "PAUSE" when to pause, "RESUME when to resume
        self.ctrl_pause_pipe, self.ctrl_pause_pipe_opp = Pipe()
        self.pump_pause_pipe, self.pump_pause_pipe_opp = Pipe()
        self.dxl_pause_pipe, self.dxl_pause_pipe_opp = Pipe()
        
        pipeset_sl = [self.sl_fast_pipe_opp, self.sl_fast_pipe,
                      self.sl_slow_pipe_opp, self.sl_slow_pipe,
                      self.sl_pause_pipe_opp, self.sl_pause_pipe]
        
        pipeset_ctrl = [self.ctrl_fast_pipe_opp, self.ctrl_fast_pipe,
                      self.ctrl_slow_pipe_opp, self.ctrl_slow_pipe,
                      self.ctrl_pause_pipe_opp, self.ctrl_pause_pipe]
        
        pipeset_pump = [self.pump_fast_pipe_opp, self.pump_fast_pipe,
                      self.pump_slow_pipe_opp, self.pump_slow_pipe,
                      self.pump_pause_pipe_opp, self.pump_pause_pipe]
        
        pipeset_dxl = [self.dxl_fast_pipe_opp, self.dxl_fast_pipe,
                       self.dxl_slow_pipe_opp, self.dxl_slow_pipe,
                       self.dxl_pause_pipe_opp, self.dxl_fast_pipe]

        self.sl_handle = sensor_read.SensorRead(pipeset_sl)
        self.ctrl_handle = control.Control(pipeset_ctrl)
        self.pump_handle = pump.Pump(pipeset_pump)
        self.dxl_handle = dxl_control.DXLControl(pipeset_dxl)

        self.sl_proc_handle = Process(target=self.sl_handle.run, daemon=True)
        self.ctrl_proc_handle = Process(target=self.ctrl_handle.run, daemon=True)
        self.pump_proc_handle = Process(target=self.pump_handle.run, daemon=True)
        self.dxl_proc_handle = Process(target=self.dxl_handle.run, daemon=True)
        
        signal.signal(signal.SIGINT, self.sigint_handler)
        print(os.getpid())

        self.mask = np.array([False] * 16).reshape(2, 8)
        self.ms_state = {"joint_pos": np.zeros(16, dtype=float),
                         "track_cmd": np.array([500., 500.]),
                         "lift_cmd": np.zeros(2),
                         "frame": 0}
        self.sl_state = {"joint_pos": np.zeros(16, dtype=float), "frame": 0}
        self.ctrl_state = {"cmd": np.zeros(16, dtype=float),
                           "err": np.zeros(16, dtype=float), 
                           "err_i": np.zeros(16, dtype=float),
                           "err_d": np.zeros(16, dtype=float),
                           "err_norm": 0, 
                           "frame": 0}
        self.ms_state[3] = 20
        self.ms_state[10] = 20
        self.ref_pos = np.zeros(16, dtype=float)
        self.pump_state = {"act_rpm": 0, "des_rpm": 0, "des_cur": 0., "temp": 0., "frame": 0}
        
        self.cmd_override = np.array([np.nan]*16, dtype=float)

        self.ms_joint_pos = np.zeros(16)
        self.flag_is_hold = np.array([False] * 16)
        
        self.pub = node.create_publisher(Robot2UI, "robot_to_ui", qos_profile_system_default)
        node.create_subscription(ArmMasterCommInt, "cmd_input", callback=self.cb_ms_state, qos_profile=qos_profile_sensor_data)
        
        self.moveit_pub = node.create_publisher("joint_state", JointState, qos_profile_system_default)
        self.hold_sub = node.create_subscription(Hold, 'hold', callback=self.cb_hold_sub, qos_profile=qos_profile_system_default)
        
        
        self.is_on_hold = np.array([False] * 16)
        self.is_on_hold_prev = self.is_on_hold
        
        self.joint_on_smooth = np.array([False] * 16)
        self.joint_when_on = np.zeros(16)

        self.preset_sub = node.create_subscription(Preset, "preset", callback=self.cb_preset_sub, qos_profile=qos_profile_system_default)
        self.preset_pos = np.zeros(16)
        self.is_preset_mode = False
        
        self.is_ref_init = False
        
        self.ui_action_sub = node.create_subscription(UIAction, 'ui_action', callback=self.cb_ui_action_sub, qos_profile=qos_profile_system_default)
        self.ui_state = UIAction(
            False,  # pump_pwr
            0,  # pump_mode
            0,  # pump_tgt_speed
            [False] * 16,  # joint_power
            [False] * 16,  # joint_to_smooth
            2800,  # pump_max_rpm
            500,  # pump_min_rpm
            50,  # pump_max_err
            0,  # ctrl_mode
        )    
        
        self.pose_iter_sub = node.create_subscription(PoseIteration, "pose_iter", callback=self.cb_pose_iter, qos_profile=qos_profile_system_default)
        self.pose_iter_enabled = False
        self.pose_iter_dxl = 0
        
        self.enable_logging = True
        if True:
            header = (["timestamp"] + 
                      [f"L{i} pos" for i in range(1,9)] + [f"R{i} pos" for i in range(1,9)] + 
                      [f"L{i} ref" for i in range(1,9)] + [f"R{i} ref" for i in range(1,9)] + 
                      [f"L{i} cmd" for i in range(1,9)] + [f"R{i} cmd" for i in range(1,9)] +
                      ["FB rpm", "des rpm", "des cur", "temp"])
            self.log_dir = os.curdir + '/log/values_log/'
            self.log_name = self.log_dir + f"log{datetime.now().strftime('%y%m%d%H%M%S')}.csv"
            self.log_file = open(self.log_name, "a")
            self.log_file.write(",".join(header)+"\n")
            self.log_timer = node.create_timer(0.01, self.log_data)
    
    def log_data(self, timer_event):
        logdata = [time.time(), *self.sl_state["joint_pos"], *self.ref_pos.tolist(), *self.ctrl_state["cmd"].tolist()]
        pump_state_str  = ",".join(map("{:.4f}".format,
                                   [self.pump_state["act_rpm"], self.pump_state["des_rpm"],
                                    self.pump_state["des_cur"], self.pump_state["temp"]]))
        self.log_file.write(",".join(map("{:.4f}".format, logdata)) + "," + pump_state_str + "\n")

    def start_proc(self):
        # self.ms_proc_handle.start()
        self.sl_proc_handle.start()
        self.ctrl_proc_handle.start()
        self.pump_proc_handle.start()
        self.dxl_proc_handle.start()
        print(f"proc::state started with pid {self.sl_proc_handle.pid}")
        print(f"proc::control started with pid {self.ctrl_proc_handle.pid}")
        print(f"proc::pump started with pid {self.pump_proc_handle.pid}")

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
            self.joint_when_on = np.broadcast_to(time.time(), 16) * joint_to_smooth + self.joint_when_on * ~joint_to_smooth
    
    def cb_preset_sub(self, data: Preset):
        self.is_preset_mode = data.enabled
        self.preset_pos = np.array(data.preset_pos)
        print(self.is_preset_mode, self.preset_pos)
        
    def cb_ms_state(self, data: ArmMasterCommInt):
        ms_joint_state = np.array([data.L1, data.L2, data.L3, data.L4, data.L5, data.L6,
                             self.ms_state["joint_pos"][6], self.ms_state["joint_pos"][7],  # L7~8
                             data.R1, data.R2, data.R3, data.R4, data.R5, data.R6,
                             self.ms_state["joint_pos"][14], self.ms_state["joint_pos"][15]])  # R7~8
        if data.L7 != 0:  # if master l7 lever is not zero
            ms_joint_state[6] = self.sl_state["joint_pos"][6]  # ref pos is slave's pos
            self.cmd_override[6] = data.L7 * 3  # overrides cmd value to 3
            self.flag_is_hold[6] = False  # this axis is not held
        elif not self.flag_is_hold[6]:  # if master l7 lever is zero and is not held
            ms_joint_state[6] = self.sl_state["joint_pos"][6]  # ref pos is slave's pos
            self.cmd_override[6] = np.nan  # not overrides cmd value
            self.flag_is_hold[6] = True  # this axis is held            
            
        if data.L8 != 0:
            ms_joint_state[7] = self.sl_state["joint_pos"][7]
            self.cmd_override[7] = data.L8 * 3
            self.flag_is_hold[7] = False
        elif not self.flag_is_hold[7]:
            ms_joint_state[7] = self.sl_state["joint_pos"][7]
            self.cmd_override[7] = np.nan
            self.flag_is_hold[7] = True
           
        if data.R7 != 0:
            ms_joint_state[14] = self.sl_state["joint_pos"][14] 
            self.cmd_override[14] = 3 * data.R7
            self.flag_is_hold[14] = False
        elif not self.flag_is_hold[14]:
            ms_joint_state[14] = self.sl_state["joint_pos"][14]
            self.cmd_override[14] = np.nan
            self.flag_is_hold[14] = True
           
        if data.R8 != 0: 
            ms_joint_state[15] = self.sl_state["joint_pos"][15]
            self.cmd_override[15] = 3 * data.R8
            self.flag_is_hold[15] = False
        elif not self.flag_is_hold[15]:
            ms_joint_state[15] = self.sl_state["joint_pos"][15]
            self.cmd_override[15] = np.nan
            self.flag_is_hold[15] = True
           
        self.ms_state["joint_pos"] = ms_joint_state
        # FIXME
        self.ms_state["lift_cmd"] = np.array([data.lifter, data.pump])
        self.ms_state["track_cmd"] = np.array([data.lever_0, data.lever_1])     
        
    def init_ref(self):
        pos_now = copy.deepcopy(self.sl_state["joint_pos"])
        if np.array(pos_now, dtype=bool).any():
            self.ms_state["joint_pos"] = pos_now
            self.is_ref_init = True
            
    def cb_pose_iter(self, data: PoseIteration):
        self.pose_iter_enabled = data.enabled
        self.pose_iter_pose = data.poses[:16]
        # FIXME
        self.pose_iter_dxl = data.trigger
        
    def sigint_handler(self, signum, frame):
        raise KeyboardInterrupt
    
    def homing(self, tgt_joint: Union[list, int]) -> None:
        """
        function for homing joints. while on action, blocks code until homing finished
        """
        joint_list = [f"L{i}" for i in range(1,9)] + [f"R{i}" for i in range(1,9)]
        pump_rpm = 300
        volt_cmd = 3

        if type(tgt_joint) == int: 
            tgt_joint = [tgt_joint]
        
        for i in tgt_joint:
            print(f"Start Homing for Joint {joint_list[i]}...")
            pump_cmd: PumpCommand = {"power": True, "mode": 0, "tgt_rpm": pump_rpm, 
                                     "min_rpm": 0, "max_rpm": 0, "max_err": 0, "err_norm": 0}
            self.pump_fast_pipe.send(pump_cmd)
            sl_state_prev = self.sl_fast_pipe.recv()

            cmd_override = np.zeros(16)
            cmd_override[i] = volt_cmd  # needs to be specified
            joint_pwr = np.zeros(16)
            joint_pwr[i] = 1
            
            ext_cmd = chr(ord(self.sl_handle.cui_addr[i].decode())+2).encode()
            set_zero_cmd = b'\x5e'
            
            ctrl_cmd: CtrlCommand = {"joint_power": joint_pwr, 
                                     "joint_pos": sl_state_prev["joint_pos"],
                                     "ref_pos": sl_state_prev["joint_pos"],
                                     "cmd_override": cmd_override,
                                     "smooth_factor": np.ones(16),}
            self.ctrl_fast_pipe.send(ctrl_cmd)
            while True:
                sl_state = self.sl_fast_pipe.recv()
                diff = sl_state["joint_pos"][i] - sl_state_prev["joint_pos"][i]
                if is_homed(i):
                    break
            self.sl_pause_pipe.send("PAUSE")
            self.sl_handle.ser.write(ext_cmd)
            
            self.sl_handle.ser.write(set_zero_cmd)
            self.sl_pause_pipe.send("RESUME")
            
            # offset 
            
                
        def is_homed(joint_idx: int) -> bool:
            pass

    def run(self):
        self.start_proc()
        time_frame = np.array([0, 0, 0, 0])
        ms_joint_pos = np.zeros(16)
        while True:
            try:
                t_ros = rclpy.time.Time()
                # if self.ms_conn_p.poll():
                #    self.ms_state = self.ms_conn_p.recv()
                if self.sl_fast_pipe.poll():
                    self.sl_state = self.sl_fast_pipe.recv()
                    if not self.is_ref_init: self.init_ref()
                if self.ctrl_fast_pipe.poll():
                    self.ctrl_state = self.ctrl_fast_pipe.recv()
                if self.pump_fast_pipe.poll():
                    self.pump_state = self.pump_fast_pipe.recv()
                is_on_hold = self.is_on_hold
                ui_state = self.ui_state
                ms_state = copy.deepcopy(self.ms_state)
                
                if self.is_preset_mode:
                    ms_state["joint_pos"] = self.preset_pos
                    
                if self.pose_iter_enabled:
                    ms_state["joint_pos"] = self.pose_iter_pose
                    is_on_hold = np.where(np.isnan(self.pose_iter_pose), True, is_on_hold)
                    ms_state["lift_cmd"][1] = 1
                    ms_state["track_cmd"][1] = self.pose_iter_dxl *  5 + 500 if self.pose_iter_dxl != -1 else -1

                ms_joint_pos = (np.where(~is_on_hold, ms_state["joint_pos"], 0) +
                                np.where((self.is_on_hold_prev & is_on_hold) ^ is_on_hold, self.sl_state["joint_pos"], 0) + 
                                np.where((~self.is_on_hold_prev & is_on_hold) ^ is_on_hold, ms_joint_pos, 0))
                
                ms_state["joint_pos"] = ms_joint_pos
                self.is_on_hold_prev = is_on_hold
                
                smooth_factor = np.clip((t_ros.to_sec() - self.joint_when_on)/2 ,0, 1)

                ctrl_cmd: CtrlCommand = {"joint_pos": self.sl_state["joint_pos"],
                                        "ref_pos": ms_state["joint_pos"],
                                        "joint_power": ui_state.joint_power,
                                        "smooth_factor": smooth_factor,
                                        "cmd_override": self.cmd_override,
                                        "max_rpm": ui_state.pump_max_rpm,
                                        "des_rpm": self.pump_state["des_rpm"],}
                pump_cmd: PumpCommand = {"power": ui_state.pump_pwr,
                                         "mode": ui_state.pump_mode,
                                         "tgt_rpm": ui_state.pump_tgt_speed,
                                         "err_norm": self.ctrl_state["err_norm"],
                                         "max_rpm": ui_state.pump_max_rpm,
                                         "min_rpm": ui_state.pump_min_rpm,
                                         "max_err": ui_state.pump_max_err,}
                dxl_pos_cmd = np.array([ms_state["track_cmd"][0],
                                        ms_state["track_cmd"][1] if ms_state["lift_cmd"][1]==0 else 500,
                                        ms_state["lift_cmd"][0],
                                        ms_state["track_cmd"][1] if not ms_state["lift_cmd"][1]==0 else -1])
                dxl_cmd: DXLCommand = {"pos_cmd": dxl_pos_cmd}
                
                if not self.ctrl_fast_pipe_opp.poll():
                    self.ctrl_fast_pipe.send(ctrl_cmd)
                if not self.pump_fast_pipe_opp.poll():
                    self.pump_fast_pipe.send(pump_cmd)
                if not self.dxl_fast_pipe_opp.poll():
                    self.dxl_fast_pipe.send(dxl_cmd)
                    
                time_frame = np.vstack((time_frame, [t_ros.to_nsec(),
                                                     self.sl_state["frame"],
                                                     self.ctrl_state["frame"],
                                                     self.pump_state["frame"]
                                                     ]))[-5000:]
                freq = (time_frame[-1][1:] - time_frame[0][1:]) / (time_frame[-1][0] - time_frame[0][0]) * 1e9
                
                self.pub.publish(
                    Robot2UI(
                        header=Header(stamp=t_ros),
                        pump_act_rpm=self.pump_state["act_rpm"],
                        pump_des_rpm=self.pump_state["des_rpm"],
                        pump_des_cur=self.pump_state["des_cur"],
                        pump_temp=self.pump_state["temp"],
                        arm_state=self.sl_state["joint_pos"],
                        ref_state=ms_state["joint_pos"],
                        track_left_state=int(ms_state["track_cmd"][0]),
                        track_right_state=int(ms_state["track_cmd"][1]),
                        cmd_voltage=self.ctrl_state["cmd"],
                        err=self.ctrl_state["err"],
                        err_i=self.ctrl_state["err_i"],
                        err_d=self.ctrl_state["err_d"],
                        err_norm=self.ctrl_state["err_norm"],
                        freq=freq
                    )
                )
                
                sl_proc_exitcode = self.sl_proc_handle.exitcode
                ctrl_proc_exitcode = self.ctrl_proc_handle.exitcode
                pump_proc_exitcode = self.pump_proc_handle.exitcode
                if sl_proc_exitcode:
                    raise UserWarning(f"process state exits with exitcode {self.sl_proc_handle.exitcode}")

                if ctrl_proc_exitcode:
                    raise UserWarning(f"process control exits with exitcode {self.sl_proc_handle.exitcode}")

                if pump_proc_exitcode:
                    raise UserWarning(f"process pump exits with exitcode {self.sl_proc_handle.exitcode}")

            except KeyboardInterrupt:
                print("interrupting...")
                break
            except UserWarning as e:
                print(traceback.format_exc())
                print(e)
                self.log_timer.shutdown()
                self.stop_proc()
                exit(1)
                
        self.stop_proc()
        if self.enable_logging:
            self.log_timer.shutdown()
            print(f"log{datetime.now().strftime('%y%m%d%H%M%S')}.csv saved")
        exit(0)
        
    
class PoseIterator():
    def __init__(self, poses) -> None:
        self.converge_criteria = np.array([0.5] * 16)
        self.is_converged = np.array([False] * 16)
        self.poses = poses
    
    def __next__():
        pass
    
    def converged(self, position):
        pass


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
    
class CtrlCommand(TypedDict):
    joint_pos: np.ndarray
    ref_pos: np.ndarray
    joint_power: List[bool]
    smooth_factor: np.ndarray
    cmd_override: np.ndarray
    max_rpm: np.ndarray

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

if __name__ == "__main__":
    robot = Robot()
    robot.run()