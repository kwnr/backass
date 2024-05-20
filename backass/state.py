import time
from typing import List

import serial
import serial.rs485

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from multiprocessing.connection import Connection

from utils import *
import traceback


class SensorRead:
    def __init__(self, pipeset: List[Connection], parent_node: Node, port_name="/dev/ttyCUI"):
        self.ser = serial.rs485.RS485(port_name, baudrate=2000000, timeout=0.01)
        left_cui_addr = [b'\x6C', b'\x7C', b'\x8C', b'\x9C',
                         b'\xAC', b'\xBC', b'\xEC', b'\xFC']  # L1 to 8
        right_cui_addr = [b'\x0C', b'\x1C', b'\x2C', b'\x3C',
                          b'\x4C', b'\x5C', b'\xCC', b'\xDC']  # R1 to 8
        self.cui_addr = left_cui_addr + right_cui_addr
        self.joint_pos = JointPosition()

        self.conn = pipeset[0]
        self.conn_opp = pipeset[1]
        self.conn_slow = pipeset[2]
        self.conn_slow_opp = pipeset[3]
        self.conn_pause = pipeset[4]
        self.conn_pause_opp = pipeset[5]

    def __del__(self):
        self.ser.close()
        print(f"class {__name__} deleted")

    def run(self):
        frame = 0
        lpf = LowPassFilter(1, 1/100)
        timestamp_prev = time.perf_counter()
        
        while True:
            try:
                timestamp = time.perf_counter()
                dt = timestamp - timestamp_prev
                timestamp_prev = timestamp
                if self.conn_pause.poll():
                    res = self.conn_pause.recv()
                    if res == "PAUSE":
                        res = self.conn_pause.recv()
                        if res == "RESUME":
                            pass

                joint_pos = self.read()
                if joint_pos is None:
                    continue
                elif not self.conn_opp.poll():
                    frame += 1                    
                    joint_pos = lpf.filter(joint_pos)
                    self.joint_pos.append(joint_pos)
                    self.conn.send({"joint_pos": self.joint_pos.pos,
                                    "joint_vel": self.joint_pos.vel,
                                    "joint_acc": self.joint_pos.acc,
                                    "frame": frame})

            except KeyboardInterrupt:
                print(f"Inturrupted by user. Process {__name__} closed.")
                break

            except Exception:
                print(traceback.format_exc())

    def read(self) -> np.ndarray:
        self.ser.reset_input_buffer()
        t_s = time.time_ns()
        for cui_addr in self.cui_addr:
            self.ser.reset_output_buffer()
            self.ser.write(cui_addr)
            time.sleep(1e-4)
        while True:  # blocks until buffer filled or 3e6 ns (3ms) passed
            in_waiting = self.ser.in_waiting
            if in_waiting == 32:
                break
            if time.time_ns() - t_s >= 5e7:
                break

        if self.ser.in_waiting == 32:
            raw_value: bytes = self.ser.read(32)  # should be 32 len bytes
            raw_array = np.frombuffer(buffer=raw_value, dtype=np.uint8).reshape(-1, 2)
            if self.checksum(raw_array):
                return self.raw_to_deg(raw_array)
            else:
                print("CHECKSUM FAILED")
        else:
            raw_value = False

    def read_one_by_one(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        raw_value = b''
        for i, cui_addr in enumerate(self.cui_addr):
            self.ser.write(cui_addr)
            raw_value += self.ser.read(2)
        raw_array = np.frombuffer(buffer=raw_value, dtype=np.uint8).reshape(-1, 2)
        if self.checksum(raw_array):
            return self.raw_to_deg(raw_array)
        else:
            return self.read_one_by_one()

    @staticmethod
    def raw_to_deg(a: np.ndarray) -> np.ndarray:
        a = a & [0b11111111, 0b00111111]
        total_bytes = (a @ np.array([[1], [2 ** 8]])).reshape(-1)
        val_ori = total_bytes * 360 / 2**14
        mask = np.array([
            0, 1, 1, 1, 0, 1,
            1, 1,
            1, 0, 0, 0, 1, 0,
            0, 0,
            ])
        val = mask * 360 - (mask * 2 - 1) * val_ori
        val = val - (val > 180) * 360
        return val

    @staticmethod
    def checksum(a: np.ndarray) -> bool:
        total_bytes = a @ np.array([[1], [2**8]])  # shape of total_bytes: (-1,1)
        odd_masked = total_bytes & np.uint16(0b1010101010101010)
        even_masked = total_bytes & np.uint16(0b0101010101010101)
        return (
            np.char.count((list(map(bin, even_masked[:, 0]))), "1") % 2
            & np.char.count((list(map(bin, odd_masked[:, 0]))), "1") % 2
        ).all()



        

if __name__ == "__main__":
    from multiprocessing.connection import Pipe
    rclpy.init()
    node = Node("test_state")
    conn, conn_opp = Pipe()
    state = SensorRead([conn, conn_opp, conn, conn_opp, conn, conn_opp,])
    state.run()
