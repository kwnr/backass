import time
from typing import List
import serial
import serial.rs485
import numpy as np
import rclpy
from rclpy.node import Node
import traceback
import logging
from multiprocessing.connection import Connection
from scipy.signal import medfilt


class SensorRead:
    def __init__(self, pipeset: List[Connection], port_name="/dev/ttyCUI"):
        self.ser = serial.rs485.RS485(port_name, baudrate=2000000, timeout=0.05)
        left_cui_addr = [b'\x6C', b'\x7C', b'\x8C', b'\x9C', b'\xAC', b'\xBC', b'\xEC', b'\xFC']  # L1 to 8
        right_cui_addr = [b'\x0C', b'\x1C', b'\x2C', b'\x3C', b'\x4C', b'\x5C', b'\xCC', b'\xDC']  # R1 to 8
        self.cui_addr =  left_cui_addr + right_cui_addr
        self.cui_value = np.zeros(16, dtype=float)
        
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
        nf = NoiseFilter()
        nf.prev = self.read_one_by_one()
        while True:
            try:
                if self.conn_pause.poll():
                    res = self.conn_pause.recv()
                    if res == "PAUSE":
                        res = self.conn_pause.recv()
                        if res == "RESUME":
                            pass
                       
                self.cui_value = self.read()
                if self.cui_value is None:
                    pass
                elif not self.conn_opp.poll():
                    frame += 1
                    # self.cui_value = nf.update(self.cui_value)
                    self.conn.send({"joint_pos": self.cui_value, "frame": frame})
                
            except KeyboardInterrupt:
                print(f"Inturrupted by user. Process {__name__} closed.")
                break
            
            except Exception as e:
                print(traceback.format_exc())

    def read(self) -> np.ndarray:
        self.ser.reset_input_buffer()
        t_s = time.time_ns()
        for cui_addr in self.cui_addr:
            self.ser.reset_output_buffer()
            self.ser.write(cui_addr)
            time.sleep(.1e-3)
        while self.ser.in_waiting != 32 and (time.time_ns() - t_s) <= 5e6:  # blocks until buffer filled or 3e6 ns (3ms) passed
            pass
        if self.ser.in_waiting == 32:
            raw_value: bytes = self.ser.read(32)  # should be 32 len bytes
        else:
            raw_value = False
        if raw_value:
            raw_array = np.frombuffer(buffer=raw_value, dtype=np.uint8).reshape(-1, 2)
            if self.checksum(raw_array):
                return self.raw_to_deg(raw_array)
            
    def read_one_by_one(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        for i, cui_addr in enumerate(self.cui_addr):
            self.ser.write(cui_addr)
            raw_value = self.ser.read(2)
        raw_array = np.frombuffer(buffer=raw_value, dtype=np.uint8).reshape(-1,2)
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

if __name__ == "__main__":
    from multiprocessing.connection import Pipe
    rclpy.init()
    node = Node("test_state")
    conn, conn_opp = Pipe()
    state = SensorRead([conn, conn_opp, conn, conn_opp, conn, conn_opp,])
    state.run()