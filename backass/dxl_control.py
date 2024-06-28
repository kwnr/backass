import dynamixel_sdk
import numpy as np
from rclpy.node import Node

# TODO make write function for dxl

DXL_ID = [11, 12, 14, 15, 16]  # left, right, lift, trigger, trigger-directional
NUM_DXL = len(DXL_ID)

DXL_BAUDRATE = 115200
DXL_PORT = "/dev/ttyDXL"

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_PRESENT_POSITION = 36
ADDR_MX_P_GAIN = 28
ADDR_MX_I_GAIN = 27
ADDR_MX_D_GAIN = 26

LEN_MX_GOAL_POSITION = 4
LEN_MX_PRESENT_POSITION = 2

# track_left, track_right, lifter, trigger, trigger_direction
POS_DXL_IDLE = np.array([1050, 1710, 2600, 2300, 1500])
POS_DXL_MIN = np.array([1600, 1160, 2310, 2210, 1200])
POS_DXL_MAX = np.array([500, 2260, 2800, 2100, 1800])


class DXLControl:
    def __init__(self, pipeset, parent_node: Node):
        self.conn = pipeset[0]
        self.conn_opp = pipeset[1]
        self.conn_slow = pipeset[2]
        self.conn_slow_opp = pipeset[3]
        self.conn_pause = pipeset[4]
        self.conn_pause_opp = pipeset[5]

        self.port = dynamixel_sdk.PortHandler(DXL_PORT)
        self.packet = dynamixel_sdk.PacketHandler(protocol_version=1.0)
        self.group_write_pos = dynamixel_sdk.GroupSyncWrite(
            self.port, self.packet, ADDR_MX_GOAL_POSITION, LEN_MX_GOAL_POSITION
        )

        self.cmd = np.zeros(NUM_DXL)
        self.dxl_init()

    def __del__(self):
        for i, id in enumerate(DXL_ID):
            res, err = self.packet.write1ByteTxRx(
                self.port, id, ADDR_MX_TORQUE_ENABLE, 0
            )
        self.port.closePort()

    def run(self):
        val_min = np.array([0, 0, 1, 500, 0])
        val_max = np.array([1000, 1000, 2, 1000, 1])
        while True:
            if self.conn.poll():
                cmd = self.pipe_reciever()
                goal_pos = self.array_map(
                    cmd, val_min, val_max, POS_DXL_MIN, POS_DXL_MAX
                )
                self.move(goal_pos)
            pos_cur = self.read_pos()
            if self.conn_opp.poll():
                self.conn.send(pos_cur)

    def pipe_reciever(self):
        data = self.conn.recv()
        pos_cmd = data["pos_cmd"]
        # pos_cmd[i]
        # i = 0: left, 1: right, 2: lift, 3: pump
        # cmd[i]
        # i = 0: left, 1: right, 2: lift, 3: trigger, 4: trigger-directional
        cmd = np.zeros(5)
        if pos_cmd[3] == 0:  # when trigger not activated
            cmd = np.array([pos_cmd[0], pos_cmd[1], pos_cmd[2], -1, -1])
        else:
            trigger = pos_cmd[1] if pos_cmd[1] > 510 else -1
            direction = 0 if pos_cmd[3] == -1 else 1
            cmd = np.array([-1, -1, -1, trigger, direction])

        if pos_cmd[2] == 0:
            cmd[2] = -1
        return cmd

    @staticmethod
    def array_map(value: np.ndarray, val_min, val_max, out_min, out_max) -> np.ndarray:
        ret = np.where(
            value != -1.0,
            (value - val_min) * (out_max - out_min) / (val_max - val_min) + out_min,
            POS_DXL_IDLE,
        )
        return ret

    def dxl_init(self):
        self.port.openPort()
        self.port.setBaudRate(DXL_BAUDRATE)
        for i, id in enumerate(DXL_ID):
            res, err = self.packet.write1ByteTxRx(
                self.port, id, ADDR_MX_TORQUE_ENABLE, 1
            )
            if res != 0:
                print(self.packet.getTxRxResult(res))
            if err != 0:
                print(self.packet.getRxPacketError(err))
        self.packet.write2ByteTxOnly(self.port, 15, ADDR_MX_P_GAIN, 100)
        self.packet.write1ByteTxOnly(self.port, 15, ADDR_MX_I_GAIN, 10)
        self.packet.write2ByteTxOnly(self.port, 15, ADDR_MX_D_GAIN, 5)

    def move(self, goal_pos: np.ndarray):
        for i, id in enumerate(DXL_ID):
            res = self.group_write_pos.addParam(
                id, self.make_word(goal_pos[i].astype(int))
            )
        res = self.group_write_pos.txPacket()
        if res != 0:
            print(self.packet.getTxRxResult(res))
        self.group_write_pos.clearParam()
        return res

    def read_pos(self):
        pos = np.zeros_like(DXL_ID)
        for i, id in enumerate(DXL_ID):
            data, res, err = self.packet.read2ByteTxRx(
                self.port, id, ADDR_MX_PRESENT_POSITION
            )
            pos[i] = data
        return pos

    @staticmethod
    def make_word(value):
        return [
            dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_LOWORD(value)),
            dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_LOWORD(value)),
            dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_HIWORD(value)),
            dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_HIWORD(value)),
        ]


if __name__ == "__main__":
    from multiprocessing import Pipe
    node = Node("dxl")
    m, o = Pipe()
    dxl = DXLControl([m, o] * 3, node)
    dxl.run()
