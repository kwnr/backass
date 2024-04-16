import numpy as np
import rospy
import time
from arm_msgs.msg import arm_master_comm_int
from multiprocessing.connection import Pipe


class MasterState:
    def __init__(self, conn, conn_opp):
        self.conn = conn
        self.conn_opp = conn_opp
        self.joint_pos = np.zeros(16)
        self.lift_cmd = np.zeros(2)
        self.track_cmd = np.zeros(2)

    def __del__(self):
        print("MasterState terminated")

    def run(self):
        print("proc::master_state started")
        rospy.Subscriber(
            "cmd_input", callback=self.master_callback, data_class=arm_master_comm_int, tcp_nodelay=True, queue_size=10
        )
        while True:
            if not self.conn_opp.poll():
                self.conn.send({"joint_pos": self.joint_pos, "lift_cmd": self.lift_cmd, "track_cmd": self.track_cmd})

    def p_run(self):
        rate = rospy.Rate(1000)
        while True:
            t_ros = rospy.get_rostime()
            t = time.time()
            data = arm_master_comm(t_ros, *(np.sin(np.arange(1, 16 + 1) * t) * 10),
                                   *(np.sin([t, t * 2]) * 100),
                                   *(list(map(int, np.sin([t, t * 2]) * 100 + 500))))
            self.master_callback(data)
            rate.sleep()

    def master_callback(self, data: arm_master_comm_int):
        self.joint_pos = np.array([data.L1, data.L2, data.L3, data.L4, data.L5, data.L6, data.L7, data.L8,
                                   data.R1, data.R2, data.R3, data.R4, data.R5, data.R6, data.R7, data.R8])
        self.lift_cmd = np.array([data.lifter, data.pump])
        self.track_cmd = np.array([data.lever_0, data.lever_1])
        print("asdf")


if __name__ == "__main__":
    conn1, conn2 = Pipe()
    ms = MasterState(conn1, conn2)
    ms.run()