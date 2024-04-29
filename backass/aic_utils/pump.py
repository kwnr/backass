import serial
import serial.rs485
import numpy as np


class Pump:
    def __init__(self, pipeset):
        try:
            self.ser = serial.Serial("/dev/ttyELMO", baudrate=115200, timeout=1)
        except serial.SerialException:
            print("FATAL: Could not open serial")
            raise SystemExit(1)
            
        self.elmo = Elmo(self.ser)
        self.elmo.write("EC=0;")
        self.elmo.flush()

        self.conn = pipeset[0]
        self.conn_opp = pipeset[1]
        self.conn_slow = pipeset[2]
        self.conn_slow_opp = pipeset[3]
        self.conn_pause = pipeset[4]
        self.conn_pause_opp = pipeset[5]

        # received from pipe
        self.power = 0
        self.mode = 0
        self.tgt_rpm = 0
        self.err_norm = 0

        # return
        self.act_rpm = 0
        self.des_rpm = 0
        self.des_cur = 0.0
        self.elmo_temp = 0.

        # pump params
        self.min_rpm = 300
        self.max_rpm = 2000
        self.max_err = 50
        self.max_acc = 2000
        self.max_dec = 2000
        self.tgt_stop_dec = 2500
        self.has_motor_enabled = False
        self.has_motor_disabled = False

    def __del__(self):
        self.ser.close()
        print(f"class {__name__} deleted")
        del self.elmo

    def control(self):
        if self.mode == 0:  # manual
            self.elmo.set_motor_tgt_rpm(self.tgt_rpm)
        elif self.mode == 1:  # auto
            if self.err_norm > 100:
                self.tgt_rpm = self.max_rpm
                # print(f"auto rpm saturated @ {self.tgt_rpm}")
            else:
                self.tgt_rpm = (
                    (self.max_rpm - self.min_rpm) // self.max_err * self.err_norm
                ) + self.min_rpm
                self.tgt_rpm = np.clip(self.tgt_rpm, self.min_rpm, self.max_rpm)
                # print(f"auto rpm tgts @ {self.tgt_rpm}")
            self.elmo.set_motor_tgt_rpm(self.tgt_rpm)
        self.elmo.begin_motion()

    def receiver(self):
        data = self.conn.recv()
        self.power = data["power"]
        self.mode = data["mode"]
        self.tgt_rpm = data["tgt_rpm"]
        self.err_norm = data["err_norm"]
        self.max_rpm = data["max_rpm"]
        self.min_rpm = data["min_rpm"]

    def get_elmo_state(self):
        self.act_rpm = self.elmo.get_elmo_fb_rpm()
        self.des_rpm = self.elmo.get_elmo_des_rpm()
        self.des_cur = self.elmo.get_elmo_des_cur()
        self.elmo_temp = self.elmo.get_elmo_temp()

    def run(self):
        frame = 0
        try:
            while True:
                if self.conn.poll():
                    self.receiver()
                self.get_elmo_state()

                if self.power == 1:
                    if not self.has_motor_enabled:
                        print("MOTOR ON")
                        self.elmo.set_motor_on()
                        self.elmo.set_motor_max_acc(self.max_acc)
                        self.elmo.set_motor_max_dec(self.max_dec)
                        self.elmo.set_motor_tgt_stop_dec(self.tgt_stop_dec)
                        self.elmo.set_motor_tgt_rpm(0)
                        self.elmo.begin_motion()
                        self.has_motor_enabled = True
                        self.has_motor_disabled = False

                    # print(f"err norm: {self.err_norm}")
                    self.control()
                    # rate needs to be specified

                else:
                    if not self.has_motor_disabled:
                        print("MOTOR OFF")
                        self.elmo.set_motor_tgt_rpm(0)
                        self.elmo.begin_motion()
                        self.elmo.set_motor_off()
                        self.has_motor_enabled = False
                        self.has_motor_disabled = True

                frame += 1
                if not self.conn_opp.poll():
                    self.conn.send(
                        {
                            "frame": frame,
                            "act_rpm": self.act_rpm,
                            "des_rpm": self.des_rpm,
                            "des_cur": self.des_cur,
                            "elmo_temp": self.elmo_temp,
                        }
                    )

            self.elmo.set_motor_off()
        except KeyboardInterrupt:
            self.elmo.set_motor_off()
            print(f"Inturrupted by user. Process {__name__} closed.")
            return


class Elmo:
    def __init__(self, ser: serial.Serial):
        self.ser = ser
        self.cnt_rpm = 21 * 6 / 60

    def get_elmo_temp(self):
        res = self.write("TI[1];").split(";")
        if len(res) == 3:
            temp = float(res[1])
        else:
            temp = -1.0
        return temp

    def get_elmo_des_rpm(self):
        res = self.write("DV[2];").split(";")
        if len(res) == 3:
            rpm = int(int(res[1]) // self.cnt_rpm)
        else:
            rpm = -1
        return rpm

    def get_elmo_fb_rpm(self):
        res = self.write("FV[1];").split(";")
        if len(res) == 3:
            rpm = int(int(res[1]) // self.cnt_rpm)
        else:
            rpm = -1
        return rpm

    def get_elmo_des_cur(self):
        res = self.write("IQ;").split(";")
        if len(res) == 3:
            cur = float(res[1])
        else:
            cur = -1.0
        return cur

    def write(self, command: str):
        self.ser.flush()
        self.ser.write(command.encode("ASCII"))
        fb1 = self.ser.read_until(b";").decode()
        fb2 = self.ser.read_until(b";").decode()
        feedback = fb1 + fb2
        self.ser.flush()
        # print(f"feedback for command {command}: {feedback}")
        if "?" in feedback:
            print(feedback)
            self.ser.write(b"EC;")
            errmsg = self.ser.read_until(b";").decode()
            errmsg += self.ser.read_until(b";").decode()
            print(f"ELMO returns err: {errmsg}")
        return feedback

    def set_motor_on(self):
        return self.write("MO=1;")

    def set_motor_off(self):
        return self.write("MO=0;")

    def set_motor_tgt_rpm(self, tgt_rpm: int):
        value = int(tgt_rpm * self.cnt_rpm)
        return self.write(f"JV = {value};")

    def set_motor_max_acc(self, max_acc: int):
        return self.write(f"AC = {max_acc};")

    def set_motor_max_dec(self, max_acc: int):
        return self.write(f"DC = {max_acc};")

    def set_motor_tgt_stop_dec(self, tgt_stop_dec: int):
        return self.write(f"SD = {tgt_stop_dec};")
        # is SD correct? could be SF

    def begin_motion(self):
        return self.write("BG;")

    def flush(self):
        self.ser.reset_input_buffer()


if __name__ == "__main__":
    from multiprocessing.connection import Pipe

    p1, p2 = Pipe()
    pump = Pump(p1, p2)
    pump.run()
