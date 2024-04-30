from yoctopuce.yocto_api import *
from yoctopuce.yocto_pwmoutput import *

PWM_FRQ_MAX = 0
PWM_FRQ_MIN = 0
PWM_DTY_CYC = 0

class HPUPWM():
    def __init__(self, pipeset):
        self.conn = pipeset[0]
        self.conn_opp = pipeset[1]
        self.conn_slow = pipeset[2]
        self.conn_slow_opp = pipeset[3]
        self.conn_pause = pipeset[4]
        self.conn_pause_opp = pipeset[5]
        
        self.pwm = YPwmOutput.FirstPwmOutput()
        if self.pwm is None:
            assert "cannot find yoctoPWM..."
        self.pwm.set_enabled(YPwmOutput.ENABLED_TRUE)
        
        # received from pipe
        self.power = 0
        self.mode = 0
        self.tgt_rpm = 0
        self.err_norm = 0

        # return
        self.act_rpm = 0
        self.des_rpm = 0
        self.des_cur = 0.0

        # pump params
        self.min_rpm = 300
        self.max_rpm = 2000
        self.max_err = 50
        self.max_acc = 2000
        self.max_dec = 2000
        self.tgt_stop_dec = 2500
        self.has_motor_enabled = False
        self.has_motor_disabled = False
    
    def motor_startup(self):
        pass
        
    def set_motor_off(self):
        self.pwm.set_enabled(YPwmOutput.ENABLED_FALSE)
    
    def get_motor_speed(self):
        pass
        