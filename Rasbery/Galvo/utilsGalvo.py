import time
import board
import busio
from digitalio import DigitalInOut, Direction
from .mcp48xx import MCP4822

class GalvoController:
    def __init__(
        self,
        max_angle_deg=40.0,
        laser_pin=board.D17,
        gain=2,
        safe_start=True,
        max_code=4095
    ):
        self.max_angle = max_angle_deg

        # SPI
        self.spi = busio.SPI(board.SCK, board.MOSI)
        self.cs = None
        self.dac = MCP4822(self.spi, self.cs)

        # gain DAC
        self.dac.channel_a.gain = gain
        self.dac.channel_b.gain = gain

        # laser
        self.laser = DigitalInOut(laser_pin)
        self.laser.direction = Direction.OUTPUT
        self.laser.value = False  # laser OFF par défaut
        self.max_code = max_code

        if safe_start:
            self.set_angles(0.0, 0.0)

    def angle_to_dac(self,angle_deg, max_angle_deg=20.0, ):
        # limiter angle
        angle_deg = max(-max_angle_deg,min(max_angle_deg,angle_deg))
        
        # -max -> 0.0, 0 -> 0.5, +max -> 1.0
        normalized = (angle_deg + max_angle_deg) / (2.0 * max_angle_deg)
        return int(round(normalized * self.max_code)) 


    def set_angles(self, theta_x, theta_y):
        code_x = self.angle_to_dac(theta_x, self.max_angle)
        code_y = self.angle_to_dac(theta_y, self.max_angle)

        self.dac.channel_a.raw_value = code_x
        self.dac.channel_b.raw_value = code_y

    def laser_on(self):
        self.laser.value = True

    def laser_off(self):
        self.set_angles(0.0, 0.0)
        self.laser.value = False

    def shutdown(self):
        """Arrêt sécurisé"""
        self.laser_off()
        self.set_angles(0.0, 0.0)
        time.sleep(0.1)
