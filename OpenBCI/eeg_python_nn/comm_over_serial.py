import serial
import time
import random

port = "/dev/ttyS1"

def init_serial():
    ser = serial.Serial(port, 9600)
    time.sleep(1)
    return ser

def action_callback(ser, action):
    ser.write((action + '\n').encode('utf-8'))
    print(ser, action)
    time.sleep(random.uniform(4, 6))
    print(ser, action)
    ser.write((action + '\n').encode('utf-8'))
    # time.sleep(0.1)
