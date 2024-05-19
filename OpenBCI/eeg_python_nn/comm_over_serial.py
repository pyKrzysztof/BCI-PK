import serial
import time
import random

port = "/dev/ttyS1"

# parameters returned by the initialization function will always be passed 
# as first argument of the on_action and on_close functions.
def init_serial():
    ser = serial.Serial(port, 9600)
    time.sleep(1)
    return ser

def send_action(ser, action: str):
    print(ser, action)
    ser.write((action + '\n').encode('utf-8'))
    time.sleep(random.uniform(4, 6))
    print(ser, action)
    ser.write((action + '\n').encode('utf-8'))

def close_serial(ser):
    print("Closing connection", ser)
    ser.close()