import serial
import time
import random

port = "/dev/ttyACM0"

# parameters returned by the initialization function will always be passed 
# as first argument of the on_action and on_close functions.
def init_serial():
    ser = serial.Serial(port, 9600)
    time.sleep(3)
    return ser

def send_action(ser, action: str):
    print(ser, action)
    ser.write((action + '\n').encode('utf-8'))
    time.sleep(random.uniform(4, 6))
    print(ser, action)
    ser.write((action + '\n').encode('utf-8'))
    time.sleep(1)

def close_serial(ser):
    print("Closing connection", ser)
    ser.close()

if __name__ == "__main__":
    ser = init_serial()
    send_action(ser, "L")
    send_action(ser, "R")
    close_serial(ser)