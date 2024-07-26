import playsound
import time
import random


def init():
    return lambda _ : _

def play(func, action):
    print(action, func)
    if action == "L":
        func("sound_files/left.mp3")
    elif action == "R":
        func("sound_files/right.mp3")
    elif action == "U":
        func("sound_files/up.mp3")
    elif action == "D":
        func("sound_files/down.mp3")
    time.sleep(1)

def close(func):
    return None
