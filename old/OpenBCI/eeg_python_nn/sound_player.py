import playsound
import time
import random


def init():
    return playsound.playsound

def play(func, action):
    print(action, func)
    if action == "L":
        func("left.mp3")
    elif action == "R":
        func("right.mp3")
    elif action == "U":
        func("up.mp3")
    elif action == "D":
        func("down.mp3")
    time.sleep(random.uniform(4, 6))

def close(func):
    return None
