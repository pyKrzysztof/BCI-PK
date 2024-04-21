import random as rd
import time
from brainflow.board_shim import BoardShim

from eeglib.action import ActionEEG



class PipelineEEG:

    def __init__(self, nSamples, tBetweenRange, actionHandles : list[ActionEEG]) -> None:
        self.nSamples = nSamples
        self.actionHandles = actionHandles
        self.nActions = len(actionHandles)
        self.tb1 = tBetweenRange[0]
        self.tb2 = tBetweenRange[1]

    # this function should be called manually if not used together with SessionEEG
    def prepare(self):
        # prepare randomized actions and delays, return estimated session time and the action sequence.
        delays = [rd.uniform(self.tb1, self.tb2) for _ in range(self.nSamples*self.nActions)]
        actionMarkerPairs = [action for action in self.actionHandles for _ in range(self.nSamples)]
        rd.shuffle(actionMarkerPairs)
        self.action_sequence = list(zip(actionMarkerPairs, delays))
        approx_time = sum([action.tHold + action.tWait for action in self.actionHandles]) + sum(delays)
        print(f"Expected session time: {approx_time:.2f} seconds.")
        return approx_time, self.action_sequence


    def start(self, boardHandle, timeout):
        for i, (action, delay) in enumerate(self.action_sequence):
            action()
            time.sleep(action.tWait)
            PipelineEEG.insertMarker(boardHandle, action.marker)
            time.sleep(action.tHold)
            PipelineEEG.insertMarker(boardHandle, action.endMarker)
            action.endHandle()
            
            if i == len(self.action_sequence)-1:
                time.sleep(0.25)
                continue
            time.sleep(delay)
            
        return None

    @staticmethod
    def insertMarker(boardHandle: BoardShim, value):
        boardHandle.insert_marker(value)
