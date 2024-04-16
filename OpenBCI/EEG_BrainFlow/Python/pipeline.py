import random as rd
import time
import numpy as np

from brainflow.board_shim import BoardShim



class PipelineEEG:

    def __init__(self, nSamples, tBetweenRange, actionHandles : list, markerValues : list, bias=None) -> None:
        self.nSamples = nSamples
        self.actionHandles = actionHandles
        self.nActions = len(actionHandles)
        self.markers = markerValues
        self.tb1 = tBetweenRange[0]
        self.tb2 = tBetweenRange[1]
        self.bias = [1,]*self.nActions if not bias else bias

        assert len(self.markers) == self.nActions

    def prepare(self):
        # prepare randomized actions with bias
        # create a handle generator, include marker values
        pass

    def start(self, boardHandle, timeout):
        start_time = time.time
        current_time = start_time

        for action, marker in self.generated_actions:
            PipelineEEG.insertMarker(boardHandle, marker)
            

        

    @staticmethod
    def insertMarker(boardHandle : BoardShim, value):
        boardHandle.insert_marker(value)