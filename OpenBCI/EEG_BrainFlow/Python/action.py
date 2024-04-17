from typing import Any, Callable



class ActionEEG:
    
    
    # handle should be a function handle without args, kwargs.
    # marker is the value inserted to the training data
    # actionWait is the interval (in seconds) since the handle function returned, after which the marker is placed in the data.
    # actionHold is the time (in seconds) for how long the data should be analysed after placing the marker. After this, the endHandle function is called, and the data is no longer processed once it returns.
    def __init__(self, handle: Callable, marker: float, tActionHold: float = 2.0, tActionWait: float = 1.0, endMarker: Any = None, endHandle: Callable = None) -> None:
        self.handle = handle
        self.marker = marker
        self.tHold = tActionHold
        self.tWait = tActionWait
        if not endMarker:
            self.endMarker = -marker
        if not endHandle:
            try:
                self.endHandle = ActionEEG.common_end_handle
            except:
                print("No end handle provided for an action, and a common end handle was not set. Defaulting to empty function.")
                self.endHandle = lambda: None
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        status = self.handle()
        
        return status

    def __str__(self) -> str:
        return f"ActionEEG({self.handle}, marker = {self.marker})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @staticmethod
    def set_common_end_handle(handle):
        ActionEEG.common_end_handle = handle