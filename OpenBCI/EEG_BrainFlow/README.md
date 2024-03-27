
# Information.

#### 1. [Brainflow Installation](https://brainflow.readthedocs.io/en/stable/BuildBrainFlow.html#python).

#### 2. [Supported Boards](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#supported-boards) - and parameters for configuration in code.

For Cyton board, the following initial configuration is enough:
```
params = BrainFlowInputParams()
params.serial_port = "COMx"
board = BoardShim(BoardIds.CYTON_BOARD, params)
```

#### 3. [Code Samples](https://brainflow.readthedocs.io/en/stable/Examples.html#python) of all BrainFlow bindings.

#### 4. [Brainflow API](https://brainflow.readthedocs.io/en/stable/UserAPI.html#).

