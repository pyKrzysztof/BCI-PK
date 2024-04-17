from session import SessionEEG
from pipeline import PipelineEEG
from action import ActionEEG




ActionEEG.set_common_end_handle(lambda: print("STOP, and wait...\n\n"))
    
action1 = ActionEEG(lambda: print("LEFT"), 1.0, tActionHold=2)
action2 = ActionEEG(lambda: print("RIGHT"), 2.0, tActionHold=2)

pipe = PipelineEEG(1, (1, 2), [action1, action2])

session = SessionEEG(config=[], simulated=True)
session.process_pipeline(pipe)
session.export_data("my_data.csv")