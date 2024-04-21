from eeglib import ActionEEG, SessionEEG, PipelineEEG


# common function handle executed after finishing one of the actions.
ActionEEG.set_common_end_handle(lambda: print("STOP, and wait...\n\n"))

# actions to classify.
action1 = ActionEEG(lambda: print("LEFT"), 1.0, tActionHold=2)
action2 = ActionEEG(lambda: print("RIGHT"), 2.0, tActionHold=2)

# pipeline passed to a session object or started manually via pipe.start(boardHandle, timeout)
pipeline1 = PipelineEEG(4, (1, 2), [action1, action2])

# to work with a real board, omit 'simulated' and set correct configuration for the Cyton electrodes 
# (see SessionEEG configurations or create custom command chains according to https://docs.openbci.com/Cyton/CytonSDK/#channel-setting-commands)
session = SessionEEG(simulated=True)
session.process_pipeline(pipeline1)
session.export_data("my_data.csv")

