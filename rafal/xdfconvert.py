import os

def convert():
    import mybci.xdfconversion
    for filename in [os.path.join("session/", filename) for filename in os.listdir("session/") if filename.endswith(".xdf")]:
        for idx in [0, 1, 2]:
            try:
                mybci.xdfconversion.xdf_to_csv(filename, idx, 3, {"L": 1, "R": 2, "S": 3}, timestamp_tolerance=0.01, output_file=filename.replace(".xdf", f"_{idx}.csv"))
            except:
                pass


def process():
    from preprocess import prep1, filter1
    import mybci

    config = mybci.get_base_config()

    config['name'] = "rafal_1"
    config['session_file'] = [os.path.join("session/", filename) for filename in os.listdir("session/") if filename.endswith(".csv")]
    config['channel_column_ids'] = [0, 1, 2, 3, 4, 5, 6, 7]
    config['action_markers'] = [1, 2, 3]
    config['filter_func'] = {"f1": filter1}
    config['ml_prepare_func'] = {'ml1': prep1}
    config['ml_prepare_chunk_start_offset'] = 100
    config['packet_size'] = 128
    config['filter_size'] = 128
    config['ml_prepare_size'] = 128
    config['save_training_dataset'] = True
    config['keep_seperate_training_data'] = False
    config['output_path_training_dataset'] = "data/datasets/"

    processor = mybci.DataProcessor(config)
    processor.process()


# convert()
process()