import mybci




# params
epochs = 11
batch_size = 32
passes = 10

# dataset directories by filter/process functions.
datasets_path = "data/datasets/processed_data/"
datasets_directories = [f"f{i}_p1" for i in [1, 2, 3, 4]]

# models for each dataset directory
my_models = {name: None for name in datasets_directories}

