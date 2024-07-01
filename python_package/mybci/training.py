import numpy as np
from keras import models


def custom_evaluate(X, Y, model: models.Model, batch_size, step, return_indexed=False, no_overhead=False):
    # if no_overhead:
        # return custom_evaluate_no_overhead(X, Y, model)
    if isinstance(X, np.ndarray):
        X = [X]

    num_samples = len(Y)
    for start in range(0, num_samples - batch_size + 1, step):
        batch_X = [x[start:start + batch_size] for x in X]
        batch_Y = Y[start:start + batch_size]
        reshaped_batch_X = [input.reshape(-1, *input.shape[1:], 1) for input in batch_X]

        outputs = model([input for input in reshaped_batch_X]).numpy()

        if return_indexed:
            yield range(start, start + batch_size), batch_Y, outputs
        else:
            yield batch_Y, outputs
