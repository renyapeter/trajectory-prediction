import numpy as np

def create_sequences(data, past=4, future=6):
    X, y = [], []
    for i in range(len(data) - past - future):
        X.append(data[i : i + past])
        y.append(data[i + past : i + past + future])
    return np.array(X), np.array(y)

def agent_centric_transform(X, y):
    X_new, y_new = [], []
    for i in range(len(X)):
        origin = X[i][-1]  # last observed point
        X_shifted = X[i] - origin
        y_shifted = y[i] - origin
        X_new.append(X_shifted)
        y_new.append(y_shifted)
    return np.array(X_new), np.array(y_new)

def add_velocity(X):
    X_new = []
    for seq in X:
        vel = np.diff(seq, axis=0, prepend=seq[0:1])
        new_seq = np.concatenate([seq, vel], axis=1)
        X_new.append(new_seq)
    return np.array(X_new)

def inverse_transform_predictions(predictions_ac_normalized, past_trajectories_normalized_non_ac, mean_original, std_original):
    predictions_world = []
    for i in range(len(predictions_ac_normalized)):
        origin = past_trajectories_normalized_non_ac[i][-1]
        shifted_predictions_normalized = predictions_ac_normalized[i] + origin
        denormalized_predictions_world = shifted_predictions_normalized * std_original + mean_original
        predictions_world.append(denormalized_predictions_world)
    return np.array(predictions_world)
