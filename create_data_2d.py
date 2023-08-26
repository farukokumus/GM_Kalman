import matplotlib.pyplot as plt
import numpy as np

pos_x = np.linspace(0, 20, 100)
pos_y = 5 + 1.5 * pos_x
pos_y[25:] = -pos_y[25:] + pos_y[24] + pos_y[25]
speed_x = []
speed_y = []

for i in range(1, len(pos_x)):
    speed_x += [pos_x[i] - pos_x[i-1]]
    speed_y += [pos_y[i] - pos_y[i-1]]
speed_x += [speed_x[-1]]
speed_y += [speed_y[-1]]

true_state = np.zeros((pos_x.shape[0], 4))
true_state[:, 0] = pos_x
true_state[:, 1] = speed_x
true_state[:, 2] = pos_y
true_state[:, 3] = speed_y

measurements = np.zeros((pos_x.shape[0], 2))
measurements[:, 0] = pos_x + np.random.normal(0, 0.5, pos_x.shape[0])
measurements[:, 1] = pos_y + np.random.normal(0, 0.5, pos_x.shape[0])

np.save("true_states_2d.npy", true_state)
np.save("t_vec_2d.npy", pos_x)
np.save("measurements_2d.npy", measurements)


print(true_state)