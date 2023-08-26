from genericpath import exists
from ime_fgs import messages
from ime_fgs.gaussian_mixture_reduction import reduction_algorithm
import numpy as np
from matplotlib import pyplot as plt
from ime_fgs.messages import GaussianMixtureMeanCovMessage, GaussianMixtureWeightedMeanInfoMessage
from numpy import linalg
from scipy.stats import multivariate_normal
import os
import tqdm
import shutil

def calc_prob_dens_2d(msg: GaussianMixtureMeanCovMessage, bounds: tuple=(0, 20)):

    x, y = np.mgrid[bounds[0]:bounds[1]:1, bounds[0]:bounds[1]:1]
    pos = np.dstack((x, y))


    p = np.zeros(pos.shape[:2])

    for k in range(msg.weights.shape[0]):
        gaussian = multivariate_normal(msg.mean[k, :, :].reshape(2),msg.cov[k, :, :])
        p += msg.weights[k] * gaussian.pdf(pos)
    return x, p


def save_to_file(f:str, msg:GaussianMixtureMeanCovMessage):
    np.savez(f, weights=msg.weights, mean=msg.mean, cov=msg.cov)


def load_from_file(f:str) -> GaussianMixtureMeanCovMessage:
    msg_data = np.load(f)
    return GaussianMixtureMeanCovMessage(msg_data["weights"], msg_data["mean"], msg_data["cov"])

street_map = plt.imread("street_map.jpg")

street_map = street_map[30:105,25:100,0] > 10

i = 0
num_gaussians = np.count_nonzero(street_map)
weights = []
means = []
cov = []
for x in range(street_map.shape[0]):
    for y in range(street_map.shape[1]):
        if street_map[x, y]:
            means += [[[x], [y]]]
            cov += [[[0.5,0],[0,0.5]]]

weights = np.full((len(means),1), 1/len(means))

map = GaussianMixtureMeanCovMessage(weights, means, cov)

shutil.rmtree('raw_measurements', ignore_errors=True)
shutil.rmtree('map_measurements', ignore_errors=True)
shutil.rmtree('true_measurements', ignore_errors=True)

os.makedirs("raw_measurements", exist_ok=True)
os.makedirs("map_measurements", exist_ok=True)
os.makedirs("true_measurements", exist_ok=True)


meas_x = np.linspace(0,75, 75)
meas_y = 2 + 0.32*meas_x
meas_y[16:] = -15 + 1.35*meas_x[16:]

speed = 2
meas_x = [0]
meas_y = [1]
i = 0
direction = 0.35* np.pi/4
while(meas_x[-1] < street_map.shape[0] and meas_y[-1] < street_map.shape[0]):
    meas_x += [meas_x[-1] + speed * np.cos(direction)]
    meas_y += [meas_y[-1] + speed * np.sin(direction)]
    if i == 6:
        direction = 1.2 * np.pi/4

    i += 1

meas_x = np.array(meas_x)
meas_y = np.array(meas_y)

#_, p_map = calc_prob_dens_2d(map, bounds=(0, street_map.shape[0]))
#plt.imshow(p_map)
#plt.scatter(meas_x, meas_y)
#plt.show()
#exit()

print("Save true values")
for i in range(meas_x.shape[0]):
    #if meas_y[i] > street_map.shape[0]:
    #    break
    #if meas_x[i] > street_map.shape[0]:
    #    break
    pos = GaussianMixtureMeanCovMessage([[1]], [[[meas_y[i]],[meas_x[i]]]], [[[3, 0],[0, 3]]])
    save_to_file(f"true_measurements/{i}", pos)

meas_x += np.random.normal(0,3, meas_x.shape)
meas_y += np.random.normal(0,3, meas_y.shape)

print("Save measurement values")
for i in range(meas_x.shape[0]):
    #if meas_y[i] > street_map.shape[0]:
    #    break
    #if meas_x[i] > street_map.shape[0]:
    #    break
    pos = GaussianMixtureMeanCovMessage([[1]], [[[meas_y[i]],[meas_x[i]]]], [[[3, 0],[0, 3]]])
    save_to_file(f"raw_measurements/{i}", pos)
measurements = []


print("Save mapped measurement values")
for i in tqdm.tqdm(range(meas_x.shape[0])):
    #if meas_y[i] > street_map.shape[0]:
    #    break
    #if meas_x[i] > street_map.shape[0]:
    #    break

    pos = GaussianMixtureMeanCovMessage([[1]], [[[meas_y[i]],[meas_x[i]]]], [[[3, 0],[0, 3]]])

    measurement = GaussianMixtureMeanCovMessage(weights, means, cov)
    pos_gaussian = multivariate_normal(pos.mean[0, :, :].reshape(2), pos.cov[0, :, :].reshape((2,2)))

    new_weights = pos_gaussian.pdf(measurement.mean.reshape(measurement.mean.shape[:2]))

    new_weights = new_weights.reshape((*new_weights.shape, 1))
    new_weights /= np.sum(new_weights)
    measurement.weights = new_weights

    measurement = reduction_algorithm(measurement, 20)
    save_to_file(f"map_measurements/{i}", measurement)
    measurements += [measurement]

pos = GaussianMixtureMeanCovMessage([[1]], [[[meas_y[25]],[meas_x[25]]]], [[[3, 0],[0, 3]]])

x_map, p_map = calc_prob_dens_2d(map, bounds=(0, street_map.shape[0]))
x_pos, p_pos = calc_prob_dens_2d(pos, bounds=(0, street_map.shape[0]))
p_measurements = np.zeros(p_map.shape)

for i in range(50):
    _, p_measurement = calc_prob_dens_2d(measurements[i], bounds=(0, street_map.shape[0]))
    p_measurements += p_measurement

measurement = reduction_algorithm(measurements[-1],5)

plt.subplot(2,2,1)
plt.imshow(p_map)
plt.scatter(meas_x, meas_y)
plt.subplot(2,2,2)
plt.imshow(p_pos)
plt.subplot(2,2,3)
plt.imshow(p_measurements)
plt.show()