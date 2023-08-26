from re import S
from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.gaussian_mixture_reduction import reduction_algorithm
from ime_fgs.messages import GaussianMixtureMeanCovMessage, GaussianMixtureWeightedMeanInfoMessage
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import glob

# State estimation with a Kalman filter, using the ime-fgs toolbox
# This is a simple example to introduce the toolbox, for more serious uses please take a look at the Kalman class
#
# Single time slice of a Kalman filter:
#
#
#               process_noise_in_node
#                     +---+
#                     |   |
#                     +---+
#                       |
#                       | I_k
#                       v             equality_node
#  X_k +---+  X''_k   +---+     X_k+1     +---+ X'_k+1
# ---->| A |--------->| + |-------------->| = |------->
#      +---+          +---+               +---+
#      A_node   add_process_noise_node      |
#                                           | X''_k+1
#                                           v
#                                         +---+
#                                         | C | C_node
#                                         +---+
#                                           |
#                                           | Y_k+1
#                                           v
#                             +---+ D_k+1 +---+
#                             |   |------>| + | add_meas_noise_node
#                             +---+       +---+
#                       meas_noise_node     |
#                                           | Z_k+1
#                                           v
#                                          +-+
#                                          +-+
#                                       meas_in_node
#



def load_from_file(f:str) -> GaussianMixtureMeanCovMessage:
    msg_data = np.load(f)
    return GaussianMixtureMeanCovMessage(msg_data["weights"], msg_data["mean"], msg_data["cov"])

# System parameters
A = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]  # adjust naming convention for the slice above
C = [[1, 0, 0, 0],[0, 0, 1, 0]]  # only the first state was measured
meas_noise_cov = [[0, 0], [0, 0]]  # the measurement noise
process_noise_cov = [[0.0001, 0.001, 0, 0], [0, 0.001, 0, 0], [0, 0, 0.0001, 0.001], [0, 0, 0, 0.001]]  # we assume no noise for the process

true_states = []
measurements = []
meas_in_list_msg = []
files = glob.glob("map_measurements/*.npz")
for i in range(len(files)):
    meas_in_list_msg += [load_from_file(f"map_measurements/{i}.npz")]#[GaussianMixtureMeanCovMessage([[1]], [measurements[i,:].reshape(2,1)], [[[0,0],[0,0]]]) ]
    true_states += [load_from_file(f"true_measurements/{i}.npz").mean.reshape(2)]#[GaussianMixtureMeanCovMessage([[1]], [measurements[i,:].reshape(2,1)], [[[0,0],[0,0]]]) ]
    measurements += [reduction_algorithm(meas_in_list_msg[-1],1).mean.reshape(2)]

true_states = np.array(true_states)
measurements = np.array(measurements)
# Create all relevant nodes for a single Kalman slice
# including an additional PriorNode for the state input and state output, which act as a terminator
# The name is optional, but may come handy if you are trying to debug your graph
state_in_node = PriorNode(name="x_in")
state_out_node = PriorNode(name="x_out")
A_node = MatrixNode(A, name="A")
C_node = MatrixNode(C, name="C")
meas_noise_node = PriorNode(GaussianMixtureMeanCovMessage([[1]], [[[0],[0]]], [meas_noise_cov]), name="N_D")
equality_node = EqualityNode(name="=")

add_process_noise_node = AdditionNode(name="add_process_noise_node")
add_meas_noise_node = AdditionNode(name="add_meas_noise_node")
process_noise_in_node = PriorNode(
    GaussianMixtureMeanCovMessage([[1]], [[[0], [0], [0], [0]]], [process_noise_cov]), name="process_noise_in_node"
)
meas_in_node = PriorNode(name="meas_in_node")

# Connect the nodes together with the .connect function
equality_node.ports[1].connect(C_node.port_a)
equality_node.ports[2].connect(state_out_node.port_a)

C_node.port_b.connect(add_meas_noise_node.port_a)


state_in_node.port_a.connect(A_node.port_a)
A_node.port_b.connect(add_process_noise_node.port_a)

process_noise_in_node.port_a.connect(add_process_noise_node.port_b)
add_process_noise_node.port_c.connect(equality_node.ports[0])

meas_noise_node.port_a.connect(add_meas_noise_node.port_b)
add_meas_noise_node.port_c.connect(meas_in_node.port_a)
# set a (wrong) start state with high variances (low confidence)
start_state = GaussianMixtureMeanCovMessage([[1]], [[[0], [1.41], [1], [1.41]]], [[[99, 99, 0, 0], [0, 99, 0, 0], [0, 0, 99, 99], [0, 0, 0, 99]]])

# Create a list of estimated messages by updating all ports for the Kalman filter for every time step
estimated_state_list = [start_state]
for meas in tqdm.tqdm(meas_in_list_msg):
    # use last state estimation for new state estimation
    state_in_node.update_prior(estimated_state_list[-1])
    meas_in_node.update_prior(meas)

    A_node.port_b.update(GaussianMixtureMeanCovMessage)
    add_process_noise_node.port_c.update(GaussianMixtureWeightedMeanInfoMessage)
    add_meas_noise_node.port_a.update(GaussianMixtureWeightedMeanInfoMessage)
    C_node.port_a.update(GaussianMixtureWeightedMeanInfoMessage)
    # append new estimated state to list
    estimated_state_list.append(reduction_algorithm(equality_node.ports[2].update(GaussianMixtureMeanCovMessage), 3))

# Plot result
# Extract results
estimated_position_list = [reduction_algorithm(x, 2).mean[0, (0, 2)] for x in estimated_state_list]
estimated_speed_list = [reduction_algorithm(x, 2).mean[0,(1, 3)] for x in estimated_state_list]
estimated_positions = np.array(estimated_position_list)
estimated_speed = np.array(estimated_speed_list)

# Use two subplots to plot the position and the speed of the mass

street_map = plt.imread("street_map.jpg")
street_map = street_map[30:105,25:100,0] > 10

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(street_map)
ax1.set_aspect(aspect="equal")
ax1.scatter(measurements[:, 1], measurements[:, 0], label="simulated measurement")
ax1.scatter(true_states[:, 1], true_states[:, 0], label="true state")
ax1.scatter(estimated_positions[:, 1], estimated_positions[:, 0], label="estimated state")
ax1.legend()

ax2 = plt.subplot(1, 2, 2, sharex=ax1)
ax2.plot(estimated_speed[:-1, 0], label="estimated velocity X")
ax2.plot(estimated_speed[:-1, 1], label="estimated velocity Y")
ax2.legend()

ax1.set(xlabel="position p in m", ylabel="position p in m")
ax2.set(xlabel="time in seconds", ylabel="velocity in m/s")

ax1.grid()
ax2.grid()

plt.show()
