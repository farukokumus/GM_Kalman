from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.gaussian_mixture_reduction import reduction_algorithm
from ime_fgs.messages import GaussianMixtureMeanCovMessage, GaussianMixtureWeightedMeanInfoMessage
import matplotlib.pyplot as plt
import numpy as np

from ime_fgs.plot import draw_graph

# State estimation with a Kalman filter, using the ime-fgs toolbox
# This is a simple example to introduce the toolbox, for more serious uses please take a look at the Kalman class
#
# Single time slice of a Kalman filter:
#
#
#               input_in_node
#                    +-+
#                    +-+
#                     |
#                     | U_k
#                     v       process_noise_in_node
#                   +---+           +---+
#                   | B | B_node    |   |
#                   +---+           +---+
#                     |               |
#                     |               | I_k
#                     v               v             equality_node
#  X_k +---+  X_k'' +---+   X'''_k  +---+     X_k+1     +---+ X'_k+1
# ---->| A |------->| + |---------->| + |-------------->| = |------->
#      +---+        +---+           +---+               +---+
#      A_node  add_input_node add_process_noise_node      |
#                                                         | X''_k+1
#                                                         v
#                                                       +---+
#                                                       | C | C_node
#                                                       +---+
#                                                         |
#                                                         | Y_k+1
#                                                         v
#                                           +---+ D_k+1 +---+
#                                           |   |------>| + | add_meas_noise_node
#                                           +---+       +---+
#                                     meas_noise_node     |
#                                                         | Z_k+1
#                                                         v
#                                                        +-+
#                                                        +-+
#                                                     meas_in_node
#


# System parameters
A = [[1, 0.1], [-0.01, 0.99]]  # adjust naming convention for the slice above
B = [[0.0], [0.01]]
C = [[1, 0]]  # only the first state was measured
meas_noise_cov = [[1e3]]  # the measurement noise
process_noise_cov = [[0, 0], [0, 0]]  # we assume no noise for the process

# read measurements of of the first state of the system
meas_list_with_noise = np.load("data_meas_list_with_noise.npy")
# read true states of the system (as reference)
state_list = np.load("data_state_list.npy")
# read time vector corresponding to measurements and states
t_vec = np.load("data_t_vec.npy")
# read system input
input_u = np.load("data_input_u.npy")

meas_in_list_msg = [GaussianMixtureMeanCovMessage([[1]], [[[m]]], [[[0]]]) for m in meas_list_with_noise]
input_u_list_msg = [GaussianMixtureMeanCovMessage([[1]], [[[u]]], [[[0]]]) for u in input_u]

# Create all relevant nodes for a single Kalman slice
# including an additional PriorNode for the state input and state output, which act as a terminator
# The name is optional, but may come handy if you are trying to debug your graph
state_in_node = PriorNode(name="x_in")
state_out_node = PriorNode(name="x_out")
A_node = MatrixNode(A, name="A")
B_node = MatrixNode(B, name="B")
C_node = MatrixNode(C, name="C")
meas_noise_node = PriorNode(GaussianMixtureMeanCovMessage([[1]], [[[0]]], [meas_noise_cov]), name="N_D")
equality_node = EqualityNode(name="=")

add_process_noise_node = AdditionNode(name="add_process_noise_node")
add_meas_noise_node = AdditionNode(name="add_meas_noise_node")
process_noise_in_node = PriorNode(
    GaussianMixtureMeanCovMessage([[1]], [[[0], [0]]], [process_noise_cov]), name="process_noise_in_node"
)
add_input_node = AdditionNode(name="add_input_node")
input_in_node = PriorNode(name="input_in_node")
meas_in_node = PriorNode(name="meas_in_node")

# Connect the nodes together with the .connect function
equality_node.ports[1].connect(C_node.port_a)
equality_node.ports[2].connect(state_out_node.port_a)

C_node.port_b.connect(add_meas_noise_node.port_a)

input_in_node.port_a.connect(B_node.port_a)
B_node.port_b.connect(add_input_node.port_b)

state_in_node.port_a.connect(A_node.port_a)
A_node.port_b.connect(add_input_node.port_a)

add_input_node.port_c.connect(add_process_noise_node.port_a)

process_noise_in_node.port_a.connect(add_process_noise_node.port_b)
add_process_noise_node.port_c.connect(equality_node.ports[0])

meas_noise_node.port_a.connect(add_meas_noise_node.port_b)
add_meas_noise_node.port_c.connect(meas_in_node.port_a)
# You can uncomment the next line to get a simple plot of the graph
draw_graph(state_in_node)

# set a (wrong) start state with high variances (low confidence)
start_state = GaussianMixtureMeanCovMessage([[1]], [[[100], [20]]], [[[99, 0], [0, 99]]])

# Create a list of estimated messages by updating all ports for the Kalman filter for every time step
estimated_state_list = [start_state]
for idx, t in enumerate(t_vec):
    # use last state estimation for new state estimation
    state_in_node.update_prior(estimated_state_list[-1])
    input_in_node.update_prior(input_u_list_msg[idx])
    meas_in_node.update_prior(meas_in_list_msg[idx])

    A_node.port_b.update(GaussianMixtureMeanCovMessage)
    B_node.port_b.update(GaussianMixtureMeanCovMessage)
    add_input_node.port_c.update(GaussianMixtureMeanCovMessage)
    add_process_noise_node.port_c.update(GaussianMixtureWeightedMeanInfoMessage)
    add_meas_noise_node.port_a.update(GaussianMixtureWeightedMeanInfoMessage)
    C_node.port_a.update(GaussianMixtureWeightedMeanInfoMessage)
    # append new estimated state to list
    estimated_state_list.append(reduction_algorithm(equality_node.ports[2].update(GaussianMixtureMeanCovMessage), 3))

# Plot result
# Extract results
estimated_position_list = [x.mean[0,0] for x in estimated_state_list]
estimated_speed_list = [x.mean[0,1] for x in estimated_state_list]

# Use two subplots to plot the position and the speed of the mass
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t_vec, meas_list_with_noise, color="C0", label="simulated measurement")
ax1.plot(t_vec, state_list[0, :], color="C2", label="true state")
ax1.plot(t_vec, estimated_position_list[:-1], color="C1", label="estimated state")
ax1.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(t_vec, state_list[1, :], color="C2", label="true state")
ax2.plot(t_vec, estimated_speed_list[:-1], color="C1", label="estimated state")
ax2.legend()

plt.suptitle("Mass Spring Damper System")
ax1.set(ylabel="position p in m")
ax2.set(xlabel="time in s", ylabel="speed v in m/s")

ax1.grid()
ax2.grid()

plt.show()
