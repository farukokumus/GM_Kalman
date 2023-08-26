# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from ime_fgs.base import NodePort
from numpy import asarray, atleast_2d, linspace, inf, identity, transpose, kron
from numpy.random import normal, seed

import time
import matplotlib.pyplot as plt

from ime_fgs.basic_nodes import *
from ime_fgs.compound_nodes import *
from ime_fgs.messages import *
from ime_fgs.em_node import *

import numpy as np

"""
EM-Matrix-Node a lá Loeliger
"""


class EMMatrixSlice(Node):
    def __init__(self, A_0, A_theta, theta, B, C, matrix_parameterization='full', name=None):
        super().__init__(name=name)
        # init external ports of this slice
        self.port_state_in = NodePort(self, self.calc_msg_state_in)
        self.port_state_out = NodePort(self, self.calc_msg_state_out)
        self.port_process_noise = NodePort(self, self.calc_msg_process_noise)
        self.port_input = NodePort(self, self.calc_msg_input)
        self.port_meas = NodePort(self, self.calc_msg_meas)

        # init PriorNodes as interfaces
        self.state_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.state_out_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.process_noise_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.input_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.meas_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))

        # init remaining SSM-nodes
        # 1. matrices
        self.A_node = EmMatrixNodeAffineParametrized(A_0, A_theta, theta, matrix_parameterization)
        self.B_node = MatrixNode(B)
        self.C_node = MatrixNode(C)
        # 2. connectors
        self.add_process_noise_node = AdditionNode()
        self.add_input_node = AdditionNode()
        self.equality_node = EqualityNode()
        # 3. port_theta connector
        self.theta_connector = EqualityNode()
        # connect the model starting from state_in
        self.state_in_node.port_a.connect(self.A_node.port_a)
        self.A_node.port_theta.connect(self.theta_connector.ports[0])
        self.A_node.port_b.connect(self.add_process_noise_node.port_a)
        self.process_noise_node.port_a.connect(self.add_process_noise_node.port_b)
        self.add_process_noise_node.port_c.connect(self.add_input_node.port_a)
        self.input_node.port_a.connect(self.B_node.port_a)
        self.B_node.port_b.connect(self.add_input_node.port_b)
        self.add_input_node.port_c.connect(self.equality_node.ports[0])
        self.meas_node.port_a.connect(self.C_node.port_b)
        self.C_node.port_a.connect(self.equality_node.ports[1])
        self.equality_node.ports[2].connect(self.state_out_node.port_a)

    def calc_msg_state_in(self):
        # 1. update all the priors
        self.process_noise_node.update_prior(self.port_process_noise.in_msg)
        self.input_node.update_prior(self.port_input.in_msg)
        self.meas_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        self.state_out_node.update_prior(self.port_state_out.in_msg, GaussianWeightedMeanInfoMessage)
        # 2. reverse through the branches to get to state_in
        self.C_node.port_a.update()
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.B_node.port_b.update()
        self.add_input_node.port_a.update()
        self.add_process_noise_node.port_a.update(GaussianWeightedMeanInfoMessage)
        return self.A_node.port_a.update()

    def calc_msg_state_out(self):
        # 1. update all the priors
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.process_noise_node.update_prior(self.port_process_noise.in_msg)
        self.input_node.update_prior(self.port_input.in_msg)
        self.meas_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        # 2. forward through the branches to get to state_out
        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update()
        self.B_node.port_b.update()
        self.add_input_node.port_c.update(GaussianWeightedMeanInfoMessage)
        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        return self.equality_node.ports[2].update(GaussianMeanCovMessage)

    def calc_msg_process_noise(self):
        # 1. update all the priors
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.input_node.update_prior(self.port_input.in_msg)
        self.meas_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        # 2. iterate through branches
        self.A_node.port_b.update()
        self.C_node.port_b.update()
        self.equality_node.ports[0].update()
        self.B_node.port_b.update()
        self.add_input_node.port_a.update()
        return self.add_process_noise_node.port_b.update()

    def calc_msg_input(self):
        # 1. update all the priors
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.process_noise_node.update_prior(self.port_process_noise.in_msg)
        self.meas_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        # 2. iterate through branches
        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update()
        self.C_node.port_b.update()
        self.equality_node.ports[0].update()
        self.add_input_node.port_b.update()
        self.B_node.port_a.update()
        return self.B_node.port_a.out_msg

    def calc_msg_meas(self):
        # 1. update all the priors
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.process_noise_node.update_prior(self.port_process_noise.in_msg)
        self.input_node.update_prior(self.port_input.in_msg)
        # self.meas_node.update_prior(self.port_meas.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        # 2. iterate through branches
        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update()
        self.B_node.port_b.update()
        self.add_input_node.port_c.update()
        self.C_node.port_a.update()
        return self.C_node.port_a.out_msg

    # after smoothing over the Filter calculate updated theta_msg for this slice
    def e_step(self):
        self.A_node.port_theta.update()

    # update the matrix in A(theta) according to updated values
    def m_step(self):
        self.A_node.update_theta()

    def eq_forward(self):
        self.theta_connector.ports[2].update()

    def eq_backward(self):
        self.theta_connector.ports[0].update()
        self.theta_connector.ports[1].update()

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas, self.port_input, self.port_process_noise]


class EMMatrixFilter(object):
    def __init__(self, A_0, A_theta, theta, B, C, initial_state_msg, input_noise_cov=None, process_noise_cov=None,
                 meas_noise_cov=None,
                 slice_type=EMMatrixSlice,
                 matrix_parameterization='full'):
        """
        Initialize a Kalman filter object given an initial state.

        :param A: Initial system matrix to be used as the default in all time slices.
        :param B: Input matrix to be used as the default in all time slices.
        :param C: Output matrix to be used as the default in all time slices.
        :param uncertainty_variance: 'uncertainty' for EM-Node with mean = 0, but given variance
        :param initial_state_msg: Message specifying information about the initial system state.
        :param input_noise_cov: Input noise covariance to be used as the default in all time slices. If not provided,
          input noise messages must be provided in each time slice.
        :param process_noise_cov: Process noise covariance to be used as the default in all time slices. If not
          provided, process noise messages must be provided in each time slice.
        :param meas_noise_cov: Measurement noise covariance to be used as the default in all time slices. If not
          provided, measurement noise messages must be provided in each time slice.
        :param slice_type: The Kalman filter time slice model to be used.
        """
        # Store default parameter values
        self.A_0 = A_0
        self.A_theta = A_theta
        self.theta = theta
        self.B = B
        self.C = C
        self.slice_type = slice_type
        self.input_noise_cov = input_noise_cov
        self.process_noise_cov = process_noise_cov
        self.meas_noise_cov = meas_noise_cov
        self.matrix_parameterization = matrix_parameterization

        # check for right dimensions
        self.rows = shape(A_0)[0]
        self.cols = shape(A_0)[1]
        if self.cols != shape(initial_state_msg.convert(GaussianMeanCovMessage).mean)[0]:
            raise Exception("Dimensions of initial state and matrix A do not match!")

        # Initialize factor graph
        self.slices = []
        self.initial_state = PriorNode(initial_state_msg)
        self.initial_state_msg = initial_state_msg
        final_mean = [[0], [0]]
        final_cov = [[inf, 0], [0, inf]]
        self.final_state_msg = GaussianMeanCovMessage(final_mean, final_cov).convert(GaussianWeightedMeanInfoMessage)

    def add_slice(self, input_val, meas_val, A_k=None, A_theta_k=None, B_k=None, C_k=None, process_noise_msg=None):
        if A_k is None:
            A_k = self.A_0
        if A_theta_k is None:
            A_theta_k = self.A_theta
        if B_k is None:
            B_k = self.B
        if C_k is None:
            C_k = self.C
        if process_noise_msg is None:
            process_noise_msg = GaussianMeanCovMessage(np.zeros((2, 1)), np.identity(2) * self.process_noise_cov)

        self.eq_start = PriorNode(GaussianWeightedMeanInfoMessage(
            np.zeros((len(self.theta), 1)), np.zeros((len(self.theta), len(self.theta)))))
        self.eq_end = PriorNode(GaussianWeightedMeanInfoMessage(
            np.zeros((len(self.theta), 1)), np.zeros((len(self.theta), len(self.theta)))))

        # construct a new slice
        new_slice = self.slice_type(A_k, A_theta_k, theta, B_k, C_k,
                                    matrix_parameterization=self.matrix_parameterization)
        # add input, measurements and process noise
        new_slice.port_meas.connect(PriorNode(GaussianMeanCovMessage(
            meas_val, [[0.001, 0], [0, 0.001]])).port_a)
        new_slice.port_input.connect(PriorNode(GaussianMeanCovMessage(input_val, [[1e-10]])).port_a)
        new_slice.port_process_noise.connect(
            PriorNode(process_noise_msg).port_a)
        # if it is the first slice - set an initial state - else connect to preceding slice
        if len(self.slices) == 0:
            self.initial_state.port_a.connect(new_slice.port_state_in)
            self.eq_start.port_a.connect(new_slice.theta_connector.ports[1])
        else:
            self.slices[-1].theta_connector.ports[2].disconnect()
            self.slices[-1].theta_connector.ports[2].connect(new_slice.theta_connector.ports[1])
            self.slices[-1].port_state_out.disconnect()
            self.slices[-1].port_state_out.connect(new_slice.port_state_in)
        # connect the 'terminator' node to this slice in case it is the last one
        self.final_node = PriorNode(self.final_state_msg, name="final_node")
        new_slice.port_state_out.connect(self.final_node.port_a)

        new_slice.theta_connector.ports[2].connect(self.eq_end.port_a)

        # add newly created slice to our list
        self.slices.append(new_slice)

        # there is no return

    # get out_msg of each slice in the filter
    def get_state_out_msgs(self):
        return [slice.port_state_out.out_msg for slice in self.slices]

    # iterate from left to right and calculate out_msg of each slice
    def do_forward(self):
        for slice in self.slices:
            slice.port_state_out.update()
            # self.final_node.update_prior(self.final_node.port_a.in_msg, GaussianWeightedMeanInfoMessage)
            # update the final_node with the outgoing msg (right) of the last slice for backward iteration
            # self.final_node.update_prior(self.slices[-1].port_state_out.out_msg)

    # iterate from right to left and calculate out_msg of each slice
    def do_backward(self):
        for slice in reversed(self.slices):
            slice.port_state_in.update()
            # self.initial_state.update_prior(self.initial_state.port_a.in_msg, GaussianMeanCovMessage)
            # update the initial_state with the outgoing msg (left) of the first slice for next forward iteration
            # self.initial_state.update_prior(self.slices[0].port_state_in.out_msg.convert(GaussianMeanCovMessage))

    # after smoothing over the filter is done - do expectation maximisation
    def do_update(self):
        slice_counter = 0
        # 1. calculate theta_msg for each slice
        for slice in self.slices:
            slice.e_step()
        # 2. if there is an equality-cycle --> perform smoothing over it
        for slice in self.slices:
            slice.eq_forward()
        for slice in reversed(self.slices):
            slice.eq_backward()
        # 3. update A(theta) matrices in each slice
        for slice in self.slices:
            slice.m_step()

    def get_state_msgs(self):
        return [slice.port_state_out.out_message() for slice in self.slices]

    def get_A_value(self):
        return [slice.get_matrix() for slice in self.slices]


if __name__ == '__main__':

    """
        Test for EM-Matrix-Node

        Matlab generated linear state space model with parameters:

        A = [0.75, 0.1; 0, 0.5]
        B = [1; 2]
        C = [1, 0; 0, 1]
        D = 0

        input u is constantly 1

        matrix to be estimated is A
    """

    measurements1 = []
    file_meas = open("meas.txt", "r")
    for line in file_meas:
        measurements1.append(float(line))

    measurements2 = []
    file_meas2 = open("meas2.txt", "r")
    for line in file_meas2:
        measurements2.append(float(line))

    input1 = np.ones(99)

    # Define system model
    B = [[1], [2]]  # B, C-matrix are the correct System matrices
    C = [[1, 0], [0, 1]]
    # aka slice-count
    num_meas = len(input1)

    # init A_0, A_theta, theta
    A_0 = np.array([[0, 0], [0, 0]])
    A_theta = np.array([[1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]])
    theta = transpose(np.array([[1, 2, 4, 4]]))

    # initialize the system
    initial_mean = [[0], [0]]
    initial_cov = [[1e-1, 0], [0, 1e-1]]
    process_noise_cov = 1e-2

    initial_state = GaussianMeanCovMessage(initial_mean, initial_cov)

    # switch between Loelliger EMMatrixNode and alternative EMMatrixNode
    embase = EMMatrixFilter(A_0, A_theta, theta, B, C, initial_state, matrix_parameterization='full', input_noise_cov=0,
                            process_noise_cov=process_noise_cov, meas_noise_cov=0)

    for ii in range(num_meas):
        embase.add_slice([[input1[ii]]], [[measurements1[ii + 1]], [measurements2[ii + 1]]])

    oldmatrix = 0
    newmatrix = 1
    counter = 0

    while not np.allclose(oldmatrix, newmatrix):
        oldmatrix = embase.slices[0].A_node.get_matrix()
        counter = counter + 1
        embase.do_forward()
        embase.do_backward()
        embase.do_update()
        newmatrix = embase.slices[0].A_node.get_matrix()
        print("updated matrix:", embase.slices[0].A_node.get_matrix())
        print("finished loop #:" + str(counter))

    print("EM converged on matrix: ", embase.slices[0].A_node.get_matrix())
    print("Theta:", "\n", embase.slices[0].A_node.port_theta.in_msg.convert(GaussianMeanCovMessage).mean)
