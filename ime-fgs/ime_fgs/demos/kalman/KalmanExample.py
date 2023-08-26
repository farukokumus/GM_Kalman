# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Peterson, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#

import time
import copy
import gc
from numpy import atleast_2d, linspace, zeros, zeros_like, concatenate, ndim, nan, abs, float32, shape, squeeze, sqrt
from numpy.random import seed, multivariate_normal
from collections import namedtuple
from inspect import signature

from ime_fgs.messages import GaussianMeanCovMessage
from ime_fgs.utils import col_vec, row_vec

SimResult = namedtuple('SimResult', ['inputs', 'inputs_disturbed', 'input_noise', 'states', 'process_noise',
                                     'measurements', 'measurements_disturbed', 'measurement_noise', 'time'])


def simulate_discrete_system(A, B, C, initial_state, inputs, input_noise_cov, process_noise_cov, meas_noise_cov, n_meas,
                             dT=1, AofXandU=False, seed_val=1):
    '''
    Simulates a discrete state space system
    :param A: system matrix or system function. The potentially non linear system function can either only depend on the
              state x or also on the input u. In the later case you have to set AofXandU=True and B will be ignored.
    :param B: input matrix
    :param C: output matrix
    :param initial_state: initial state of the system
    :param inputs: The system input can be defined in different ways. If a single input vector is passed it is used for
                   all time steps. Alternatively a callable (e.g. function) which returns an input vector for a given
                   time can be passed.
    :param input_noise_cov: input noise covariance matrix
    :param process_noise_cov: process noise convariance matrix
    :param meas_noise_cov: measurement noise covariance matrix
    :param n_meas: number of simulation steps
    :param dT: time step size
    :param AofXandU: set to True if A is a fuction of the state x AND the input u
    :param seed_val: seed to use for the noise sampling
    :return: the simulation result
    '''
    C = atleast_2d(C)
    if not callable(A):
        B = atleast_2d(B)

        def A_fun(state, input):
            return atleast_2d(A) @ state + B @ input
    else:
        if AofXandU is True:
            A_fun = A
        else:
            B = atleast_2d(B)

            def A_fun(state, input):
                return A(state) + B @ input

    initial_state = atleast_2d(initial_state)
    input_noise_cov = atleast_2d(input_noise_cov)
    process_noise_cov = atleast_2d(process_noise_cov)
    meas_noise_cov = atleast_2d(meas_noise_cov)

    time = linspace(0, n_meas * dT, n_meas + 1)  # 0 to n_meas*dT with n_meas+1 steps
    n_inputs = B.shape[1]
    M_C, N_C = C.shape
    n_states = shape(initial_state)[0]

    if callable(inputs):
        # create input matrix with column vectors from input function for each time step
        inputs = concatenate([col_vec(inputs(instant)) for instant in time], axis=1)
    else:
        # blow up input vector to create a matrix of column vectors for each time step
        if ndim(inputs) <= 1:
            inputs = col_vec(inputs)
        inputs_adjusted = zeros((n_inputs, n_meas + 1))
        inputs_adjusted[:, 1:] = inputs
        inputs = inputs_adjusted
    inputs = inputs.astype(float32)  # If the inputs provided are all ints, the following line doesn't work
    inputs[:, [0]] = nan  # Set the zeroth entry to nan, since the input vector has size n_meas+1

    assert inputs.shape == (n_inputs, n_meas + 1)

    # initialization for all time steps
    states = zeros((n_states, n_meas + 1))
    states[:, [0]] = initial_state
    inputs_disturbed = zeros_like(inputs)
    inputs_disturbed[:, [0]] = nan
    input_noise = zeros_like(inputs)
    input_noise[:, [0]] = nan
    process_noise = zeros((n_states, n_meas + 1))
    process_noise[:, [0]] = nan
    measurement_noise = zeros((M_C, n_meas + 1))
    measurement_noise[:, [0]] = nan
    measurements = zeros((M_C, n_meas + 1))
    measurements[:, [0]] = nan
    measurements_disturbed = zeros((M_C, n_meas + 1))
    measurements_disturbed[:, [0]] = nan

    # set seed for noise generation
    seed(seed_val)

    # simulate system for  all time steps
    for i in range(1, n_meas + 1):
        # sample input noise
        input_noise[:, [i]] = col_vec(multivariate_normal(zeros((n_inputs,)), input_noise_cov))
        # add noise to inputs
        inputs_disturbed[:, [i]] = inputs[:, [i]] + input_noise[:, [i]]
        # sample process noise
        process_noise[:, [i]] = col_vec(multivariate_normal(zeros((n_states,)), process_noise_cov))

        # compute next state
        states[:, [i]] = A_fun(states[:, [i - 1]], inputs_disturbed[:, [i]]) + process_noise[:, [i]]
        # sample measurement noise
        measurement_noise[:, [i]] = col_vec(multivariate_normal(zeros((M_C,)), meas_noise_cov))
        # compute measurement
        measurements[:, [i]] = C @ states[:, [i]]
        # add noise to measurements
        measurements_disturbed[:, [i]] = measurements[:, [i]] + measurement_noise[:, [i]]

    return SimResult(inputs, inputs_disturbed, input_noise, states, process_noise, measurements, measurements_disturbed,
                     measurement_noise, time)


class KalmanExample:

    def __init__(self, A, B, C, initial_state, inputs, input_noise_cov, process_noise_cov, meas_noise_cov, n_meas,
                 dT=1, output_noise_cov=None, outputs=None):
        self.A = A
        self.B = B
        self.C = C
        self.initial_state = initial_state
        self.dT = dT
        self.n_meas = n_meas
        self.output_noise_cov = output_noise_cov
        self.input_noise_cov = input_noise_cov
        self.process_noise_cov = process_noise_cov
        self.meas_noise_cov = meas_noise_cov

        if callable(self.A):
            Asig = signature(self.A)
            if len(Asig.parameters) == 1:
                self.AofXandU = False
            elif len(Asig.parameters) == 2:
                self.AofXandU = True
            else:
                raise NotImplementedError('State transition functions with more than 2 or 0 parameters is not'
                                          'supported!')
        else:
            self.AofXandU = False

        self.sim_result = simulate_discrete_system(A, B, C, initial_state.mean, inputs, input_noise_cov,
                                                   process_noise_cov, meas_noise_cov, n_meas, dT,
                                                   AofXandU=self.AofXandU)

        time = linspace(0, (n_meas + 1) * dT, n_meas + 1)
        M_C, N_C = atleast_2d(self.C).shape
        if outputs is not None:
            if callable(outputs):
                self.outputs = concatenate([col_vec(outputs(instant)) for instant in time], axis=1)
            else:
                if ndim(outputs) == 1:
                    self.outputs = row_vec(outputs)
                    self.outputs_adjusted = zeros((N_C, n_meas + 1))
                    self.outputs_adjusted[:, 1:] = self.outputs
                    self.outputs = self.outputs_adjusted
        else:
            self.outputs = self.sim_result.measurements_disturbed

        if callable(inputs):
            # create input matrix with column vectors from input function for each time step
            inputs = concatenate([col_vec(inputs(instant)) for instant in time], axis=1)
        else:
            # blow up input vector to create a matrix of column vectors for each time step
            if ndim(inputs) <= 1:
                inputs = col_vec(inputs)
            n_inputs = B.shape[1]
            inputs_adjusted = zeros((n_inputs, n_meas + 1))
            inputs_adjusted[:, 1:] = inputs
            inputs = inputs_adjusted
        self.inputs = inputs

    def setup_kf(self, kf_obj, input_estimation_cov=None, input_prior=None):
        # Assumes correct model specifications
        kf_obj_loc = copy.deepcopy(kf_obj)

        if input_estimation_cov is not None:
            # Perform input estimation
            input_noise_cov = input_estimation_cov
            inputs = input_prior
        else:
            input_noise_cov = self.input_noise_cov
            if input_prior is None:
                inputs = self.sim_result.inputs_disturbed[:, 1:]
            else:
                inputs = input_prior

        kf_obj_loc.set_model(self.A, self.B, self.C, initial_state_msg=self.initial_state,
                             input_noise_cov=input_noise_cov, process_noise_cov=self.process_noise_cov,
                             meas_noise_cov=self.meas_noise_cov)
        kf_obj_loc.add_slices(self.outputs[:, 1:], inputs)
        return kf_obj_loc

    def run_kf_on_example(self, kf_obj, do_backward=True, input_estimation_cov=None, input_prior=None,
                          num_iterations=1):
        kf_obj_loc = self.setup_kf(kf_obj, input_estimation_cov=input_estimation_cov, input_prior=input_prior)

        for i in range(num_iterations):
            kf_obj_loc.do_forward()
            if do_backward:
                kf_obj_loc.do_backward()
            if input_estimation_cov is not None:
                kf_obj_loc.do_input_estimation()

            # Output RMSE values
            state_marginals = kf_obj_loc.get_state_marginals()
            input_marginals = kf_obj_loc.get_input_marginals()

            input_marginals_means = atleast_2d(squeeze([u.mean for u in input_marginals]))
            state_marginals_means = atleast_2d(squeeze([x.mean for x in state_marginals]))

            state_errors = (state_marginals_means.T - self.sim_result.states)
            state_rmse = [None] * shape(state_errors)[0]
            for j in range(shape(state_errors)[0]):
                state_rmse[j] = sqrt((state_errors[j, :] @ state_errors[j, :].T) / self.n_meas)
                print(str(i) + ': RMSE of state ' + str(j) + ': ' + str(state_rmse[j]))

            input_errors = (input_marginals_means.T - self.inputs[:, 1:])
            input_rmse = [None] * shape(input_errors)[0]
            for j in range(shape(input_errors)[0]):
                input_rmse[j] = sqrt((input_errors[j, :] @ input_errors[j, :].T) / self.n_meas)
                print(str(i) + ': RMSE of input ' + str(j) + ': ' + str(input_rmse[j]))

        return kf_obj_loc

    def benchmark_kf(self, kf_obj, n_runs=None, do_backward=True, input_estimation_cov=None):
        if n_runs is None:
            n_runs = 10
        assert n_runs >= 1
        total_forward_time = 0
        total_backward_time = 0
        total_input_time = 0
        for i in range(0, n_runs):
            gc.collect()
            kf_obj_setup = self.setup_kf(copy.deepcopy(kf_obj))

            start = time.perf_counter()
            kf_obj_setup.do_forward()
            end = time.perf_counter()
            total_forward_time += end - start
            if do_backward:
                start = time.perf_counter()
                kf_obj_setup.do_backward()
                end = time.perf_counter()
                total_backward_time += end - start
            if input_estimation_cov is not None:
                start = time.perf_counter()
                kf_obj_setup.do_input_estimation()
                end = time.perf_counter()
                total_input_time += end - start

        mean_forward_time = total_forward_time / n_runs
        mean_backward_time = total_backward_time / n_runs
        mean_input_time = total_input_time / n_runs
        if not do_backward:
            return mean_forward_time
        else:
            if input_estimation_cov is None:
                return mean_forward_time, mean_backward_time
            else:
                return mean_forward_time, mean_backward_time, mean_input_time

    def compare_filters(self, kfs, show_forward=True, do_backward=True, input_estimation_cov=None,
                        n_benchmark_runs=None):
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            from ime_fgs.demos.kalman.plot_dataset import plot_dataset
        except ImportError:
            raise RuntimeError('Optional packages matplotlib and pandas must be present for this operation.')

        filter_results = {}
        if do_backward:
            smoother_results = {}
            first_smoother_est = None
        if input_estimation_cov is not None:
            input_estimation_results = {}
        first_kf_name = None
        first_filter_est = None
        for kf_name, kf_obj in kfs.items():

            if do_backward:
                forward_time, backward_time = self.benchmark_kf(kf_obj, n_runs=n_benchmark_runs)
                if show_forward:
                    print('Mean filtering time elapsed with ' + kf_name + ': ' + str(forward_time))
                print('Mean smoothing time elapsed with ' + kf_name + ': ' + str(backward_time))
            else:
                forward_time = self.benchmark_kf(kf_obj, do_backward=False, n_runs=n_benchmark_runs)
                print('Mean filtering time elapsed with ' + kf_name + ': ' + str(forward_time))

            kf_obj_loc = self.run_kf_on_example(kf_obj, do_backward=do_backward,
                                                input_estimation_cov=input_estimation_cov)
            filter_results[kf_name] = kf_obj_loc.get_state_fwd_messages()
            curr_filter_est = GaussianMeanCovMessage.get_means(filter_results[kf_name])

            if show_forward:
                for state_idx in range(0, curr_filter_est.shape[0]):
                    print('Mean abs ' + kf_name + ' filter error in state ' + str(state_idx) + ': '
                          + str(abs(curr_filter_est[state_idx, :] - self.sim_result.states[state_idx, :]).mean()))

            if do_backward:
                smoother_results[kf_name] = kf_obj_loc.get_state_marginals()
                curr_smoother_est = GaussianMeanCovMessage.get_means(smoother_results[kf_name])
                for state_idx in range(0, curr_smoother_est.shape[0]):
                    print('Mean abs ' + kf_name + ' smoother error in state ' + str(state_idx) + ': '
                          + str(abs(curr_smoother_est[state_idx, :] - self.sim_result.states[state_idx, :]).mean()))

            if input_estimation_cov is not None:
                input_estimation_results[kf_name] = kf_obj_loc.get_input_marginals()
                curr_input_est = GaussianMeanCovMessage.get_means(input_estimation_results[kf_name])
                for input_idx in range(0, curr_input_est.shape[0]):
                    print('Mean abs ' + kf_name + ' filter/smoother error in input ' + str(input_idx) + ': '
                          + str(abs(curr_input_est[input_idx, :] - self.sim_result.inputs[input_idx, 1:]).mean()))

            if first_kf_name is None:
                first_kf_name = kf_name
                first_filter_est = curr_filter_est
                if do_backward:
                    first_smoother_est = curr_smoother_est
            else:
                if show_forward:
                    print('Max filter diff ' + kf_name + ' vs. ' + first_kf_name + ' across all states: '
                          + str(abs(curr_filter_est - first_filter_est).max()))
                if do_backward:
                    print('Max smoother diff ' + kf_name + ' vs. ' + first_kf_name + ' across all states: '
                          + str(abs(curr_smoother_est - first_smoother_est).max()))

        # Plot of system dynamics + estimates
        df_to_plot = pd.DataFrame()
        ref_sigs = {}
        for input_idx in range(0, self.sim_result.inputs.shape[0]):
            input_name = 'Input ' + str(input_idx)
            df_to_plot[input_name] = self.sim_result.inputs[input_idx, :]
            ref_sigs[input_name] = pd.DataFrame(
                {input_name + ' (disturbed)': self.sim_result.inputs_disturbed[input_idx, :]}
            )
            if input_estimation_cov is not None:
                for kf_name, results in input_estimation_results.items():
                    ref_sigs[input_name].loc[1:, kf_name] = \
                        GaussianMeanCovMessage.get_means(results)[input_idx, :]
                    ref_sigs[input_name].loc[0, kf_name] = nan
        for meas_idx in range(0, self.sim_result.measurements.shape[0]):
            meas_name = 'Measurement ' + str(meas_idx)
            df_to_plot[meas_name] = self.sim_result.measurements[meas_idx, :]
            ref_sigs[meas_name] = pd.DataFrame(
                {meas_name + ' (disturbed)': self.outputs[meas_idx, :]}
            )
        n_states = self.sim_result.states.shape[0]
        for state_idx in range(0, n_states):
            state_name = 'State ' + str(state_idx)
            df_to_plot[state_name] = self.sim_result.states[state_idx, :]
            ref_sigs[state_name] = pd.DataFrame()
            if show_forward:
                for kf_name, results in filter_results.items():
                    ref_sigs[state_name][kf_name + ' filter'] = \
                        GaussianMeanCovMessage.get_means(results)[state_idx, :]
            if do_backward:
                for kf_name, results in smoother_results.items():
                    ref_sigs[state_name][kf_name + ' smoother'] = \
                        GaussianMeanCovMessage.get_means(results)[state_idx, :]

        plot_dataset(df_to_plot, x_series=pd.Series(self.sim_result.time, name='Time'), ref_sigs=ref_sigs,
                     title='Kalman Filter/Smoother Example')

        # Plot of system estimate covariances
        df_to_plot_covs = pd.DataFrame()
        ref_sigs_covs = {}
        for col_idx in range(0, n_states):
            for row_idx in range(col_idx, n_states):
                first_col = None
                if show_forward:
                    for kf_name, results in filter_results.items():
                        cov_name = 'Cov[' + str(row_idx) + ',' + str(col_idx) + '] ' + kf_name + ' filter'
                        if first_col is None:
                            first_col = cov_name
                            ref_sigs_covs[first_col] = pd.DataFrame()
                            df_to_plot_covs[cov_name] = \
                                GaussianMeanCovMessage.get_covs(results)[row_idx, col_idx, :]
                        else:
                            ref_sigs_covs[first_col][cov_name] = \
                                GaussianMeanCovMessage.get_covs(results)[row_idx, col_idx, :]
                if do_backward:
                    for kf_name, results in smoother_results.items():
                        cov_name = 'Cov[' + str(row_idx) + ',' + str(col_idx) + '] ' + kf_name + ' smoother'
                        if first_col is None:
                            first_col = cov_name
                            ref_sigs_covs[first_col] = pd.DataFrame()
                            df_to_plot_covs[cov_name] = \
                                GaussianMeanCovMessage.get_covs(results)[row_idx, col_idx, :]
                        else:
                            ref_sigs_covs[first_col][cov_name] = \
                                GaussianMeanCovMessage.get_covs(results)[row_idx, col_idx, :]

        plot_dataset(df_to_plot_covs, x_series=pd.Series(self.sim_result.time, name='Time'), ref_sigs=ref_sigs_covs,
                     title='Estimation error covariance estimates')

        plt.show()
