# For research and educational use only, do not distribute without permission.
# LCSS / CDC Paper
# Created by:
# Eike Petersen, Christian Hoffmann
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine

import numpy as np
from numpy.linalg import inv
from numpy import cos, sin, squeeze, shape, eye
from ime_fgs.messages import GaussianMeanCovMessage
from ime_fgs.demos.control.ControlExample import ControlExample
from ime_fgs.demos.kalman.KalmanExample import KalmanExample
# from ime_fgs.kalman import KalmanFilter, KalmanSlice
from ime_fgs.kalman_nonlin_input import KalmanFilter, KalmanSlice
from ime_fgs.utils import col_vec
from ime_fgs.unscented_utils import SigmaPointScheme
import time


def setup_example():
    # Sampling time
    Ts = 0.1

    def system_dimensions():
        nx = 2
        ny = 1
        nu = 1
        return nx, ny, nu

    # Define output function
    C = [[1, 0]]

    # Linear input gain matrix
    B = np.identity(1)

    def Ac(x, u):
        x = squeeze(x)
        u = squeeze(u)

        Ac = col_vec([x[1], - x[0] + 0.5 * (1.0 - x[0] ** 2) * x[1] + 1.0 * (x[0] / 2.0 + 1.0) * u])

        return col_vec(Ac)

    def dAcdx(x, u):
        x = squeeze(x)
        u = squeeze(u)

        dAcdx = squeeze([[0, 1, 0], [-1 - x[0] * x[1] + 0.5 * u, 0.5 * (1 - x[0] ** 2), 0.5 * x[0] + 1]])

        return dAcdx

    # def A(x, u):
    #     # First-Order Euler Style Discretization
    #     x = squeeze(x)
    #     u = squeeze(u)
    #     vx = np.atleast_2d(x).T
    #     A  = vx + Ts * Ac(x, u)
    #
    #     return A

    def A(x, u):
        # Runge-Kutta Style Discretization
        x = squeeze(x)
        u = squeeze(u)
        k1 = squeeze(Ts * Ac(x, u))
        # k2 = squeeze(Ts * Ac(x + k1/2.0, u))
        # k3 = squeeze(Ts * Ac(x + k2/2.0, u))
        # k4 = squeeze(Ts * Ac(x + k3/1.0, u))
        vx = squeeze(x)
        # A = col_vec(vx + 1.0/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4))
        A = col_vec(vx + k1)

        return A

    def dAdx(x, u):
        # Runge-Kutta Style Discretization
        x = squeeze(x)
        u = squeeze(u)
        # k1 = squeeze(Ts * dAcdx(x, u))
        # k2 = squeeze(Ts * dAcdx(x + k1/2.0, u))
        # k3 = squeeze(Ts * dAcdx(x + k2/2.0, u))
        # k4 = squeeze(Ts * dAcdx(x + k3/1.0, u))
        # vx = squeeze(x)
        # dAdx = vx + 1.0/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

        dAdx = squeeze([[1, 0, 0], [0, 1, 0]]) + squeeze(Ts * dAcdx(x, u))

        return dAdx

    # Initial state
    x0_mean = [[0.1], [0.1]]
    nx = shape(x0_mean)[0]
    x0_cov = eye(nx) * 0.05
    x0 = GaussianMeanCovMessage(x0_mean, x0_cov)
    u0 = np.array([[0.0]])

    def input_fun(t):
        if 0 <= t < 5:
            return col_vec(0)
        elif 5 <= t < 15:
            return col_vec(1.0)
        elif 20 <= t < 30:
            return col_vec(1.0)
        else:
            return col_vec(0)

    def ref(t):
        if 0 <= t < 5:
            return col_vec(0)
        elif 5 <= t < 25:
            return col_vec(0.5)
        elif 25 <= t < 75:
            return col_vec(0)
        else:
            return col_vec(0)

    def output_fun(t):
        return col_vec([ref(t)])

    nx, ny, nu = system_dimensions()

    R = 1.0
    Q = 0.5
    VR = np.identity(nu) * 1 / R
    VU = np.identity(nu) * 0.0
    VZ = np.identity(ny) * 1 / Q
    VI = np.identity(nx) * 0.00000005
    VD = np.identity(ny) * 1.0

    Tfinal = 45.0
    n_meas = int(np.round(Tfinal / Ts, 0))

    return ControlExample(A=A, B=B, C=C, initial_state=x0, outputs=None, output_noise_cov=None, input_noise_cov=VU,
                          process_noise_cov=VI,
                          meas_noise_cov=VD, inputs=input_fun, n_meas=n_meas, dT=Ts, dAdx=dAdx)


def generate_paper_plots():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from ime_fgs.demos.kalman.plot_dataset import plot_dataset
    sns.set_style("white")

    print('\n\nNONLINEAR 2-D KF/KS vdP INPUT ESTIMATION EXAMPLE')

    VR = np.identity(1) * 0.75  # 0.75

    num_iterations = 7
    range_iterations = range(1, num_iterations)
    state_rmse = [None] * (num_iterations - 1)
    input_rmse = [None] * (num_iterations - 1)

    df_rmse_to_plot = pd.DataFrame()

    for repetitions in range(10):
        start = time.time()
        for iterations in range_iterations:
            example = setup_example()
            # kf_obj, input_rmse[iterations-1], state_rmse[iterations-1] = example.run_kf_on_example(
            #     KalmanFilter(slice_type=KalmanSlice,
            #     backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde,
            #     sigma_point_scheme=SigmaPointScheme.GaussHermite,
            #     regularized_input_estimation=False,
            #     linearization_about_marginal=True,
            #     expectation_propagation=False,
            #     jacobian_linearization=False),
            #     do_backward=True, input_estimation_cov=VR, num_iterations=iterations)
            kf_obj, input_rmse[iterations - 1], state_rmse[iterations - 1] = example.run_kf_on_example(
                KalmanFilter(slice_type=KalmanSlice,
                             backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde,
                             sigma_point_scheme=SigmaPointScheme.GaussHermite,
                             regularized_input_estimation=False,
                             linearization_about_marginal=True,
                             expectation_propagation=False,
                             jacobian_linearization=False),
                do_backward=True, input_estimation_cov=VR, num_iterations=iterations)
            smoother_results = kf_obj.get_state_marginals()
            input_estimation_results = kf_obj.get_input_marginals()

            # Plot of system dynamics + estimates
            n_samples_to_omit = 1
            df_to_plot = pd.DataFrame()
            ref_sigs = {}
            # Input signals
            df_to_plot['u'] = GaussianMeanCovMessage.get_means(input_estimation_results)[0, :-n_samples_to_omit]
            ref_sigs['u'] = pd.DataFrame({'ur': example.inputs[0, :-n_samples_to_omit]})
            df_to_plot['ur'] = example.inputs[0, :-n_samples_to_omit]
            # ref_sigs['u (t)'].loc[1:, 'Inferred Control Input'] = \
            #     GaussianMeanCovMessage.get_means(input_estimation_results)[0, 1:-n_samples_to_omit]
            # ref_sigs['u (t)'].loc[0, 'Inferred Control Input'] = np.nan

            # Measurement signal
            # df_to_plot['y (t)'] = example.sim_result.measurements_disturbed[0, :-n_samples_to_omit]
            # df_to_plot['t'] = example.sim_result.time[:-n_samples_to_omit] \
            # range(len(example.sim_result.measurements_disturbed[0, :-n_samples_to_omit])) * Ts

            # State signals
            df_to_plot['x1'] = GaussianMeanCovMessage.get_means(smoother_results)[0, 1:-n_samples_to_omit]
            ref_sigs['x1'] = \
                pd.DataFrame({'x1m': example.outputs[0, :-n_samples_to_omit]})
            df_to_plot['x1m'] = example.outputs[0, :-n_samples_to_omit]
            df_to_plot['x1r'] = example.sim_result.states[0, 0:-n_samples_to_omit]
            df_to_plot['x2'] = GaussianMeanCovMessage.get_means(smoother_results)[1, 1:-n_samples_to_omit]
            ref_sigs['x2'] = \
                pd.DataFrame({'x2r': example.sim_result.states[1, 0:-n_samples_to_omit]})
            df_to_plot['x2r'] = example.sim_result.states[1, 0:-n_samples_to_omit]

            mpl.rcParams.update({'font.size': 10})
            mpl.rcParams['lines.linewidth'] = 2
            plot_dataset(df_to_plot, x_series=pd.Series(example.sim_result.time[:-n_samples_to_omit], name='Time (s)'),
                         ref_sigs=ref_sigs, num_cols=1, bare_plot=False)
            fig = plt.gcf()
            fig.set_size_inches(13, 12)
            # fig.savefig('van-der-pol.png', dpi=600, bbox_inches='tight')
            # plt.show()

            # for j in range(len(state_rmse[0])):
            #    print(str(iterations) + ': RMSE of state ' + str(j) + ': ' + str(state_rmse[iterations-1][j]))
            # for j in range(len(input_rmse[0])):
            #    print(str(iterations) + ': RMSE of input ' + str(j) + ': ' + str(input_rmse[iterations-1][j]))

            df_to_plot.to_csv(path_or_buf='TikZ-vdP2_IterativeControlInputEstimationGauss_i' + str(iterations) + '.dat',
                              sep='\t')

        end = time.time()
        print(end - start)

    df_rmse_to_plot['uRMSE'] = input_rmse
    for j in range(len(state_rmse[0])):
        df_rmse_to_plot['x' + str(j + 1) + 'RMSE'] = [state_rmse[i][j] for i in range(num_iterations - 1)]

    plot_dataset(df_rmse_to_plot, x_series=pd.Series(range(1, num_iterations), name='Number of Iterations'),
                 ref_sigs=None, num_cols=1, bare_plot=False)

    fig = plt.gcf()
    fig.set_size_inches(13, 12)
    plt.show()

    df_rmse_to_plot.to_csv(
        path_or_buf='TikZ-vdP2_IterativeControlInputEstimationGauss_RMSE_i' + str(num_iterations) + '.dat', sep='\t')

    # Manual save as csv
    # with open('TikZ-vanDerPolOscillator_IterativeSmoother.dat', 'w', newline='') as csvfile:
    #     fieldnames = ['t', 'x1', 'x2', 'u']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
    #     writer.writeheader()
    #     for ii in range(len(t)): #[0::4] # to pick every 4th entry
    #         writer.writerow({fieldnames[0]: str(t[ii]), fieldnames[1]: str(squeeze(y_fg[ii])),
    #                          fieldnames[2]: str(squeeze(u_fg[ii])), fieldnames[3]: str(squeeze(z[ii]))})


if __name__ == '__main__':
    generate_paper_plots()
