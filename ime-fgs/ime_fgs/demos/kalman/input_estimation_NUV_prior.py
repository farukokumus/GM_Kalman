# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Christian Hoffmann
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine

import numpy as np
from ime_fgs.messages import GaussianMeanCovMessage
from ime_fgs.demos.kalman.KalmanExample import KalmanExample
from ime_fgs.kalman import KalmanFilter, KalmanSlice
from ime_fgs.utils import col_vec, row_vec
from ime_fgs.unscented_utils import SigmaPointScheme


def setup_example():
    # Sampling time
    Ts = 0.1
    dampfac = 1

    # Define state transition function
    # def A(x):
    #     # Van der Pol Oscillator
    #     x = np.squeeze(x)
    #     if x[2] > 0:
    #         pass
    #     return col_vec([x[0] + Ts * x[1], x[1] - Ts * x[0] + Ts * 0.25 * (1 - x[0] ** 2) * x[1] + Ts * x[2],
    #                     dampfac * x[2]])
    #
    # # Define output function
    # C = row_vec([1, 0, 0])
    #
    # # Linear input gain matrix
    # B = col_vec([0, 0, 1])
    #
    # # Initial state
    # x0 = GaussianMeanCovMessage([np.pi / 4, np.pi / 4, 0], [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    # Define state transition matrix
    A = np.array([[0.3, 1.0],
                  [0.0, 1.0]])

    # Define input gain matrix
    B = np.array([[1.0, 0.0],
                  [0.0, 1.0]])

    # Define output function
    C = row_vec([1, 0])

    # Initial state
    x0 = GaussianMeanCovMessage([[1.0], [2.0]], [[1.0, 0.0],
                                                 [0.0, 1.0]])

    n_meas = 240
    inputs = np.zeros((2, n_meas))
    inputs[1, [55]] = 5.0
    inputs[1, [115]] = -5.0

    return KalmanExample(A=A, B=B, C=C, initial_state=x0, input_noise_cov=[[0, 0], [0, 0]],
                         process_noise_cov=[[0.005, 0], [0, 0.005]],
                         meas_noise_cov=0.02, inputs=inputs, n_meas=n_meas, dT=Ts)


def generate_paper_plots():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from ime_fgs.demos.kalman.plot_dataset import plot_dataset
    sns.set_style("white")
    example = setup_example()

    # Can be used to specify an offset for the input prior
    n_meas = 240
    inputs = np.zeros((2, n_meas))

    kf_obj = example.run_kf_on_example(KalmanFilter(slice_type=KalmanSlice,
                                                    backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde,
                                                    sigma_point_scheme=SigmaPointScheme.GaussHermite,
                                                    regularized_input_estimation=True,
                                                    linearization_about_marginal=False),
                                       do_backward=True, input_estimation_cov=[[1, 0], [0, 1]], input_prior=inputs,
                                       num_iterations=20)

    smoother_results = kf_obj.get_state_marginals()
    input_estimation_results = kf_obj.get_input_marginals()

    # Plot of system dynamics + estimates
    n_samples_to_omit = 20
    df_to_plot = pd.DataFrame()
    ref_sigs = {}
    # Input signals
    df_to_plot['u1 (t)'] = example.sim_result.inputs[0, :-n_samples_to_omit]
    ref_sigs['u1 (t)'] = pd.DataFrame({'Estimate': np.zeros((len(example.sim_result.time) - n_samples_to_omit,))})
    ref_sigs['u1 (t)'].loc[1:, 'Estimate'] = \
        GaussianMeanCovMessage.get_means(input_estimation_results)[0, :-n_samples_to_omit]
    ref_sigs['u1 (t)'].loc[0, 'Estimate'] = np.nan
    df_to_plot['u2 (t)'] = example.sim_result.inputs[1, :-n_samples_to_omit]
    ref_sigs['u2 (t)'] = pd.DataFrame({'Estimate': np.zeros((len(example.sim_result.time) - n_samples_to_omit,))})
    ref_sigs['u2 (t)'].loc[1:, 'Estimate'] = \
        GaussianMeanCovMessage.get_means(input_estimation_results)[1, :-n_samples_to_omit]
    ref_sigs['u2 (t)'].loc[0, 'Estimate'] = np.nan

    # Measurement signal
    df_to_plot['y (t)'] = example.sim_result.measurements_disturbed[0, :-n_samples_to_omit]

    # State signals
    df_to_plot['x1 (t)'] = example.sim_result.states[0, :-n_samples_to_omit]
    ref_sigs['x1 (t)'] = \
        pd.DataFrame({'Estimate': GaussianMeanCovMessage.get_means(smoother_results)[0, :-n_samples_to_omit]})
    df_to_plot['x2 (t)'] = example.sim_result.states[1, :-n_samples_to_omit]
    ref_sigs['x2 (t)'] = \
        pd.DataFrame({'Estimate': GaussianMeanCovMessage.get_means(smoother_results)[1, :-n_samples_to_omit]})
    # df_to_plot['x3 (t)'] = example.sim_result.states[2, :-n_samples_to_omit]
    # ref_sigs['x3 (t)'] = \
    #     pd.DataFrame({'Estimate': GaussianMeanCovMessage.get_means(smoother_results)[2, :-n_samples_to_omit]})

    mpl.rcParams.update({'font.size': 20})
    mpl.rcParams['lines.linewidth'] = 3
    plot_dataset(df_to_plot, x_series=pd.Series(example.sim_result.time[:-n_samples_to_omit], name='Time (s)'),
                 ref_sigs=ref_sigs, num_cols=1, bare_plot=True)
    fig = plt.gcf()
    fig.set_size_inches(13, 12)
    fig.savefig('van-der-pol.png', dpi=600, bbox_inches='tight')
    plt.show()


def run_example(n_benchmark_runs=None):
    print('\n\nNONLINEAR 2-D KF/KS VAN DER POL OSCILLATOR EXAMPLE')

    example = setup_example()
    example.compare_filters({'Tilde': KalmanFilter(slice_type=KalmanSlice,
                                                   backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde,
                                                   regularized_input_estimation=True)},
                            do_backward=True,
                            n_benchmark_runs=n_benchmark_runs,
                            input_estimation_cov=10)


if __name__ == '__main__':
    # run_example(n_benchmark_runs=1)
    generate_paper_plots()
