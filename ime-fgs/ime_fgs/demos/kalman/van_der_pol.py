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
from ime_fgs.utils import col_vec


def setup_example():
    # Sampling time
    Ts = 0.2

    # Define state transition function
    def A(x):
        # Van der Pol Oscillator
        x = np.squeeze(x)
        return col_vec(x + Ts * np.array([x[1], -x[0] + 0.95 * (1 - x[0] ** 2) * x[1]]))

    # Define output function
    C = [[1, 0]]

    # Linear input gain matrix
    B = Ts * np.array([[0], [1]])

    # Initial state
    x0 = GaussianMeanCovMessage([[np.pi / 4], [np.pi / 4]], [[0.5, 0], [0, 0.5]])

    def input_fun(t):
        if t < 10:
            return 0
        elif 10 <= t < 15:
            return 3
        else:
            return 0

    return KalmanExample(A=A, B=B, C=C, initial_state=x0, input_noise_cov=0, process_noise_cov=[[0, 0], [0, 0]],
                         meas_noise_cov=0.1, inputs=input_fun, n_meas=120, dT=Ts)


def run_example(n_benchmark_runs=None):
    print('\n\nNONLINEAR 2-D KF/KS VAN DER POL OSCILLATOR EXAMPLE')

    example = setup_example()
    example.compare_filters({'Tilde': KalmanFilter(slice_type=KalmanSlice,
                                                   backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde)},
                            do_backward=True,
                            n_benchmark_runs=n_benchmark_runs,
                            input_estimation_cov=10)


if __name__ == '__main__':
    run_example()
