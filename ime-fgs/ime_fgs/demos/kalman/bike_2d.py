# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Peterson, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#

from numpy import array
from ime_fgs.kalman import KalmanFilter, KalmanSlice
from ime_fgs.messages import GaussianMeanCovMessage
from ime_fgs.demos.kalman.KalmanExample import KalmanExample


def setup_example():

    mass = 100
    damping = 0.9
    T = 0.1

    # Define system model
    A = array([[1, T], [0, damping]])
    B = array([[0], [T / mass]])
    C = array([[0, 1]])
    meas_noise_cov = 0.5
    process_noise_cov = array([[1e-5, 0], [0, 0.1]])
    input_noise_cov = 50
    num_meas = 200

    def input_fun(t):
        if 0.5 < t <= 10:
            return 100
        elif 10 < t <= 12:
            return 200
        elif 12 < t <= 15:
            return 1000
        else:
            return 0

    initial_state = GaussianMeanCovMessage(mean=[[0], [0]], cov=[[0.01, 0], [0, 0.1]])

    return KalmanExample(A=A, B=B, C=C, initial_state=initial_state, input_noise_cov=input_noise_cov,
                         process_noise_cov=process_noise_cov, meas_noise_cov=meas_noise_cov, inputs=input_fun,
                         n_meas=num_meas, dT=T)


def run_example(n_benchmark_runs=None):
    print('\n\nLINEAR 2-D KF/KS INPUT ESTIMATION BICYCLE TACHOMETER EXAMPLE')
    #
    # Example: Estimate distance from noisy velocity measurement and infer the force exerted by the cyclist.
    #
    example = setup_example()
    example.compare_filters({'Tilde': KalmanFilter(slice_type=KalmanSlice,
                                                   backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde)},
                            do_backward=True, input_estimation_cov=1e6,
                            n_benchmark_runs=n_benchmark_runs)


if __name__ == '__main__':
    run_example()
