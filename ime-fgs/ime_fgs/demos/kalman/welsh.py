# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Peterson, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#

from ime_fgs.kalman import KalmanFilter, KalmanSlice
from ime_fgs.messages import GaussianMeanCovMessage
from ime_fgs.demos.kalman.KalmanExample import KalmanExample


def setup_example():
    return KalmanExample(A=1, B=1, C=1, initial_state=GaussianMeanCovMessage([[10]], [[1]]), input_noise_cov=1e-1,
                         process_noise_cov=1e-1, meas_noise_cov=1, inputs=lambda t: 1, n_meas=50)


def run_example(n_benchmark_runs=None):
    print('\n\nLINEAR 1-D KF/KS EXAMPLE')
    #
    # Example due to Welsh, Bishop (2006): An Introduction to the Kalman Filter
    # http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    #
    # Corresponds to estimation of a constant given noisy measurements of that constant.
    #
    # Code in parts due to http://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
    #
    example = setup_example()
    example.compare_filters({'MeanCov': KalmanFilter(slice_type=KalmanSlice,
                                                     backward_pass_mode=KalmanSlice.BackwardPassMode.MeanCov),
                             'Tilde': KalmanFilter(slice_type=KalmanSlice,
                                                   backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde)},
                            do_backward=True,
                            n_benchmark_runs=n_benchmark_runs)


if __name__ == '__main__':
    run_example()
