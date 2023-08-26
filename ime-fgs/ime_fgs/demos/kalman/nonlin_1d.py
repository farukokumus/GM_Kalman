# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Peterson, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#

import numpy as np
from ime_fgs.demos.kalman.KalmanExample import KalmanExample
from ime_fgs.messages import GaussianMeanCovMessage
from ime_fgs.kalman import KalmanFilter, KalmanSlice
from ime_fgs.unscented_utils import SigmaPointScheme


def setup_example():
    def A(x):
        return x + 1 / (1 + x * x) + 5 * np.cos(x)

    initial_state = GaussianMeanCovMessage(mean=[[10]], cov=[[1]])

    return KalmanExample(A=A, B=1, C=1, initial_state=initial_state, input_noise_cov=1e-9, process_noise_cov=0.5,
                         meas_noise_cov=10, inputs=lambda t: 0, n_meas=200)


def run_example(n_benchmark_runs=None):
    print('\n\nNONLINEAR (SIGMA POINT) 1-D KF/KS EXAMPLE')
    example = setup_example()
    example.compare_filters({'UKS-classic': KalmanFilter(slice_type=KalmanSlice,
                                                         sigma_point_scheme=SigmaPointScheme.SigmaPointsClassic),
                             'UKS-reduced': KalmanFilter(slice_type=KalmanSlice,
                                                         sigma_point_scheme=SigmaPointScheme.SigmaPointsReduced)},
                            do_backward=True,
                            n_benchmark_runs=n_benchmark_runs)


if __name__ == '__main__':
    run_example()
