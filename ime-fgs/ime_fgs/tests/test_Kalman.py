# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#

import unittest
from unittest.mock import patch
from numpy import linspace
from numpy.random import normal, seed
import numpy as np

from ime_fgs.demos.kalman import welsh, bike_2d, nonlin_1d, van_der_pol
from ime_fgs.messages import GaussianMeanCovMessage
from ime_fgs.advanced_nodes import UnscentedNode, StatisticalLinearizationNode
from ime_fgs.kalman import KalmanSliceNaive, KalmanFilter, KalmanSlice
from ime_fgs.utils import row_vec


# TODO: Add input + noise estimation tests (+demo in kalman.py)
# TODO: Add 2d test case
# TODO: Add validation by comparison with output of classical KF


class TestKalmanFilter(unittest.TestCase):

    def compare_filter_results(self, kf_1, kf_2, compare_marginals=False):
        msgs_1 = kf_1.get_state_fwd_messages() if not compare_marginals else kf_1.get_state_marginals()
        filtered_means_1 = GaussianMeanCovMessage.get_means(msgs_1)
        filtered_cov_1 = GaussianMeanCovMessage.get_covs(msgs_1)
        msgs_2 = kf_2.get_state_fwd_messages() if not compare_marginals else kf_2.get_state_marginals()
        filtered_means_2 = GaussianMeanCovMessage.get_means(msgs_2)
        filtered_cov_2 = GaussianMeanCovMessage.get_covs(msgs_2)
        self.assertTrue(np.allclose(filtered_means_1, filtered_means_2))
        self.assertTrue(np.allclose(filtered_cov_1, filtered_cov_2))

    def test_init_kalman(self):
        initial_state = GaussianMeanCovMessage(mean=[[10], [1]], cov=np.identity(2))
        kf = KalmanFilter(a=1, b=1, c=1, initial_state_msg=initial_state,
                          input_noise_cov=1e-1,
                          process_noise_cov=1e-1,
                          meas_noise_cov=1,
                          slice_type=KalmanSliceNaive)
        self.assertTrue(len(kf.slices) == 0)
        self.assertTrue(kf.initial_state._prior == initial_state)

    def test_kalman_add_slices(self):
        # Define system model
        a = 1
        b = 1
        c = 1
        # Number of time instants
        num_meas = 50
        seed(1)
        z_real = linspace(0, num_meas, num_meas)
        z = row_vec(z_real + normal(0, 1, num_meas))
        u = row_vec(linspace(1, 1, num_meas))
        initial_state = GaussianMeanCovMessage(mean=[[10]], cov=[[1]])
        kf = KalmanFilter(a, b, c, initial_state,
                          input_noise_cov=1e-1,
                          process_noise_cov=1e-1,
                          meas_noise_cov=1,
                          slice_type=KalmanSliceNaive)
        for ii in range(num_meas):
            kf.add_slice(z[0, ii], u[0, ii])

        kf2 = KalmanFilter(a, b, c, initial_state,
                           input_noise_cov=1e-1,
                           process_noise_cov=1e-1,
                           meas_noise_cov=1,
                           slice_type=KalmanSliceNaive)
        kf2.add_slices(z, u)
        self.assertEqual(len(kf.slices), len(kf2.slices))
        self.assertEqual(len(kf.slices), num_meas)
        self.assertEqual(kf.slices[0].port_input.other_port.out_msg.mean, u[0, 0])
        self.assertEqual(kf.slices[0].port_input.other_port.out_msg.cov, 1e-1)
        self.assertEqual(kf.slices[-1].port_input.other_port.out_msg.mean, u[0, -1])
        self.assertEqual(kf.slices[-1].port_input.other_port.out_msg.cov, 1e-1)
        self.assertEqual(kf.slices[0].port_meas.other_port.out_msg.mean, z[0, 0])
        self.assertEqual(kf.slices[0].port_meas.other_port.out_msg.cov, 1)
        self.assertEqual(kf.slices[-1].port_meas.other_port.out_msg.mean, z[0, -1])
        self.assertEqual(kf.slices[-1].port_meas.other_port.out_msg.cov, 1)
        kf.do_forward()
        kf2.do_forward()
        self.assertTrue(np.all(kf.get_state_fwd_messages() == kf2.get_state_fwd_messages()))

    def test_Kalman_filt_smooth_1d(self):
        # Define system model
        A = 1
        C = 1
        B = 1

        # Number of time instants
        num_meas = 50

        seed(1)
        z_real = linspace(0, num_meas, num_meas)
        z = row_vec(z_real + normal(0, 1, num_meas))
        u = row_vec(linspace(1, 1, num_meas))

        # KF with compound slices
        initial_state = GaussianMeanCovMessage(mean=[[10]], cov=[[1]])
        kf1 = KalmanFilter(A, B, C, initial_state,
                           input_noise_cov=1e-1,
                           process_noise_cov=1e-1,
                           meas_noise_cov=1,
                           slice_type=KalmanSlice,
                           backward_pass_mode=KalmanSlice.BackwardPassMode.MeanCov)
        kf1.add_slices(u, z)
        kf1.do_forward()
        filtered_means = GaussianMeanCovMessage.get_means(kf1.get_state_fwd_messages())
        smoothed_means_after_filt = GaussianMeanCovMessage.get_means(kf1.get_state_marginals())
        filtered_cov = GaussianMeanCovMessage.get_covs(kf1.get_state_fwd_messages())
        self.assertTrue(np.all(np.logical_not(np.isnan(filtered_means))))
        self.assertEqual(filtered_means.shape, (1, num_meas + 1))
        self.assertEqual(kf1.get_state_fwd_messages()[0], initial_state)
        self.assertTrue(np.all(filtered_means == smoothed_means_after_filt))
        kf1.do_backward()
        mean_cov_smoothed_est = GaussianMeanCovMessage.get_means(kf1.get_state_marginals())
        self.assertEqual(kf1.get_state_marginals()[-1], kf1.get_state_fwd_messages()[-1])

        kf2 = KalmanFilter(A, B, C, initial_state,
                           input_noise_cov=1e-1,
                           process_noise_cov=1e-1,
                           meas_noise_cov=1,
                           slice_type=KalmanSlice,
                           backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde)
        kf2.add_slices(u, z)
        kf2.do_forward()
        kf2.do_backward()
        tilde_smoothed_est = GaussianMeanCovMessage.get_means(kf2.get_state_marginals())
        tilde_smoothed_cov = GaussianMeanCovMessage.get_covs(kf2.get_state_marginals())
        self.assertTrue(np.allclose(mean_cov_smoothed_est, tilde_smoothed_est))
        self.assertTrue(np.all(tilde_smoothed_cov <= filtered_cov))

    def test_unscented_filt_smooth_lin(self):
        # Compare results of standard an unscented filters and smoothers for a linear example (where both should yield
        # the same results)
        # Define system model
        A = 1
        C = 1
        B = 1

        # Number of time instants
        num_meas = 50

        seed(1)
        z_real = linspace(0, num_meas, num_meas)
        z = row_vec(z_real + normal(0, 1, num_meas))
        u = row_vec(linspace(1, 1, num_meas))

        # Linear KF with compound slices and MeanCov backwards pass
        initial_state = GaussianMeanCovMessage(mean=[[10]], cov=[[1]])
        kf = KalmanFilter(A, B, C, initial_state,
                          input_noise_cov=1e-1,
                          process_noise_cov=1e-1,
                          meas_noise_cov=1,
                          slice_type=KalmanSlice,
                          backward_pass_mode=KalmanSlice.BackwardPassMode.MeanCov)
        kf.add_slices(u, z)
        kf.do_forward()
        kf.do_backward()

        # UKF with compound slices and (naive) MeanCov backwards pass
        ukf = KalmanFilter(lambda x: A * x, B, C, initial_state,
                           input_noise_cov=1e-1,
                           process_noise_cov=1e-1,
                           meas_noise_cov=1,
                           slice_type=KalmanSlice,
                           backward_pass_mode=KalmanSlice.BackwardPassMode.MeanCov)
        ukf.add_slices(u, z)
        self.assertIsInstance(ukf.slices[0].A_node, StatisticalLinearizationNode)
        ukf.do_forward()
        ukf.do_backward()
        self.compare_filter_results(kf, ukf)
        self.compare_filter_results(kf, ukf, compare_marginals=True)

    def test_unscented_smoother_equivalence(self):
        # Compare the results of the different unscented nonlinear RTS backwards pass implementations
        # Define system model
        def A(x):
            return x + 1 / (1 + x * x) + 5 * np.cos(x)

        C = 1
        B = 1

        process_noise = 0.5
        measurement_noise = 10
        input_noise = 1e-9

        # Number of time instants
        num_meas = 200

        np.random.seed(2)

        z_real = np.ones(num_meas)
        for i in range(1, num_meas):
            z_real[i] = A(z_real[i - 1]) + np.random.normal(0, process_noise)

        z = row_vec(z_real + np.random.normal(0, measurement_noise, num_meas))
        u = row_vec(np.zeros(num_meas))

        initial_state = GaussianMeanCovMessage(mean=[[10]], cov=[[1]])

        # UKS with compound slices and MeanCov backwards pass and MeanCov RTS implementation
        uks1 = KalmanFilter(A, B, C, initial_state,
                            input_noise_cov=input_noise,
                            process_noise_cov=process_noise,
                            meas_noise_cov=measurement_noise,
                            slice_type=KalmanSlice,
                            backward_pass_mode=KalmanSlice.BackwardPassMode.MeanCov)
        uks1.add_slices(u, z)
        # assert backward pass mode
        self.assertEqual(uks1.slices[-1].A_node.backward_pass_mode,
                         StatisticalLinearizationNode.BackwardPassMode.MeanCov)
        uks1.do_forward()
        uks1.do_backward()

        # UKS with compound slices, MeanCov backwards pass and WeightedMeanInfo RTS pass
        uks2 = KalmanFilter(A, B, C, initial_state,
                            input_noise_cov=input_noise,
                            process_noise_cov=process_noise,
                            meas_noise_cov=measurement_noise,
                            slice_type=KalmanSlice,
                            backward_pass_mode=KalmanSlice.BackwardPassMode.MeanCov,
                            unscented_backward_pass_mode=StatisticalLinearizationNode.BackwardPassMode.WeightedMeanInfo)
        uks2.add_slices(u, z)
        self.assertEqual(uks2.slices[-1].A_node.backward_pass_mode,
                         StatisticalLinearizationNode.BackwardPassMode.WeightedMeanInfo)
        uks2.do_forward()
        uks2.do_backward()

        self.compare_filter_results(uks1, uks2)
        self.compare_filter_results(uks1, uks2, compare_marginals=True)

        # UKS with compound slices, Tilde backwards pass and Tilde RTS pass
        uks3 = KalmanFilter(A, B, C, initial_state,
                            input_noise_cov=input_noise,
                            process_noise_cov=process_noise,
                            meas_noise_cov=measurement_noise,
                            slice_type=KalmanSlice,
                            backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde)
        uks3.add_slices(u, z)
        self.assertEqual(uks3.slices[-1].A_node.backward_pass_mode, StatisticalLinearizationNode.BackwardPassMode.Tilde)
        uks3.do_forward()
        uks3.do_backward()

        self.compare_filter_results(uks1, uks3)
        self.compare_filter_results(uks1, uks3, compare_marginals=True)

    @patch("matplotlib.pyplot.show")  # mock the show function to not open plot windows
    @patch("matplotlib.pyplot.figure")  # mock the figure function
    @patch("builtins.print")  # mock the print function to avoid useless print
    def test_examples_work(self, mock_show, mock_figure, mock_print):
        # test if examples run
        welsh.run_example(n_benchmark_runs=1)
        bike_2d.run_example(n_benchmark_runs=1)
        nonlin_1d.run_example(n_benchmark_runs=1)
        van_der_pol.run_example(n_benchmark_runs=1)


if __name__ == '__main__':
    unittest.main()
