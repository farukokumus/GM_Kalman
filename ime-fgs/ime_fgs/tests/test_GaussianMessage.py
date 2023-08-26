# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import unittest
import numpy as np

from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage
from ime_fgs.messages import Message, MultipleCombineMessage


class GaussianMeanCovMessageTest(unittest.TestCase):
    """ Test for GaussianMeanCovMessages """

    def test_init_1(self):
        with self.assertRaises(AssertionError):
            GaussianMeanCovMessage([[1], [2]], np.identity(7) * 5)

    def test_init_2(self):
        with self.assertRaises(AssertionError):
            GaussianMeanCovMessage([[1], [2 + 1j]], [[9]])

    def test_combine(self):
        """
        Testing exception raising when combining two MeanCov
        """
        msg_a = GaussianMeanCovMessage([[1], [3], [5]], np.identity(3) * 5)
        msg_b = GaussianMeanCovMessage([[1], [9], [5]], np.identity(3) * 13)
        with self.assertRaises(NotImplementedError):
            msg_a.combine(msg_b)

    def test_convert_to_MeanCov(self):
        """
        Test: converting MeanCov to MeanCov
        """
        msg_a = GaussianMeanCovMessage([[1], [3], [5]], np.identity(3) * 2)
        msg_b = msg_a.convert(GaussianMeanCovMessage)
        self.assertEqual(msg_b, msg_a)

    def test_convert_to_WeightedMeanInfo(self):
        """
        Test: converting MeanCov to WeightedMeanInfo
        """
        msg_a = GaussianMeanCovMessage([[1], [3], [5]], np.identity(3) * 2)
        msg_c = msg_a.convert(GaussianWeightedMeanInfoMessage)
        msg_d = GaussianWeightedMeanInfoMessage([[0.5], [1.5], [2.5]], np.identity(3) * 0.5)
        self.assertEqual(msg_d, msg_c)

    def test_convert_Error(self):
        """
        Test: Error when converting to wrong target type
        """
        msg_a = GaussianMeanCovMessage([[1], [3], [5]], np.identity(3) * 2)
        with self.assertRaisesRegex(NotImplementedError,
                                    'This kind of message type conversion has not been implemented yet.'):
            msg_a.convert(list)

    def test_multiply_deterministic(self):
        """
        Matrix Multiplication Node
        """
        msg_a = GaussianMeanCovMessage([[1], [3], [5]], np.identity(3) * 2)
        matrix = (np.identity(3) * (1 + 2j))
        matrix_h = (np.identity(3) * (1 - 2j))
        msg_b = msg_a.multiply_deterministic(matrix)
        cov = matrix @ (np.identity(3) * 2) @ matrix_h
        mean = matrix @ ([[1], [3], [5]])
        msg_c = GaussianMeanCovMessage(mean, cov)
        self.assertEqual(msg_b, msg_c)

    def test_multiply_deterministic_exception(self):
        """
        Test: Exception when inverse direction
        """
        msg_a = GaussianMeanCovMessage([[1], [3], [5]], np.identity(3) * 2)
        matrix = (np.identity(3) * (1 + 2j))
        with self.assertRaises(NotImplementedError):
            msg_a.multiply_deterministic(matrix, 1)

    def test_add_1(self):
        """
        Test: Addition of two MeanCov
        """
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 2)
        msg_b = GaussianMeanCovMessage([[2], [2]], np.identity(2) * 6)
        msg_d = msg_a + msg_b
        msg_e = GaussianMeanCovMessage([[3], [4]], np.identity(2) * 8)
        self.assertEqual(msg_d, msg_e)

    def test_add_2(self):
        """
        Test: Exception when adding MeanCov and scalar
        """
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 2)
        with self.assertRaises(TypeError):
            msg_a + 3

    def test_add_3(self):
        """
        Test: Exception when adding MeanCov and WeightedMeanInfo
        """
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 2)
        msg_c = GaussianWeightedMeanInfoMessage([[1], [3]], np.identity(2) * 3)
        with self.assertRaises(TypeError):
            msg_a + msg_c

    def test_sub(self):
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 2)
        msg_b = GaussianMeanCovMessage([[2], [2]], np.identity(2) * 6)
        msg_c = msg_a - msg_b
        msg_d = msg_a + (-msg_b)
        msg_e = GaussianMeanCovMessage([[-1], [0]], np.identity(2) * (-4))
        self.assertEqual(msg_c, msg_d, msg_e)

    def test_neg(self):
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 2)
        msg_b = GaussianMeanCovMessage([[-1], [-2]], np.identity(2) * 2)
        self.assertEqual(msg_b, msg_a.__neg__())

    def test_eq(self):
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 2)
        msg_b = msg_a
        msg_c = GaussianWeightedMeanInfoMessage([[1], [2]], np.identity(2) * 2)
        self.assertTrue(msg_a == msg_b != msg_c)


class GaussianWeightedMeanInfoMessageTest(unittest.TestCase):
    """ Test for GaussianWeightedMeanMessages"""

    def test_init_1(self):
        with self.assertRaises(AssertionError):
            GaussianWeightedMeanInfoMessage([[1]], np.identity(2) * 5)

    def test_init_2(self):
        with self.assertRaises(AssertionError):
            GaussianWeightedMeanInfoMessage([[1], [2], [6]], np.identity(2) * 2)

    def test_combine_two_WeightedMeanInfo_1(self):
        """
        Test: Combining two WeightedMeanInfo
        """
        msg_a = GaussianWeightedMeanInfoMessage([[0.5], [1.5], [2 + 3j]], np.identity(3) * 0.5)
        msg_b = GaussianWeightedMeanInfoMessage([[0.5], [1.5 + 1j], [2.5]], np.identity(3) * 7)
        msg_c = msg_a.combine(msg_b)
        msg_d = GaussianWeightedMeanInfoMessage([[1], [3 + 1j], [4.5 + 3j]], np.identity(3) * 7.5)
        self.assertEqual(msg_c, msg_d)

    def test_combine_two_WeightedMeanInfo_2(self):
        """
        Test: Exception when Combining two WeightedMeanInfo with different seizes
        """
        msg_a = GaussianWeightedMeanInfoMessage([[0.5], [1.5], [2 + 3j]], np.identity(3) * 0.5)
        msg_b = GaussianWeightedMeanInfoMessage([[1.5 + 1j], [2.5]], np.identity(2))
        with self.assertRaises(AssertionError):
            msg_a.combine(msg_b)

    def test_combine_WeightedMeanInfo_MeanCov(self):
        """
        Test: Exception when combining A WeightedMeanInfo and Mean Cov
        """
        msg_a = GaussianWeightedMeanInfoMessage([[0.5], [1.5], [2 + 3j]], np.identity(3) * 0.5)
        msg_b = GaussianMeanCovMessage([[2], [4], [7]], np.identity(3) * 7)
        with self.assertRaises(NotImplementedError):
            msg_a.combine(msg_b)

    def test_convert_to_GaussianWeightedMeanInfo(self):
        msg_a = GaussianWeightedMeanInfoMessage([[0.5], [1.5], [2.5]], np.identity(3) * 2)
        msg_b = msg_a.convert(GaussianWeightedMeanInfoMessage)
        self.assertEqual(msg_b, msg_a)

    def test_convert_to_GaussianMeanCov(self):
        msg_a = GaussianWeightedMeanInfoMessage([[0.5], [1.5], [2.5]], np.identity(3) * 2)
        msg_b = msg_a.convert(GaussianMeanCovMessage)
        msg_c = GaussianMeanCovMessage([[0.25], [0.75], [1.25]], np.identity(3) * 0.5)
        self.assertEqual(msg_b, msg_c)

    def test_convert_Error(self):
        """
        Test: Error when converting to wrong target type
        """
        msg_a = GaussianWeightedMeanInfoMessage([[1], [3], [5]], np.identity(3) * 2)
        with self.assertRaises(NotImplementedError):
            msg_a.convert(list)

    def test_multiply_deterministic(self):
        """
        Matrix Multiplication Node
        """
        msg_a = GaussianWeightedMeanInfoMessage([[1], [3], [5]], np.identity(3) * 2)
        matrix = (np.identity(3) * (1 + 2j))
        matrix_h = (np.identity(3) * (1 - 2j))
        msg_b = msg_a.multiply_deterministic(matrix, 1)
        info = matrix_h @ (np.identity(3) * 2) @ matrix
        w_mean = matrix_h @ ([[1], [3], [5]])
        msg_c = GaussianWeightedMeanInfoMessage(w_mean, info)
        self.assertEqual(msg_b, msg_c)

    def test_multiply_deterministic_exception(self):
        """
        Test: Exception when not inverse direction
        """
        msg_a = GaussianWeightedMeanInfoMessage([[1], [3], [5]], np.identity(3) * 2)
        matrix = (np.identity(3) * (1 + 2j))
        with self.assertRaises(NotImplementedError):
            msg_a.multiply_deterministic(matrix)

    def test_add(self):
        """
        Test: Exception when adding WeightedMeanInfoMessages
        """
        msg_a = GaussianWeightedMeanInfoMessage([[1], [3]], np.identity(2) * 3)
        msg_b = GaussianWeightedMeanInfoMessage([[3], [4]], np.identity(2) * 8)
        with self.assertRaises(TypeError):
            msg_a + msg_b

    def test_sub(self):
        msg_a = GaussianWeightedMeanInfoMessage([[1], [3]], np.identity(2) * 3)
        msg_b = GaussianWeightedMeanInfoMessage([[3], [4]], np.identity(2) * 8)
        with self.assertRaises(TypeError):
            msg_a - msg_b

    def test_neg(self):
        msg_a = GaussianWeightedMeanInfoMessage([[1], [3]], np.identity(2) * 3)
        msg_b = GaussianWeightedMeanInfoMessage([[-1], [-3]], np.identity(2) * 3)
        self.assertEqual(msg_b, msg_a.__neg__())

    def test_eq(self):
        msg_a = GaussianWeightedMeanInfoMessage([[2], [3]], np.identity(2) * (2 + 2j))
        msg_b = msg_a
        msg_c = GaussianMeanCovMessage([[2], [3]], np.identity(2) * (2 + 2j))
        self.assertTrue(msg_a == msg_b != msg_c)


if __name__ == '__main__':
    unittest.main()
