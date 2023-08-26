# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import unittest
from ime_fgs.messages import GaussianWeightedMeanInfoMessage, GaussianMeanCovMessage, GaussianMixtureWeightedMeanInfoMessage, GaussianMixtureMeanCovMessage


class ConversionTests(unittest.TestCase):
    """Test Convert"""

    def test_convert(self):
        """
        Does message equalization work for simple, scalar messages in weighted mean and information matrix
        parameterization?
        """

        msg_gwmi = GaussianWeightedMeanInfoMessage([[0.5]], [[2]])
        msg_gmc  = GaussianMeanCovMessage([[0.25]], [[0.5]])

        msg_gmwmi = GaussianMixtureWeightedMeanInfoMessage([[1]], [[[0.5]]], [[[2]]])
        msg_gmmc  = GaussianMixtureMeanCovMessage([[1]], [[[0.25]]], [[[0.5]]])


        msg_gmwmi2gwmi  = msg_gmwmi.convert(GaussianWeightedMeanInfoMessage)
        msg_gmwmi2gmc   = msg_gmwmi.convert(GaussianMeanCovMessage)
        msg_gmwmi2gmwmi = msg_gmwmi.convert(GaussianMixtureWeightedMeanInfoMessage)
        msg_gmwmi2gmmc  = msg_gmwmi.convert(GaussianMixtureMeanCovMessage)

        self.assertEqual(msg_gwmi,  msg_gmwmi2gwmi)
        self.assertEqual(msg_gmc,   msg_gmwmi2gmc)
        self.assertEqual(msg_gmwmi, msg_gmwmi2gmwmi)
        self.assertEqual(msg_gmmc,  msg_gmwmi2gmmc)

        msg_gmmc2gwmi  = msg_gmmc.convert(GaussianWeightedMeanInfoMessage)
        msg_gmmc2gmc   = msg_gmmc.convert(GaussianMeanCovMessage)
        msg_gmmc2gmwmi = msg_gmmc.convert(GaussianMixtureWeightedMeanInfoMessage)
        msg_gmmc2gmmc  = msg_gmmc.convert(GaussianMixtureMeanCovMessage)

        self.assertEqual(msg_gwmi,  msg_gmmc2gwmi)
        self.assertEqual(msg_gmc,   msg_gmmc2gmc)
        self.assertEqual(msg_gmwmi, msg_gmmc2gmwmi)
        self.assertEqual(msg_gmmc,  msg_gmmc2gmmc)

        msg_gwmi2gwmi  = msg_gwmi.convert(GaussianWeightedMeanInfoMessage)
        msg_gwmi2gmc   = msg_gwmi.convert(GaussianMeanCovMessage)
        msg_gwmi2gmwmi = msg_gwmi.convert(GaussianMixtureWeightedMeanInfoMessage)
        msg_gwmi2gmmc  = msg_gwmi.convert(GaussianMixtureMeanCovMessage)

        self.assertEqual(msg_gwmi,  msg_gwmi2gwmi)
        self.assertEqual(msg_gmc,   msg_gwmi2gmc)
        self.assertEqual(msg_gmwmi, msg_gwmi2gmwmi)
        self.assertEqual(msg_gmmc,  msg_gwmi2gmmc)

        msg_gmc2gwmi  = msg_gmc.convert(GaussianWeightedMeanInfoMessage)
        msg_gmc2gmc   = msg_gmc.convert(GaussianMeanCovMessage)
        msg_gmc2gmwmi = msg_gmc.convert(GaussianMixtureWeightedMeanInfoMessage)
        msg_gmc2gmmc  = msg_gmc.convert(GaussianMixtureMeanCovMessage)

        self.assertEqual(msg_gwmi,  msg_gmc2gwmi)
        self.assertEqual(msg_gmc,   msg_gmc2gmc)
        self.assertEqual(msg_gmwmi, msg_gmc2gmwmi)
        self.assertEqual(msg_gmmc,  msg_gmc2gmmc)

if __name__ == '__main__':
    unittest.main()
