import unittest

import numpy as np

from blood_gas import (DRY_PRESSURE_MMHG, INSPIRED, METABOLISM_ML_S, TISSUE_ML, LungInputs,
                       dissociation, lung_exchange, tissue_exchange)


class BloodGasTests(unittest.TestCase):
    def test_dissociation_jacobian_and_cross_effect(self):
        pressures = np.array([100., 40.])
        concentration, jacobian = dissociation(pressures)
        epsilon = 1e-4
        columns = np.eye(2) * epsilon
        difference = [(dissociation(pressures + shift)[0] - dissociation(pressures - shift)[0])
                      / (2 * epsilon) for shift in columns]
        np.testing.assert_allclose(jacobian, np.array(difference).T, rtol=1e-7)
        self.assertTrue((concentration > 0).all())
        self.assertTrue((np.diagonal(jacobian) > 0).all())
        self.assertLess(jacobian[0, 1], 0.)
        self.assertLess(jacobian[1, 0], 0.)

    def test_lung_equilibrium_mixing(self):
        f_alveolar = np.asarray(INSPIRED)
        inputs = LungInputs(.1, 2.25, .3, .2, 80., 100., 78.64, 1.36, .017)
        venous = dissociation(f_alveolar * DRY_PRESSURE_MMHG)[0]
        derivative, arterial = lung_exchange(np.r_[f_alveolar, f_alveolar], venous, inputs)
        np.testing.assert_allclose(derivative, 0., atol=1e-12)
        np.testing.assert_allclose(arterial, venous, atol=1e-12)

    def test_metabolic_change_at_equal_concentrations(self):
        concentration = np.array([.15, .48])
        pools = np.tile(concentration, (5, 1))
        a = np.ones(5) * 100
        d_tissue, d_veins, d_central = tissue_exchange(
            pools, pools, concentration, concentration, peripheral_volume_ml=a,
            venous_volume_ml=a, central_volume_ml=150., peripheral_inflow_ml_s=a,
            peripheral_outflow_ml_s=a, venous_flow_ml_s=a
        )
        np.testing.assert_allclose(d_tissue * (TISSUE_ML + a)[:, None], METABOLISM_ML_S)
        np.testing.assert_array_equal(d_veins, 0.)
        np.testing.assert_array_equal(d_central, 0.)


if __name__ == "__main__":
    unittest.main()
