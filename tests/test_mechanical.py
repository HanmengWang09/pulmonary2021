"""核对原文公式、闭合体液守恒和步长敏感性。"""

import unittest

import numpy as np

import heart
import systemic
from mechanical import COLUMNS, MechanicalModel


class MechanicalTests(unittest.TestCase):
    def test_pericardium_2021_equation(self):
        self.assertAlmostEqual(heart.pericardial_pressure([40, 40, 40, 40], -4.), -4.)
        self.assertAlmostEqual(heart.pericardial_pressure([80, 80, 80, 80], -4.),
                               -4. + .5 * np.expm1(.8))

    def test_septum_force_balance(self):
        for activation in (0., .2, 1.):
            _, septal, residual = heart.ventricular_pressures(110., 120., activation, heart.HeartParameters())
            self.assertGreater(septal, -110.)
            self.assertLess(septal, 120.)
            self.assertLess(abs(residual), 1e-9)

    def test_septum_uses_actual_chamber_gradient_when_valves_are_open(self):
        parameters = heart.HeartParameters()
        time_s = (0.5 - 0.075 / parameters.period_s) / 2.0
        pressure, flow, details = heart.evaluate(
            time_s, np.array([75., 80., 110., 120.]), -4., 15., 90., parameters
        )
        self.assertTrue((flow[2:] > 0).all())
        self.assertAlmostEqual(pressure[3] - pressure[2], 75.)
        self.assertAlmostEqual(details[1], 75. / parameters.septal_elastance)
        self.assertAlmostEqual(heart.septal_volume(-3., .4, 32.4),
                               -heart.septal_volume(3., .4, 32.4))

    def test_blood_conservation_and_refinement(self):
        model = MechanicalModel()
        derivative, _ = model.evaluate(.205, model.initial_state(), 0., -2.)
        self.assertLess(abs(model.blood_volume(derivative)), 1e-10)
        self.assertAlmostEqual(model.blood_volume(model.initial_state()), systemic.TOTAL_BLOOD_ML, places=9)
        names = COLUMNS.split(",")
        course = model.simulate(.85, lambda _: 0., lambda _: -1.)
        refined = model.simulate(.85, lambda _: 0., lambda _: -1., step_s=.00025, sample_every=20)
        self.assertTrue(np.isfinite(course).all())
        np.testing.assert_allclose(course[:, names.index("blood_volume")], 5300., atol=1e-8, rtol=0.)
        self.assertLess(np.max(abs(course[:, names.index("p_pa")] - refined[:, names.index("p_pa")])), .02)


if __name__ == "__main__":
    unittest.main()
