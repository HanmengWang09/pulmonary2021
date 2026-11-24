"""复核论文参数、解析稳态及基于通量的守恒。"""

import unittest
from dataclasses import replace

import numpy as np

from pulmonary import Boundary, NOMINAL, evaluate, example_boundary, resistances, simulate, steady_state
import respiration


class PulmonaryPaperTests(unittest.TestCase):
    def setUp(self):
        self.boundary = Boundary(85., 4.35, -3.68, 0., 2.25)

    def test_table_and_nominal_steady_solution(self):
        self.assertEqual(NOMINAL.compliance_ml_mmhg, (.76, 1.93, 3.87, 25.37))
        self.assertEqual(NOMINAL.unstressed_volume_ml, (0., 43.30, 64.94, 105.6))
        self.assertEqual((NOMINAL.r_pa, NOMINAL.r_pv, NOMINAL.r_periphery, NOMINAL.l_pa),
                         (.023, .0056, .0894, .00018))
        state = steady_state(self.boundary)
        derivative, pressure, flow = evaluate(state, self.boundary)
        np.testing.assert_allclose(derivative, 0., atol=1e-9)
        np.testing.assert_allclose(pressure, [14.38, 12.425, 8.6255, 4.826], atol=1e-10)
        np.testing.assert_allclose(flow, [85., 83.555, 83.555, 1.445, 85.], atol=1e-10)

    def test_external_pressures_lung_volume_and_reverse_flow(self):
        state = steady_state(self.boundary)
        change = replace(self.boundary, pleural_pressure_mmhg=-2.68,
                         alveolar_pressure_mmhg=2.0, left_atrium_pressure_mmhg=7.)
        derivative, pressure, flow = evaluate(state, change)
        old_pressure = evaluate(state, self.boundary)[1]
        np.testing.assert_allclose(pressure - old_pressure, [1., 1., 2., 1.])
        self.assertLess(flow[-1], 0.)
        self.assertAlmostEqual(derivative[:4].sum(), 85. - flow[-1])
        r_pal, r_pc = resistances(4.5)
        self.assertAlmostEqual(r_pal, 4 * r_pc)

    def test_dynamic_volume_conservation_and_step_refinement(self):
        original = list(simulate(example_boundary, 0.85))
        fine = list(simulate(example_boundary, 0.85, step_s=0.00025))
        initial_volume = original[0][1][:4].sum()
        error = [abs(state[:4].sum() - initial_volume - net)
                 for _, state, _, _, net in original]
        self.assertLess(max(error), 1e-9)
        np.testing.assert_allclose(original[-1][2], fine[-1][2], atol=1e-7, rtol=0.)


class RespirationPaperTests(unittest.TestCase):
    def test_nominal_exhalation_and_flow_conservation(self):
        parameters = respiration.ALBANESE_2016
        self.assertAlmostEqual(sum(parameters.resistance_cmh2o_s_l), 1.7459)
        state = respiration.end_expiratory_state()
        derivative, pressure, pleural, _ = respiration.evaluate(state, 0.0, 0.0)
        np.testing.assert_allclose(pressure, 0.0, atol=1e-12)
        np.testing.assert_allclose(derivative, 0.0, atol=1e-12)
        self.assertEqual(pleural, -5.0)
        muscle = respiration.quiet_muscle_pressure(1.875)
        self.assertAlmostEqual(muscle, -5.0)
        derivative, _, _, flow = respiration.evaluate(state, 3.0, muscle)
        self.assertAlmostEqual(derivative[:4].sum(), flow[0])
        self.assertAlmostEqual(derivative[1:4].sum(), derivative[4])


if __name__ == "__main__":
    unittest.main()
