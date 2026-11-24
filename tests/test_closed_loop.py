from dataclasses import replace
import unittest

import numpy as np

import autonomic
import blood_gas
import closed_loop


class ClosedLoopTests(unittest.TestCase):
    def test_dissociation_inverse_includes_shunt_mixture(self):
        for expected in ((20., 60.), (40., 46.), (100., 40.), (200., 30.)):
            np.testing.assert_allclose(
                blood_gas.partial_pressures(blood_gas.dissociation(expected)[0]), expected
            )
        venous = blood_gas.dissociation((40., 46.))[0]
        alveolar = blood_gas.dissociation((100., 40.))[0]
        mixed = .017 * venous + .983 * alveolar
        np.testing.assert_allclose(blood_gas.dissociation(blood_gas.partial_pressures(mixed))[0], mixed)

    def test_reversing_capillary_branch_does_not_subtract_gas_from_shunt(self):
        inputs = blood_gas.LungInputs(.15, 2.25, 0., 0., 40., 65., -.158, .197, .017)
        venous = blood_gas.dissociation((40., 46.))[0]
        previous = blood_gas.dissociation((100., 40.))[0]
        fractions = np.tile(np.asarray([100., 40.]) / blood_gas.DRY_PRESSURE_MMHG, 2)
        _, arterial = blood_gas.lung_exchange(fractions, venous, inputs, last_arterial=previous)
        np.testing.assert_allclose(arterial, venous)
        _, stalled = blood_gas.lung_exchange(fractions, venous, replace(inputs, shunt_outflow_ml_s=0.),
                                             last_arterial=previous)
        np.testing.assert_allclose(stalled, previous)

    def test_pressure_reflex_direction(self):
        low = autonomic.initial_state(70., (100., 40.), .53)
        high = autonomic.initial_state(105., (100., 40.), .53)
        self.assertGreater(low[10], high[10])  # 左室主动弹性
        self.assertGreater(low[12], high[12])  # 内脏阻力
        self.assertLess(low[14], high[14])     # 静脉无应力容积
        self.assertLess(autonomic.heart_period(low), autonomic.heart_period(high))

    def test_delay_uses_only_accepted_history(self):
        history = closed_loop.DelayHistory(np.array([2.]), 3, .1)
        self.assertEqual(history.sample(.05, .2, 0), 2.)
        history.commit(0, np.array([4.]))
        history.commit(1, np.array([8.]))
        self.assertAlmostEqual(history.sample(.25, .2, 0), 6.)
        with self.assertRaises(ValueError):
            history.sample(.32, .2, 0)

    def test_rk4_conserves_blood_and_advances_heart_phase(self):
        rows, _ = closed_loop.simulate(.3, sample_every=10)
        columns = {name: index for index, name in enumerate(closed_loop.OUTPUT_FIELDS)}
        np.testing.assert_allclose(rows[:, columns["blood_volume"]], 5300., atol=1e-8, rtol=0.)
        self.assertTrue(np.all(np.diff(rows[:, columns["heart_cycle"]]) > 0.))
        self.assertTrue(np.isfinite(rows).all())


if __name__ == "__main__":
    unittest.main()
