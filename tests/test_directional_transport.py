import unittest

import numpy as np

import blood_gas
from breathing import AdaptiveBreathing
import closed_loop


class DirectionalTransportTests(unittest.TestCase):
    def test_pulmonary_gas_inventory_closes_during_reversal(self):
        from pulmonary_transport import evaluate

        volumes = np.array([16., 70., 80., 160., 130., 100., 25.])
        fractions = np.tile([.14, .055], 2)
        venous = blood_gas.dissociation((40., 46.))[0]
        arterial = blood_gas.dissociation((100., 40.))[0]
        pools = np.vstack(([venous] * 2, [arterial] * 4))
        inventories = volumes[[0, 1, 3, 4, 5, 6], None] * pools
        # 毛细血管和肺静脉同时逆流，另两条边仍正向。
        flow = np.array([35., 23., -4., -2., 1., 0., 35., .4])
        result = evaluate(fractions, inventories, volumes, 23. - (-4.), 2.25, -.21, flow,
                          40., venous, arterial, .15, blood_gas.INSPIRED)
        derivative, balances = result
        self.assertTrue(np.isfinite(derivative).all())
        np.testing.assert_allclose(balances, 0., atol=1e-12, rtol=0.)
        # V_pv 的 O₂ 通量来自左房与分流，并以肺静脉浓度流向毛细血管。
        pv_flux = derivative[8:10]
        np.testing.assert_allclose(pv_flux, .4 * venous + 2. * arterial - 4. * arterial)

    def test_blood_gases_change_the_inspiratory_cutoff_threshold(self):
        law = lambda po2, pco2: .25 + .01 * (pco2 - 40.)
        drive = AdaptiveBreathing(.53, threshold_at=law)
        drive.observe(0., 2., -5., 12., arterial_gases=(100., 40.))
        drive.observe(.5, 2.3, -5., 12., arterial_gases=(100., 50.))
        self.assertIsNone(drive.cutoff)
        drive.observe(.6, 2.3, -5., 12., arterial_gases=(100., 40.))
        self.assertEqual(drive.events_s, [.6])

    def test_inspired_composition_reaches_the_closed_loop(self):
        from gas_transport import GasTransport

        norm, _ = closed_loop.simulate(1., sample_every=100)
        challenge, _ = closed_loop.simulate(1., sample_every=100, gas_model=GasTransport(inspired=(.16, .03)))
        columns = {name: index for index, name in enumerate(closed_loop.OUTPUT_FIELDS)}
        self.assertLess(challenge[-1, columns["fd_o2"]], norm[-1, columns["fd_o2"]])
        self.assertGreater(challenge[-1, columns["fd_co2"]], norm[-1, columns["fd_co2"]])
        for name in ("pulmonary_o2_balance", "pulmonary_co2_balance"):
            self.assertLess(np.max(np.abs(challenge[:, columns[name]])), 1e-11)

    def test_closed_loop_supplies_aortic_gases_to_the_cutoff(self):
        seen = []

        def threshold(po2, pco2):
            seen.append((po2, pco2))
            return .55 if abs(po2 - 100.) <= .01 else .25

        _, events = closed_loop.simulate(2., tidal_threshold_at=threshold, sample_every=100)
        np.testing.assert_allclose(seen[0], (100., 40.), rtol=1e-12)
        self.assertGreater(np.ptp(np.array(seen)[:, 0]), .01)
        self.assertEqual(len(events), 1)
        self.assertLess(events[0], 1.875)


if __name__ == "__main__":
    unittest.main()
