import unittest
from dataclasses import replace

import numpy as np

from breathing import HeringBreuerDrive
from respiration import quiet_muscle_pressure
from respiration_single import SingleCompartment
import systemic


class ThesisMechanicsTests(unittest.TestCase):
    def test_passive_respiration_reference_and_ards_pleural_pressure(self):
        nominal = SingleCompartment()
        volume = nominal.equilibrium_volume()
        self.assertAlmostEqual(volume, 2.25)
        np.testing.assert_allclose(nominal.evaluate(volume, 0., 0.), [0., 0., -5.], atol=1e-12)
        ards = replace(nominal, lung_compliance_l_cmh2o=.06, chest_wall_compliance_l_cmh2o=.11,
                       lung_unstressed_l=.4, chest_wall_unstressed_l=1.)
        flow, alveolar, pleural = ards.evaluate(ards.equilibrium_volume(5.), 5., 0.)
        self.assertAlmostEqual(flow, 0.)
        self.assertAlmostEqual(alveolar, 5.)
        self.assertAlmostEqual(pleural, -1.7647058824)

    def test_tidal_volume_crossing_terminates_current_breath(self):
        drive = HeringBreuerDrive(lambda _: .5)
        drive.observe(0., 2.25)
        drive.observe(.9, 2.74)
        self.assertEqual(drive.events_s, [])
        self.assertAlmostEqual(drive(.9), quiet_muscle_pressure(.9))
        drive.observe(1., 2.75)
        self.assertEqual(drive.events_s, [1.])
        self.assertAlmostEqual(drive(1.), quiet_muscle_pressure(1.))
        self.assertGreater(drive(1.4), drive(1.))
        drive.observe(5., 2.3)
        self.assertEqual(drive.cutoff, None)
        self.assertAlmostEqual(drive(5.1), quiet_muscle_pressure(5.1))

    def test_two_bed_projection_preserves_group_flow_and_volume(self):
        model = systemic.NOMINAL
        pressure = 85.
        venous = np.array([7., 8.])
        flows = systemic.bed_flows(pressure, venous, 3., 75.)
        volume = np.asarray(model.venous_unstressed_ml) + venous * model.venous_compliance
        vp, vv, fi, fo, fv = systemic.tissue_inputs(pressure, volume, *flows)
        np.testing.assert_allclose(systemic.parallel_sum(vv), volume)
        np.testing.assert_allclose(systemic.parallel_sum(vp),
                                   np.asarray(model.peripheral_unstressed_ml) + pressure * np.array(model.peripheral_compliance))
        for before, projected in zip(flows, (fi, fo, fv)):
            np.testing.assert_allclose(systemic.parallel_sum(projected), before)
