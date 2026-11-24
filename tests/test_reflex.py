import unittest
from math import log

import numpy as np

from reflex import respiratory_derivative, respiratory_setpoints, septal_derivative


class ReflexTests(unittest.TestCase):
    def test_septal_threshold_and_decay(self):
        self.assertEqual(septal_derivative(0., 2.66), 0.)
        self.assertEqual(septal_derivative(2., 1.), -.25)
        self.assertAlmostEqual(septal_derivative(0., 3.66), 6.44 * log(2.) / 8.)

    def test_respiratory_effectors_are_independent(self):
        state = np.zeros(4)
        np.testing.assert_array_equal(respiratory_derivative(state, 3.7, 40.), state)
        self.assertEqual(respiratory_setpoints(state), (-5., 12.))
        peripheral = respiratory_derivative(state, 4.7, 40.)
        self.assertTrue((peripheral[[0, 2]] > 0).all())
        np.testing.assert_array_equal(peripheral[[1, 3]], 0.)


if __name__ == "__main__":
    unittest.main()
