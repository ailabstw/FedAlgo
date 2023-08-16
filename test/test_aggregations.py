import unittest
import gwasprs
import numpy as np
import jax.numpy as jnp

class SumUpTestCase(unittest.TestCase):

    def setUp(self):
        self.A = np.random.rand(2, 3, 4)
        self.B = np.random.rand(2, 3, 4)
        self.C = np.random.rand(2, 3, 4)

    def tearDown(self):
        self.A = None
        self.B = None
        self.C = None

    def test_sum_of_numpy_arrays(self):
        result = gwasprs.aggregations.SumUp()(self.A, self.B, self.C)
        ans = self.A + self.B + self.C
        np.testing.assert_array_almost_equal(ans, result)

    def test_sum_of_jax_arrays(self):
        result = gwasprs.aggregations.SumUp()(jnp.array(self.A), jnp.array(self.B), jnp.array(self.C))
        ans = jnp.array(self.A) + jnp.array(self.B) + jnp.array(self.C)
        np.testing.assert_array_almost_equal(ans, result)

    def test_sum_of_numbers(self):
        result = gwasprs.aggregations.SumUp()(1, 2.5, 3.4)
        ans = 6.9
        self.assertEqual(ans, result)

    def test_sum_of_list_of_numpy_arrays(self):
        result = gwasprs.aggregations.SumUp()([self.A, self.B], [self.C, self.C])
        ans = [self.A + self.C, self.B + self.C]
        np.testing.assert_array_almost_equal(ans, result)

    def test_sum_of_list_of_jax_arrays(self):
        result = gwasprs.aggregations.SumUp()([jnp.array(self.A), jnp.array(self.B)], [jnp.array(self.C), jnp.array(self.C)])
        ans = [jnp.array(self.A + self.C), jnp.array(self.B + self.C)]
        np.testing.assert_array_almost_equal(ans, result)
