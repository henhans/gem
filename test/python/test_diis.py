'''DIIS extrapolation.

* On a linear map with an eigenvalue 1.5 (an unstable fixed point) damped
  linear mixing diverges for any mixing, while DIIS finds the fixed point.
* Coefficients sum to one and are real, so Hermitian states stay Hermitian.
* The history is capped at nvec.
'''

import unittest

import numpy as np

from gem.diis import DIIS


class test_diis(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(11)
        n = 6
        Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        self.A = Q @ np.diag([1.5, 0.6, 0.3, -0.2, -0.5, 0.1]) @ Q.T
        self.b = rng.normal(size=n)
        self.x_star = np.linalg.solve(np.eye(n) - self.A, self.b)
        self.G = lambda x: self.A @ x + self.b

    def test_mixing_diverges(self):
        x = np.zeros(6)
        for _ in range(60):
            x = 0.5 * self.G(x) + 0.5 * x
        self.assertGreater(np.abs(x - self.x_star).max(), 1e3)

    def test_unstable_fixed_point(self):
        diis = DIIS(8)
        x = np.zeros(6)
        for _ in range(20):
            x_new = self.G(x)
            diis.append(x_new - x, x_new)
            x = diis.solve()
        np.testing.assert_allclose(x, self.x_star, atol=1e-10)

    def test_coefficients_and_history(self):
        rng = np.random.default_rng(2)
        diis = DIIS(3)
        for _ in range(5):
            M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            H = M + M.conj().T
            diis.append(rng.normal(size=8) + 1j * rng.normal(size=8), H)
        self.assertEqual(len(diis.states), 3)
        c = diis.coefficients()
        self.assertTrue(np.isrealobj(c))
        self.assertAlmostEqual(c.sum(), 1.0, places=12)
        H = diis.solve()
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)
        diis.flush()
        with self.assertRaises(ValueError):
            diis.solve()


if __name__ == '__main__':
    unittest.main()
