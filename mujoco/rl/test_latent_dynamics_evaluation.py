import unittest

import numpy as np

try:
    import latent_dynamics_evaluation as evaluation
except ImportError:
    evaluation = None


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(evaluation, 'delta evaluation is not implemented')

    def test_metrics_have_independently_checked_values(self):
        truth = np.array([[0.] * 7, [2.] * 7])
        result = evaluation.metrics(np.zeros_like(truth), truth)
        self.assertAlmostEqual(result['overall']['rmse'], 2 ** .5)
        self.assertEqual(result['overall']['mae'], 1.)
        self.assertEqual(result['overall']['r2'], -1.)
        self.assertEqual(result['per_dimension']['vx']['rmse'], 2 ** .5)

    def test_constant_target_r2_is_null_not_nan(self):
        result = evaluation.metrics(np.ones((2, 7)), np.zeros((2, 7)))
        self.assertIsNone(result['overall']['r2'])

    def test_gap_requires_oracle_better_than_markov(self):
        self.assertEqual(evaluation.gap_closure(2., 1.5, 1.), .5)
        self.assertIsNone(evaluation.gap_closure(1., 1., 1.))
        self.assertIsNone(evaluation.gap_closure(1., 1., 2.))

    def test_region_boundaries_are_unambiguous(self):
        mask = evaluation.distance_masks(np.array([.099, .1, .199, .2, .499, .5]))
        self.assertEqual([int(x.sum()) for x in mask.values()], [1, 2, 2, 1])
        np.testing.assert_array_equal(sum(mask.values()), np.ones(6))

    def test_frozen_linear_probe_fits_only_provided_train_rows(self):
        z = np.arange(10.).reshape(-1, 1)
        pi = np.column_stack([2 * z[:, 0] + 1, -z[:, 0], np.zeros(10)])
        probe = evaluation.fit_linear_probe(z, pi)
        out = evaluation.apply_linear_probe(np.array([[20.], [30.]]), probe)
        np.testing.assert_allclose(out, [[41, -20, 0], [61, -30, 0]], atol=1e-9)


if __name__ == '__main__':
    unittest.main()
