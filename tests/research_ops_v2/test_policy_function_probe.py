"""No tensor import: fixed diagnostic scope and trusted released artifacts."""
import copy
import importlib
import importlib.util
import unittest


class PolicyFunctionProbeTests(unittest.TestCase):
    def module(self):
        name = 'research_regression.authorized_rl.policy_function_probe'
        self.assertIsNotNone(importlib.util.find_spec(name), 'missing fixed no-learning R29 interface')
        return importlib.import_module(name)

    def test_fixed_grid_has_no_duplicate_states_or_hidden_yaw(self):
        rows = self.module().probe_grid()
        self.assertEqual(len(rows), 79)
        self.assertEqual(len({tuple(v['state']) for v in rows}), 79)
        self.assertTrue(all(len(v['state']) == 7 and v['state'][6] == 0 for v in rows))
        self.assertEqual(rows[0]['state'], [-1., 0., 0., -.2, 0., 0., 0.])
        self.assertTrue(all(v['source'] == 'COUNTERFACTUAL_POLICY_FUNCTION_PROBE' for v in rows))

    def test_fixed_scope_is_not_new_evaluation_or_training(self):
        p = self.module().configuration()
        self.assertEqual(p['checkpoint_steps'], [0, 32768, 65536])
        self.assertEqual(p['model_input_rows'], 237)
        for k in ('training_timesteps', 'optimizer_updates', 'physics_ticks', 'new_episodes', 'test_episodes'):
            self.assertEqual(p[k], 0)

    def test_exact_published_checkpoint_membership_not_arbitrary_path(self):
        m = self.module(); refs = m.references()
        self.assertEqual([v['step'] for v in refs['models']], [0, 32768, 65536])
        self.assertEqual(refs['models'][2]['checkpoint_sha256'], '0bcbdca60b131b749ed3a3b0101fbd149e47b58244f99e13e92c0916da5dbd0a')
        self.assertEqual(refs['R28_scientific_result'], 'FAIL')
        with self.assertRaises(ValueError):
            m.require_model({**refs['models'][0], 'checkpoint_path': '/tmp/untrusted.zip'}, refs)

    def test_bad_reference_digest_is_rejected_before_deserialization(self):
        m = self.module(); refs = m.references(); wrong = copy.deepcopy(refs['models'][0])
        wrong['checkpoint_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            m.require_model(wrong, refs)

    def test_assessment_requires_complete_finite_ordered_rows(self):
        m = self.module()
        with self.assertRaises(ValueError):
            m.assess([])
        rows = [{'step': s, 'grid_index': i, 'raw_mean': [0., 0., 0., 0.],
                 'clipped_action': [0., 0., 0., 0.]} for s in (0, 32768, 65536) for i in range(79)]
        value = m.assess(rows)
        self.assertEqual(value['claim'], 'FUNCTION_PROBE_ONLY_NOT_CLOSED_LOOP_OR_CAUSAL_STABILITY')
        for change in ('duplicate', 'nonfinite', 'wrong_clip'):
            bad = copy.deepcopy(rows)
            if change == 'duplicate': bad[-1] = copy.deepcopy(bad[-2])
            if change == 'nonfinite': bad[-1]['raw_mean'][0] = float('nan')
            if change == 'wrong_clip': bad[-1]['clipped_action'][0] = .1
            with self.assertRaises(ValueError): m.assess(bad)


if __name__ == '__main__':
    unittest.main()
