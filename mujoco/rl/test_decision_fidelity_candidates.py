"""Reconstruct actual saved first decisions; no new generator definition."""
import importlib
import importlib.util
import unittest
from pathlib import Path
import json
import numpy as np
from run_latent_random_shooting_mpc import context,planner_seed
from latent_mpc_core import sample_candidates
from ppo_pi_env import MineUAVPIEnv


class CandidateTests(unittest.TestCase):
    def test_previous_first_decision_cost_action_noise_and_sampler_reproduce(self):
        self.assertIsNotNone(importlib.util.find_spec('decision_fidelity_candidates'),'candidate reconstruction missing')
        api=importlib.import_module('decision_fidelity_candidates'); root=Path(__file__).resolve().parents[2]
        ctx=context(root)
        ctx['previous_report']=json.loads((root/'mujoco/reports/latent_mpc_train_supported_action_control_seed0.json').read_text())
        ctx['support']=ctx['previous_report']['train_support']; env=MineUAVPIEnv(reward_version='v2')
        try:
            row=api.reconstruct(ctx,env,'benchmark',0)
            self.assertTrue(row['evidence']['previous_first_decision_exact'])
            expected=sample_candidates(np.zeros(4),ctx['stats']['action']['std'],-np.ones(4),np.ones(4),np.random.default_rng(planner_seed('benchmark',0)))
            np.testing.assert_array_equal(row['conditions']['unconstrained']['candidates'],expected)
            self.assertEqual(row['conditions']['train_supported']['candidates'].shape,(512,10,4))
            self.assertEqual(row['evidence']['raw_epsilon_sha256'][0],row['evidence']['raw_epsilon_sha256'][1])
            again=api.reconstruct(ctx,env,'benchmark',0)
            for c in row['conditions']:
                np.testing.assert_array_equal(row['conditions'][c]['candidates'],again['conditions'][c]['candidates'])
                self.assertEqual(row['evidence']['candidate_sha256'][c],again['evidence']['candidate_sha256'][c])
        finally: env.close()


if __name__=='__main__': unittest.main()
