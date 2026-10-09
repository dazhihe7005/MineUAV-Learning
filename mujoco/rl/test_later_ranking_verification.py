"""Fail closed on changed costs, eligibility, recorded actions or source identity."""
import importlib
import importlib.util
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from decision_fidelity_metrics import ranking

class VerificationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('later_ranking_verification'))
        return importlib.import_module('later_ranking_verification')

    def test_derived_metric_corruption_is_rejected(self):
        api=self.api(); p=np.arange(512,dtype=float)[::-1]; t=np.arange(512,dtype=float)
        good=ranking(p,t); api.verify_ranking(good,p,t)
        bad=dict(good,normalized_regret=0)
        with self.assertRaises(AssertionError): api.verify_ranking(bad,p,t)

    def test_candidate_snapshot_hash_corruption_is_rejected(self):
        api=self.api(); d=dict(candidate_sha256='a',snapshot_sha256='b',epsilon_sha256='c',
            planner_rng_before_sha256='d',planner_rng_after_sha256='e',decision_index=10)
        api.verify_decision_identity(d,d)
        with self.assertRaises(AssertionError): api.verify_decision_identity(dict(d,decision_index=11),d)
        with self.assertRaises(AssertionError): api.verify_decision_identity(dict(d,candidate_sha256='z'),d)

    def test_cache_identity_and_episode_swap_are_rejected(self):
        api=self.api(); saved=dict(identity={'source':'a'},episode=dict(condition='unconstrained',split='benchmark',index=0))
        api.verify_cache_identity(saved,{'source':'a'},('unconstrained','benchmark',0))
        with self.assertRaises(ValueError): api.verify_cache_identity(saved,{'source':'b'},('unconstrained','benchmark',0))
        with self.assertRaises(ValueError): api.verify_cache_identity(saved,{'source':'a'},('unconstrained','holdout',0))

    def test_S0_prior_candidate_and_snapshot_identity_are_required(self):
        api=self.api(); self.assertTrue(hasattr(api,'verify_prior_S0'))
        previous=dict(snapshot_sha256='original_state',reconstruction={'candidate_sha256':{'unconstrained':'original_candidates'}},
            conditions={'unconstrained':{'spearman':.6}})
        row=dict(condition='unconstrained',snapshot_sha256='original_state',candidate_sha256='original_candidates',metrics={'spearman':.6})
        api.verify_prior_S0(row,previous)
        with self.assertRaises(AssertionError): api.verify_prior_S0(dict(row,candidate_sha256='new_candidates'),previous)
        with self.assertRaises(AssertionError): api.verify_prior_S0(dict(row,snapshot_sha256='new_state'),previous)


class OutcomeVerificationTests(unittest.TestCase):
    """Real archived S3 outcomes must not be trusted as verification inputs."""
    @classmethod
    def setUpClass(cls):
        cls.api=importlib.import_module('later_ranking_verification')
        root=Path(__file__).resolve().parents[2]
        directory=root/'mujoco/reports'
        saved=json.loads((directory/'world_model_later_decision_ranking_seed0_parts/unconstrained_benchmark_003.json').read_text())
        cls.episode=saved['episode']
        cls.row=next(r for r in saved['records'] if r['stage']=='S3')
        geometry=json.loads((directory/'later_decision_train_distribution_seed0.json').read_text())
        cls.api.init_worker(root,geometry,saved['identity'])
        # Assert the unmodified real fixture passes before testing corruption.
        cls.api.verify_episode((cls.episode,[cls.row]))

    @classmethod
    def tearDownClass(cls):
        cls.api.ENV.close()

    def test_scripted_task_utility_corruption_is_rejected(self):
        bad=copy.deepcopy(self.row)
        bad['scripted']['distance_reduction']+=123
        with self.assertRaises(AssertionError): self.api.verify_episode((self.episode,[bad]))

    def test_candidate_termination_map_corruption_is_rejected(self):
        bad=copy.deepcopy(self.row)
        bad['metrics']['termination_reason_counts']={'success':512}
        with self.assertRaises(AssertionError): self.api.verify_episode((self.episode,[bad]))

    def test_candidate_early_termination_corruption_is_rejected(self):
        bad=copy.deepcopy(self.row)
        bad['candidate_early_termination_count']=512
        with self.assertRaises(AssertionError): self.api.verify_episode((self.episode,[bad]))

    def test_state_termination_metadata_corruption_is_rejected(self):
        bad=copy.deepcopy(self.row)
        bad['termination_reason']='success'
        with self.assertRaises(AssertionError): self.api.verify_episode((self.episode,[bad]))

    def test_episode_outcome_metadata_corruption_is_rejected(self):
        bad=copy.deepcopy(self.episode)
        bad['success']=True
        bad['termination_reason']='success'
        with self.assertRaises(AssertionError): self.api.verify_episode((bad,[self.row]))

if __name__=='__main__': unittest.main()
