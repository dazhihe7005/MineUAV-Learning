"""Literal support metrics, fixed window identity and paired cohort integrity."""
import importlib
import importlib.util
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
from pathlib import Path
from joint_autonomous_consistency_training import load_autonomous
from latent_mpc_evaluation import realized_prefix_check


class SupportEvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('latent_mpc_support_evaluation'),
                             'missing paired support evaluation metrics')
        return importlib.import_module('latent_mpc_support_evaluation')

    def test_support_departure_tolerance_and_distribution(self):
        api=self.api()
        train=dict(mean=[0]*4,std=[.5]*4)
        support=dict(lower=[-.2]*4,upper=[.2]*4,tolerance=1e-7)
        a=np.array([[.2,0,0,0],[.3,0,0,0],[-.20000003,0,0,0]])
        out=api.support_metrics(a,train,support)
        self.assertAlmostEqual(out['outside_train_95_any_dimension_fraction'],1/3)
        self.assertAlmostEqual(out['outside_train_95_element_fraction'],1/12)
        self.assertEqual(out['outside_train_95_per_dimension_fraction'],[1/3,0,0,0])
        self.assertAlmostEqual(out['standardized_norm']['max'],.6)

    def test_projection_global_histogram_mean_and_quantile_error_bound(self):
        api=self.api(); accumulator=api.ProjectionAccumulator()
        mag=np.zeros((512,10)); mag[:256]=.3
        audit=dict(projection_magnitudes=mag,projected_elements=np.repeat((mag>0)[...,None],4,axis=2),anchor_projected_candidates=1)
        candidates=np.zeros((512,10,4))
        accumulator.update(audit,candidates,(-np.ones(4)*.2,np.ones(4)*.2))
        out=accumulator.summary()
        self.assertAlmostEqual(out['candidate_timestep_projection_fraction'],.5)
        self.assertAlmostEqual(out['projection_magnitude_all']['mean'],.15)
        self.assertLessEqual(abs(out['projection_magnitude_all']['p95']-.3),.0001+1e-12)
        self.assertEqual(out['anchor_projection_count'],1)
        self.assertEqual(out['candidate_timesteps'],5120)

    def test_fixed_windows_include_failures_and_never_cross_terminal_boundary(self):
        api=self.api()
        self.assertEqual(api.window_manifest('benchmark',0,12),[dict(split='benchmark',episode_index=0,decision_index=0,horizon=10)])
        self.assertEqual(api.window_manifest('holdout',2,40),[dict(split='holdout',episode_index=2,decision_index=k,horizon=10) for k in (0,25)])
        self.assertEqual(api.window_manifest('holdout',0,9),[])
        self.assertEqual(api.window_manifest('benchmark',3,100),[])

    def test_pairing_rejects_changed_noise_initial_state_and_missing_cohort(self):
        api=self.api()
        row=dict(env_seed=1,target=[1,0,1],target_index=0,first_observation_sha256='obs',first_latent_sha256='z',epsilon_hashes=['a','b'])
        u=[row]; s=[dict(row,epsilon_hashes=['a','b','c'])]
        self.assertEqual(api.verify_pairing(u,s)['matched_noise_decisions'],2)
        for key,value in [('epsilon_hashes',['a','x']),('first_latent_sha256','other'),('first_observation_sha256','other')]:
            with self.subTest(key=key),self.assertRaises(AssertionError): api.verify_pairing(u,[dict(row,**{key:value})])
        with self.assertRaises(AssertionError): api.verify_pairing(u,[])

    def test_future_observation_perturbation_cannot_change_realized_prefix_prediction(self):
        api=self.api()
        model,stats,_=load_autonomous(Path(__file__).parent/'models/joint_latent_world_model_v3_autonomous_consistency.pt')
        trace=dict(actions=np.zeros((12,4),np.float32),latents=np.zeros((12,64),np.float32),observations=np.zeros((13,7),np.float32))
        first=realized_prefix_check(model,stats,trace)
        changed=realized_prefix_check(model,stats,dict(trace,observations=np.ones((13,7),np.float32)*999))
        np.testing.assert_array_equal(first['predicted'],changed['predicted'])
        self.assertFalse(np.array_equal(first['actual'],changed['actual']))

    def test_projection_merge_matches_pooled_candidate_counts(self):
        api=self.api(); a=api.ProjectionAccumulator(); b=api.ProjectionAccumulator()
        mag=np.ones((512,10))*.2
        audit=dict(projection_magnitudes=mag,projected_elements=np.ones((512,10,4),bool),anchor_projected_candidates=2)
        a.update(audit,np.zeros((512,10,4)),None); b.update(audit,np.zeros((512,10,4)),None)
        a.merge(b)
        self.assertEqual(a.summary()['candidate_timesteps'],10240)
        self.assertEqual(a.summary()['anchor_projection_count'],4)
        self.assertAlmostEqual(a.summary()['projection_magnitude_all']['mean'],.2)

    def test_cache_rejects_changed_support_and_incomplete_cohort(self):
        self.assertIsNotNone(importlib.util.find_spec('run_latent_mpc_action_support'),'missing support control runner')
        from run_latent_mpc_action_support import validate_cache
        identity=dict(support_sha256='original',execution_hardware=dict(cpu='original'))
        row=dict(env_seed=1,target=[1,0,1],episode_steps=375,termination_reason='time_limit',success=False,episode_duration_s=15.)
        saved=dict(identity=identity,result=dict(episodes=[row]))
        validate_cache(saved,identity,[(1,[1,0,1])])
        for changed in (dict(identity,support_sha256='new'),dict(identity,execution_hardware=dict(cpu='other'))):
            with self.assertRaises(ValueError): validate_cache(saved,changed,[(1,[1,0,1])])
        with self.assertRaises(AssertionError): validate_cache(dict(saved,result=dict(episodes=[])),identity,[(1,[1,0,1])])

    def test_cache_binds_executing_model_forward_and_checkpoint_loader_sources(self):
        import run_latent_mpc_action_support as runner
        from run_latent_random_shooting_mpc import condition_identity
        captured=[]
        def identity_with_evidence(value,sources,hardware):
            result=condition_identity(value,sources,hardware); captured.append(result)
            return result
        with tempfile.TemporaryDirectory() as directory:
            ctx=dict(root=Path(directory),immutable={},checkpoint=Path(runner.__file__),
                     normalization_sha256='fixed',target_meta={'benchmark':dict(target_sha256='fixed')})
            with patch.object(runner,'context',return_value=ctx), \
                 patch.object(runner,'compute_support',return_value=dict(sha256='fixed')), \
                 patch.object(runner,'write_json'), \
                 patch.object(runner,'condition_identity',side_effect=identity_with_evidence), \
                 patch.object(runner,'evaluate_condition',side_effect=RuntimeError('stop before simulation')):
                ctx['train_stats']=dict(source_sha256='fixed')
                with self.assertRaisesRegex(RuntimeError,'stop before simulation'): runner.run(directory,False)
        identity=captured[0]; dependencies=identity['immutable_execution_dependencies']
        row=dict(env_seed=1,target=[1,0,1],episode_steps=375,termination_reason='time_limit',success=False,episode_duration_s=15.)
        saved=dict(identity=identity,result=dict(episodes=[row]))
        for name in ('explicit_latent_models.py','joint_autonomous_consistency_training.py'):
            path=str(Path(runner.__file__).with_name(name))
            with self.subTest(source=name):
                self.assertIn(path,dependencies,'executing model dependency missing from cache identity')
                changed=dict(identity,immutable_execution_dependencies=dict(dependencies,**{path:'changed forward/loader'}))
                with self.assertRaises(ValueError): runner.validate_cache(saved,changed,[(1,[1,0,1])])


if __name__=='__main__': unittest.main()
