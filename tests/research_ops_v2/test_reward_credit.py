"""R19 actual-data accounting math, no ML or simulator imports."""
import importlib,importlib.util,unittest,tempfile,math
from pathlib import Path
from research_ops.storage import OpsError


class RewardCreditTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.reward_credit'),
                             'missing R19 fixed credit audit')
        return importlib.import_module('research_regression.authorized_rl.reward_credit')

    def row(self,reward):
        parts={k:0. for k in ('progress','action','brake','success','collision','timeout_or_time_cost','physical_failure','sensor_failure')}
        parts['progress']=reward;parts['total']=reward
        return {'reward':reward,'reward_breakdown':parts}

    def test_discounted_components_match_hand_calculation_without_counting_total_twice(self):
        m=self.module();v=m.discounted_components([self.row(1),self.row(2),self.row(3)],.99)
        self.assertAlmostEqual(v['return'],1+1.98+2.9403);self.assertAlmostEqual(v['components']['progress'],5.9203)
        self.assertNotIn('total',v['components'])

    def test_invalid_gamma_nonfinite_and_inconsistent_breakdown_fail_closed(self):
        m=self.module()
        for g in (True,0,-.1,float('nan'),1.1,.9):
            with self.assertRaises(ValueError):m.discounted_components([self.row(1)],g)
        for r in (self.row(float('inf')),{**self.row(1),'reward':2}):
            with self.assertRaises(ValueError):m.discounted_components([r],.99)

    def test_late_success_bonus_is_discounted_at_index_minus_one(self):
        m=self.module();rows=[self.row(0) for _ in range(3)]
        rows[-1]['reward']=rows[-1]['reward_breakdown']['total']=10
        rows[-1]['reward_breakdown']['success']=10
        self.assertAlmostEqual(m.discounted_components(rows,.99)['components']['success'],9.801)

    def test_closed_episode_steps_plus_tail_are_not_fabricated_terminal_samples(self):
        m=self.module();rows=[{'base_scenario_identity_sha256':'a','distance_stage':'ORIGINAL','episode_policy_steps':10,
            'terminal_class':'time_limit','episode_return':-2.1,'reward_components':{**self.row(0)['reward_breakdown'],'timeout_or_time_cost':-2.1,'total':-2.1}}]
        v=m.episode_exposure(rows,{'a':'empty'},tail_steps=5,tail_base='a',expected_policy_steps=15)
        self.assertEqual(v['by_layout']['empty']['policy_steps_with_tail'],15)
        self.assertEqual(v['by_layout']['empty']['ended_episodes'],1);self.assertEqual(v['unfinished_tail_steps'],5)

    def test_unknown_scene_or_inconsistent_total_steps_rejected(self):
        m=self.module()
        with self.assertRaises(ValueError):m.episode_exposure([],{'a':'empty'},tail_steps=1,tail_base='validation',expected_policy_steps=1)
        with self.assertRaises(ValueError):m.episode_exposure([],{'a':'empty'},tail_steps=1,tail_base='a',expected_policy_steps=2)

    def test_missing_frozen_history_cannot_create_output(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.references(Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_fixed_scope_is_mathematical_audit_not_new_training(self):
        c=self.module().configuration()
        self.assertEqual(c['gammas'],[.99,.995,.999,1.]);self.assertEqual(c['training_timesteps'],0)
        for k in ('optimizer_updates','physics_steps','new_episodes','test_episodes','model_inference_calls'):self.assertEqual(c[k],0)

    def test_actual_scheduler_task_identity_strings_not_fabricated_dicts(self):
        m=self.module();v=m.reset_exposure(['short','a','b','a'],{'a':'empty','b':'box'},closed_episodes=2,tail_base='a')
        self.assertEqual(v['original_reset_counts'],{'box':1,'empty':2});self.assertEqual(v['original_resets'],3)

    def test_missing_tail_reset_or_mismatched_episode_count_rejected(self):
        m=self.module()
        for ids,count,tail in ((['a'],1,'a'),(['a','b'],1,'a')):
            with self.assertRaises(ValueError):m.reset_exposure(ids,{'a':'empty','b':'box'},closed_episodes=count,tail_base=tail)
