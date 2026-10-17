"""Real pure-statistics contracts; no framework/model/physics imports."""
import copy
import importlib
import importlib.util
import math
import unittest


def tasks(split):
    return [dict(identity_sha256=(str(i) if split=='train' else str(i+4))*64,
                 split=split,layout=layout,start_yaw_rad=yaw)
            for i,(layout,yaw) in enumerate((('empty',math.radians(179)),('front_box',0.)))]


class DistributionAuditTests(unittest.TestCase):
    def module(self):
        name='research_regression.authorized_rl.distribution_audit'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing closed read-only distribution analysis')
        return importlib.import_module(name)

    def test_circular_yaw_distance_wraps_and_rejects_invalid_values(self):
        m=self.module()
        self.assertAlmostEqual(m.yaw_distance(math.radians(179),math.radians(-179)),math.radians(2))
        self.assertAlmostEqual(m.yaw_distance(0,math.pi),math.pi)
        for bad in (True,float('nan'),float('inf'),'0'):
            with self.assertRaises(ValueError):m.yaw_distance(bad,0.)

    def test_nearest_yaw_uses_same_layout_as_well_as_global_train(self):
        m=self.module();tr=tasks('train');va=tasks('validation');va[0]['start_yaw_rad']=math.radians(-179)
        va[1]['start_yaw_rad']=math.radians(178)
        out=m.heading_coverage(tr,va)
        self.assertIn('nearest_same_layout_yaw_gap_rad',out[0])
        self.assertIn('nearest_any_train_yaw_gap_rad',out[0])
        self.assertAlmostEqual(out[0]['nearest_same_layout_yaw_gap_rad'],math.radians(2))
        self.assertAlmostEqual(out[0]['nearest_same_layout_yaw_gap_deg'],2.)
        self.assertAlmostEqual(out[1]['nearest_any_train_yaw_gap_deg'],1.)
        self.assertAlmostEqual(out[1]['nearest_same_layout_yaw_gap_deg'],178.)

    def test_split_identity_duplicate_and_wrong_split_fail_closed(self):
        m=self.module();tr=tasks('train');va=tasks('validation')
        for changed in ('overlap','duplicate','test'):
            a,b=copy.deepcopy(tr),copy.deepcopy(va)
            if changed=='overlap':b[0]['identity_sha256']=a[0]['identity_sha256']
            elif changed=='duplicate':a.append(copy.deepcopy(a[0]))
            else:b[0]['split']='test'
            with self.assertRaises(ValueError):m.heading_coverage(a,b)

    def row(self,step,n,kind):
        return dict(base_scenario_identity_sha256='0'*64,scenario_identity_sha256='0'*64,
            distance_stage='ORIGINAL',policy_end_step=step,episode_policy_steps=n,
            terminal_class=kind,outcome=kind,episode_return=1.,minimum_goal_error_m=.08,
            final_target_distance_m=.09,near_goal_steps=5 if kind=='success' else 0,
            goal_stop_diagnostics=dict(max_consecutive_joint_samples=5 if kind=='success' else 0))

    def test_grouping_keeps_episode_and_policy_step_denominators_distinct(self):
        m=self.module();rows=[self.row(20,20,'success'),self.row(120,100,'collision')]
        out=m.original_train_groups(rows,tasks('train'));empty=out['by_layout']['empty']
        self.assertEqual((empty['episodes'],empty['policy_steps'],empty['terminal_counts']['success']), (2,120,1))
        self.assertEqual(empty['episodes_with_sampled_goal_entry'],1)
        self.assertEqual(empty['mean_final_goal_error_m'],.09)
        self.assertEqual(out['by_identity']['1'*64]['episodes'],0)

    def test_short_foreign_unknown_contradictory_and_invalid_rows_rejected(self):
        m=self.module()
        for patch in ({'distance_stage':'SHORT'},{'base_scenario_identity_sha256':'4'*64},
                      {'terminal_class':'unknown'},{'outcome':'success','terminal_class':'collision'},
                      {'episode_policy_steps':0},{'final_target_distance_m':float('nan')}):
            row=self.row(20,20,'collision');row.update(patch)
            with self.assertRaises(ValueError):m.original_train_groups([row],tasks('train'))
        with self.assertRaises(ValueError):m.original_train_groups([self.row(20,20,'collision')]*2,tasks('train'))

    def test_evaluation_requires_all_actual_validation_identities_and_preserves_contacts(self):
        m=self.module();va=tasks('validation')
        rows=[dict(scenario_identity_sha256=r['identity_sha256'],layout=r['layout'],terminal_class='collision',outcome='collision',
                   collision_contact_pairs=[['wall','coarse_collision']],policy_steps=20,final_target_error_m=.5)
              for r in va]
        result=m.validation_groups(rows,va)
        self.assertEqual(result['by_layout']['empty']['terminal_counts']['collision'],1)
        self.assertEqual(result['by_identity']['4'*64]['collision_contact_pairs'],[['wall','coarse_collision']])
        for bad in (rows[:1],[rows[0],rows[0]],[{**rows[0],'layout':'channel'},rows[1]],
                    [{**rows[0],'outcome':'success'},rows[1]]):
            with self.assertRaises(ValueError):m.validation_groups(bad,va)
