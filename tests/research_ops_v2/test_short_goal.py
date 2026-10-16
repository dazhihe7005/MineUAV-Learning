"""Distance-only tasks must not change scene/authority or accept stale evidence."""
import copy
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from research_ops.storage import OpsError,read_json,atomic_json,digest,sha256


class ShortGoalTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.short_goal'),'missing frozen distance task module')
        from research_regression.authorized_rl import short_goal
        return short_goal

    def prereg(self):
        return read_json(Path('mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json'))

    def test_distance_only_target_has_hand_checked_direction_and_height(self):
        self.assertEqual(self.module().short_target([0,0,1],[2,0,1]),[.20,0.,1.])
        self.assertEqual(self.module().short_target([0,0,1],[0,-2,1]),[0.,-.20,1.])

    def test_only_twelve_train_and_six_validation_rows_and_original_unchanged(self):
        p=self.prereg();old=copy.deepcopy(p);m=self.module().short_manifest(p)
        self.assertEqual(set(m),{'train','validation'})
        self.assertEqual([len(m[k]) for k in ('train','validation')],[12,6]);self.assertEqual(p,old)
        for split in m:
            for row,base in zip(m[split],p['scenario_manifest'][split]):
                self.assertNotEqual(row['identity_sha256'],base['identity_sha256'])
                self.assertEqual(row['base_identity_sha256'],base['identity_sha256'])
                self.assertAlmostEqual(math.dist(row['start_xyz'],row['target_xyz']),.20,places=12)
                for key in ('start_xyz','start_yaw_rad','obstacle_configuration_hash','scene_seed','obstacle_variation'):
                    self.assertEqual(row[key],base[key])

    def test_nonfinite_degenerate_and_already_short_target_refused(self):
        for a,b in [([0,0,1],[0,0,1]),([0,0,1],[.1,0,1]),([0,0,1],[1,0,float('nan')]),([0,0],[2,0,1]),([0,0,1],[2,0,2])]:
            with self.subTest(a=a,b=b),self.assertRaises(OpsError):self.module().short_target(a,b)

    def test_changed_manifest_or_base_cannot_be_rehashed_into_frozen_identity(self):
        p=self.prereg();m=self.module().short_manifest(p);self.module().verify_short_manifest(m,p)
        m['train'][0]['target_xyz'][0]+=.01
        with self.assertRaises(OpsError):self.module().verify_short_manifest(m,p)
        p['scenario_manifest']['train'][0]['start_yaw_rad']+=.1
        with self.assertRaises(OpsError):self.module().short_manifest(p)

    def test_distance_stage_changes_at_reset_threshold_not_from_wall_time(self):
        f=self.module().stage_for_steps
        self.assertEqual([f(n) for n in [0,32767,32768,65536]],['SHORT','SHORT','ORIGINAL','ORIGINAL'])
        for n in [-1,True,.1]:
            with self.assertRaises(OpsError):f(n)

    def test_missing_current_no_update_admission_refuses_before_execution(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_short_checks({'source_identity':{}},Path(d))

    def valid_fixture(self):
        m=self.module()
        manifest=m.short_manifest(self.prereg())
        rows=[{'task_identity_sha256':r['identity_sha256'],'base_identity_sha256':r['base_identity_sha256'],
               'start_xyz':r['start_xyz'],'target_xyz':r['target_xyz'],'terminal_class':'success',
               'raw_termination_reason':'success','steps':5,
               'trace':[{'distance_m':.05,'speed_m_s':.1,'sim_time':i*.04,'physics_ticks':20,
                         'position':[r['target_xyz'][0]-.05,*r['target_xyz'][1:]],'velocity':[.1,0,0]}
                        for i in range(1,6)]}
              for r in manifest['train']]
        good={'status':'PASS','experiment_id':'AUTORL_R5_SHORT_FEASIBILITY_ONLY','rows':rows,
              'physics_ticks':1200,'training_timesteps':0,'optimizer_updates':0,'test_episodes':0,'validation_episodes':0}
        return good,manifest

    def test_truncated_or_false_success_feasibility_cannot_authorize_training(self):
        m=self.module();good,manifest=self.valid_fixture()
        m.validate_feasibility(good,manifest)
        for change in ('truncate','fast','far','failure','status','raw_reason','physics','target','position','velocity'):
            bad=copy.deepcopy(good)
            if change=='truncate':bad['rows'].pop()
            elif change=='failure':bad['rows'][0]['terminal_class']='collision'
            elif change=='status':bad['status']='FAIL'
            elif change=='raw_reason':bad['rows'][0]['raw_termination_reason']='collision'
            elif change=='physics':bad['physics_ticks']=0
            elif change=='target':bad['rows'][0]['target_xyz']=[100,100,1]
            elif change=='position':bad['rows'][0]['trace'][-1]['position']=[100,100,1]
            elif change=='velocity':bad['rows'][0]['trace'][-1]['velocity']=[100,0,0]
            else:bad['rows'][0]['trace'][-1]['speed_m_s' if change=='fast' else 'distance_m']=.2
            with self.subTest(change=change),self.assertRaises(OpsError):m.validate_feasibility(bad,manifest)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):m.verify_feasibility(Path(d))

    def test_wrong_proof_runtime_identity_refused_even_with_matching_complete_sha(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'.research_local/autorl_r5_short_goal/run_01';run.mkdir(parents=True)
            reports=root/'mujoco/reports';reports.mkdir(parents=True)
            atomic_json(reports/'mujoco_lidar_ppo_training_prereg_seed0.json',self.prereg())
            reg={'source_identity':{'fixture':'SYNTHETIC_NOT_RUNTIME'}};spec={'registration_sha256':digest(reg)}
            atomic_json(reports/'autonomous_rl_short_goal_r5_registration.json',reg)
            atomic_json(reports/'autonomous_rl_short_goal_r5_prereg.json',spec)
            result,_=self.valid_fixture();result.update(registration_sha256='0'*64,source_identity_sha256='0'*64,
                protocol_sha256='0'*64,short_manifest_sha256=digest(m.short_manifest(self.prereg())))
            atomic_json(run/'result.json',result)
            atomic_json(run/'complete.json',{'result_sha256':sha256(run/'result.json')})
            with self.assertRaises(OpsError):m.verify_feasibility(root)


if __name__=='__main__':unittest.main()
