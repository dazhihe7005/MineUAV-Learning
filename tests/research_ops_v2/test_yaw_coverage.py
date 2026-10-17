import copy
import importlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from research_ops.storage import read_json,digest


class YawCoverageTests(unittest.TestCase):
    def module(self):
        name='research_regression.authorized_rl.yaw_coverage'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing independent Train yaw task contract')
        return importlib.import_module(name)

    def prereg(self):
        return read_json(Path(__file__).parents[2]/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')

    def test_five_yaws_per_frozen_train_base_two_stages_no_validation_or_test(self):
        m=self.module();p=self.prereg();before=digest(p);manifest=m.yaw_manifest(p)
        self.assertEqual((len(manifest['SHORT']),len(manifest['ORIGINAL'])),(60,60))
        for stage in ('SHORT','ORIGINAL'):
            for base in p['scenario_manifest']['train']:
                rows=[v for v in manifest[stage] if v['base_identity_sha256']==base['identity_sha256']]
                self.assertEqual(len(rows),5)
                self.assertEqual([v['start_yaw_rad'] for v in rows],[base['start_yaw_rad'],0.,math.pi/2,math.pi,-math.pi/2])
                self.assertTrue(all(v['split']=='train' and v['obstacle_configuration_hash']==base['obstacle_configuration_hash']
                    and v['start_xyz']==base['start_xyz'] for v in rows))
        self.assertEqual(len({v['identity_sha256'] for s in ('SHORT','ORIGINAL') for v in manifest[s]}),120)
        self.assertEqual(digest(p),before)

    def test_target_geometry_and_old_metadata_preserved_with_new_explicit_identity(self):
        m=self.module();p=self.prereg();manifest=m.yaw_manifest(p)
        from research_regression.authorized_rl.short_goal import short_manifest
        short={v['base_identity_sha256']:v for v in short_manifest(p)['train']}
        for stage in ('SHORT','ORIGINAL'):
            for base in p['scenario_manifest']['train']:
                row=m.resolve_yaw_task(manifest,stage,base['identity_sha256'],3)
                self.assertEqual(row['target_xyz'],short[base['identity_sha256']]['target_xyz'] if stage=='SHORT' else base['target_xyz'])
                self.assertEqual(row['obstacle_variation'],base['obstacle_variation'])
                self.assertEqual(row['start_yaw_rad'],math.pi)
                self.assertNotEqual(row['identity_sha256'],base['identity_sha256'])
                self.assertEqual(row['identity_sha256'],digest({k:v for k,v in row.items() if k!='identity_sha256'}))

    def test_manifest_tampering_rejected_even_with_recomputed_row_identity(self):
        m=self.module();p=self.prereg();a=m.yaw_manifest(p);a['SHORT'][0]['target_xyz'][0]+=.01
        a['SHORT'][0]['identity_sha256']=digest({k:v for k,v in a['SHORT'][0].items() if k!='identity_sha256'})
        with self.assertRaises(ValueError):m.verify_yaw_manifest(a,p)
        changed=copy.deepcopy(p);changed['scenario_manifest']['train'][0]['start_yaw_rad']=0.
        with self.assertRaises(ValueError):m.yaw_manifest(changed)

    def test_lookup_rejects_nontrain_invalid_option_and_stage_and_returns_independent_copy(self):
        m=self.module();p=self.prereg();a=m.yaw_manifest(p);base=p['scenario_manifest']['train'][0]['identity_sha256']
        for stage,identity,option in (('test',base,0),('SHORT',p['scenario_manifest']['validation'][0]['identity_sha256'],0),
                                      ('SHORT',base,True),('SHORT',base,-1),('SHORT',base,5)):
            with self.assertRaises(ValueError):m.resolve_yaw_task(a,stage,identity,option)
        row=m.resolve_yaw_task(a,'SHORT',base,0);row['start_xyz'][0]+=1
        self.assertNotEqual(row['start_xyz'],m.resolve_yaw_task(a,'SHORT',base,0)['start_xyz'])

    def test_temporal_configuration_checks_model_and_rollout_buffer_no_gae_carryover(self):
        m=self.module();model=SimpleNamespace(gamma=.99,gae_lambda=.95,rollout_buffer=SimpleNamespace(gamma=.99,gae_lambda=.95))
        m.verify_yaw_temporal_configuration(model)
        for owner,name in ((model,'gamma'),(model,'gae_lambda'),(model.rollout_buffer,'gamma'),(model.rollout_buffer,'gae_lambda')):
            old=getattr(owner,name);setattr(owner,name,.995)
            with self.assertRaises(ValueError):m.verify_yaw_temporal_configuration(model)
            setattr(owner,name,old)
