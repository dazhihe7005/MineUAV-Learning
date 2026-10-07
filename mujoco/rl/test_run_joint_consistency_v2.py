"""Saved untrained fixture baseline; only v2 temporary training is executed."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_run_joint_latent_composition_audit import fixture as audit_fixture
from latent_dynamics_data import file_hash,json_hash,episodes_from_arrays
from latent_dynamics_multistep import start_manifest
from latent_dynamics_models import HistoryLatentDynamics,save_model
from explicit_latent_models import ResidualTransition,ObservationDecoder,save_component
from joint_latent_world_model import initialize_joint,save_joint,component_hashes
from joint_latent_evaluation import encoded_sequences,evaluate_joint
from joint_latent_training import validation_loss
from joint_composition_core import freeze_joint
from run_joint_latent_composition_audit import run_experiment as audit_reference

try:
    import run_joint_consistency_v2 as runner
    import joint_consistency_evaluation as evaluation
except ImportError:
    runner=evaluation=None


def fixture(root):
    source=audit_fixture(root); r=json.loads(source.read_text()); stats=r['normalization']
    latent=dict(mean=[0.]*64,std=[1.]*64); torch.manual_seed(0)
    modeldir=root/'mujoco/rl/models'; paths=[modeldir/n for n in ('encoder.pt','transition.pt','decoder.pt')]
    save_model(paths[0],HistoryLatentDynamics(),'history',stats,{})
    save_component(paths[1],ResidualTransition(),'transition',stats,latent,{})
    save_component(paths[2],ObservationDecoder(),'decoder',stats,latent,{})
    model,_,initialization=initialize_joint(*paths); freeze_joint(model)
    save_joint(r['model']['path'],model,stats,{'fixture':'pretrained initialization weights, not optimized baseline'})
    r['model']['sha256']=file_hash(r['model']['path'])
    r['initialization']=initialization
    r['training'].update(parameter_hashes_before=component_hashes(model),parameter_hashes_best=component_hashes(model))
    r['config']=dict(seed=0,epochs=2,batch_size=2,learning_rate=.0003,horizon=10)
    splits={}; windows={}
    for split in ('train','val','test'):
        p=Path(r['dataset']['source_directory'])/f'{split}.npz'
        with np.load(p,allow_pickle=False) as a:
            eps=episodes_from_arrays(a)
            for ep,start,stop in zip(eps,a['offsets'][:-1],a['offsets'][1:]): ep['next_obs']=a['inputs'][start+1:stop,:7].copy()
        splits[split]=eps
        if split!='test':
            data=start_manifest(eps,[10],r['dataset']['source_dataset_sha256'][split]); data.pop('manifest_sha256')
            data['split']=split; data['dataset_split_sha256']=data.pop('dataset_test_sha256'); data['manifest_sha256']=json_hash(data)
            p=root/f'{split}_windows.json'; p.write_text(json.dumps(data))
            windows[split]=dict(path=str(p),sha256=file_hash(p),window_count=data['window_counts']['10'])
    r['window_manifests']=windows
    r['training']['initial_validation_loss']=validation_loss(model,splits['val'],stats,10,2)
    r['joint']=evaluate_joint(model,splits['test'],stats,r['pi_magnitude_bins']['thresholds_m_s2'],r['horizons_steps'],encoded_sequences(model,splits['train'],stats))
    r['immutable_artifacts']={r['model']['path']:r['model']['sha256']}
    r['immutable_artifacts'].update({str(p):file_hash(p) for p in paths})
    source.write_text(json.dumps(r)); audited=audit_reference(root,source,make_figures=False)
    return source,Path(audited['report_path'])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner,'v2 runner missing'); self.assertIsNotNone(evaluation,'v2 evaluation missing')
        torch.set_num_threads(1)

    def test_loader_initial_hashes_and_scale_fit_do_not_open_test_or_refit_observation_statistics(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); v1,audit=fixture(root); original=np.load; opened=[]
            def checked(path,*a,**kw):
                opened.append(Path(path).name); self.assertNotEqual(Path(path).name,'test.npz')
                return original(path,*a,**kw)
            with patch('numpy.load',side_effect=checked): c=runner.load_context(root,v1,audit)
            self.assertEqual(opened,['train.npz','val.npz'])
            self.assertEqual(component_hashes(c['model']),c['baseline']['training']['parameter_hashes_before'])
            self.assertEqual(c['initial_scale']['split'],'train'); self.assertEqual(c['initial_scale']['frame_count'],30)
            self.assertEqual(c['statistics'],c['baseline']['normalization'])
            self.assertAlmostEqual(c['initial_observation_validation'],c['baseline']['training']['initial_validation_loss'],places=8)
            bad=json.loads(v1.read_text()); bad['training']['parameter_hashes_before']['encoder']='wrong'; v1.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): runner.load_context(root,v1,audit)

    def test_full_selection_replay_audit_figures_no_source_mutation_and_manifest_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); v1,audit=fixture(root); saved=json.loads(v1.read_text())
            before={str(p):file_hash(p) for p in [v1,audit,saved['model']['path']]}
            result=runner.run_experiment(root,v1,audit)
            self.assertEqual(len(result['figures']),12)
            self.assertTrue(all(Path(p).stat().st_size>1000 for p in result['figures']))
            self.assertEqual(result['config'],saved['config']); self.assertEqual(result['lambda_consistency'],.1)
            self.assertEqual(result['windows'],saved['windows'])
            self.assertEqual(result['window_manifests'],saved['window_manifests'])
            self.assertEqual(result['selected_window'],saved['selected_window'])
            self.assertEqual(result['training']['parameter_hashes_before'],saved['training']['parameter_hashes_before'])
            self.assertEqual(result['audit']['periodic_correction']['50'],result['audit']['periodic_correction']['never'])
            verified=runner.verify_experiment(result['report_path'])
            self.assertTrue(all(verified.values()))
            self.assertIn('selected_checkpoint_and_evaluation_metrics_reproduced',verified)
            self.assertNotIn('all_metrics_reproduced',verified)
            for p,h in before.items(): self.assertEqual(file_hash(p),h)
            with self.assertRaises(FileExistsError): runner.run_experiment(root,v1,audit)
            path=Path(result['report_path'])
            for field in ('lambda_consistency','selected_window','initial_latent_scale','audit','best_epoch',
                          'baseline_empty','baseline_omission','history_total','history_epoch','final_summary','history_gradient'):
                wrong=copy.deepcopy(result)
                if field=='lambda_consistency': wrong[field]=1.
                elif field=='selected_window': wrong[field]['start']+=1
                elif field=='initial_latent_scale': wrong[field]['std'][0]*=2
                elif field=='audit': wrong[field]['local_one_step']['latent']['normalized_rmse']+=1
                elif field=='baseline_empty': wrong['baseline']={}
                elif field=='baseline_omission': del wrong['baseline']['local_one_step']
                elif field=='history_total': wrong['training']['history'][0]['train']['total']+=1
                elif field=='history_epoch': wrong['training']['history'][0]['epoch']=99
                elif field=='final_summary': wrong['training']['final_train']['observation']+=1
                elif field=='history_gradient': wrong['training']['history'][0]['gradient_norms']['encoder']['mean']=-1
                else: wrong['training'][field]=99
                path.write_text(json.dumps(wrong))
                with self.assertRaises((ValueError,AssertionError)): runner.verify_experiment(path)

    def test_relative_ratio_literal_and_stationary_denominator_disclosure(self):
        source=np.zeros((2,64)); ref=source.copy(); ref[0,0]=2; pred=ref.copy(); pred[:,0]+=1
        r=evaluation.relative_residual(pred,source,ref,epsilon=1.)
        # ratios1/3 and1; stationary reference retained, not silently dropped.
        self.assertAlmostEqual(r['all']['mean'],2/3)
        self.assertAlmostEqual(r['all']['median'],2/3)
        self.assertEqual(r['stationary_reference_count'],1)
        self.assertEqual(r['all']['count'],2)


if __name__=='__main__': unittest.main()
