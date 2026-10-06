"""Matched starts/scaling, frozen references, reset, selection and yaw contracts."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import file_hash, json_hash, fit_statistics, load_dataset
from latent_dynamics_models import make_model, save_model
from latent_dynamics_multistep import rollout_windows, start_manifest
from latent_multistep_training import observation_loss, predict_windows
from run_latent_dynamics_multistep_eval import evaluate_model, run_evaluation
from run_latent_memory_ablation import parameter_hash
from test_latent_dynamics_multistep import episode, statistics


def long_memory_fixture(root):
    """Real temporary NPZ episodes, long enough for genuine K10 manifests."""
    source=root/'source'; source.mkdir(); hashes={}
    for split,target in [('train',1),('val',2),('test',3)]:
        inputs=[]; actions=[]; states=[]; offsets=[0]; metadata=[]
        for episode_id in range(2):
            time=np.arange(13)[:,None]
            obs=(.2*np.sin(time*.17+np.arange(7)[None,:])+.03*time).astype(np.float32)
            action=(.1*np.cos(time*.2+np.arange(4)[None,:])).astype(np.float32)
            prev=np.vstack([np.zeros((1,4),np.float32),action[:-1]])
            pi=np.repeat(.01*time,3,axis=1).astype(np.float32)
            inputs.append(np.concatenate([obs,prev],axis=1)); actions.append(action); states.append(pi)
            offsets.append(offsets[-1]+len(obs)); metadata.append(dict(target_id=target,source=f'fixture_{episode_id}'))
        path=source/f'{split}.npz'
        np.savez_compressed(path,inputs=np.concatenate(inputs),actions=np.concatenate(actions),
            truth=np.concatenate(states),offsets=np.array(offsets),metadata=json.dumps(metadata))
        hashes[split]=file_hash(path)
    (source/'manifest.json').write_text(json.dumps(dict(dataset_sha256=hashes,seed=0)))
    splits,manifest=load_dataset(source); stats=fit_statistics(splits['train'])
    from run_latent_memory_ablation import evaluate
    data=dict(dataset=manifest,normalization=stats,normalization_sha256=json_hash(stats),
        integral_bins=dict(thresholds_m_s2=[.1,.2]),frozen_source_hashes={},models={})
    for kind in ('markov','history','oracle','no_memory'):
        net=make_model(kind); path=root/f'{kind}.pt'; save_model(path,net,kind,stats,dict(epoch=0))
        data['models'][kind]=dict(artifact=dict(path=str(path),sha256=file_hash(path)),
            test=evaluate(net,kind,splits['test'],stats,[.1,.2]))
    path=root/'memory.json'; path.write_text(json.dumps(data)); return path

try:
    import run_latent_objective_matched_control as control
except ImportError:
    control = None


def reference_fixture(root):
    """Untrained temporary reference ONLY; never calls a K10 optimizer/trainer."""
    memory = long_memory_fixture(root); data = json.loads(memory.read_text()); torch.manual_seed(0)
    data['architecture'] = dict(initial_parameter_sha256=parameter_hash(make_model('history')))
    memory.write_text(json.dumps(data))
    frozen = run_evaluation(root, memory, horizons=(1,2), make_figures=False)
    source_path = root/'mujoco/reports/latent_dynamics_multistep_frozen_eval_seed0.json'
    cfg = dict(seed=0, epochs=2, batch_size=2, learning_rate=.001, horizon=10)
    net = make_model('history'); path=root/'k10.pt'; stats=data['normalization']
    save_model(path, net, 'history', stats, dict(config=cfg, initial_parameter_sha256=data['architecture']['initial_parameter_sha256']))
    from run_latent_dynamics_multistep_eval import load_inputs
    ev=load_inputs(root, memory)
    row=evaluate_model(net,'history',ev['episodes'],stats,[.1,.2],[1,2])
    row['artifact']=dict(path=str(path),sha256=file_hash(path))
    immutable=dict(frozen['immutable_artifacts']); immutable[str(source_path)]=file_hash(source_path)
    immutable[str(path)]=file_hash(path)
    parent=dict(config=cfg,source_report=str(source_path),normalization=stats,
        normalization_sha256=json_hash(stats),training=dict(initial_parameter_sha256=data['architecture']['initial_parameter_sha256']),
        models=dict(history=frozen['models']['history'],multistep_history=row),
        dataset=data['dataset'],immutable_artifacts=immutable)
    manifests={}
    from run_latent_multistep_training import load_training_inputs
    ctx=load_training_inputs(root,source_path,horizon=10)
    for split,manifest in ctx['manifests'].items():
        p=root/f'{split}_windows.json'; p.write_text(json.dumps(manifest))
        manifests[split]=dict(path=str(p),sha256=file_hash(p),semantic_sha256=manifest['manifest_sha256'],window_count=manifest['window_counts']['10'])
    parent['train_validation_manifests']=manifests
    p=root/'k10_report.json'; p.write_text(json.dumps(parent)); return p


class MatchedTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(control,'objective-matched control runner missing')
        torch.set_num_threads(1); torch.manual_seed(0)

    def test_matched_prefixes_use_exact_k10_start_ids_not_extra_tail_starts(self):
        eps=[episode(12),episode(15,2)]
        manifest=start_manifest(eps,[10],'fixture'); before=copy.deepcopy(eps)
        result=control.match_start_prefixes(eps,manifest,10)
        self.assertEqual([len(e['obs']) for e in result],[3,6])
        for original,matched,unchanged in zip(eps,result,before):
            np.testing.assert_array_equal(original['obs'],unchanged['obs'])
            np.testing.assert_array_equal(matched['obs'],original['obs'][:len(matched['obs'])])
        manifest['episodes'][0]['valid_start_count']['10']=4
        with self.assertRaisesRegex(ValueError,'start'):
            control.match_start_prefixes(eps,manifest,10)

    def test_matched_k1_first_predictions_equal_k10_first_steps(self):
        eps=[episode(12),episode(15,2)]; stats=statistics(); net=make_model('history')
        matched=control.match_start_prefixes(eps,start_manifest(eps,[10],'fixture'),10)
        k1=predict_windows(net,matched,stats,1); k10=predict_windows(net,eps,stats,10)
        torch.testing.assert_close(k1['predictions'][:,0],k10['predictions'][:,0],atol=1e-7,rtol=1e-7)
        self.assertAlmostEqual(observation_loss(k1['predictions'],k1['truth'],stats).item(),
            observation_loss(k10['predictions'][:,:1],k10['truth'][:,:1],stats).item(),places=7)

    def test_config_and_initialization_anchored_to_frozen_k10(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=reference_fixture(root); ctx=control.load_reference(root,p)
            self.assertEqual({**ctx['config'],'horizon':10},ctx['source']['config'])
            data=json.loads(p.read_text()); data['models']['multistep_history']['artifact']['sha256']='0'*64
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'hash'):
                control.load_reference(root,p)

    def test_reference_loader_never_opens_test_values_or_recomputes_statistics(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=reference_fixture(root); original_load=np.load; opened=[]
            def tracked_load(path,*args,**kwargs):
                opened.append(Path(path).name)
                self.assertNotEqual(Path(path).name,'test.npz')
                return original_load(path,*args,**kwargs)
            with patch('numpy.load',side_effect=tracked_load):
                ctx=control.load_reference(root,source)
            self.assertEqual(opened,['train.npz','val.npz'])
            self.assertEqual(ctx['statistics'],ctx['source']['normalization'])
            self.assertEqual([len(ep['obs']) for ep in ctx['matched']['train']],[3,3])

    def test_wrong_initial_hash_or_changed_reference_windows_block_training(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=reference_fixture(root); original=source.read_text()
            data=json.loads(original); data['training']['initial_parameter_sha256']='0'*64
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'initial parameter hash'):
                control.load_reference(root,source)
            source.write_text(original); data=json.loads(original)
            manifest=Path(data['train_validation_manifests']['train']['path'])
            manifest.write_text(manifest.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'manifest'):
                control.load_reference(root,source)

    def test_yaw_audit_keeps_coordinate_subtraction_and_detects_wrap_jumps(self):
        ep=episode(3); ep['obs'][:,6]=[3.0,-3.0,-2.9]; ep['next_obs'][:,6]=[-3.0,-2.9,-2.8]
        result=control.yaw_audit([ep])
        self.assertEqual(result['adjacent_wrap_jumps'],1)
        self.assertEqual(result['metric'],'raw coordinate error, identical frozen evaluator; no new circular wrapping')
        with self.assertRaisesRegex(ValueError,'yaw'):
            ep['obs'][0,6]=4.0; control.yaw_audit([ep])

    def test_end_to_end_k1_only_reload_selection_and_frozen_reference_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=reference_fixture(root)
            before={str(p):file_hash(p) for p in root.glob('*.pt')}
            report=control.run_control(root,source,make_figures=True)
            self.assertEqual(len(report['figures']),3)
            self.assertTrue(all(Path(p).is_file() for p in report['figures']))
            self.assertEqual(report['training']['initial_parameter_sha256'],report['reference_initial_parameter_sha256'])
            self.assertEqual(report['training']['best_epoch'],np.argmin([r['validation_loss'] for r in report['training']['history']])+1)
            self.assertTrue(all(control.verify_control(Path(report['report_path'])).values()))
            report_path=Path(report['report_path']); original_report=report_path.read_text()
            changed=json.loads(original_report); changed['horizons_steps']=[1]
            changed['comparison']={'1':changed['comparison']['1']}
            report_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(AssertionError,'horizon'):
                control.verify_control(report_path)
            report_path.write_text(original_report)
            for p,h in before.items(): self.assertEqual(file_hash(p),h)
            manifest=Path(report['matched_start_manifests']['train']['path']); original=manifest.read_text()
            manifest.write_text(original+' ')
            with self.assertRaisesRegex(ValueError,'manifest'):
                control.verify_control(Path(report['report_path']))
            manifest.write_text(original)
            with self.assertRaises(FileExistsError): control.run_control(root,source,make_figures=False)


if __name__=='__main__': unittest.main()
