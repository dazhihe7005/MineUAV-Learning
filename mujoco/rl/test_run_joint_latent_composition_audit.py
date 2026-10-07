"""Saved fixed-weight fixtures, no training, check the entire audit boundary."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from latent_dynamics_data import episodes_from_arrays,fit_statistics,file_hash,json_hash,load_dataset
from latent_dynamics_multistep import start_manifest
from joint_latent_world_model import JointLatentWorldModel,save_joint,component_hashes
from joint_latent_evaluation import encoded_sequences,evaluate_joint
from joint_composition_core import freeze_joint

try:
    import run_joint_latent_composition_audit as runner
except ImportError:
    runner=None


def fixture(root):
    torch.manual_seed(0); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    data=root/'mujoco/rl/datasets/fixture'; data.mkdir(parents=True)
    rng=np.random.default_rng(0); splits={}
    for si,split in enumerate(('train','val','test')):
        episodes=[]; offsets=[0]; metadata=[]; obsrows=[]; actions=[]; truth=[]
        for ei in range(2):
            n=56 if split=='test' else 14
            o=rng.normal(0,.2,(n+1,7)).astype(np.float32)
            a=rng.uniform(-.3,.3,(n+1,4)).astype(np.float32)
            prev=np.vstack([np.zeros(4,dtype=np.float32),a[:-1]])
            pi=rng.normal(0,.05,(n+1,3)).astype(np.float32); pi[0]=0
            obsrows.append(np.concatenate([o,prev],1)); actions.append(a); truth.append(pi)
            offsets.append(offsets[-1]+n+1); metadata.append({'target_id':10*si+ei,'source':'fixed_random'})
        arrays=dict(inputs=np.concatenate(obsrows),actions=np.concatenate(actions),truth=np.concatenate(truth),
                    offsets=np.asarray(offsets),metadata=np.asarray(json.dumps(metadata)))
        np.savez(data/f'{split}.npz',**arrays)
        episodes=episodes_from_arrays(arrays)
        for ep,start,stop in zip(episodes,offsets[:-1],offsets[1:]): ep['next_obs']=arrays['inputs'][start+1:stop,:7].copy()
        splits[split]=episodes
    manifest=dict(seed=0,dataset_sha256={s:file_hash(data/f'{s}.npz') for s in splits})
    (data/'manifest.json').write_text(json.dumps(manifest))
    _,dataset=load_dataset(data); stats=fit_statistics(splits['train'])
    model=freeze_joint(JointLatentWorldModel()); path=root/'mujoco/rl/models/joint_latent_world_model_v1.pt'
    save_joint(path,model,stats,{'fixture':'fixed weights, no optimizer'})
    windows=start_manifest(splits['test'],(1,5,10,25,50),manifest['dataset_sha256']['test'])
    wd=root/'mujoco/reports'; wd.mkdir(parents=True); wm=wd/'starts.json'; wm.write_text(json.dumps(windows))
    train_z=encoded_sequences(model,splits['train'],stats)
    original=evaluate_joint(model,splits['test'],stats,[.05,.15],(1,5,10,25,50),train_z)
    report=dict(model={'path':str(path),'sha256':file_hash(path)},normalization=stats,normalization_sha256=json_hash(stats),
        dataset=dataset,windows=windows,horizons_steps=[1,5,10,25,50],test_start_manifest={'path':str(wm),'sha256':file_hash(wm)},
        pi_magnitude_bins={'thresholds_m_s2':[.05,.15]},joint=original,selected_window={'episode_id':0,'target_id':20,'start':0,'horizon':50},
        training={'parameter_hashes_best':component_hashes(model)},immutable_artifacts={str(path):file_hash(path)})
    source=wd/'joint_latent_world_model_v1_seed0.json'; source.write_text(json.dumps(report)); return source


class RunnerTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(runner,'composition runner missing')

    def test_frozen_audit_reproduces_original_windows_no_optimizer_or_model_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=fixture(root); old=json.loads(source.read_text())
            before={p:file_hash(p) for p in [str(source),old['model']['path'],old['test_start_manifest']['path']]}
            with patch('torch.optim.Adam',side_effect=AssertionError('audit optimizer forbidden')):
                report=runner.run_experiment(root,source,make_figures=False)
                self.assertTrue(all(runner.verify_experiment(Path(report['report_path'])).values()))
            self.assertEqual(report['windows'],old['windows'])
            self.assertTrue(report['verification']['all_modules_frozen_and_unchanged'])
            self.assertEqual(report['local_one_step']['latent']['count'],112)
            self.assertEqual(report['periodic_correction']['never']['observation']['window_count'],14)
            self.assertEqual(report['periodic_correction']['50'],report['periodic_correction']['never'])
            self.assertTrue(report['verification']['encoder_reference_A_B_A_reset'])
            self.assertEqual(report['train_manifold']['provenance']['split'],'train')
            for p,h in before.items(): self.assertEqual(file_hash(p),h)
            with self.assertRaises(FileExistsError): runner.run_experiment(root,source,make_figures=False)
            for field in ['local_one_step','periodic_correction','train_manifold']:
                wrong=json.loads(json.dumps(report))
                if field=='train_manifold': wrong[field]['std'][0]*=10
                elif field=='local_one_step': wrong[field]['latent']['normalized_rmse']+=1
                else: wrong[field]['1']['observation']['normalized_observation_rmse']+=1
                p=Path(report['report_path']); p.write_text(json.dumps(wrong))
                with self.assertRaises((AssertionError,ValueError)): runner.verify_experiment(p)

    def test_train_stats_independent_of_test_and_pure_future_mutation_no_leak(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=fixture(root)
            context=runner.load_context(root,source); before=context['manifold']
            for ep in context['test']: ep['obs'][:]*=1000; ep['next_obs'][:]*=1000
            self.assertEqual(before,runner.load_context(root,source)['manifold'])
            self.assertTrue(all(runner.leakage_checks(runner.load_context(root,source)).values()))
            self.assertEqual(context['dataset']['target_ids']['train'],[0,1])
            self.assertEqual(context['dataset']['target_ids']['test'],[20,21])

    def test_actual_figures_and_hash_tamper_fail_before_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=fixture(root)
            report=runner.run_experiment(root,source)
            self.assertEqual(len(report['figures']),9)
            self.assertTrue(all(Path(p).stat().st_size>1000 for p in report['figures']))
            old=json.loads(source.read_text()); old['model']['sha256']='wrong'; source.write_text(json.dumps(old))
            with self.assertRaises(ValueError): runner.load_context(root,source)


if __name__=='__main__': unittest.main()
