"""Frozen Joint v1 composition audit. No optimizer, backward or model writes."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch

from latent_dynamics_data import episodes_from_arrays,file_hash,json_hash
from latent_dynamics_multistep import start_manifest
from joint_latent_world_model import load_joint,component_hashes,encode_episode
from joint_latent_evaluation import encoded_sequences
from explicit_latent_models import array_hash
from joint_composition_core import freeze_joint,fit_manifold,distribution_scores,describe,compose
from joint_composition_evaluation import audit

BASE='59c0a28468c81a28c90283200c6acef27eb3362c'
BRANCH='feat/joint-latent-composition-audit'
REPORT='joint_latent_composition_audit_seed0.json'
STATISTICS='joint_latent_composition_train_statistics_seed0.json'


def compare(actual,expected):
    if isinstance(expected,dict):
        if set(actual)!=set(expected): raise AssertionError('diagnostic metric keys changed')
        for k in expected: compare(actual[k],expected[k])
    elif isinstance(expected,list):
        if len(actual)!=len(expected): raise AssertionError('diagnostic metric length changed')
        for a,b in zip(actual,expected): compare(a,b)
    elif expected is None or isinstance(expected,(str,bool)):
        if actual!=expected: raise AssertionError('diagnostic metadata mismatch')
    elif not np.isclose(actual,expected,rtol=1e-7,atol=1e-10): raise AssertionError(f'diagnostic mismatch {actual} != {expected}')


def load_context(root,source_report):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    path=Path(source_report).resolve(); source=json.loads(path.read_text()); modelrow=source['model']
    immutable=dict(source['immutable_artifacts']); immutable[str(path)]=file_hash(path); immutable[modelrow['path']]=modelrow['sha256']
    for p,h in immutable.items():
        if file_hash(p)!=h: raise ValueError(f'immutable source hash mismatch: {p}')
    model,stats,_=load_joint(modelrow['path']); freeze_joint(model); hashes=component_hashes(model)
    if hashes!=source['training']['parameter_hashes_best'] or stats!=source['normalization'] or json_hash(stats)!=source['normalization_sha256']:
        raise ValueError('Joint parameters/Train normalization provenance mismatch')
    dataset=source['dataset']; directory=Path(dataset['source_directory']); manifest=directory/'manifest.json'
    if file_hash(manifest)!=dataset['source_manifest_sha256']: raise ValueError('dataset manifest hash mismatch')
    immutable[str(manifest)]=file_hash(manifest); splits={}
    # Validation values not required; Train fits diagnostics only, Test evaluates.
    for split in ('train','test'):
        p=directory/f'{split}.npz'; h=dataset['source_dataset_sha256'][split]
        if file_hash(p)!=h: raise ValueError('dataset split hash mismatch')
        immutable[str(p)]=h
        with np.load(p,allow_pickle=False) as arrays:
            eps=episodes_from_arrays(arrays)
            for ep,start,stop in zip(eps,arrays['offsets'][:-1],arrays['offsets'][1:]):
                ep['next_obs']=arrays['inputs'][start+1:stop,:7].copy()
        identities=[dict(target_id=e['metadata']['target_id'],source=e['metadata']['source'],rows=e['source_rows']) for e in eps]
        if identities!=dataset['split_identity'][split]: raise ValueError('original split identity changed')
        splits[split]=eps
    if {e['metadata']['target_id'] for e in splits['train']}&{e['metadata']['target_id'] for e in splits['test']}:
        raise ValueError('Train/Test target overlap')
    horizons=source['horizons_steps']; windows=start_manifest(splits['test'],horizons,dataset['source_dataset_sha256']['test'])
    wm=source['test_start_manifest']
    if windows!=source['windows'] or windows!=json.loads(Path(wm['path']).read_text()) or file_hash(wm['path'])!=wm['sha256']:
        raise ValueError('original Test manifest changed')
    immutable[wm['path']]=wm['sha256']
    train_sequences=encoded_sequences(model,splits['train'],stats)
    manifold=fit_manifold(train_sequences,[ep['action'] for ep in splits['train']])
    scores=distribution_scores(np.concatenate(train_sequences),manifold)
    manifold['train_score_summary']={k:describe(v) for k,v in scores.items()}
    manifold['train_score_quantiles']={k:np.quantile(v,[1/3,2/3]).tolist() for k,v in scores.items()}
    manifold['provenance'].update(dataset_sha256=dataset['source_dataset_sha256']['train'],encoder_parameter_sha256=hashes['encoder'])
    return dict(source=source,source_path=str(path),model=model,stats=stats,hashes_before=hashes,immutable=immutable,
        train=splits['train'],test=splits['test'],dataset=dataset,windows=windows,manifold=manifold,
        train_reference_hashes=[array_hash(z) for z in train_sequences])


@torch.inference_mode()
def leakage_checks(context):
    model=context['model']; stats=context['stats']; selected=context['source']['selected_window']
    ep=context['test'][selected['episode_id']]; t=selected['start']; h=selected['horizon']
    def initial(e): return encode_episode(model,e['obs'][:t+1],e['previous_action'][:t+1],stats)[-1:]
    z0=initial(ep); actions=torch.from_numpy(ep['action'][None,t:t+h].astype(np.float64)); original=compose(model,z0,actions,stats)
    changed=copy.deepcopy(ep); changed['obs'][t+1:]+=1000; changed['next_obs'][:]=np.nan
    corrupted=compose(model,initial(changed),actions,stats)
    changed['obs']=changed['obs'][:t+1]; changed['previous_action']=changed['previous_action'][:t+1]; del changed['next_obs']
    deleted=compose(model,initial(changed),actions,stats)
    reference=torch.zeros(1,h+1,64); reference[:,1:]=float('nan')
    never=compose(model,z0,actions,stats,h,reference)
    good_reference=encoded_sequences(model,[ep],stats)[0][None,t:t+h+1]
    good_reference=torch.from_numpy(good_reference)
    c1=compose(model,z0,actions,stats,1,good_reference)
    good_reference[:,-1]=float('nan'); endpoint_altered=compose(model,z0,actions,stats,1,good_reference)
    A=encoded_sequences(model,[ep],stats)[0]
    encoded_sequences(model,[context['test'][-1]],stats)
    Aagain=encoded_sequences(model,[ep],stats)[0]
    checks=dict(corrupted_future_observation_unchanged=torch.equal(original['latents'],corrupted['latents']),
        deleted_future_observation_unchanged=torch.equal(original['normalized_observations'],deleted['normalized_observations']),
        never_future_reference_ignored=torch.equal(original['latents'],never['latents']),
        corrected_C1_endpoint_reference_unused=torch.equal(c1['latents'],endpoint_altered['latents']),
        encoder_reference_A_B_A_reset=np.array_equal(A,Aagain))
    if not all(checks.values()): raise AssertionError('causal/reference reset correctness failed')
    return checks


def evaluate_context(context):
    source=context['source']; out,plot=audit(context['model'],context['train'],context['test'],context['stats'],
        source['pi_magnitude_bins']['thresholds_m_s2'],source['horizons_steps'],source['selected_window'],context['manifold'])
    compare(out['encoder_reconstruction'],source['joint']['current_reconstruction'])
    for h in source['horizons_steps']:
        got=out['autonomous'][str(h)]; old=source['joint']['horizons'][str(h)]
        compare(got['observation'],old['observation'])
        for newkey,oldkey in [('normalized_rmse','train_std_normalized_distance'),('mean_l2','mean_l2'),('mean_cosine','mean_cosine')]:
            compare(got['latent'][newkey],old['latent_consistency'][oldkey])
        for field in ('distance_regions','pi_magnitude_regions'):
            for name,row in old[field].items(): compare(got[field][name]['observation'],row['observation'])
    local=out['local_one_step']['latent']; auto=out['autonomous']['1']['latent']
    compare(local,auto)
    checks=leakage_checks(context); after=component_hashes(context['model'])
    frozen=(after==context['hashes_before'] and all(not p.requires_grad and p.grad is None for p in context['model'].parameters())
            and all(not m.training for m in context['model'].modules()))
    immutable=all(file_hash(p)==h for p,h in context['immutable'].items())
    if not frozen or not immutable: raise AssertionError('read-only model/source contract violated')
    return out,plot,dict(checks,all_modules_frozen_and_unchanged=frozen,source_hashes_unchanged=immutable,
        original_Joint_metrics_reproduced=True,original_Test_manifest_reused=True,Train_only_statistics=True)


def run_experiment(root,source_report,make_figures=True):
    root=Path(root).resolve(); directory=root/'mujoco/reports'; reportpath=directory/REPORT; statpath=directory/STATISTICS
    if reportpath.exists() or statpath.exists(): raise FileExistsError('audit artifacts exist; no overwrite')
    started=time.monotonic(); context=load_context(root,source_report); out,plot,checks=evaluate_context(context)
    statpath.write_text(json.dumps(context['manifold'],indent=2,allow_nan=False)+'\n')
    source=context['source']; report=dict(experiment='Deterministic Latent Composition / Consistency Audit',branch=BRANCH,
        base_branch='feat/joint-latent-world-model-v1',base_commit=BASE,seed=0,source_report=context['source_path'],report_path=str(reportpath),
        checkpoint=source['model'],parameter_hashes_before=context['hashes_before'],parameter_hashes_after=component_hashes(context['model']),
        dataset=context['dataset'],dataset_regenerated=False,windows=context['windows'],test_start_manifest=source['test_start_manifest'],
        horizons_steps=source['horizons_steps'],normalization=context['stats'],normalization_sha256=json_hash(context['stats']),
        train_manifold=context['manifold'],train_statistics_file=dict(path=str(statpath),sha256=file_hash(statpath),semantic_sha256=json_hash(context['manifold'])),
        train_reference_hashes=context['train_reference_hashes'],pi_magnitude_bins=source['pi_magnitude_bins'],selected_window=source['selected_window'],
        immutable_artifacts=context['immutable'],verification=checks,**out,
        periodic_protocol='sameH50cohort; correct current input at0<k<50 divisible byC beforeT stepk+1; no terminalcorrection; C50==never',
        correction_assistance='diagnostic real-history encoder references only atscheduledcurrentinput steps; not deployable prediction',
        reference_definition='Joint Encoder on real history, episode reset; encoder-reference latent, NOT physical ground truth',
        metrics_definition='scale-aware latent errors use posttrainingTrainstd only; observation errors use immutableoriginalTrainstd; manifoldstatistics descriptive only',
        optional_jacobian='not executed; no autograd in audit',
        limitations=['Single seed nominal recorded actions and overlapping windows; no training/control/planning.',
            'Corrections inject real-history information diagnostically and are not a deployable solution.',
            'Train PCA/covariance describe linear distribution proxies, not a proven nonlinear latent manifold.',
            'Empirical decoder sensitivity ratio depends on latent/output units, not Jacobian spectral norm.',
            'Pearson correlations and observation/latent drift association do not identify causality.'])
    if make_figures:
        from joint_composition_plots import render_figures
        report['figures']=render_figures(report,plot,directory)
    else: report['figures']=[]
    report['elapsed_seconds']=time.monotonic()-started
    reportpath.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(f'Saved {reportpath}',flush=True)
    return report


def verify_experiment(report_path):
    path=Path(report_path).resolve(); root=path.parents[2]; report=json.loads(path.read_text()); context=load_context(root,report['source_report'])
    for key,actual in [('checkpoint',context['source']['model']),('normalization',context['stats']),('dataset',context['dataset']),
        ('windows',context['windows']),('test_start_manifest',context['source']['test_start_manifest']),('horizons_steps',context['source']['horizons_steps']),
        ('parameter_hashes_before',context['hashes_before']),('parameter_hashes_after',context['hashes_before']),('train_manifold',context['manifold']),
        ('immutable_artifacts',context['immutable']),('train_reference_hashes',context['train_reference_hashes']),('pi_magnitude_bins',context['source']['pi_magnitude_bins'])]:
        compare(actual,report[key])
    if report['normalization_sha256']!=json_hash(context['stats']): raise ValueError('audit normalization hash mismatch')
    stat=report['train_statistics_file']
    if file_hash(stat['path'])!=stat['sha256'] or json_hash(context['manifold'])!=stat['semantic_sha256']:
        raise ValueError('Train statistics file/hash mismatch')
    compare(json.loads(Path(stat['path']).read_text()),context['manifold'])
    out,_,checks=evaluate_context(context)
    for field,value in out.items(): compare(value,report[field])
    compare(checks,report['verification'])
    return dict(parameters_and_checkpoint_unchanged=True,all_diagnostics_reproduced=True,Train_only_statistics=True,
        manifest_reused=True,C50_equals_never=True,no_autonomous_future_input=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    if args.verify: print(json.dumps(verify_experiment(root/'mujoco/reports'/REPORT),indent=2))
    else: run_experiment(root,root/'mujoco/reports/joint_latent_world_model_v1_seed0.json')
