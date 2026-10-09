"""Supported future collection entrypoint; refuse unstamped historical caches.

The original immutable collectors are retained as the generation implementation
of this completed experiment. Never stamp/relabel their historical artifacts.
New collection/resumption must use this runtime/dependency guard.
"""
import argparse,json,platform
from pathlib import Path
import mujoco,numpy,torch
from latent_dynamics_data import file_hash
from run_latent_random_shooting_mpc import write_json

def runtime_signature(root):
    from run_onpolicy_adaptation_data import collection_context
    ctx=collection_context(root);sources=dict(ctx['identity']['source_sha256'])
    # All local dynamics/encoder/normalization/physics execution dependencies,
    # plus the guard itself. Exclude training, tests and analysis-only helpers.
    names=('joint_latent_world_model.py','explicit_latent_models.py','latent_dynamics_data.py',
        'joint_autonomous_consistency_training.py','joint_latent_training.py','latent_dynamics_multistep.py',
        'onpolicy_adaptation_final_data.py','onpolicy_adaptation_evaluation.py','later_ranking_core.py',
        'joint_composition_core.py','latent_mpc_support_evaluation.py',Path(__file__).name)
    for name in names:
        path=Path(root)/'mujoco/rl'/name;sources[str(path)]=file_hash(path)
    return dict(runtime=dict(python=platform.python_version(),mujoco=mujoco.__version__,
        numpy=numpy.__version__,torch=torch.__version__),source_sha256=sources)

def guard_cache(directory,signature):
    directory=Path(directory);stamp=directory/'collection_runtime_signature.json'
    if stamp.exists():
        if json.loads(stamp.read_text())['signature']!=signature:
            raise ValueError('runtime/execution dependencies changed; use a fresh cache, never relabel')
        return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('historical cache has no runtime stamp; read-only verification permitted, collection reuse refused')
    directory.mkdir(parents=True,exist_ok=True)
    write_json(stamp,dict(signature=signature,meaning='recorded before first collection; not an inferred historical runtime'))

def run(root,phase,workers):
    root=Path(root).resolve()
    guard_cache(root/'mujoco/reports/world_model_onpolicy_adaptation_seed0_parts',runtime_signature(root))
    if phase=='adaptation':
        from run_onpolicy_adaptation_data import run as collect
    else:
        from onpolicy_adaptation_final_data import run as collect
    return collect(root,workers)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--phase',choices=('adaptation','final'),required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();run(a.root,a.phase,a.workers)
