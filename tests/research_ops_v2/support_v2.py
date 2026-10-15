import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/research_ops'))
from support import project,git


def fixture(root):
    from research_ops.lifecycle import Operations
    project(root)
    import shutil
    shutil.copytree(ROOT/'research_ops_v2',root/'research_ops_v2',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'research_regression',root/'research_regression',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'tests/research_ops_v2',root/'tests/research_ops_v2',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'researchctl-v2',root/'researchctl-v2')
    for p in (ROOT/'research_ops').glob('*.py'):shutil.copy2(p,root/'research_ops'/p.name)
    (root/'research_ops_v2/core.py').write_text('# FIXTURE, no ML\n')
    old=Operations(root);old.initialize()
    return root


def proposal(root,phase='OPS-02',**changes):
    from research_ops_v2.registration import build_registration
    values={'phase_id':phase,'tasks':[{'task_id':'audit','task_type':'offline-audit',
              'parameters':{'scope':'history'},'wall_time_s':60,'workers':1}],
            'sync_files':['README.md','docs/new.md','docs/PROJECT_PROGRESS.md',
              'docs/CODEX_NEXT_ACTION.md','docs/DAILY_PROGRESS.md','docs/RESEARCH_STATUS_V2.json'],
            'source_paths':['research_ops_v2'],'approved_changes':{}}
    values.update(changes)
    if 'mujoco/reports/research_operations_v2.json' not in values['sync_files']:
        values['sync_files']=values['sync_files']+['mujoco/reports/research_operations_v2.json']
    return build_registration(root,**values)


def review(registration):
    from research_ops.storage import digest
    return {'verdict':'GO','method':'SYNTHETIC_TEST_REVIEW_FIXTURE',
            'registration_sha256':digest(registration),
            'source_identity_sha256':digest(registration['source_identity'])}
