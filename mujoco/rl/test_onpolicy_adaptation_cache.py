"""Future collection must fail closed; historical caches are never relabeled."""
import importlib.util,json,tempfile,unittest
from pathlib import Path

class CacheTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_guarded_onpolicy_adaptation_collection'))
        import run_guarded_onpolicy_adaptation_collection as api
        return api

    def test_historical_unstamped_cache_refused_without_write(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'mpc_state_train_000.json').write_text('{}')
            before={x.name:x.read_bytes() for x in p.iterdir()}
            with self.assertRaises(ValueError):api.guard_cache(p,{'runtime':{'mujoco':'x'}})
            self.assertEqual(before,{x.name:x.read_bytes() for x in p.iterdir()})

    def test_runtime_or_dependency_drift_refused(self):
        api=self.api()
        signature={'runtime':{'mujoco':'3.3','numpy':'2'},'source_sha256':{'joint_latent_world_model.py':'a'}}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);api.guard_cache(p,signature);api.guard_cache(p,signature)
            for changed in ({**signature,'runtime':{'mujoco':'3.4','numpy':'2'}},
                            {**signature,'source_sha256':{'joint_latent_world_model.py':'b'}}):
                with self.assertRaises(ValueError):api.guard_cache(p,changed)
            self.assertEqual(json.loads((p/'collection_runtime_signature.json').read_text())['signature'],signature)

    def test_actual_signature_covers_execution_dependencies(self):
        api=self.api();root=Path(__file__).resolve().parents[2]
        signature=api.runtime_signature(root)
        self.assertTrue({'mujoco','numpy','torch','python'}<=set(signature['runtime']))
        for name in ('joint_latent_world_model.py','explicit_latent_models.py','latent_dynamics_data.py',
                     'onpolicy_adaptation_final_data.py','joint_autonomous_consistency_training.py'):
            self.assertIn(str(root/'mujoco/rl'/name),signature['source_sha256'])

if __name__=='__main__':unittest.main()
