"""Grouped labels, indexing, recoverable records and real resource limits."""
import importlib,importlib.util,json,tempfile,unittest,sys,subprocess,os,signal,time
from unittest.mock import patch
from pathlib import Path
import numpy as np

class BCDataSafetyTests(unittest.TestCase):
    def api(self,name):
        self.assertIsNotNone(importlib.util.find_spec(name))
        return importlib.import_module(name)
    def test_budget_refuses_eight_workers_and_low_headroom(self):
        s=self.api('uav_bc_safety');g=1024**3
        s.check_budget(1,6*g);s.check_budget(2,6*g)
        for workers,free in ((8,12*g),(1,5*g),(0,8*g)):
            with self.assertRaises((ValueError,MemoryError)):s.check_budget(workers,free)
        for free,rss,oom in ((1.9*g,1*g,0),(8*g,4.1*g,0),(8*g,1*g,1)):
            with self.assertRaises(MemoryError):s.check_running(free,rss,oom)
    def test_atomic_files_roundtrip_and_no_partial_publish(self):
        s=self.api('uav_bc_safety')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';s.atomic_json(p,{'complete':True})
            with self.assertRaises(ValueError):s.atomic_json(p,{'bad':float('nan')})
            self.assertEqual(json.loads(p.read_text()),{'complete':True})
            q=Path(d)/'b.npz';s.atomic_npz(q,obs=np.arange(7))
            with np.load(q) as a:np.testing.assert_array_equal(a['obs'],np.arange(7))
            self.assertEqual({x.name for x in Path(d).iterdir()},{'a.json','b.npz'})
    def test_target_coordinate_leakage_rejected(self):
        a=self.api('uav_bc_data')
        groups={'train':[{'target_id':1,'target':[0.,0.,1.]}],
                'val':[{'target_id':2,'target':[.1,0.,1.]}],'test':[{'target_id':3,'target':[.2,0.,1.]}]}
        a.check_splits(groups,{'benchmark':[(4,[.3,0.,1.])]})
        groups['val'][0]['target']=[0.,0.,1.]
        with self.assertRaises(ValueError):a.check_splits(groups,{})
    def test_existing_target_manifests_reused_disjoint(self):
        a=self.api('uav_bc_data');groups,evaluation,meta=a.split_targets(Path(__file__).resolve().parents[2])
        self.assertEqual({k:len(v) for k,v in groups.items()},{'train':84,'val':18,'test':18})
        self.assertEqual({k:len(v) for k,v in evaluation.items()},{'benchmark':100,'holdout':100})
        a.check_splits(groups,evaluation);self.assertEqual(meta['overlap_count'],0)
    def test_expert_actions_previous_index_and_complete_failure_accounting(self):
        a=self.api('uav_bc_data')
        from test_env_scripted_policy import scripted_action
        from ppo_pi_env import MineUAVPIEnv
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            target=dict(target_id=5,env_seed=55,target=[1.,-.5,1.1])
            row,raw=a.collect_episode(env,target,scripted_action)
            self.assertEqual(row['termination_reason'],'time_limit');self.assertFalse(row['success'])
            self.assertTrue(row['failure']);self.assertEqual(len(raw['actions']),2)
            self.assertEqual(len(raw['observations']),3)
            np.testing.assert_array_equal(raw['previous_actions'][0],np.zeros(4))
            np.testing.assert_array_equal(raw['previous_actions'][1:],raw['actions'][:-1])
            np.testing.assert_array_equal(raw['step_index'],[0,1])
            for o,act in zip(raw['observations'],raw['actions']):np.testing.assert_array_equal(act,scripted_action(o))
        finally:env.close()
    def test_resume_rejects_identity_hash_and_duplicate_target(self):
        a=self.api('uav_bc_data');s=self.api('uav_bc_safety')
        from latent_dynamics_data import file_hash
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';s.atomic_npz(p,actions=np.zeros((1,4)))
            r=dict(identity={'runtime':'one'},target_id=1,path=str(p),sha256=file_hash(p))
            a.validate_record(r,{'runtime':'one'},1)
            with self.assertRaises(ValueError):a.validate_record(r,{'runtime':'two'},1)
            with self.assertRaises(ValueError):a.validate_record(r,{'runtime':'one'},2)
            r['sha256']='0'*64
            with self.assertRaises(ValueError):a.validate_record(r,{'runtime':'one'},1)
    def test_supervisor_actually_stops_child_under_memory_pressure(self):
        s=self.api('uav_bc_safety');g=1024**3
        with tempfile.TemporaryDirectory() as d:
            # Start is legal; simulated runtime pressure trips the real child
            # termination path, without consuming actual large memory.
            with patch.object(s,'available_memory',side_effect=[8*g,8*g,g]),patch.object(s,'oom_count',return_value=0):
                with self.assertRaises(RuntimeError):s.supervise([sys.executable,'-c','import time; time.sleep(20)'],d,'pressure')
            r=json.loads((Path(d)/'resource_pressure.json').read_text())
            self.assertIsNotNone(r['failure']);self.assertNotEqual(r['exit_code'],0)
            self.assertEqual(r['minimum_available_bytes'],g)
    def test_supervisor_rejects_concurrent_phase(self):
        import fcntl
        s=self.api('uav_bc_safety')
        with tempfile.TemporaryDirectory() as d:
            with (Path(d)/'phase.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):s.supervise([sys.executable,'-c','pass'],d,'second')
    def test_supervisor_sigterm_reaps_owned_child_and_records_interruption(self):
        s=self.api('uav_bc_safety')
        with tempfile.TemporaryDirectory() as d:
            child=None
            command=[sys.executable,s.__file__,'--directory',d,'--phase','terminated','--',
                sys.executable,'-c','import os,time; print(os.getpid(),flush=True); time.sleep(20)']
            supervisor=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                child=int(supervisor.stdout.readline().strip())
                supervisor.send_signal(signal.SIGTERM);supervisor.wait(timeout=5)
                try:os.kill(child,0);alive=True
                except ProcessLookupError:alive=False
                self.assertFalse(alive,'terminated supervisor left its child unmonitored')
                r=json.loads((Path(d)/'resource_terminated.json').read_text())
                self.assertNotEqual(r['exit_code'],0);self.assertIn('signal',r['failure'])
            finally:
                if supervisor.poll() is None:supervisor.kill();supervisor.wait()
                if child is not None:
                    try:os.killpg(child,signal.SIGTERM)
                    except ProcessLookupError:pass
                supervisor.stdout.close();supervisor.stderr.close()

if __name__=='__main__':unittest.main()
