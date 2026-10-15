import tempfile
import subprocess
import unittest
from pathlib import Path
from support_v2 import fixture,proposal,review,git
from research_ops.storage import OpsError,atomic_json,sha256,digest,read_json
from research_ops.identity import source_identity
from research_ops_v2.manager import OperationsV2


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();base=Path(self.tmp.name)
        self.root=fixture(base/'repo');self.remote=base/'remote.git'
        subprocess.run(['git','init','--bare',str(self.remote)],capture_output=True,check=True)
        git(self.root,'remote','add','origin',str(self.remote))
        self.m=OperationsV2(self.root);self.reg=proposal(self.root)
        self.m.register(self.reg,review(self.reg),migrating=True)
        self.ready()
    def tearDown(self):self.tmp.cleanup()

    def ready(self):
        for name in self.reg['sync_files']:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_text('{}\n' if path.suffix=='.json' else '# 测试夹具\n')
        local=self.root/'.research_local';evidence=local/'tests.json';review_path=local/'review.json'
        atomic_json(evidence,{'complete':True,'tests_run':1,'failures':0,'errors':0,'optimizer_updates':0,
            'source_identity_sha256':digest(self.reg['source_identity']),
            'test_source_identity':source_identity(self.root,['tests/research_ops_v2']),
            'scope':'SYNTHETIC PUBLICATION TEST FIXTURE, not real stage verification'})
        atomic_json(review_path,review(self.reg))
        atomic_json(self.root/self.reg['report_path'],{'implementation_status':'PASS',
            'registration_sha256':digest(self.reg),'source_identity_sha256':digest(self.reg['source_identity']),
            'tests':{'status':'PASS','evidence':{'path':str(evidence.relative_to(self.root)),'sha256':sha256(evidence)}},
            'review':{'status':'GO','source_identity_sha256':digest(self.reg['source_identity']),
                'evidence':{'path':str(review_path.relative_to(self.root)),'sha256':sha256(review_path)}}})

    def test_real_feature_commit_push_and_restart_verify(self):
        (self.root/'personal.txt').write_text('keep local')
        result=self.m.sync();self.assertEqual(result['status'],'PASS');self.assertTrue(result['phase_complete'])
        head=git(self.root,'rev-parse','HEAD')
        self.assertEqual(head,git(self.root,'ls-remote','origin','refs/heads/feat/ops-test').split()[0])
        self.assertEqual(OperationsV2(self.root).recover()['state']['latest_successful_push']['commit'],head)
        self.assertFalse(git(self.root,'ls-files','personal.txt'))

    def test_failed_push_then_recover_retries_exact_commit(self):
        git(self.root,'remote','set-url','origin',str(self.remote.parent/'missing.git'))
        self.assertEqual(self.m.sync()['status'],'SYNC_PENDING')
        head=git(self.root,'rev-parse','HEAD');self.m.recover()
        self.assertEqual(self.m.status()['state']['stage_status'],'SYNC_PENDING')
        with self.assertRaises(OpsError):self.m.start('audit')
        git(self.root,'remote','set-url','origin',str(self.remote))
        self.assertEqual(self.m.sync()['status'],'PASS');self.assertEqual(git(self.root,'rev-parse','HEAD'),head)

    def test_pre_registration_unpublished_secret_commit_is_not_trusted(self):
        # New registration base is deliberately after the unsafe local commit.
        (self.root/'private.txt').write_text('must never publish')
        git(self.root,'add','private.txt');git(self.root,'commit','-m','unreviewed local history')
        new=proposal(self.root,phase='OPS-03');self.m.register(new,review(new));self.reg=new;self.ready()
        with self.assertRaisesRegex(OpsError,'unregistered path'):self.m.sync()
        self.assertEqual(git(self.root,'ls-remote','origin'),'')

    def test_user_staging_not_changed(self):
        (self.root/'mine.txt').write_text('mine');git(self.root,'add','mine.txt')
        with self.assertRaisesRegex(OpsError,'staged'):self.m.sync()
        self.assertEqual(git(self.root,'diff','--cached','--name-only'),'mine.txt')

    def test_changed_test_source_invalidates_readiness(self):
        from research_ops_v2.publication import readiness
        p=self.root/'tests/research_ops_v2/test_versions.py';p.write_text('# changed tests\n')
        with self.assertRaisesRegex(OpsError,'not complete'):readiness(self.m,self.reg)

    def test_failing_test_evidence_and_forged_positive_strings_refused(self):
        path=self.root/'.research_local/tests.json';data=read_json(path);data['failures']=1;atomic_json(path,data)
        report=read_json(self.root/self.reg['report_path']);report['tests']['evidence']['sha256']=sha256(path)
        atomic_json(self.root/self.reg['report_path'],report)
        with self.assertRaisesRegex(OpsError,'not complete'):self.m.sync()

    def test_secret_raw_large_and_png_payload_refused(self):
        from research_ops_v2.publication import check_blob
        for name,data in [('docs/a.md',('ghp_'+'x'*40).encode()),('docs/a.md',b'a'*(2*1024**2+1)),
                          ('mujoco/reports/figures/a.png',b'not png')]:
            with self.assertRaises(OpsError):check_blob(name,data)

    def test_historical_report_not_publishable_even_if_registered(self):
        p=self.root/'history/failure.json';before=p.read_bytes()
        # delivery path whitelist already refuses this protected path.
        with self.assertRaises(OpsError):proposal(self.root,sync_files=['history/failure.json'])
        self.assertEqual(p.read_bytes(),before)


if __name__=='__main__':unittest.main()
