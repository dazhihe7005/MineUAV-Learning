import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from support import project,git


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();base=Path(self.temp.name)
        self.root=project(base/'repo');self.remote=base/'remote.git'
        import subprocess
        subprocess.run(['git','init','--bare',str(self.remote)],check=True,capture_output=True)
        git(self.root,'remote','add','origin',str(self.remote))

    def tearDown(self):self.temp.cleanup()

    def manager(self):
        assert importlib.util.find_spec('research_ops.lifecycle'), 'synchronization lifecycle missing'
        from research_ops.lifecycle import Operations
        m=Operations(self.root);m.initialize();return m

    def test_real_commit_push_and_remote_sha_are_verified(self):
        m=self.manager();(self.root/'personal.txt').write_text('keep local')
        result=m.sync()
        self.assertEqual(result['status'],'PASS')
        head=git(self.root,'rev-parse','HEAD')
        remote=git(self.root,'ls-remote','origin','refs/heads/feat/ops-test').split()[0]
        self.assertEqual(head,remote)
        self.assertEqual(m.status()['state']['latest_successful_push']['commit'],head)
        self.assertFalse(git(self.root,'ls-files','personal.txt'))
        self.assertEqual((self.root/'personal.txt').read_text(),'keep local')
        self.assertTrue((self.root/'docs/DAILY_PROGRESS.md').is_file())

    def test_failed_push_preserves_commit_and_sync_pending_retries_same_head(self):
        m=self.manager();git(self.root,'remote','set-url','origin',str(self.remote.parent/'missing.git'))
        result=m.sync();self.assertEqual(result['status'],'SYNC_PENDING')
        head=git(self.root,'rev-parse','HEAD')
        git(self.root,'remote','set-url','origin',str(self.remote))
        self.assertEqual(m.sync()['status'],'PASS')
        self.assertEqual(git(self.root,'rev-parse','HEAD'),head)

    def test_unrelated_staged_file_blocks_without_unstaging_it(self):
        m=self.manager();(self.root/'private.txt').write_text('mine');git(self.root,'add','private.txt')
        with self.assertRaises(Exception):m.sync()
        self.assertEqual(git(self.root,'diff','--cached','--name-only'),'private.txt')

    def test_secret_and_symlink_never_reach_remote(self):
        m=self.manager();(self.root/'README.md').write_text('ghp_'+'x'*40)
        before=git(self.root,'rev-parse','HEAD')
        with self.assertRaises(Exception):m.sync()
        self.assertEqual(git(self.root,'rev-parse','HEAD'),before)
        (self.root/'README.md').unlink();(self.root/'README.md').symlink_to(self.root/'history/failure.json')
        with self.assertRaises(Exception):m.sync()

    def test_main_branch_sync_is_rejected(self):
        m=self.manager();git(self.root,'switch','-c','main')
        with self.assertRaises(Exception):m.sync()

    def test_recover_pending_push_does_not_change_sealed_document_commit(self):
        m=self.manager();git(self.root,'remote','set-url','origin',str(self.remote.parent/'missing.git'))
        self.assertEqual(m.sync()['status'],'SYNC_PENDING')
        before=(self.root/'docs/DAILY_PROGRESS.md').read_bytes()
        m.recover()
        self.assertEqual((self.root/'docs/DAILY_PROGRESS.md').read_bytes(),before)
        git(self.root,'remote','set-url','origin',str(self.remote))
        self.assertEqual(m.sync()['status'],'PASS')
        self.assertEqual(git(self.root,'status','--porcelain'),'')

    def test_remote_ahead_is_never_force_pushed(self):
        m=self.manager();m.sync()
        import subprocess
        other=self.remote.parent/'other'
        subprocess.run(['git','clone','--branch','feat/ops-test',str(self.remote),str(other)],check=True,capture_output=True)
        git(other,'config','user.name','Other');git(other,'config','user.email','other@example.invalid')
        (other/'remote-note').write_text('preserve')
        git(other,'add','remote-note');git(other,'commit','-m','remote change');git(other,'push')
        remote_head=git(other,'rev-parse','HEAD')
        self.assertEqual(m.sync()['status'],'SYNC_PENDING')
        self.assertEqual(git(self.root,'ls-remote','origin','refs/heads/feat/ops-test').split()[0],remote_head)

    def test_phase_report_is_generated_from_explicit_sections(self):
        from research_ops import documents
        self.assertTrue(hasattr(documents,'render_phase'),'phase report renderer missing')
        sections={str(i): 'evidence '+str(i) for i in range(1,12)}
        text=documents.render_phase('OPS-01',sections)
        self.assertIn('evidence 11',text)
        del sections['7']
        with self.assertRaises(Exception):documents.render_phase('OPS-01',sections)

    def test_existing_unpublished_commit_outside_allowlist_is_never_pushed(self):
        m=self.manager();(self.root/'private.txt').write_text('private data')
        git(self.root,'add','private.txt');git(self.root,'commit','-m','unrelated local commit')
        with self.assertRaises(Exception):m.sync()
        self.assertEqual(git(self.root,'ls-remote','origin','refs/heads/feat/ops-test'),'')

    def test_alternate_pushurl_is_rejected_before_any_publication(self):
        m=self.manager();other=self.remote.parent/'wrong-destination.git'
        import subprocess
        subprocess.run(['git','init','--bare',str(other)],check=True,capture_output=True)
        git(self.root,'remote','set-url','--push','origin',str(other))
        result=m.sync()
        self.assertEqual(result['status'],'SYNC_PENDING')
        self.assertEqual(git(self.root,'ls-remote',str(other),'refs/heads/feat/ops-test'),'')

    def test_three_positive_strings_do_not_prove_stage_readiness(self):
        config_path=self.root/'research_ops/operations_config.json'
        config=json.loads(config_path.read_text());config['readiness_report']='docs/fake.json'
        config['required_delivery_paths']=['docs/fake.json']
        config_path.write_text(json.dumps(config))
        (self.root/'docs/fake.json').write_text(json.dumps({'implementation_status':'PASS','tests':{'status':'PASS'},'review':{'status':'GO'}}))
        m=self.manager()
        with self.assertRaises(Exception):m.sync()

    def test_only_matching_runtime_and_test_evidence_can_complete_delivery(self):
        from research_ops.identity import source_identity
        from research_ops.storage import digest,sha256
        config_path=self.root/'research_ops/operations_config.json'
        config=json.loads(config_path.read_text());config['readiness_report']='docs/evidence.json'
        config['required_delivery_paths']=['docs/evidence.json'];config['sync_files'].append('docs/evidence.json')
        config_path.write_text(json.dumps(config))
        cases=self.root/'tests/research_ops';cases.mkdir(parents=True)
        (cases/'case.py').write_text('# SYNTHETIC_MANAGER_FIXTURE, no ML\n')
        m=self.manager();protocol=json.loads((m.local/'protocol.json').read_text())
        source=digest(protocol['source_identity']);test_identity=source_identity(self.root,['tests/research_ops'])
        evidence=m.local/'unit-fixture.json'
        evidence.write_text(json.dumps({'fixture':'SYNTHETIC_MANAGER_FIXTURE','complete':True,
            'failures':0,'errors':0,'optimizer_updates':0,'runtime_identity_sha256':source,
            'test_source_identity':test_identity,'results':[{'nodeid':'synthetic-case','outcome':'PASS'}]}))
        report={'project_id':'MineUAV-Learning','phase_id':'OPS-01','implementation_status':'PASS',
            'protocol_sha256':m.status()['state']['protocol_sha256'],'runtime_identity_sha256':source,
            'tests':{'status':'PASS','runtime_identity_sha256':source,'test_source_identity':test_identity,
                'evidence':{'path':str(evidence.relative_to(self.root)),'sha256':sha256(evidence)}},
            'review':{'status':'GO','runtime_identity_sha256':source}}
        (self.root/'docs/evidence.json').write_text(json.dumps(report))
        self.assertTrue(m.sync()['state']['phase_complete'])
        (cases/'case.py').write_text('# changed test after evidence\n')
        with self.assertRaises(Exception):m.sync()

    def test_pending_delivery_refuses_new_working_tree_document_edits(self):
        m=self.manager();git(self.root,'remote','set-url','origin',str(self.remote.parent/'missing.git'))
        self.assertEqual(m.sync()['status'],'SYNC_PENDING')
        with (self.root/'docs/DAILY_PROGRESS.md').open('a') as f:f.write('\nnew user note\n')
        git(self.root,'remote','set-url','origin',str(self.remote))
        with self.assertRaises(Exception):m.sync()


if __name__=='__main__':unittest.main()
