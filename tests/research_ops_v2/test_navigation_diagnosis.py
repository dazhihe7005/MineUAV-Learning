"""AUDIT-01 pure offline evidence tests; synthetic rows are explicitly fixtures."""
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


def api():
    assert importlib.util.find_spec('research_regression.navigation_diagnosis'), 'offline diagnosis analyzer missing'
    from research_regression import navigation_diagnosis
    return navigation_diagnosis


def scene(identity='a',layout='empty'):
    return {'identity_sha256':identity,'split':'train','layout':layout,'start_xyz':[0,0,1],
            'target_xyz':[2,0,1]}


def episode(**updates):
    row={'scenario_identity_sha256':'a','outcome':'time_limit','policy_end_step':375,
         'episode_policy_steps':375,'episode_return':-6.0,'final_target_distance_m':2.,
         'mean_action_norm':.5,'trajectory_length_m':1.,'net_displacement_m':.3,
         'motionless_episode':False,'sim_end_time_s':15.}
    row.update(updates);return row


class NavigationDiagnosisTests(unittest.TestCase):
    def test_episode_and_layout_counts_and_terminal_distance_are_distinct_from_closest(self):
        m=api();out=m.summarize_episodes([episode()],{'a':scene()})
        self.assertEqual(out['counts']['time_limit'],1);self.assertEqual(out['by_layout']['empty']['episodes'],1)
        self.assertEqual(out['final_distance_threshold_counts']['<0.1m'],0)
        self.assertEqual(out['closest_approach'],'NOT OBSERVABLE')

    def test_validation_or_unknown_identity_cannot_be_counted_as_train(self):
        m=api()
        for manifest in ({'a':scene('a')|{'split':'validation'}},{'b':scene('b')}):
            with self.assertRaises(ValueError):m.summarize_episodes([episode()],manifest)

    def test_unknown_nonfinite_or_inconsistent_terminal_rows_fail_closed(self):
        m=api()
        for row in [episode(outcome='weird'),episode(episode_return=math.nan),
                    episode(terminal_class='collision'),episode(mean_action_norm=2.1),
                    episode(policy_end_step=1)]:
            with self.assertRaises(ValueError):m.summarize_episodes([row],{'a':scene()})

    def test_one_snapshot_is_not_a_learning_curve(self):
        m=api();out=m.summarize_ppo_snapshot({'train/std':.98,'train/approx_kl':.0001},100096)
        self.assertEqual(out['temporal_trend'],'NOT OBSERVABLE')
        self.assertEqual(out['snapshot_count'],1)

    def test_time_bins_keep_exact_step_intervals_and_scene_groups(self):
        m=api();rows=[episode(outcome='collision',policy_end_step=100,episode_policy_steps=100),
                     episode(policy_end_step=30375)]
        out=m.summarize_episodes(rows,{'a':scene()})
        self.assertEqual([b['episodes'] for b in out['step_blocks']],[1,1,0,0])
        self.assertEqual(out['by_layout']['empty']['counts']['collision'],1)

    def test_reward_timeout_telescopes_but_collision_progress_not_reconstructed(self):
        m=api();out=m.reward_bounds(episode(),scene())
        self.assertAlmostEqual(out['progress_sum'],0.)
        self.assertAlmostEqual(out['time_and_timeout_sum'],-5.75)
        out=m.reward_bounds(episode(outcome='collision'),scene())
        self.assertEqual(out['progress_sum'],'NOT OBSERVABLE')

    def test_action_bounds_do_not_square_mean_as_exact_energy(self):
        m=api();bounds=m.reward_bounds(episode(mean_action_norm=.5),scene())['action_penalty_bounds']
        self.assertLessEqual(bounds[0],-.005*375*.5**2);self.assertNotEqual(bounds[0],bounds[1])

    def test_causal_rows_reject_future_or_bad_time_without_inferring_positions(self):
        m=api();row={'policy_step':1,'simulation_time':.04,'lidar_sample_time':0.,'lidar_age_s':.04,
                      'lidar_status':'FRESH','physics_steps':20,'scenario_identity_sha256':'a'}
        self.assertEqual(m.summarize_causality([row],{'a':scene()})['max_age_s'],.04)
        for change in ({'lidar_sample_time':.1},{'lidar_age_s':.3},{'lidar_status':'STALE'}):
            with self.assertRaises(ValueError):m.summarize_causality([row|change],{'a':scene()})

    def test_corrupt_duplicate_key_jsonl_is_rejected(self):
        m=api()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'rows.jsonl';p.write_text('{"x":1,"x":2}\n')
            with self.assertRaises(ValueError):m.read_jsonl(p)

    def test_sealed_raw_file_tampering_is_rejected(self):
        m=api()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'raw.json';p.write_text('{}')
            with self.assertRaises(ValueError):m.verify_seal(Path(directory),{'raw.json':'0'*64})

    def test_figure_serialization_removes_metadata_not_pixel_measurements(self):
        import struct,zlib
        assert importlib.util.find_spec('research_regression.navigation_diagnosis_plots'),'audit plot serializer missing'
        from research_regression.navigation_diagnosis_plots import clean_png
        def chunk(kind,payload):return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
        pixel=zlib.compress(b'\x00\x01\x02\x03')
        source=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0))+chunk(b'tEXt',b'Software\x00fixture')+chunk(b'IDAT',pixel)+chunk(b'IEND',b'')
        out=clean_png(source);self.assertIn(pixel,out);self.assertNotIn(b'Software',out)

    def test_jsonl_overflow_in_unconsumed_field_is_not_silently_accepted(self):
        m=api()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'rows.jsonl';p.write_text('{"x":1e309}\n')
            with self.assertRaises(ValueError):m.read_jsonl(p)


if __name__=='__main__':unittest.main()
