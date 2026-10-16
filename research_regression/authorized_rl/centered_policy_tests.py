"""R13 no-update neural contracts; no environment reset/step or checkpoint reuse."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from .prior_tests import ObservationOnly


class CenteredPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a, **kw):
            raise AssertionError('R13_NO_OPTIMIZER_OR_LEARNING')
        for cls, name in ((PPO, 'learn'), (PPO, 'train'), (torch.optim.Adam, 'step')):
            guard = patch.object(cls, name, forbidden)
            guard.start()
            self.addCleanup(guard.stop)

    def module(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.state_centered_policy'),
                             'missing R13 centered Actor implementation')
        from . import state_centered_policy
        return state_centered_policy

    def obs(self, n=4):
        rng = np.random.default_rng(72)
        return {'lidar': rng.uniform(0, 1, (n, 3, 12, 24)).astype(np.float32),
                'state': rng.normal(0, .5, (n, 7)).astype(np.float32),
                'heading': np.array([[np.sin(t), np.cos(t)] for t in np.linspace(-np.pi, np.pi, n)], dtype=np.float32),
                'lidar_age': np.linspace(0, .14, n, dtype=np.float32)[:, None]}

    def test_zero_state_arbitrary_background_and_constant_head_bias_cancel(self):
        m = self.module().make_centered_ppo(ObservationOnly(), seed=0)
        obs = self.obs(7)
        obs['state'][:] = 0
        with torch.no_grad():
            m.policy.action_net.linear.bias.fill_(8.)  # FORCED_TEST_STATE, not optimizer update.
        tensor, _ = m.policy.obs_to_tensor(obs)
        mean = m.policy.get_distribution(tensor).distribution.mean.detach().numpy()
        self.assertTrue(np.array_equal(mean, np.zeros((7, 4))))

    def test_initial_values_and_actions_match_frozen_prior_but_class_identity_is_new(self):
        from lidar_integration.ppo_preflight import make_ppo
        from .actor_prior import initialize_actor_prior
        mod = self.module()
        m = mod.make_centered_ppo(ObservationOnly(), seed=0)
        self.assertEqual(mod.canonical_policy_sha(m.policy), 'bc1adbd197b52b4bfd0bb759a749c16e351092960557f2d341ba3321876b39ba')
        identity = mod.initialize_centered_prior(m)
        self.assertEqual(identity['canonical_policy_sha256'], '6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74')
        old = make_ppo(ObservationOnly(), seed=0)
        initialize_actor_prior(old)
        with torch.no_grad():
            old.policy.log_std.fill_(np.log(.3))
        obs = self.obs(62)
        t, _ = m.policy.obs_to_tensor(obs)
        new_dist, old_dist = m.policy.get_distribution(t), old.policy.get_distribution(t)
        self.assertTrue(torch.equal(new_dist.distribution.mean, old_dist.distribution.mean))
        self.assertTrue(torch.equal(new_dist.distribution.stddev, old_dist.distribution.stddev))
        self.assertTrue(torch.equal(m.policy.predict_values(t), old.policy.predict_values(t)))
        self.assertTrue(np.array_equal(m.predict({k:v[0] for k,v in obs.items()}, deterministic=True)[0], m.predict(obs, deterministic=True)[0][0]))

    def test_all_sb3_actor_paths_agree_and_backward_is_finite_without_update(self):
        mod = self.module()
        m = mod.make_centered_ppo(ObservationOnly(), seed=0)
        mod.initialize_centered_prior(m)
        before = copy.deepcopy(m.policy.state_dict())
        obs = self.obs(8)
        t, _ = m.policy.obs_to_tensor(obs)
        actions, values, logp = m.policy(t, deterministic=True)
        dist = m.policy.get_distribution(t)
        ev, ep, entropy = m.policy.evaluate_actions(t, actions)
        self.assertTrue(torch.equal(actions, dist.distribution.mean))
        self.assertTrue(torch.equal(values, ev))
        self.assertTrue(torch.equal(logp, ep))
        self.assertTrue(torch.equal(logp, dist.log_prob(actions)))
        self.assertTrue(np.array_equal(np.clip(actions.detach().numpy(), -1, 1), m.predict(obs, deterministic=True)[0]))
        loss = (actions - .1).square().mean() + (values - 1).square().mean() - .001 * entropy.mean()
        loss.backward()
        grads = {k:p.grad for k,p in m.policy.named_parameters() if p.grad is not None}
        self.assertTrue(all(torch.isfinite(v).all() for v in grads.values()))
        self.assertTrue(any(k.startswith(('mlp_extractor.base.policy_net', 'action_net.linear')) and torch.count_nonzero(v) for k,v in grads.items()))
        self.assertTrue(any(k.startswith(('mlp_extractor.base.value_net', 'value_net')) and torch.count_nonzero(v) for k,v in grads.items()))
        self.assertTrue(all(torch.equal(v, m.policy.state_dict()[k]) for k,v in before.items()))
        self.assertFalse(m.policy.optimizer.state)
        self.assertEqual(m.num_timesteps, 0)

    def test_wrapped_parameters_stay_in_optimizer_and_roundtrip_preserves_policy(self):
        mod = self.module()
        m = mod.make_centered_ppo(ObservationOnly(), seed=0)
        mod.initialize_centered_prior(m)
        policy_ids = [id(p) for p in m.policy.parameters()]
        opt_ids = [id(p) for g in m.policy.optimizer.param_groups for p in g['params']]
        self.assertEqual(policy_ids, opt_ids)
        self.assertEqual(len(opt_ids), len(set(opt_ids)))
        obs = self.obs(3)
        before = m.predict(obs, deterministic=True)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'R13_TEMP_NEW_INITIALIZATION.zip'
            m.save(path)
            loaded = PPO.load(path, device='cpu')
            self.assertIs(type(loaded.policy), mod.StateCenteredPolicy)
            self.assertTrue(np.array_equal(before, loaded.predict(obs, deterministic=True)[0]))
            self.assertEqual(mod.canonical_policy_sha(m.policy), mod.canonical_policy_sha(loaded.policy))
            mod.verify_centered_policy(loaded)
            self.assertEqual(loaded.learning_rate, 3e-5)
            self.assertEqual(loaded.policy.optimizer.param_groups[0]['lr'], 3e-5)
            self.assertFalse(loaded.policy.optimizer.state)

    def test_nonfresh_or_wrong_seed_initialization_refused_without_mutation(self):
        mod = self.module()
        for seed, steps in ((1, 0), (0, 128)):
            m = mod.make_centered_ppo(ObservationOnly(), seed=seed)
            m.num_timesteps = steps  # FORCED_TEST_STATE; no training.
            before = copy.deepcopy(m.policy.state_dict())
            with self.assertRaises(ValueError):
                mod.initialize_centered_prior(m)
            self.assertTrue(all(torch.equal(v, m.policy.state_dict()[k]) for k,v in before.items()))

    def test_structure_corruption_refused_not_silently_unanchored(self):
        mod = self.module()
        m = mod.make_centered_ppo(ObservationOnly(), seed=0)
        m.policy.action_net = m.policy.action_net.linear  # FORCED_TEST_STATE.
        with self.assertRaises(ValueError):
            mod.verify_centered_policy(m)

    def test_current_lidar_features_can_condition_nonzero_state_response(self):
        mod = self.module()
        m = mod.make_centered_ppo(ObservationOnly(), seed=0)
        net = m.policy.mlp_extractor.base.policy_net
        head = m.policy.action_net.linear
        with torch.no_grad():  # FORCED_TEST_STATE, demonstrates expressivity, not learning.
            for p in (*net.parameters(), *head.parameters()):
                p.zero_()
            net[0].weight[0, 0] = 1
            net[0].weight[0, 64] = 1
            net[2].weight[0, 0] = 1
            head.weight[0, 0] = 1
        features = torch.zeros((2, 74))
        features[:, 64] = .2
        features[1, 0] = 1
        response = m.policy.action_net(m.policy.mlp_extractor.forward_actor(features))
        self.assertGreater(float(abs(response[0, 0] - response[1, 0]).detach()), .01)
        self.assertTrue(torch.isfinite(response).all())

    def test_nonfinite_parameter_and_incompatible_features_are_rejected(self):
        mod = self.module()
        m = mod.make_centered_ppo(ObservationOnly(), seed=0)
        with torch.no_grad():
            m.policy.log_std[0] = float('inf')  # FORCED_TEST_STATE.
        with self.assertRaises(ValueError):
            mod.verify_centered_policy(m)
        with self.assertRaises(ValueError):
            mod.StateCenteredPolicy(ObservationOnly.observation_space, ObservationOnly.action_space,
                                    lambda _: 3e-5, share_features_extractor=False)

    def test_only_actor_function_and_declared_lr_differ_from_frozen_factory(self):
        from lidar_integration.ppo_preflight import make_ppo
        mod = self.module()
        old = make_ppo(ObservationOnly(), seed=0)
        new = mod.make_centered_ppo(ObservationOnly(), seed=0)
        for name in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef',
                     'vf_coef', 'max_grad_norm', 'normalize_advantage', 'target_kl', 'use_sde'):
            self.assertEqual(getattr(new, name), getattr(old, name))
        self.assertEqual(new.clip_range(.5), old.clip_range(.5))
        self.assertEqual(new.policy.optimizer.defaults, {**old.policy.optimizer.defaults, 'lr': 3e-5})
        self.assertEqual(sum(p.numel() for p in new.policy.parameters()), sum(p.numel() for p in old.policy.parameters()))

    def test_stateless_actor_critic_cnn_corruption_is_refused(self):
        mod = self.module()
        for part in ('actor', 'critic', 'cnn'):
            m = mod.make_centered_ppo(ObservationOnly(), seed=0)
            if part == 'actor':
                m.policy.mlp_extractor.base.policy_net[1] = torch.nn.ReLU()
            elif part == 'critic':
                m.policy.mlp_extractor.base.value_net[1] = torch.nn.ReLU()
            else:
                m.policy.features_extractor.lidar_net[1] = torch.nn.Tanh()
            with self.assertRaises(ValueError, msg=part):
                mod.verify_centered_policy(m)
            with self.assertRaises(ValueError, msg=part):
                mod.initialize_centered_prior(m)

    def test_actual_historical_panel_mean_std_value_log_probability_equivalence(self):
        from .feedback_followup import load_background
        from .feedback_spec import probe_cases
        from lidar_integration.ppo_preflight import make_ppo
        from .actor_prior import initialize_actor_prior
        mod = self.module()
        new = mod.make_centered_ppo(ObservationOnly(), seed=0)
        mod.initialize_centered_prior(new)
        old = make_ppo(ObservationOnly(), seed=0)
        initialize_actor_prior(old)
        with torch.no_grad():
            old.policy.log_std.fill_(np.log(.3))
        base, _, _, _ = load_background()
        cases = probe_cases()
        obs = {k:np.repeat(a[None], len(cases), axis=0) for k,a in base.items()}
        obs['state'] = np.asarray([c['state'] for c in cases], dtype=np.float32)
        t, _ = new.policy.obs_to_tensor(obs)
        nd, od = new.policy.get_distribution(t), old.policy.get_distribution(t)
        actions = torch.linspace(-.8, .8, 62*4).reshape(62, 4)
        for a,b in ((nd.distribution.mean, od.distribution.mean),
                    (nd.distribution.stddev, od.distribution.stddev),
                    (new.policy.predict_values(t), old.policy.predict_values(t)),
                    (nd.log_prob(actions), od.log_prob(actions))):
            self.assertTrue(torch.equal(a,b))

    def test_constant_head_bias_cancels_at_nonzero_state(self):
        mod = self.module()
        m = mod.make_centered_ppo(ObservationOnly(), seed=0)
        obs, _ = m.policy.obs_to_tensor(self.obs(8))
        with torch.no_grad():
            before = m.policy.get_distribution(obs).distribution.mean.clone()
            m.policy.action_net.linear.bias.add_(.25)  # FORCED_TEST_STATE.
            after = m.policy.get_distribution(obs).distribution.mean.clone()
        self.assertTrue(torch.allclose(before, after, atol=1e-7, rtol=0))


if __name__ == '__main__':
    unittest.main()
