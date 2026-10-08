"""The verifier must reject incomplete comparisons and false noise provenance."""
import importlib
import importlib.util
import hashlib
import unittest
import numpy as np


class SupportVerificationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('latent_mpc_support_verification'),'missing independent support verifier')
        return importlib.import_module('latent_mpc_support_verification')

    def test_comparison_cannot_lose_condition_split_or_episode(self):
        api=self.api()
        good={c:{s:dict(episodes=[{} for _ in range(100)]) for s in ('benchmark','holdout')} for c in ('unconstrained','train_supported')}
        self.assertEqual(api.verify_shape(good),400)
        for change in ('condition','split','episode'):
            bad={c:{s:dict(episodes=list(v['episodes'])) for s,v in g.items()} for c,g in good.items()}
            if change=='condition': del bad['unconstrained']
            elif change=='split': del bad['train_supported']['holdout']
            else: bad['train_supported']['holdout']['episodes'].pop()
            with self.subTest(change=change),self.assertRaises(AssertionError): api.verify_shape(bad)

    def test_first_epsilon_hash_rebuilt_from_original_episode_seed_and_std(self):
        api=self.api(); std=np.array([.2,.3,.4,.5]); seed=int(np.random.SeedSequence([0,1,4]).generate_state(1)[0])
        epsilon=np.random.default_rng(seed).normal(size=(510,10,4))*(.5*std)
        expected=hashlib.sha256(epsilon.astype('<f8').tobytes()).hexdigest()
        self.assertEqual(api.first_epsilon_hash('holdout',4,std),expected)
        self.assertNotEqual(api.first_epsilon_hash('benchmark',4,std),expected)
        self.assertNotEqual(api.first_epsilon_hash('holdout',4,std*.9),expected)


if __name__=='__main__': unittest.main()
