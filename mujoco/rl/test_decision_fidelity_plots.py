"""Figure protocol must not select states by model performance."""
import importlib
import importlib.util
import unittest


class PlotProtocolTests(unittest.TestCase):
    def test_required_ten_figures_and_fixed_example(self):
        self.assertIsNotNone(importlib.util.find_spec('decision_fidelity_plots'),'figure renderer missing')
        api=importlib.import_module('decision_fidelity_plots')
        self.assertEqual(len(api.FIGURES),10); self.assertEqual(len(set(api.FIGURES)),10)
        self.assertEqual(api.EXAMPLE,('benchmark',0))
        self.assertIn('normalized_regret_distribution.png',api.FIGURES)


if __name__=='__main__': unittest.main()
