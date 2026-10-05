"""Read-only reproduction of saved one-step metrics, train stats and hashes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import file_hash, fit_statistics, flatten, json_hash, load_dataset
from latent_dynamics_models import load_model
from latent_dynamics_evaluation import apply_linear_probe, metrics, predict_episodes


def verify_report(path):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    for row in report['models'].values():
        assert file_hash(row['artifact']['path']) == row['artifact']['sha256'], 'model hash mismatch'
    splits, manifest = load_dataset(report['dataset']['source_directory'])
    assert manifest == report['dataset'], 'dataset/split/derived manifest mismatch'
    stats = fit_statistics(splits['train'])
    assert stats == report['normalization'], 'Train-only normalization mismatch'
    assert json_hash(stats) == report['normalization_sha256']
    truth = flatten(splits['test'], 'delta')
    torch.set_num_threads(1)
    latent_test = None
    for kind, row in report['models'].items():
        net, saved_stats, meta = load_model(row['artifact']['path'])
        assert saved_stats == stats
        assert meta['best_epoch'] == row['training']['best_epoch']
        validation = [x['validation_loss'] for x in row['training']['history']]
        assert meta['best_epoch'] == int(np.argmin(validation)) + 1
        assert len(validation) == report['trainer']['epochs']
        predictions, latent = predict_episodes(net, splits['test'], stats, history=(kind == 'history'))
        actual = metrics(np.concatenate(predictions), truth)
        expected = row['test']['physical_delta']
        for field in ('rmse', 'mae', 'r2'):
            assert np.isclose(actual['overall'][field], expected['overall'][field], rtol=1e-7, atol=1e-10)
        for dim in actual['per_dimension']:
            for field in ('rmse', 'mae', 'r2'):
                a, b = actual['per_dimension'][dim][field], expected['per_dimension'][dim][field]
                assert (a is None and b is None) or np.isclose(a, b, rtol=1e-7, atol=1e-10)
        if kind == 'history':
            latent_test = np.concatenate(latent)
    probe_artifact = report['linear_probe']['artifact']
    assert file_hash(probe_artifact['path']) == probe_artifact['sha256']
    probe = torch.load(probe_artifact['path'], map_location='cpu', weights_only=True)
    pred_pi = apply_linear_probe(latent_test, {'coefficient': probe['coefficient'].numpy()})
    actual_probe = metrics(pred_pi, flatten(splits['test'], 'pi'), names=('pi_x', 'pi_y', 'pi_z'))
    assert np.isclose(actual_probe['overall']['rmse'], report['linear_probe']['test']['overall']['rmse'], atol=1e-10)
    for name, artifact in report['latent']['local_only_artifacts'].items():
        assert file_hash(artifact['path']) == artifact['sha256'], f'latent artifact hash: {name}'
    root = path.parents[2]
    for relative, expected_hash in report['frozen_source_hashes'].items():
        assert file_hash(root / relative) == expected_hash, f'frozen source changed: {relative}'
    return dict(dataset_hashes='pass', split_manifest='pass', train_only_statistics='pass',
                best_validation_selection='pass', saved_model_metrics='pass', linear_probe='pass',
                latent_hashes='pass', frozen_environment_controller='pass')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path(__file__).resolve().parents[2] /
                        'mujoco/reports/latent_dynamics_one_step_seed0.json')
    print(json.dumps(verify_report(parser.parse_args().report), indent=2))
