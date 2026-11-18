"""Reconstruct test metrics and audit a portable five-run benchmark archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score, recall_score, roc_auc_score
from scipy.stats import pearsonr
from bioautoml.execution import atomic_json, file_hash, utc_now
from bioautoml.homology import read_sequences


def read(path):
    return json.loads(path.read_text())


def metric_names(model):
    return ['Sn', 'Sp', 'ACC', 'MCC', 'AUC'] if model <= 50 else (
        ['Sn', 'Sp', 'ACC', 'MCC'] if model <= 57 else ['MAE', 'MSE', 'RMSE', 'R2', 'Pearson'])


def near(actual, expected, context, atol=1e-9):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=atol):
        raise ValueError(f'{context}: reconstructed {actual}, stored {expected}')


def audit(root, output, include_manifest=True):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    datasets = sorted(root.glob('dataset*'), key=lambda p: int(p.name.split('_')[0][7:]))
    assert [int(p.name.split('_')[0][7:]) for p in datasets] == list(range(1, 65))
    rows, sources, warnings, input_records = [], [], [], []
    for directory in datasets:
        model = int(directory.name.split('_')[0][7:])
        task = int(directory.name.split('_')[-1])
        dtype = 'Protein' if directory.name.split('_')[-2] == 'protein' else 'DNA/RNA'
        records = {}
        for split in ('train', 'test'):
            files = sorted(p for p in (directory / split).glob('*') if p.suffix.lower() in ('.fa', '.fasta', '.faa', '.fna'))
            records[split] = read_sequences([str(p) for p in files], [p.stem for p in files], split, dtype, task)
            input_records.extend(dict(dataset=directory.name, split=split, path=str(p.relative_to(root)),
                                      sha256=file_hash(p)) for p in files)
        classes = sorted({r['label'] for r in records['train']}) if not task else []
        assert (task == 1 and model >= 58) or (task == 0 and ((len(classes) == 2) == (model <= 50)))
        truth = {r['sequence_id']: r['label'] for r in records['test']}
        if truth and not task and set(classes) - set(truth.values()):
            warnings.append(dict(model=model, missing_test_classes=sorted(set(classes)-set(truth.values())),
                                 convention='Test macro-averages use all trained classes; absent-class sensitivity is zero.'))
        paths = sorted((directory / 'runs').glob('run_*'))
        paths = [p for p in paths if p.is_dir()]
        assert [p.name for p in paths] == [f'run_{n}' for n in range(1, 6)]
        for run, path in enumerate(paths, 1):
            record = read(path / 'run.json')
            assert record['status'] == 'completed' and all((path / p).is_file() for p in record['outputs'])
            for stage in ('stage_1', 'stage_2'):
                h = read(path / f'reports/optimization/{stage}_study.json')
                assert h['status'] == 'completed' and h['settings']['search_seed'] == 6300 + run
                if stage == 'stage_2':
                    assert len(h['trials']) == 150 and h['stop_reason'] == 'trial_budget'
            cv_path = path / 'results/metrics/optimization_cv_metrics.csv'
            cv = pd.read_csv(cv_path).iloc[0]
            metrics = metric_names(model)
            used = [cv_path, path / 'run.json']
            for metric in metrics:
                value = float(cv[metric])
                if not np.isfinite(value):
                    raise ValueError(f'Undefined CV metric: {path}/{metric}')
                rows.append(dict(model=model, dataset=directory.name, run=run, split='cv', metric=metric,
                                 value=value, source=str(cv_path.relative_to(root))))
            if truth:
                pred_path = path / 'results/predictions/test_predictions.csv'
                pred = pd.read_csv(pred_path, dtype={'internal_id':str, 'prediction':str})
                assert not pred.internal_id.duplicated().any()
                assert set(pred.internal_id) == set(truth), f'Test IDs mismatch: {path}'
                used.append(pred_path)
                if task:
                    y = np.array([float(truth[i]) for i in pred.internal_id])
                    guess = pred.prediction.astype(float).to_numpy()
                    mse = mean_squared_error(y, guess)
                    values = dict(MAE=mean_absolute_error(y, guess), MSE=mse, RMSE=math.sqrt(mse),
                                  R2=r2_score(y, guess), Pearson=pearsonr(y, guess).statistic)
                    metrics_path = path / 'results/metrics/test_metrics.csv'
                    stored = pd.read_csv(metrics_path).set_index('Metric').Value
                    for metric in metrics:
                        near(values[metric], stored[metric], f'{path}/{metric}')
                else:
                    y = np.array([truth[i] for i in pred.internal_id])
                    guess = pred.prediction.to_numpy()
                    assert set(guess) <= set(classes)
                    probabilities = pred[classes].astype(float).to_numpy()
                    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
                    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                    assert np.array_equal(np.array(classes)[probabilities.argmax(axis=1)], guess)
                    cm = confusion_matrix(y, guess, labels=classes)
                    spec = []
                    for i in range(len(classes)):
                        tn = cm.sum() - cm[i,:].sum() - cm[:,i].sum() + cm[i,i]
                        fp = cm[:,i].sum() - cm[i,i]
                        spec.append(tn / (tn+fp) if tn+fp else 0.)
                    binary = len(classes) == 2
                    sensitivity = (recall_score(y, guess, pos_label=classes[-1], zero_division=0) if binary
                                   else recall_score(y, guess, labels=classes, average='macro', zero_division=0))
                    values = dict(ACC=accuracy_score(y, guess), MCC=matthews_corrcoef(y, guess),
                                  Sn=sensitivity,
                                  Sp=spec[-1] if binary else np.mean(spec))
                    if binary:
                        values['AUC'] = roc_auc_score(y == classes[-1], probabilities[:, -1])
                    metrics_path = path / 'results/metrics/test_additional_metrics.csv'
                    stored = pd.read_csv(metrics_path).set_index('Metric').Value
                    for ours, theirs in [('ACC', 'Accuracy'), ('MCC', 'MCC')] + ([('AUC','AUC')] if binary else []):
                        near(values[ours], stored[theirs], f'{path}/{ours}', atol=1e-7 if ours=='AUC' else 1e-9)
                    cm_path = path / 'results/metrics/test_confusion_matrix.csv'
                    saved_cm = pd.read_csv(cm_path, index_col=0)
                    saved_cm.index = saved_cm.index.astype(str)
                    assert set(saved_cm.index)-{'All'} <= set(classes)
                    assert set(saved_cm.columns)-{'All'} <= set(classes)
                    assert saved_cm.loc['All','All'] == len(y)
                    assert np.array_equal(saved_cm.reindex(index=classes, columns=classes, fill_value=0).to_numpy(), cm)
                    used.append(cm_path)
                used.append(metrics_path)
                for metric in metrics:
                    if not np.isfinite(values[metric]):
                        raise ValueError(f'Undefined test metric: {path}/{metric}')
                    rows.append(dict(model=model, dataset=directory.name, run=run, split='test', metric=metric,
                                     value=float(values[metric]), source=str(pred_path.relative_to(root))))
            for p in used:
                sources.append(dict(path=str(p.relative_to(root)), sha256=file_hash(p)))
        print(f'Checked model {model}: 5 runs, '+('test metrics reconstructed' if truth else 'no external test set'), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'run_metrics.csv', index=False, float_format='%.17g')
    summary = frame.groupby(['model','dataset','split','metric'], sort=True).value.agg(['mean','std','count']).reset_index()
    assert (summary['count'] == 5).all()
    summary.to_csv(output / 'summary.csv', index=False, float_format='%.17g')
    pd.DataFrame(input_records).to_csv(output / 'input_inventory.csv', index=False)
    pd.DataFrame(sources).drop_duplicates().to_csv(output / 'metric_sources.csv', index=False)
    archive = None
    if include_manifest:
        members = []
        for directory in datasets:
            for name in ('train','test','shared','runs'):
                for p in (directory / name).rglob('*'):
                    if p.is_symlink():
                        raise ValueError(f'Symlink in archive: {p}')
                    if p.is_file() and (name != 'runs' or p.relative_to(directory / name).parts[0] in {f'run_{i}' for i in range(1,6)}):
                        members.append(p)
        members += [root / 'references.bib']
        members += [root / '.benchmark' / n for n in ('results.csv','results_index.csv','protocol.json','input_inventory.csv','post_repair_validation.json')]
        members = sorted(p for p in members if p.exists())
        print(f'Hashing {len(members)} archive members...', flush=True)
        def fingerprint(p):
            return dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=file_hash(p))
        with ThreadPoolExecutor(max_workers=4) as pool:
            manifest = list(pool.map(fingerprint, members))
        pd.DataFrame(manifest).to_csv(output / 'archive_manifest.csv', index=False)
        (output / 'archive_files.txt').write_text(''.join(r['path']+'\n' for r in manifest))
        archive = dict(files=len(manifest), bytes=sum(r['bytes'] for r in manifest), symlinks=0)
    report = dict(status='passed', checked_at=utc_now(), datasets=64, runs=320,
                  test_runs=sum(r['split']=='test' and r['metric']=='MCC' for r in rows)+sum(r['split']=='test' and r['metric']=='R2' for r in rows),
                  test_metric_values=sum(r['split']=='test' for r in rows), cv_metric_values=sum(r['split']=='cv' for r in rows),
                  cv_precision='Per-run means are archived rounded to 4 decimals; original unrounded fold scores are not generally saved.',
                  test_precision='Reconstructed from saved predictions; compared to stored metrics.',
                  aggregation='Mean and sample standard deviation (ddof=1) across five run means; no averaging of fold SDs.',
                  external_paths='Legacy metadata may contain historical absolute paths. This audit reads only the supplied dataset directory.',
                  archive=archive, warnings=warnings)
    atomic_json(output / 'validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-root', type=Path, default=PROJECT / 'App/datasets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-manifest', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.datasets_root, args.output, not args.skip_manifest), indent=2))
