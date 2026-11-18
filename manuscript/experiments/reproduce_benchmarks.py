"""Fresh benchmark searches from a portable dataset directory, without result reuse.

Defaults reproduce seeds 6301..6305, 200 Stage 1 trials (patience 80), and
150 Stage 2 trials without a wall-clock cutoff. The web/CLI defaults and the
historical benchmark scripts are deliberately left unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(Path(__file__).resolve().parent)]
from bioautoml.execution import atomic_json, file_hash, utc_now


def read(path):
    return json.loads(Path(path).read_text())


def generation():
    import bioautoml.optimization as optimization
    original = optimization.optimize_study
    def optimize(study, objective, output, stage, n_trials, timeout, n_jobs, **kwargs):
        return original(study, objective, output, stage, n_trials,
                        None if stage == 'stage_2' else timeout, n_jobs, **kwargs)
    optimization.optimize_study = optimize
    sys.argv = [str(PROJECT / 'generation.py'), *sys.argv[2:]]
    runpy.run_path(str(PROJECT / 'generation.py'), run_name='__main__')


def worker(path):
    import _experiment_support as support
    original = subprocess.run
    def run(command, *args, **kwargs):
        if isinstance(command, list) and len(command) > 1 and Path(command[1]) == PROJECT / 'generation.py':
            command = [command[0], str(Path(__file__).resolve()), '--generation', *command[2:]]
        return original(command, *args, **kwargs)
    subprocess.run = run
    support.worker(read(path))


def portable_cache(shared):
    """Resolve old recorded paths by their path WITHIN this specific shared cache."""
    manifest = shared / 'feature_inputs.json'
    if manifest.exists():
        values = read(manifest)
        inputs = {}
        for key, value in values.items():
            if key not in {'train','train_label','train_nameseq','test','test_label','test_nameseq'}:
                continue
            path = Path(value)
            if path.is_absolute():
                parts = path.parts
                positions = [i for i, part in enumerate(parts) if part == 'shared']
                if len(positions) != 1:
                    raise ValueError(f'Cannot relocate cache path: {value}')
                path = Path(*parts[positions[0] + 1:])
            resolved = (shared / path).resolve()
            if not resolved.is_relative_to(shared) or not resolved.is_file():
                raise ValueError(f'Missing local cache input: {resolved}')
            inputs[key] = str(resolved)
    else:
        inputs = {key: str(shared / 'work/features' / name) for key, name in (
            ('train', 'ftrain.csv'), ('train_label', 'flabeltrain.csv'), ('train_nameseq', 'fnameseqtrain.csv'))}
    import pandas as pd
    timings = pd.read_csv(shared / 'timings.csv')
    phases = ['training_features', 'test_features'] if 'test' in inputs else ['training_features']
    minutes = float(timings.loc[timings.phase.isin(phases), 'elapsed_seconds'].sum()) / 60
    assets = {str(path): file_hash(path) for path in [shared / 'run.json',
              *map(Path, inputs.values()), *[p for p in (shared / 'reports').rglob('*') if p.is_file()]]}
    return dict(shared=str(shared), inputs=inputs, cached_extraction_minutes=minutes,
                full_test_cache='test' in inputs), assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-root', type=Path, default=PROJECT / 'App/datasets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', type=int)
    parser.add_argument('--runs', nargs='+', type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument('--n-cpu', type=int, default=32)
    parser.add_argument('--from-raw', action='store_true', help='Re-extract descriptors from FASTA instead of using shared caches.')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.datasets_root, args.output = args.datasets_root.resolve(), args.output.resolve()
    if args.output.exists() or args.output.is_relative_to(args.datasets_root) or args.datasets_root.is_relative_to(args.output):
        parser.error('Use a new output directory outside the input dataset directory.')
    if args.n_cpu < 1 or len(set(args.runs)) != len(args.runs):
        parser.error('Use positive CPU count and unique run numbers.')
    import _experiment_support as support
    import run_benchmarks as benchmarks
    benchmarks.DATASETS_PATH = args.datasets_root
    datasets, _ = benchmarks.input_inventory()
    if args.datasets:
        if not set(args.datasets) <= {d['dataset_id'] for d in datasets}:
            parser.error('Unknown dataset ID.')
        datasets = [d for d in datasets if d['dataset_id'] in args.datasets]
    settings = dict(seed=63, n_cpu=args.n_cpu, estimations=200, tuning=150, search_jobs=1,
                    pruning=True, homology_aware=False, smoke_test=False,
                    stage_2_timeout_seconds=None, packages=support.homology.package_versions(),
                    code_sha256=support.code_hashes(), reproduction_script_sha256=file_hash(__file__))
    for d in datasets:
        if not args.from_raw:
            source, assets = portable_cache(args.datasets_root / d['dataset'] / 'shared')
            record = read(Path(source['shared']) / 'run.json')
            expected = sorted((Path(p).name, h) for p, h in d['input_hashes'].items())
            assert sorted((v['name'], v['sha256']) for v in record['inputs']) == expected
        print(f'{d["dataset"]}: fresh runs {args.runs}; '+('raw FASTA' if args.from_raw else 'local shared descriptors'), flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True)
    protocol = dict(status='running', started_at=utc_now(), datasets_root=str(args.datasets_root),
                    settings=settings, datasets=[d['dataset'] for d in datasets], runs=args.runs,
                    cached_descriptors=not args.from_raw, completed=[])
    atomic_json(args.output / 'protocol.json', protocol)
    try:
        for d in datasets:
            if args.from_raw:
                shared = args.output / d['dataset'] / 'shared'
                benchmarks.extract_cache(d, shared, settings, support)
            else:
                shared = args.datasets_root / d['dataset'] / 'shared'
            source, assets = portable_cache(shared)
            for run in args.runs:
                output = args.output / d['dataset'] / 'runs' / f'run_{run}'
                config = dict(settings, dataset=d, source=source, assets=assets, run=run,
                              search_seed=6300 + run, output=str(output), benchmark_runner=True)
                config_path = args.output / f'{d["dataset"]}_run_{run}.json'
                atomic_json(config_path, config)
                environment = dict(os.environ, PYTHONHASHSEED='63')
                environment.pop('BIOAUTOML_RUN_ROOT', None)
                with (args.output / f'{d["dataset"]}_run_{run}.log').open('x') as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(config_path)],
                                   cwd=PROJECT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                history = read(output / 'reports/optimization/stage_2_study.json')
                assert history['stop_reason'] == 'trial_budget' and len(history['trials']) == 150
                pipeline_path = output / 'reports/pipeline.json'
                pipeline = read(pipeline_path)
                pipeline['command'] = [sys.executable, str(Path(__file__).resolve()), '--generation', *pipeline['command'][2:]]
                pipeline['stage_2_timeout_seconds'] = None
                atomic_json(pipeline_path, pipeline)
                protocol['completed'].append(f'{d["dataset"]}/run_{run}')
                atomic_json(args.output / 'protocol.json', protocol)
    except BaseException:
        protocol.update(status='failed', updated_at=utc_now())
        atomic_json(args.output / 'protocol.json', protocol)
        raise
    protocol.update(status='completed', updated_at=utc_now())
    atomic_json(args.output / 'protocol.json', protocol)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--generation':
        generation()
    elif len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        main()
