#!/usr/bin/env python3
"""Generate the 64-benchmark, seed-6301 resource table using saved artifacts only.

Run from any directory:
    python manuscript/tables/resources/extract_resources.py

Only resources.csv and resources.tex in --output-dir are written. No pipeline
modules are imported, models loaded, experiments launched, or source files edited.
Requires Python >= 3.10 and the standard library only.

The composition table supplies labels, not authoritative sequence counts. Counts
come from the benchmark inventory and are checked against each run's input hashes.
For ablation-derived runs, total time is an additive estimate and memory is the
maximum separately sampled component peak. Measurement scope is retained in the CSV.
Historical CPU model, host RAM and concurrent load are not established by these
run summaries; document them separately before manuscript submission.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_IDS = set(range(1, 65))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(value, root):
    """Resolve historical absolute paths after moving a repository checkout."""
    path = Path(value)
    if not path.is_absolute():
        return root / path
    try:
        path.relative_to(root)
        return path
    except ValueError:
        for anchor in ("App", "manuscript"):
            if anchor in path.parts:
                return root.joinpath(*path.parts[path.parts.index(anchor):])
    raise ValueError(f"Cannot resolve artifact within repository: {value}")


def relative(path, root):
    return str(path.relative_to(root))


def number(value, label):
    require(value is not None, f"Missing measurement: {label}")
    result = float(value)
    require(math.isfinite(result) and result >= 0, f"Invalid {label}: {value}")
    return result


def composition(path):
    labels = {}
    for line in path.read_text().splitlines():
        match = re.match(r"Model (\d+): (.*?) & (.*?) & (.*?) & (.*?) & (.*?) &", line)
        if not match:
            continue
        identifier = int(match[1])
        require(identifier not in labels, f"Duplicate composition ID {identifier}")
        counts = [sum(int(n.replace(',', '')) for n in re.findall(r"\d[\d,]*", cell))
                  for cell in (match[5], match[6])]
        labels[identifier] = dict(model_name_tex=match[2], task=match[3],
                                  sequence_type=match[4], composition_train=counts[0],
                                  composition_test=counts[1])
    require(set(labels) == EXPECTED_IDS, "Composition table must contain IDs 1–64 exactly once")
    return labels


def load_record(path, expected_inputs):
    record = read_json(path)
    require(record['status'] == 'completed', f"Incomplete run: {path}")
    actual = Counter((item['name'], item['sha256']) for item in record['inputs'])
    require(actual == expected_inputs, f"Input hashes differ from inventory: {path}")
    performance = record['performance']
    require(performance['status'] == 'completed', f"Incomplete measurements: {path}")
    require(performance.get('total_scope') == 'end_to_end', f"Unexpected timing scope: {path}")
    require(performance['memory_method'] == 'sampled_process_tree_rss', f"Unknown memory method: {path}")
    require(not performance['memory_incomplete'], f"Incomplete memory sampling: {path}")
    require(performance['memory_samples'] > 0, f"No memory samples: {path}")
    require(performance['memory_sampling_seconds'] == 0.1, f"Unexpected sampling interval: {path}")
    require(record['settings']['seed'] == 63 and record['settings']['n_cpu'] == 32,
            f"Unexpected seed or CPU allocation: {path}")
    for field in ('total_seconds', 'peak_memory_bytes'):
        number(performance[field], f"{path}:{field}")
    return record


def check_search(path, cap):
    history = read_json(path)
    require(history['status'] == 'completed' and history['pruner'] == 'MedianPruner',
            f"Incomplete search or unexpected pruning: {path}")
    for key, expected in dict(seed=63, search_seed=6301, requested_trials=cap,
                              folds=5, search_jobs=1, model_jobs=32).items():
        require(history['settings'][key] == expected, f"Unexpected {key}: {path}")
    return history


def build_rows(root):
    labels = composition(root / 'manuscript/tables/composition/datasets.tex')
    metadata = root / 'App/datasets/.benchmark'
    inventory = defaultdict(list)
    with (metadata / 'input_inventory.csv').open() as handle:
        for item in csv.DictReader(handle):
            require(item['split'] in ('train', 'test'), 'Unknown inventory split')
            inventory[item['dataset']].append(item)
    with (metadata / 'results.csv').open() as handle:
        selected = [row for row in csv.DictReader(handle) if int(row['run']) == 1]
    identifiers = [int(re.match(r'dataset(\d+)_', row['dataset'])[1]) for row in selected]
    require(len(identifiers) == 64 and set(identifiers) == EXPECTED_IDS,
            'Expected exactly one run-1 result for each of the 64 datasets')
    require(set(inventory) == {row['dataset'] for row in selected}, 'Inventory/dataset mismatch')
    rows, warnings = [], []
    for identifier, index in sorted(zip(identifiers, selected)):
        dataset = index['dataset']
        require(int(index['search_seed']) == 6301 and index['pruning'].lower() == 'true',
                f'Unexpected seed/pruning: {dataset}')
        items = inventory[dataset]
        require(len({item['path'] for item in items}) == len(items), f'Duplicate input: {dataset}')
        expected_inputs = Counter((Path(item['path']).name, item['sha256']) for item in items)
        counts = {split: sum(int(item['records']) for item in items if item['split'] == split)
                  for split in ('train', 'test')}
        require(counts['train'] > 0, f'Missing training records: {dataset}')
        output = local_path(index['output'], root)
        record_path = output / 'run.json'
        record = load_record(record_path, expected_inputs)
        require(record['settings']['search_seed'] == 6301 and record['settings']['run'] == 1,
                f'Unexpected run settings: {dataset}')
        histories = {stage: check_search(output / f'reports/optimization/{stage}_study.json', cap)
                     for stage, cap in [('stage_1', 200), ('stage_2', 150)]}
        p = record['performance']
        # Detect the operation, not a legacy ledger name or a generic total_scope.
        additive = record['settings'].get('condition') == 'selected_tuned'
        components = [(record_path, record)]
        original_path = record_path
        source_path = output / 'reports/benchmark_source.json'
        if source_path.exists():
            source = read_json(source_path)
            original_path = local_path(source['original_output'], root) / 'run.json'
            # Historical standalone directories may have been migrated away.
            if original_path.exists():
                require(digest(original_path) == source['original_run_sha256'],
                        f'Original source hash changed: {original_path}')
                require(read_json(original_path)['performance'] == p,
                        f'Imported resource measurements changed: {dataset}')
        if additive:
            require(source_path.exists() and original_path.exists(), f'Missing ablation provenance: {dataset}')
            shared_path = local_path(record['settings']['shared'], root) / 'run.json'
            selection_path = local_path(record['settings']['stage_1'], root) / 'run.json'
            shared = load_record(shared_path, expected_inputs)
            selection = load_record(selection_path, expected_inputs)
            require(shared['settings']['operation'] == 'extraction', f'Wrong shared source: {dataset}')
            require(selection['settings']['operation'] == 'selection'
                    and selection['settings']['search_seed'] == 6301, f'Wrong selection source: {dataset}')
            selection_history = check_search(selection_path.parent / 'reports/optimization/stage_1_study.json', 200)
            require(selection_history == histories['stage_1'], f'Selection history mismatch: {dataset}')
            components = [(shared_path, shared), (selection_path, selection), (record_path, record)]
            extraction = number(shared['performance']['descriptor_extraction_seconds'], dataset)
            stage_1 = number(selection['performance']['stage_1_seconds'], dataset)
            require(p['descriptor_extraction_seconds'] is None and p['stage_1_seconds'] is None,
                    f'Unexpected overlapping component measurements: {dataset}')
        else:
            require(not (output / 'reports/pipeline.json').exists(), f'Unresolved cached pipeline: {dataset}')
            extraction = number(p['descriptor_extraction_seconds'], dataset)
            stage_1 = number(p['stage_1_seconds'], dataset)
        stage_2 = number(p['stage_2_seconds'], dataset)
        total = sum(r['performance']['total_seconds'] for _, r in components)
        peak = max(r['performance']['peak_memory_bytes'] for _, r in components)
        require(total + 1 >= extraction + stage_1 + stage_2, f'Phase/total inconsistency: {dataset}')
        disagreements = [f"{split}: composition={labels[identifier]['composition_' + split]}, inventory={counts[split]}"
                         for split in ('train', 'test')
                         if labels[identifier]['composition_' + split] != counts[split]]
        if disagreements:
            warnings.append(f"Model {identifier}: " + '; '.join(disagreements))
        row = dict(model_id=identifier, dataset=dataset, **labels[identifier],
                   train_records=counts['train'], test_records=counts['test'], run=1,
                   search_seed=6301, fold_learner_seed=63, allocated_cpus=32, search_jobs=1,
                   pruning=True, extraction_minutes=extraction / 60,
                   stage_1_minutes=stage_1 / 60, stage_2_minutes=stage_2 / 60,
                   total_minutes=total / 60, peak_ram_gib=peak / 1024**3,
                   timing_scope='additive_component_estimate' if additive else 'measured_pipeline',
                   memory_scope='maximum_separate_component_peaks' if additive else 'measured_pipeline_peak',
                   cache_status='shared_extraction_reused_cost_included' if additive else 'fresh_extraction',
                   extraction_scope=('training_only' if not counts['test'] else
                                     'all_train_and_all_test_descriptors' if additive else
                                     'all_train_and_selected_test_descriptors'),
                   memory_method=p['memory_method'], memory_sampling_seconds=0.1,
                   memory_incomplete=False, hardware_provenance='CPU model, host RAM and concurrent load not established by run summaries',
                   run_record=relative(record_path, root), run_record_sha256=digest(record_path),
                   original_run_record=relative(original_path, root),
                   original_run_available=original_path.exists(),
                   component_records=json.dumps([relative(path, root) for path, _ in components]),
                   component_sha256=json.dumps([digest(path) for path, _ in components]),
                   component_total_seconds=json.dumps([r['performance']['total_seconds'] for _, r in components]),
                   component_peak_bytes=json.dumps([r['performance']['peak_memory_bytes'] for _, r in components]),
                   input_hashes=json.dumps({relative(local_path(i['path'], root), root): i['sha256'] for i in items}, sort_keys=True),
                   composition_discrepancy='; '.join(disagreements))
        for stage, history in histories.items():
            row[f'{stage}_trial_cap'] = history['settings']['requested_trials']
            row[f'{stage}_timeout_seconds'] = history['settings']['timeout_seconds']
            row[f'{stage}_stop_reason'] = history['stop_reason']
            row[f'{stage}_trial_counts'] = json.dumps(history['trial_counts'], sort_keys=True)
        rows.append(row)
    return rows, warnings


def latex(rows):
    caption = (
        r'\textbf{Computational resources for the 64 BioAutoML-FAST benchmarks.} '
        r'Resources are reported for one run per dataset using 32 allocated CPUs. '
        r'Times are in minutes; RAM is in GiB. Extraction denotes descriptor extraction, '
        r'selection denotes Stage 1, and optimization denotes Stage 2. '
        r'Total time also includes evaluation and other pipeline processing. '
        r'Dataset composition is provided in Table~\ref{benchmark_datasets}.'
    )
    header = (r'\textbf{Model} & '
              r'\shortstack{\textbf{Extraction}\\(min)} & '
              r'\shortstack{\textbf{Selection}\\(min)} & '
              r'\shortstack{\textbf{Optimization}\\(min)} & '
              r'\shortstack{\textbf{Total}\\(min)} & '
              r'\shortstack{\textbf{Peak RAM}\\(GiB)} \\')
    lines = [
        r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3pt}',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\setlength{\LTleft}{0pt}', r'\setlength{\LTright}{0pt}',
        r'\setlength{\LTcapwidth}{\linewidth}',
        r'\begin{longtable}{@{}',
        r'  >{\raggedright\arraybackslash}p{0.35\linewidth}',
        r'  *{5}{>{\raggedleft\arraybackslash}p{\dimexpr0.13\linewidth-2\tabcolsep\relax}}',
        r'@{}}',
        r'\caption{' + caption + r'}\label{tab:computational_resources}\\',
        r'\toprule', header, r'\midrule', r'\endfirsthead',
        r'\caption[]{Computational resources for the 64 benchmarks (continued).}\\',
        r'\toprule', header, r'\midrule', r'\endhead',
        r'\midrule', r'\multicolumn{6}{r}{\textit{Continued on next page}}\\',
        r'\endfoot', r'\bottomrule', r'\endlastfoot',
    ]
    for row in rows:
        name = f"Model {row['model_id']}: {row['model_name_tex']}"
        cells = [name]
        cells += [f"{row[k]:,.2f}" for k in ('extraction_minutes', 'stage_1_minutes', 'stage_2_minutes', 'total_minutes', 'peak_ram_gib')]
        lines += [' & '.join(cells) + r' \\', r'\midrule' if row is not rows[-1] else '']
    return '\n'.join(lines + [r'\end{longtable}', r'\endgroup', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root containing archived artifacts')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    rows, warnings = build_rows(args.root.resolve())
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    table = latex(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'resources.csv').write_text(buffer.getvalue())
    (args.output_dir / 'resources.tex').write_text(table)
    estimates = sum(row['timing_scope'] == 'additive_component_estimate' for row in rows)
    print(f'Wrote 64 rows: {64 - estimates} measured pipelines and {estimates} component estimates to {args.output_dir}')
    for warning in warnings:
        print('Composition discrepancy (inventory counts used): ' + warning, file=sys.stderr)
    print('Before submission: verify historical CPU model, host RAM and concurrent workload conditions.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        sys.exit(f'Resource extraction failed: {error}')
