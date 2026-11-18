"""Resumable train/test similarity screen; no fitting or partition changes.

Uses the production MMseqs2 search and thresholds. Search identical sequences
once, retain one qualifying training witness per test sequence, then expand to
all original test records. Exact matches already establish membership and do
not need alignment. This reports record-level overlap, not pair counts.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _experiment_support as support
from bioautoml.homology import read_sequences, search_sequences, get_mmseqs


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verify(dataset):
    for path, expected in dataset['input_hashes'].items():
        if support.file_hash(path) != expected:
            raise ValueError(f'Input changed: {path}')


def assess(dataset, args, pilot=False):
    started = time.monotonic()
    verify(dataset)
    dtype = 'Protein' if dataset['dtype'] == 'protein' else 'DNA/RNA'
    train, test = [read_sequences(dataset[f'{split}_files'], dataset[f'{split}_labels'],
                                 split, dtype, dataset['task']) for split in ('train', 'test')]
    assert (len(train), len(test)) == (dataset['train_count'], dataset['test_count'])
    index = defaultdict(list)
    labels = defaultdict(Counter)
    for record in train:
        index[record['sequence']].append(record)
        labels[record['sequence']][record['label']] += 1
    targets = sorted(index)
    queries = sorted({r['sequence'] for r in test} - index.keys())
    signature = digest(json.dumps([dtype, targets, queries], separators=(',', ':')))
    cached = args.output / 'search_cache' / signature
    cached.mkdir(parents=True, exist_ok=True)
    support.atomic_json(cached / 'inputs.json', dict(dtype=dtype, target_sha256=[digest(s) for s in targets],
                                                  query_sha256=[digest(s) for s in queries]))
    witnesses = {}
    batches = list(range(0, len(queries), args.batch_size))
    if pilot:
        batches = batches[:1]
    for offset in batches:
        batch = queries[offset:offset + args.batch_size]
        directory = cached / f'batch_{offset:08d}'
        directory.mkdir(exist_ok=True)
        result = directory / 'completed.json'
        if result.exists():
            report = json.loads(result.read_text())
        else:
            begin = time.monotonic()
            best = {}
            for q, t, identity, qcov, tcov in search_sequences(
                    [{'sequence': s} for s in batch], [{'sequence': s} for s in targets],
                    dtype, directory, args.n_cpu, 90, 80):
                witness = dict(target_sha256=digest(targets[t]), identity=identity,
                               query_coverage=qcov, target_coverage=tcov)
                # Any qualifying match is enough; deterministic strongest witness.
                key = (identity, min(qcov, tcov), witness['target_sha256'])
                if q not in best or key > best[q][0]:
                    best[q] = (key, witness)
            report = dict(queries=len(batch), seconds=time.monotonic()-begin,
                          matches={digest(batch[q]): w for q, (_, w) in best.items()})
            support.atomic_json(result, report)
        witnesses.update(report['matches'])
        print(f'{dataset["dataset"]}: batch {offset // args.batch_size + 1}/{len(range(0,len(queries),args.batch_size))}; '
              f'{report["seconds"]:.1f}s; {len(report["matches"])}/{len(batch)} unique queries matched', flush=True)
    if pilot:
        return dict(dataset=dataset['dataset'], unique_targets=len(targets), unique_nonexact_queries=len(queries),
                    batch_size=args.batch_size, pilot_seconds=time.monotonic()-started)
    by_hash = {digest(s): index[s][0] for s in targets}
    rows = []
    exact_count = similar_count = conflict_count = 0
    for record in test:
        sequence = record['sequence']
        exact = sequence in index
        witness = witnesses.get(digest(sequence))
        similar = exact or witness is not None
        conflicts = sum(labels[sequence].values()) - labels[sequence][record['label']] if exact else 0
        exact_count += exact
        similar_count += similar
        conflict_count += bool(conflicts)
        target = index[sequence][0] if exact else by_hash[witness['target_sha256']] if witness else None
        rows.append(dict(test_sequence_id=record['sequence_id'], test_source=record['source'],
                         test_record_number=record['record_number'], test_original_id=record['original_id'],
                         test_sequence_sha256=digest(sequence), exact=exact, similar_including_exact=similar,
                         conflicting_exact_training_records=conflicts,
                         witness_train_sequence_id=target['sequence_id'] if target else '',
                         witness_train_source=target['source'] if target else '',
                         witness_train_record_number=target['record_number'] if target else '',
                         witness_train_sequence_sha256=digest(target['sequence']) if target else '',
                         identity=1.0 if exact else witness['identity'] if witness else '',
                         test_coverage=1.0 if exact else witness['query_coverage'] if witness else '',
                         train_coverage=1.0 if exact else witness['target_coverage'] if witness else ''))
    directory = args.output / dataset['dataset']
    directory.mkdir(exist_ok=True)
    save_csv(directory / 'test_membership.csv', rows)
    row = dict(dataset=dataset['dataset'], dataset_id=dataset['dataset_id'], task=dataset['task'], dtype=dataset['dtype'],
               train_records=len(train), test_records=len(test), status='assessed' if test else 'no test set',
               exact_test_records=exact_count if test else None,
               exact_test_percent=100*exact_count/len(test) if test else None,
               similar_test_records_including_exact=similar_count if test else None,
               similar_test_percent_including_exact=100*similar_count/len(test) if test else None,
               additional_nonexact_similar_records=similar_count-exact_count if test else None,
               test_records_without_detected_matches=len(test)-similar_count if test else None,
               conflicting_exact_test_records=conflict_count if test else None,
               minimum_identity=0.9, minimum_coverage_each=0.8, partitions_modified=False,
               sequence_search_cache=signature, seconds=time.monotonic()-started)
    verify(dataset)
    support.atomic_json(directory / 'summary.json', row)
    print(f'{dataset["dataset"]}: COMPLETE exact={exact_count}, similar={similar_count}, test={len(test)}', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=support.RESULTS_PATH / 'sequence_similarity')
    parser.add_argument('--n_cpu', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--datasets', type=int, nargs='+')
    args = parser.parse_args()
    if args.batch_size < 1 or args.n_cpu < 1:
        parser.error('batch_size and n_cpu must be positive')
    os.chdir(support.PROJECT_PATH)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _, version = get_mmseqs()
    datasets, inventory = support.benchmarks.input_inventory()
    config = dict(schema_version=1, mmseqs_version=version, n_cpu=args.n_cpu, batch_size=args.batch_size,
                  identity=0.9, coverage_each=0.8, datasets=datasets,
                  script_sha256=support.file_hash(__file__),
                  search_code_sha256=support.file_hash(support.PROJECT_PATH / 'bioautoml/homology.py'),
                  normalization='uppercase; nucleotide U -> T', nucleotide_strands='both',
                  exact_definition='full-length equality, forward orientation',
                  search='production search_sequences; unique sequences sorted lexically; exact queries skipped; batches',
                  output='one qualifying witness per original test record; no all-pairs or model performance claims',
                  limitations='Long-query searches are heuristic; absence of detected matches does not establish absence of homology.')
    frozen = args.output / 'configuration.json'
    if frozen.exists() and json.loads(frozen.read_text()) != config:
        raise ValueError('Frozen configuration changed; choose a new output directory.')
    support.atomic_json(frozen, config)
    inventory.to_csv(args.output / 'input_inventory.csv', index=False)
    chosen = [d for d in datasets if not args.datasets or d['dataset_id'] in args.datasets]
    # Pilot once on each requested dataset; normal execution smallest first to
    # complete the broad inventory while preserving resumability of large jobs.
    chosen.sort(key=lambda d: d['train_count'] * d['test_count'])
    state = dict(status='running', mode='pilot' if args.pilot else 'full', requested=[d['dataset_id'] for d in chosen])
    support.atomic_json(args.output / 'protocol.json', state)
    try:
        for dataset in chosen:
            summary = args.output / dataset['dataset'] / 'summary.json'
            if args.pilot or not summary.exists():
                result = assess(dataset, args, args.pilot)
                if args.pilot:
                    support.atomic_json(args.output / f'pilot_{dataset["dataset_id"]}.json', result)
            else:
                verify(dataset)
            all_rows = [json.loads(p.read_text()) for p in args.output.glob('dataset*/summary.json')]
            all_rows.sort(key=lambda r: r['dataset_id'])
            save_csv(args.output / 'summary.csv', all_rows)
            state['completed_datasets'] = len(all_rows)
            support.atomic_json(args.output / 'protocol.json', state)
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        support.atomic_json(args.output / 'protocol.json', state)


if __name__ == '__main__':
    main()
