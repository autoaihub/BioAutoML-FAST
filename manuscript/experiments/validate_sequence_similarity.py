"""Independently check full audit membership, provenance and earlier results."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / 'results/sequence_similarity'
RESULTS = OUTPUT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    configuration = json.loads((OUTPUT / 'configuration.json').read_text())
    assert configuration['script_sha256'] == sha(Path(__file__).with_name('audit_sequence_similarity.py'))
    assert configuration['search_code_sha256'] == sha(ROOT / 'bioautoml/homology.py')
    old_inputs = {r['path']: r['sha256'] for r in read_rows(RESULTS / 'sequence_overlap/input_inventory.csv')}
    new_inputs = {r['path']: r['sha256'] for r in read_rows(OUTPUT / 'input_inventory.csv')}
    assert old_inputs == new_inputs
    checked_batches = 0
    for cache in (OUTPUT / 'search_cache').iterdir():
        inputs = json.loads((cache / 'inputs.json').read_text())
        targets = set(inputs['target_sha256'])
        for completed in cache.glob('batch*/completed.json'):
            offset = int(completed.parent.name.split('_')[1])
            queries = inputs['query_sha256'][offset:offset + configuration['batch_size']]
            batch = json.loads(completed.read_text())
            assert batch['queries'] == len(queries)
            assert set(batch['matches']) <= set(queries)
            for witness in batch['matches'].values():
                assert witness['target_sha256'] in targets
                assert 0.9 <= witness['identity'] <= 1
                assert 0.8 <= witness['query_coverage'] <= 1
                assert 0.8 <= witness['target_coverage'] <= 1
            commands = list(completed.parent.glob('*_command.json'))
            assert commands
            for path in commands:
                command = json.loads(path.read_text())
                for flag, value in (('--min-seq-id', '0.9'), ('-c', '0.8'), ('--cov-mode', '0'),
                                    ('--alignment-mode', '3'), ('--seq-id-mode', '0'),
                                    ('--max-seqs', str(len(targets)))):
                    assert command[command.index(flag) + 1] == value
                if inputs['dtype'] == 'DNA/RNA':
                    assert command[command.index('--strand') + 1] == '2'
                if path.name == 'short_command.json':
                    assert command[command.index('--prefilter-mode') + 1] == '2'
            checked_batches += 1
    checks = []
    for dataset in configuration['datasets']:
        directory = OUTPUT / dataset['dataset']
        if not (directory / 'summary.json').exists():
            continue
        report = json.loads((directory / 'summary.json').read_text())
        source = {}
        for split in ('train', 'test'):
            for filename in dataset[f'{split}_files']:
                assert sha(filename) == dataset['input_hashes'][filename]
                for number, record in enumerate(SeqIO.parse(filename, 'fasta'), 1):
                    sequence = str(record.seq).upper()
                    if dataset['dtype'] == 'dnarna':
                        sequence = sequence.replace('U', 'T')
                    source[split, Path(filename).name, number] = hashlib.sha256(sequence.encode()).hexdigest()
        membership = read_rows(directory / 'test_membership.csv') if dataset['test_count'] else []
        assert len(membership) == dataset['test_count']
        assert len({r['test_sequence_id'] for r in membership}) == len(membership)
        observed = set()
        for row in membership:
            test_key = ('test', row['test_source'], int(row['test_record_number']))
            assert source[test_key] == row['test_sequence_sha256']
            assert test_key not in observed
            observed.add(test_key)
            if row['similar_including_exact'] == 'True':
                train_key = ('train', row['witness_train_source'], int(row['witness_train_record_number']))
                assert source[train_key] == row['witness_train_sequence_sha256']
                assert 0.9 <= float(row['identity']) <= 1
                assert 0.8 <= float(row['test_coverage']) <= 1
                assert 0.8 <= float(row['train_coverage']) <= 1
            else:
                assert not row['witness_train_sequence_id']
                assert row['exact'] == 'False'
            if row['exact'] == 'True':
                assert row['test_sequence_sha256'] == row['witness_train_sequence_sha256']
                assert all(float(row[k]) == 1 for k in ('identity', 'test_coverage', 'train_coverage'))
        old_exact = json.loads((RESULTS / 'sequence_overlap' / dataset['dataset'] / 'summary.json').read_text())
        assert old_exact['overlapping_test_records'] == report['exact_test_records']
        assert old_exact['conflicting_test_records'] == report['conflicting_exact_test_records']
        if membership:
            assert sum(r['exact'] == 'True' for r in membership) == report['exact_test_records']
            assert sum(r['similar_including_exact'] == 'True' for r in membership) == report['similar_test_records_including_exact']
            assert sum(r['similar_including_exact'] == 'False' for r in membership) == report['test_records_without_detected_matches']
        previous = RESULTS / 'homology_comparison' / dataset['dataset'] / 'preparation/reports/homology'
        previous_comparison = None
        if (previous / 'overlap_report.json').exists():
            old = json.loads((previous / 'overlap_report.json').read_text())
            assert report['similar_test_records_including_exact'] == old['similar_test_sequences_including_exact']
            old_ids = {r['test_sequence_id'] for r in read_rows(previous / 'overlap_matches.csv')}
            new_ids = {r['test_sequence_id'] for r in membership if r['similar_including_exact'] == 'True'}
            assert old_ids == new_ids
            previous_comparison = 'identical record membership'
        checks.append(dict(dataset=dataset['dataset'], input_hashes='verified', witness_provenance='verified',
                           exact_audit='agrees', previous_similarity_audit=previous_comparison))
    complete = len(checks) == len(configuration['datasets'])
    result = dict(status='passed' if complete else 'partial', validated_datasets=len(checks),
                  expected_datasets=len(configuration['datasets']),
                  exact_audit_input_files_verified=len(new_inputs),
                  validated_search_batches=checked_batches, checks=checks)
    (OUTPUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}))
    print('Earlier similarity comparisons:', sum(c['previous_similarity_audit'] is not None for c in checks))
    if '--require-complete' in sys.argv and not complete:
        raise SystemExit('Audit is not complete')


if __name__ == '__main__':
    main()
