"""Build S7/S8 from completed audits and predictions; never fit models.

Run with .venv-repro/bin/python3; --sync-manuscript updates the active LaTeX clone.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bioautoml.homology import read_sequences
from sklearn.metrics import matthews_corrcoef, r2_score

RESULTS = ROOT / 'manuscript/experiments/results'
MANUSCRIPT = ROOT / 'manuscript/revision/BioAutoML-FAST_LaTeX/oup-authoring-template'
IDS = {2, 13, 29, 55, 63}


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    audit = RESULTS / 'sequence_similarity'
    require(json.loads((audit / 'protocol.json').read_text())['status'] == 'completed', 'Incomplete audit')
    require(json.loads((audit / 'validation.json').read_text())['status'] == 'passed', 'Unvalidated audit')
    labels = dict(re.findall(r'^Model (\d+): (.*?) &',
                            (MANUSCRIPT / 'supplementary/tables/datasets.tex').read_text(), re.M))
    rows = read_csv(audit / 'summary.csv')
    require({int(r['dataset_id']) for r in rows} == set(range(1, 65)) and len(rows) == 64, 'Dataset coverage')
    for entry in read_csv(audit / 'input_inventory.csv'):
        require(sha(entry['path']) == entry['sha256'], 'Audit input changed')
    for row in rows:
        row['model_name_tex'] = labels[row['dataset_id']]
        if row['status'] != 'assessed':
            require(int(row['test_records']) == 0, 'Unexpected unavailable test set')
            continue
        members = read_csv(audit / row['dataset'] / 'test_membership.csv')
        total = int(row['test_records'])
        exact = sum(r['exact'] == 'True' for r in members)
        similar = sum(r['similar_including_exact'] == 'True' for r in members)
        require(len(members) == total and exact == int(row['exact_test_records'])
                and similar == int(row['similar_test_records_including_exact']), 'S7 membership count')
        require(total - similar == int(row['test_records_without_detected_matches']), 'S7 retained count')
        require(math.isclose(float(row['exact_test_percent']), 100 * exact / total)
                and math.isclose(float(row['similar_test_percent_including_exact']), 100 * similar / total), 'S7 percentage')
    source = RESULTS / 'homology_comparison'
    protocol = json.loads((source / 'protocol.json').read_text())
    require(protocol['status'] == 'completed' and protocol['completed_results'] == 50, 'Incomplete comparison')
    filtered = source / 'filtered_test'
    frozen = json.loads((filtered / 'protocol.json').read_text())
    require(frozen['status'] == 'completed' and frozen['completed_results'] == 100, 'Incomplete filtered evaluation')
    for path, expected in frozen['outputs'].items():
        require(sha(filtered / path) == expected, f'Filtered output changed: {path}')
    experiments = read_csv(filtered / 'results.csv')
    expected = {(i, cv, split, run) for i in IDS for cv in ('ordinary', 'homology')
                for split in ('original', 'filtered') for run in range(1, 6)}
    seen = set()
    grouped = defaultdict(list)
    for dataset in protocol['configuration']['datasets']:
        name, ident = dataset['dataset'], dataset['dataset_id']
        for path, expected_hash in dataset['input_hashes'].items():
            require(sha(path) == expected_hash, 'Comparison input changed')
        dtype = 'Protein' if dataset['dtype'] == 'protein' else 'DNA/RNA'
        test = read_sequences(dataset['test_files'], dataset['test_labels'], 'test', dtype, dataset['task'])
        truth = {r['sequence_id']: r['label'] for r in test}
        membership = read_csv(audit / name / 'test_membership.csv')
        retained = {r['test_sequence_id'] for r in membership if r['similar_including_exact'] == 'False'}
        report = json.loads((source / name / 'preparation/reports/homology/homology_report.json').read_text())
        require(report['minimum_identity'] == .9 and report['minimum_query_and_target_coverage'] == .8, 'Grouping thresholds')
        # Every frozen training group must remain within one fold at both stages.
        for count in (5, 10):
            groups = defaultdict(set)
            for item in report['rows']:
                groups[item['group_id']].add(item[f'fold_{count}'])
            require(all(len(folds) == 1 for folds in groups.values()), 'Group crosses a fold boundary')
            require(set.union(*groups.values()) == set(range(count)), 'Missing grouped fold')
        for record in (r for r in experiments if r['dataset'] == name):
            key = ident, record['cv'], record['test_set'], int(record['run'])
            require(key not in seen, 'Duplicate evaluation')
            seen.add(key)
            require(record['smoke_test'] == 'False' and record['condition'] == 'selected_tuned'
                    and int(record['search_seed']) == 6300 + int(record['run']) and record['seed'] == '63', 'Run settings')
            predictions = read_csv(Path(record['output']) / 'results/predictions/test_predictions.csv')
            require(len(predictions) == len(truth) and {p['internal_id'] for p in predictions} == set(truth), 'Prediction IDs')
            selected = [p for p in predictions if record['test_set'] == 'original' or p['internal_id'] in retained]
            actual = [truth[p['internal_id']] for p in selected]
            predicted = [p['prediction'] for p in selected]
            metric = 'R2' if ident == 63 else 'MCC'
            score = r2_score(list(map(float, actual)), list(map(float, predicted))) if ident == 63 else matthews_corrcoef(actual, predicted)
            require(math.isclose(score, float(record['primary_score']), abs_tol=1e-10, rel_tol=0), 'Prediction metric mismatch')
            require(len(selected) == int(record['test_records']), 'Filtered test size')
            cv_metrics = read_csv(Path(record['output']) / 'results/metrics/optimization_cv_metrics.csv')[0]
            cv_score = float(cv_metrics[metric])
            require(math.isclose(cv_score, float(record['optimization_cv_' + metric]), abs_tol=1e-12), 'CV metric mismatch')
            grouped[ident, record['cv'], record['test_set']].append((score, cv_score))
    require(seen == expected, 'Expected five paired repeats for each dataset/strategy/test-set combination')
    comparison = []
    for ident in sorted(IDS):
        row = next(r for r in rows if int(r['dataset_id']) == ident)
        for cv in ('ordinary', 'homology'):
            original, filtered_scores = grouped[ident, cv, 'original'], grouped[ident, cv, 'filtered']
            scores = dict(optimization_cv=[x[1] for x in original], original_test=[x[0] for x in original],
                          filtered_test=[x[0] for x in filtered_scores])
            entry = dict(model_id=ident, model_name_tex=labels[str(ident)], cv=cv,
                         metric='R2' if ident == 63 else 'MCC', repeats=5,
                         original_test_records=int(row['test_records']),
                         retained_test_records=int(row['test_records_without_detected_matches']))
            for kind, values in scores.items():
                entry[kind + '_mean'], entry[kind + '_sd'] = mean(values), stdev(values)
            comparison.append(entry)
    return rows, comparison


def table_start(widths, caption, label, header):
    columns = ['@{}']
    for i, width in enumerate(widths):
        align = r'\raggedright' if i == 0 else r'\centering'
        padding = '' if i == 0 else r'-2\tabcolsep'
        columns.append(r'>{' + align + r'\arraybackslash}p{\dimexpr' + str(width) + r'\linewidth' + padding + r'\relax}')
    columns.append('@{}')
    return [r'% Generated by manuscript/tables/sequence_redundancy/extract_redundancy.py',
            r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.15}',
            r'\setlength{\LTleft}{0pt}', r'\setlength{\LTright}{0pt}', r'\setlength{\LTcapwidth}{\linewidth}',
            r'\begin{longtable}{' + ''.join(columns) + '}',
            r'\caption{' + caption + r'}\label{' + label + r'}\\', r'\toprule', header,
            r'\midrule', r'\endfirsthead', r'\toprule', header, r'\midrule', r'\endhead', r'\midrule',
            r'\multicolumn{' + str(len(widths)) + r'}{r}{\textit{Continued on next page}}\\',
            r'\endfoot', r'\bottomrule', r'\endlastfoot']


def render_audit(rows):
    caption = (r'\textbf{Sequence redundancy between the supplied training and test partitions.} '
               r'Exact matches are full-length sequence equality after uppercasing and nucleotide U-to-T normalization. '
               r'Similarity includes exact matches and MMseqs2 matches with at least 90\% identity and 80\% coverage of each sequence; nucleotide searches consider both strands. '
               r'Counts refer to test records with at least one training match; percentages use the test-set size. '
               r'Retained denotes records without a detected training match, not a newly constructed test set. '
               r'A dash indicates no supplied test set. All 471 test records for Model 58 match training exactly. '
               r'Search details and limitations are given in the Supplementary Notes (Sequence redundancy).')
    header = (r'\textbf{Model} & \shortstack{\textbf{Training}\\\textbf{records}} & \shortstack{\textbf{Test}\\\textbf{records}} & '
              r'\shortstack{\textbf{Exact}\\$n$ (\%)} & \shortstack{\textbf{Similar}\\\textbf{(incl. exact)}\\$n$ (\%)} & \shortstack{\textbf{Retained}\\\textbf{test records}} \\')
    lines = table_start([.34,.11,.10,.15,.17,.13], caption, 'tab:sequence_overlap', header)
    for r in rows:
        cells = [f"Model {r['dataset_id']}: {r['model_name_tex']}", f"{int(r['train_records']):,}"]
        if r['status'] == 'assessed':
            cells += [f"{int(r['test_records']):,}",
                      f"{int(r['exact_test_records']):,} ({float(r['exact_test_percent']):.2f})",
                      f"{int(r['similar_test_records_including_exact']):,} ({float(r['similar_test_percent_including_exact']):.2f})",
                      f"{int(r['test_records_without_detected_matches']):,}"]
        else:
            cells += ['--'] * 4
        lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\end{longtable}', r'\endgroup', ''])


def render_comparison(rows):
    caption = (r'\textbf{Effect of similarity-grouped cross-validation and test-overlap filtering on five representative datasets.} '
               r'Ordinary and grouped strategies use the supplied training partition with random or similarity-grouped folds, respectively. '
               r'Groups are constructed from training sequences only at the Table~\ref{tab:sequence_overlap} thresholds. '
               r'Values are means $\pm$ sample SD across five paired search repeats, not fold-level SD. '
               r'Ten-fold optimization CV remains conditioned on prior selection. Original test scores use all supplied test records; '
               r'filtered test scores reuse the same predictions after excluding detected training matches. '
               r'The last column gives retained/original test sizes. MCC is used for classification and predictive $R^2$ for regression; higher is better. '
               r'Test filtering changes sample composition, so score differences are not causal estimates of redundancy effects.')
    header = (r'\textbf{Model} & \shortstack{\textbf{CV}\\\textbf{strategy}} & \textbf{Metric} & '
              r'\shortstack{\textbf{Optimization}\\\textbf{CV}} & \shortstack{\textbf{Original}\\\textbf{test}} & '
              r'\shortstack{\textbf{Filtered}\\\textbf{test}} & \shortstack{\textbf{Test size}\\retained/\\all} \\')
    lines = table_start([.26,.12,.06,.16,.16,.16,.08], caption, 'tab:homology_evaluation', header)
    previous = None
    for r in rows:
        ident = r['model_id']
        if previous is not None and previous != ident:
            lines.append(r'\midrule')
        cells = [f"Model {ident}: {r['model_name_tex']}" if ident != previous else '',
                 'Ordinary' if r['cv'] == 'ordinary' else 'Grouped', r'$R^2$' if ident == 63 else 'MCC']
        cells += [f"${r[k + '_mean']:.4f} \\pm {r[k + '_sd']:.4f}$" for k in ('optimization_cv','original_test','filtered_test')]
        cells.append(r'\shortstack{' + f"{r['retained_test_records']:,}/" + r'\\' + f"{r['original_test_records']:,}" + '}')
        lines.append(' & '.join(cells) + r' \\')
        previous = ident
    return '\n'.join(lines + [r'\end{longtable}', r'\endgroup', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync-manuscript', action='store_true')
    args = parser.parse_args()
    audit, comparison = prepare()
    destination = Path(__file__).resolve().parent
    for name, rows, render in [('sequence_overlap',audit,render_audit),('homology_evaluation',comparison,render_comparison)]:
        with (destination / (name + '.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        text = render(rows)
        (destination / (name + '.tex')).write_text(text)
        if args.sync_manuscript:
            (MANUSCRIPT / 'supplementary/tables' / (name + '.tex')).write_text(text)
    print('Validated 64 audit rows, 50 pipeline runs, 100 prediction-based scores and 5/10-fold group separation; wrote S7 and S8.')


if __name__ == '__main__':
    main()
