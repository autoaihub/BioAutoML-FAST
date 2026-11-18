"""Populate only BioAutoML-FAST rows from the independently audited five runs.

Default is preview. --write updates the active manuscript; --check fails if
the manuscript differs from the audited values. Published comparator cells
and model names are preserved. No significance or superiority bolding is added.
"""
import argparse
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from bioautoml.execution import file_hash

DEFAULT_TABLES = PROJECT / 'manuscript/revision/BioAutoML-FAST_LaTeX/oup-authoring-template/supplementary/tables'
DEFAULT_AUDIT = PROJECT / 'manuscript/experiments/results/reproducibility/archive_audit'


def render_table(text, name, summary):
    metrics = ['Sn','Sp','ACC','MCC','AUC'] if name == 'binary' else (
        ['Sn','Sp','ACC','MCC'] if name == 'multiclass' else ['MAE','MSE','RMSE','R2','Pearson'])
    expected = set(range(1,51)) if name == 'binary' else (set(range(51,58)) if name == 'multiclass' else set(range(58,65)))
    seen, model, rendered = set(), None, []
    for line in text.splitlines(keepends=True):
        match = re.search(r'Model\s+(\d+)\s*[:\-]', line)
        if match:
            model = int(match.group(1))
        if re.match(r'\s*&\s*BioAutoML-FAST\s*&', line):
            if model not in expected or model in seen:
                raise ValueError(f'Unexpected/duplicate model row: {model}')
            cells = []
            for split in ('cv','test'):
                for metric in metrics:
                    key = (model, split, metric)
                    if key not in summary.index:
                        if split == 'cv' or (model, 'test') in summary.index.droplevel('metric'):
                            raise ValueError(f'Missing metric in an available evaluation: {key}')
                        cells.append('-')
                        continue
                    row = summary.loc[key]
                    if int(row['count']) != 5 or not np.isfinite([row['mean'], row['std']]).all():
                        raise ValueError(f'Invalid aggregate: {key}')
                    mean = 0. if abs(row['mean']) < .00005 else row['mean']
                    cells.append(f'{mean:.4f} $\\pm$ {row["std"]:.4f}')
            line = '  & BioAutoML-FAST & ' + ' & '.join(cells) + ' \\\\\n'
            seen.add(model)
        rendered.append(line)
    if seen != expected:
        raise ValueError(f'Missing models in {name}: {expected-seen}')
    result = ''.join(rendered)
    # Remove the old, unused bolding rule; do not implement the unresolved SOTA
    # comparison policy while updating numerical results.
    old = (r'Dashes (-) indicate metrics that were not reported in the baseline study. '
           r'Bold entries cover two distinct cases: (i)~BioAutoML-FAST\'s point estimate strictly exceeds the reported SOTA value, '
           r'and (ii)~the SOTA value falls within our mean~$\pm$~1~SD even though the point estimate does not reach SOTA.')
    # Apostrophe is literal, not a TeX escape.
    old = old.replace("\\'", "'")
    new = (r'BioAutoML-FAST values are mean~$\pm$~sample SD across five runs. '
           r'Dashes indicate unavailable comparator metrics or absence of an external test set.')
    result = result.replace(old, new)
    convention = (' Test macro-averages include all trained classes; absent test classes '
                  'contribute zero sensitivity (Model 52: Cytosol).')
    if name == 'multiclass' and convention not in result:
        result = result.replace(new, new + convention)
    result = result.replace('Dashes indicate unavailable or unassigned source values; BioAutoML-FAST cells are left blank for completion.',
        r'BioAutoML-FAST values are mean~$\pm$~sample SD across five runs. Dashes indicate unavailable source values or absence of an external test set.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-root', type=Path, default=PROJECT / 'App/datasets')
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--tables', type=Path, default=DEFAULT_TABLES)
    parser.add_argument('--preview-dir', type=Path, default=DEFAULT_AUDIT / 'table_preview')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = json.loads((args.audit / 'validation.json').read_text())
    assert report['status'] == 'passed' and report['runs'] == 320
    root = args.datasets_root.resolve()
    fingerprints = pd.concat([pd.read_csv(args.audit / 'metric_sources.csv'),
                              pd.read_csv(args.audit / 'input_inventory.csv')], ignore_index=True)
    for row in fingerprints.itertuples():
        path = (root / row.path).resolve()
        if not path.is_relative_to(root) or file_hash(path) != row.sha256:
            raise ValueError(f'Stale or external metric source: {row.path}; rerun the audit.')
    runs = pd.read_csv(args.audit / 'run_metrics.csv')
    assert set(runs.model) == set(range(1,65))
    assert not runs.duplicated(['model','run','split','metric']).any()
    for _, group in runs.groupby(['model','split','metric']):
        assert sorted(group.run) == [1,2,3,4,5]
    summary = runs.groupby(['model','split','metric']).value.agg(['mean','std','count'])
    stored = pd.read_csv(args.audit / 'summary.csv').set_index(['model','split','metric'])[['mean','std','count']]
    pd.testing.assert_frame_equal(summary.sort_index(), stored.sort_index(), check_exact=False, rtol=1e-10, atol=1e-12)
    prepared = {}
    for name in ('binary','multiclass','regression'):
        path = args.tables / f'{name}.tex'
        before = path.read_text()
        after = render_table(before, name, summary)
        assert render_table(after, name, summary) == after, 'Non-idempotent table update.'
        prepared[path] = (before, after)
    # Prepare all tables successfully before modifying any of them.
    for path, (before, after) in prepared.items():
        if args.check:
            if before != after:
                raise ValueError(f'Table differs from audited results: {path}')
        elif args.write:
            path.write_text(after)
        else:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            (args.preview_dir / path.name).write_text(after)
        print(f'{path.name}: '+('verified' if args.check else 'updated' if args.write else 'preview generated'))


if __name__ == '__main__':
    main()
