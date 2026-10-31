"""Audit the completed fixed cohort and export descriptive scientific figures.

Read-only with respect to every training run. No policy inference or training.
Run from the project root with its original virtual environment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import binomtest
from upright_pendulum.cohort_results import exact_binomial_intervals, surface
from upright_pendulum.resume_generations import validate_full_implementation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def source_chain(final):
    chain, seen = [], set()
    current = final
    while True:
        assert current not in seen, 'source-chain cycle'
        seen.add(current)
        m = json.loads((current / 'manifest.json').read_text())
        chain.append((current, m))
        parent = m.get('recovery', {}).get('source_run')
        if parent is None:
            return list(reversed(chain))
        current = Path(parent).resolve()
        assert current.is_relative_to(ROOT / 'runs'), 'source outside project runs'


def audit_trajectory(path, record, member_ids):
    data = np.empty((100100, 12), dtype=np.float64)
    boolean = {'True': 1., 'False': 0.}
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        assert next(reader) == ['member_id', 'time_s', 'x_m', 'v_m_s', 'theta_rad_unwrapped',
                                'omega_rad_s', 'normalized_action', 'force_N', 'reward',
                                'in_region', 'has_achieved_balance', 'first_success_time_s']
        rows = 0
        for index, row in enumerate(reader):
            assert index < len(data) and len(row) == 12
            data[index, 0] = int(row[0].removeprefix('member_'))
            data[index, 1:9] = [float(value) if value else np.nan for value in row[1:9]]
            data[index, 9] = boolean[row[9]]
            data[index, 10] = boolean[row[10]]
            data[index, 11] = float(row[11]) if row[11] else np.nan
            rows += 1
    assert rows == 100100, f'incomplete trajectory: {path}'
    ids = data[:, 0].astype(int)
    assert ((0 <= ids) & (ids < 100)).all()
    time = data[:, 1]
    assert np.isfinite(time).all() and ((0 <= time) & (time <= 20)).all()
    assert np.allclose(time / .02, np.rint(time / .02), atol=1e-9, rtol=0)
    keys = np.rint(time / .02).astype(int) * 100 + ids
    assert np.array_equal(np.sort(keys), np.arange(100100)), 'missing or duplicate member/time'
    assert (np.bincount(ids) == 1001).all()
    states = data[:, 2:6]
    assert np.isfinite(states).all()
    start = time == 0
    assert start.sum() == 100
    assert np.allclose(states[start], [0, 0, np.pi, 0], atol=1e-14, rtol=0)
    for column in [6, 7, 8]:
        assert np.isnan(data[start, column]).all()
        assert np.isfinite(data[~start, column]).all()
    assert np.abs(data[~start, 7]).max() <= 10 + 1e-12
    assert np.abs(data[~start, 6]).max() <= 1 + 1e-12
    assert np.allclose(data[~start, 7], 10 * data[~start, 6], atol=1e-10, rtol=0)
    members = {m['member_id']: m for m in record['members']}
    success_map = np.array([members[key]['success_time'] if members[key]['success_time'] is not None
                            else np.nan for key in member_ids], dtype=float)
    expected_times = success_map[ids]
    observed = np.isfinite(expected_times) & (expected_times <= time + 1e-12)
    assert np.array_equal(data[:, 10].astype(bool), observed)
    assert np.allclose(data[:, 11],
                       np.where(observed, expected_times, np.nan), equal_nan=True, rtol=0, atol=1e-12)
    angle = (states[:, 2] + np.pi) % (2 * np.pi) - np.pi
    region = ((np.abs(states[:, 0]) <= .1) & (np.abs(states[:, 1]) <= .1) &
              (np.abs(angle) <= np.deg2rad(5)) & (np.abs(states[:, 3]) <= .2))
    assert np.array_equal(region, data[:, 9].astype(bool))
    last = np.flatnonzero(time == 20)
    assert len(last) == 100
    for row in last:
        assert bool(data[row, 9]) == members[member_ids[ids[row]]]['in_region']
    return rows


def audit(source, output):
    chain = source_chain(source)
    manifests = dict(chain)
    final = chain[-1][1]
    settings = final['settings']
    assert settings == dict(models=100, generations=1000, batch_steps=1000,
                            minibatch_size=250, epochs=10, seed_base=20000, evaluation_every=10)
    assert final['status'] == 'complete' and final['error'] is None
    assert final['last_training_generation'] == final['last_fully_saved_policy_generation'] == 1000
    assert final['last_complete_evaluation_generation'] == 1000
    assert final['cohort_training_steps_completed'] == 100_000_000
    launch = json.loads(Path(str(source) + '.launch.json').read_text(encoding='utf-8-sig'))
    assert launch['exit_code'] == 0 and launch['finished_utc']
    records = json.loads((source / 'generations.json').read_text())
    generations, times, percentages, n = surface(records)
    assert n == 100 and np.array_equal(generations, np.arange(0, 1001, 10))
    assert np.array_equal(generations, final['completed_generations'])
    assert np.allclose(times, np.arange(201) / 10, atol=1e-12, rtol=0)
    assert (np.diff(percentages, axis=0) >= 0).all()
    member_ids = [f'member_{i:04d}' for i in range(100)]
    seed_map = {member_id: 20000 + i for i, member_id in enumerate(member_ids)}
    final_by_g = {r['generation']: r for r in records}
    owners = {}
    total_hashes = 0
    chain_report = []
    hash_path = output / 'source_artifact_hashes.csv'
    with hash_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'relative_path', 'sha256'])
        for run, m in chain:
            for key in ['settings', 'training_recipe', 'evaluation', 'initial_state',
                        'physical_parameters', 'stabilization', 'control_dt_s', 'integration_dt_s',
                        'policy', 'member_seeds', 'inference']:
                assert m[key] == final[key], f'configuration drift: {run.name}, {key}'
            assert m['member_seeds'] == seed_map
            validate_full_implementation(run, m)
            expected_source = m.get('continuation_source_sha256', m['source_sha256'])
            for name, expected in expected_source.items():
                assert digest(run / 'source' / name) == expected, f'changed source snapshot: {run}, {name}'
            def verify_artifact(item):
                relative, expected = item
                assert digest(run / relative) == expected, f'artifact hash mismatch: {run}, {relative}'
                return relative, expected
            with ThreadPoolExecutor(max_workers=4) as pool:
                for relative, expected in pool.map(verify_artifact, m['checkpoint_sha256'].items()):
                    writer.writerow([run.relative_to(ROOT).as_posix(), relative, expected])
                    total_hashes += 1
            if 'recovery' in m:
                parent = Path(m['recovery']['source_run'])
                assert digest(parent / 'generations.json') == m['recovery']['source_event_records_sha256']
                parent_m = json.loads((parent / 'manifest.json').read_text())
                assert m['source_checkpoint_sha256'] == parent_m['checkpoint_sha256']
            local = json.loads((run / 'generations.json').read_text())
            for record in local:
                g = record['generation']
                if g in m['completed_generations']:
                    assert record == final_by_g[g], f'changed inherited/repeated measurement at {g}'
                    trajectory = run / 'trajectories' / f'generation_{g:06d}.csv'
                    if trajectory.is_file():
                        owners[g] = run
            chain_report.append({'run': str(run), 'status': m['status'],
                                 'manifest_sha256': digest(run / 'manifest.json'),
                                 'events_sha256': digest(run / 'generations.json'),
                                 'artifact_hashes_verified': len(m['checkpoint_sha256'])})
            print(f"Audited source {run.name}: {len(m['checkpoint_sha256'])} artifact hashes", flush=True)
    assert set(owners) == set(generations)
    counts = np.rint(percentages).astype(int)  # n=100, so percentages equal counts.
    lower, upper = exact_binomial_intervals(counts, 100, .95)
    for k in range(101):
        reference = binomtest(k, 100).proportion_ci(confidence_level=.95, method='exact')
        actual = exact_binomial_intervals(np.array([k], dtype=int), 100, .95)
        assert np.allclose([actual[0][0], actual[1][0]], [reference.low, reference.high], atol=1e-10, rtol=0)
    saved = np.load(source / 'surface.npz')
    for key, expected in [('generations', generations), ('evaluation_times_s', times),
                          ('percentages', percentages), ('lower_percentage', 100 * lower),
                          ('upper_percentage', 100 * upper)]:
        assert np.allclose(saved[key], expected, atol=1e-10, rtol=0), key
    saved_csv = csv_rows(source / 'confidence.csv')
    fractions_csv = csv_rows(source / 'fractions.csv')
    for rows in (saved_csv, fractions_csv):
        assert len(rows) == 101 * 201
        rows.sort(key=lambda r: (int(r['generation']), float(r['evaluation_time_s'])))
        assert len({(r['generation'], r['evaluation_time_s']) for r in rows}) == 101 * 201
        assert np.array_equal(np.array([int(r['generation']) for r in rows]), np.repeat(generations, 201))
        assert np.allclose(np.array([float(r['evaluation_time_s']) for r in rows]), np.tile(times, 101))
        assert np.array_equal(np.array([int(r['achieved_balance']) for r in rows]).reshape(101, 201).T, counts)
        assert np.allclose(np.array([float(r['percentage']) for r in rows]).reshape(101, 201).T, percentages)
    ordered = saved_csv
    for name, expected in [('lower_percentage', 100 * lower), ('upper_percentage', 100 * upper)]:
        assert np.allclose(np.array([float(r[name]) for r in ordered]).reshape(101, 201).T, expected, atol=1e-10, rtol=0)
    event_rows, checkpoint_rows, trajectory_rows = [], [], 0
    for record in records:
        g = record['generation']
        run = owners[g]
        members = {m['member_id']: m for m in record['members']}
        assert set(members) == set(member_ids)
        for member_id in member_ids:
            m = members[member_id]
            assert m['training_seed'] == seed_map[member_id]
            ts, exit_time = m['success_time'], m['first_post_success_exit_time']
            assert m['has_succeeded'] == (ts is not None)
            assert m['observed_time'] == (20 if ts is None else ts)
            if ts is None:
                assert exit_time is None
            else:
                assert np.isfinite(ts) and 2 <= ts <= 20
                assert abs(ts / .002 - round(ts / .002)) < 1e-8
                if exit_time is not None:
                    assert ts <= exit_time <= 20
            stem = run / 'models' / member_id / f'generation_{g:06d}'
            for suffix in ('.zip', '.normalization.pkl', '.resume.pkl'):
                assert Path(str(stem) + suffix).relative_to(run).as_posix() in manifests[run]['checkpoint_sha256']
            with zipfile.ZipFile(str(stem) + '.zip') as archive:
                data = json.loads(archive.read('data'))
                assert archive.testzip() is None
            assert data['num_timesteps'] == g * 1000 and data['pendulum_generation'] == g
            assert data['seed'] == seed_map[member_id]
            event_rows.append({'generation': g, **m})
        trajectory = run / 'trajectories' / f'generation_{g:06d}.csv'
        trajectory_rows += audit_trajectory(trajectory, record, member_ids)
        checkpoint_rows.append({'generation': g, 'source_run': str(run), 'trajectory': str(trajectory),
                                'trajectory_sha256': digest(trajectory), 'members': 100})
        if g % 100 == 0:
            print(f'Audited complete trajectories and checkpoint metadata through G={g}', flush=True)
    write_csv(output / 'individual_events.csv', event_rows)
    write_csv(output / 'checkpoint_sources.csv', checkpoint_rows)
    np.savez_compressed(output / 'surface.npz', generations=generations, evaluation_times_s=times,
                        counts=counts, percentages=percentages, lower_percentage=100 * lower,
                        upper_percentage=100 * upper, n_simulations=100, confidence_level=.95)
    endpoint = [{'generation': int(g), 'successes_by_20s': int(counts[-1,j]),
                 'percentage': float(percentages[-1,j]), 'lower_95_percentage': float(100*lower[-1,j]),
                 'upper_95_percentage': float(100*upper[-1,j])} for j,g in enumerate(generations)]
    write_csv(output / 'endpoint_summary.csv', endpoint)
    audit_report = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
                    'final_run': str(source), 'source_chain': chain_report,
                    'source_artifact_hashes_verified': total_hashes, 'checkpoints': 101,
                    'independent_controllers': 100, 'complete_evaluations': 10100,
                    'trajectory_rows_verified': trajectory_rows, 'canonical_policy_bundles_verified': 10100,
                    'reporting_time_points': 201, 'confidence_reference_checks': 101,
                    'source_snapshots_versions_and_configuration_match': True,
                    'saved_exports_match_recomputed_events': True,
                    'full_raw_trajectory_state_action_region_and_event_checks': True,
                    'trajectory_limitation': 'Raw state traces are at 20 ms; sustained-hold event monitoring was at 2 ms. Traces do not independently prove between-sample continuous residence.',
                    'launch_exit_code': launch['exit_code'],
                    'script_sha256': digest(Path(__file__))}
    write_json(output / 'audit.json', audit_report)
    return records, generations, times, percentages, lower, upper, final


def figures(output, records, generations, times, percentages, lower, upper):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tmp' / 'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 13, 'axes.labelsize': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})

    def save(fig, name):
        fig.savefig(output / f'{name}.png', dpi=220, facecolor='white')
        fig.savefig(output / f'{name}.pdf', dpi=300, facecolor='white')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    # Categorical x coordinates: only the 101 evaluated checkpoints are columns.
    yedges = np.r_[times[0] - .05, (times[:-1] + times[1:]) / 2, times[-1] + .05]
    mesh = ax.pcolormesh(np.arange(102) - .5, yedges, percentages, cmap='cividis',
                         vmin=0, vmax=100, shading='flat', rasterized=True)
    ax.set(ylim=(0, 20), xlim=(-.5, 100.5), ylabel='Elapsed physical evaluation time, t (s)',
           xlabel='Evaluated PPO generation, g (every 10 generations)',
           title='First completed sustained stabilization')
    ax.set_xticks(np.arange(0, 101, 10), generations[::10])
    ax.set_yticks(np.arange(0, 21, 2))
    colorbar = fig.colorbar(mesh, ax=ax, pad=.025, ticks=np.arange(0, 101, 20))
    colorbar.set_label('Controllers completing the first 2 s hold by t (%)')
    fig.get_layout_engine().set(rect=(0, .06, 1, .94))
    fig.text(.08, .025, 'n = 100 independent training seeds. Columns are measured checkpoints; intermediate generations are unmeasured.', fontsize=8)
    save(fig, 'heatmap')

    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    ax.errorbar(generations, percentages[-1],
                yerr=np.vstack([percentages[-1] - 100 * lower[-1], 100 * upper[-1] - percentages[-1]]),
                fmt='o', ms=2.7, color='#176b66', ecolor='#b5bfbd', elinewidth=.65, capsize=0)
    ax.annotate(f"{percentages[-1,-1]:.0f}%", (1000, percentages[-1,-1]), xytext=(8, 0),
                textcoords='offset points', va='center', color='#176b66', fontweight='bold')
    ax.set(xlim=(-15, 1060), ylim=(0, 100), xlabel='Evaluated PPO generation, g',
           ylabel='Controllers completing the hold by 20 s (%)', title='Learning across the fixed cohort')
    ax.set_xticks(np.arange(0, 1001, 100))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', color='#e5e5e5', linewidth=.6)
    fig.get_layout_engine().set(rect=(0, .075, 1, .925))
    fig.text(.08, .025, 'Dots: measured checkpoints. Bars: 95% pointwise exact binomial intervals; not simultaneous bands.', fontsize=8)
    save(fig, 'success_by_20s')

    fig, ax = plt.subplots(figsize=(8, 4.9), layout='constrained')
    styles = [(0, '#777777', ':'), (300, '#999999', '--'), (500, '#48769a', '-'),
              (750, '#b87937', '-'), (1000, '#176b66', '-')]
    for g, color, style in styles:
        record = records[g // 10]
        events = np.sort([m['success_time'] for m in record['members'] if m['success_time'] is not None])
        x = np.unique(np.r_[0., events, 20.])
        counts = np.searchsorted(events, x + 1e-12, side='right')
        ax.step(x, counts, where='post', label=f'g = {g}', color=color, linestyle=style,
                linewidth=2 if g == 1000 else 1.4)
        if g == 1000:
            lo, hi = exact_binomial_intervals(counts.astype(int), 100, .95)
            ax.fill_between(x, 100 * lo, 100 * hi, step='post', color=color, alpha=.12,
                            label='Final checkpoint: 95% pointwise interval')
    ax.set(xlim=(0, 20), ylim=(0, 100), xlabel='Elapsed physical evaluation time, t (s)',
           ylabel='Controllers completing the first hold by t (%)', title='When stabilization is first completed')
    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', color='#e5e5e5', linewidth=.6)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    save(fig, 'cumulative_times')


def summarize(output, records, generations, times, percentages, lower, upper, manifest):
    final = records[-1]
    events = np.array([m['success_time'] for m in final['members'] if m['success_time'] is not None])
    exit_count = sum(m['first_post_success_exit_time'] is not None for m in final['members'])
    endpoint = percentages[-1]
    first = next((int(g) for g, value in zip(generations, endpoint) if value > 0), None)
    peak = int(endpoint.max())
    report = {'final_generation': 1000, 'models': 100, 'final_successes_by_20s': int(endpoint[-1]),
              'final_pointwise_95_interval_percentage': [float(100 * lower[-1,-1]), float(100 * upper[-1,-1])],
              'final_censored_at_20s': int(100 - len(events)),
              'final_successes_with_later_exit_before_20s': exit_count,
              'final_successes_with_no_observed_later_exit': int(len(events) - exit_count),
              'first_evaluated_generation_with_any_success': first,
              'largest_observed_checkpoint_count': peak,
              'generations_with_largest_observed_count': generations[endpoint == peak].tolist(),
              'peak_note': 'Descriptive maximum across repeatedly observed checkpoints, not a selected final model or an unbiased maximum-performance estimate.',
              'final_success_conditioned_median_time_s': float(np.median(events)),
              'final_time_to_at_least_50_successes_s': float(np.sort(events)[49]) if len(events) >= 50 else None,
              'final_mean_capped_time_s': float((events.sum() + (100 - len(events)) * 20) / 100),
              'counts_at_selected_times': {str(t): int(percentages[np.flatnonzero(np.isclose(times,t))[0], -1])
                                           for t in [2, 5, 10, 13, 15, 20]},
              'training_interactions_retained': 100_000_000, 'unique_checkpoint_evaluations': 10100,
              'completed_local_time': datetime.fromisoformat(manifest['finished_utc']).astimezone(
                  timezone(timedelta(hours=5))).isoformat()}
    write_json(output / 'summary.json', report)
    ci = report['final_pointwise_95_interval_percentage']
    selected = '\n'.join(f"| {g} | {int(endpoint[g//10])}/100 |" for g in [0,100,300,500,600,750,900,1000])
    final_times = '\n'.join(f'| {time} | {count}/100 |' for time, count in report['counts_at_selected_times'].items())
    time_to_half = (f"{report['final_time_to_at_least_50_successes_s']:.3f} s"
                    if report['final_time_to_at_least_50_successes_s'] is not None else 'not reached by 20 s')
    caption = 'Each column is one measured checkpoint (0,10,...,1000); each time row is 0.1 s. No smoothing or generation interpolation. Event monitoring used 2 ms substeps.'
    text = f'''# Fixed 100-controller learning study — completed results

All 100 controllers completed the prespecified G=1000 budget. Final evaluation: **{int(endpoint[-1])}/100 completed the first qualifying two-second hold by 20 s**, with a **95% pointwise Clopper–Pearson interval of {ci[0]:.1f}–{ci[1]:.1f}%**. All valid failures remain in the denominator. No further training was started.

## Primary measured surface

H_hat(g,t) = (1/100) sum_i 1[T_s,i(g) <= t]. T_s is completion of the first hold; later exits do not undo that event. Horizontal: evaluated generation. Vertical: physical evaluation time. Color: 100 H_hat.

![Measured learning surface]({(output/'heatmap.png').as_posix()})

{caption}

## Learning and timing

| Generation | Completed hold by 20 s |
| --- | ---: |
{selected}

The first evaluated checkpoint with any event was G={first}; only evaluated checkpoints are observed, so transient successes at unmeasured policies cannot be ruled out. The largest observed count was {peak}/100 at generations {report['generations_with_largest_observed_count']}. This descriptive maximum is not a replacement for the final checkpoint. Counts can decrease with generation; all regressions are retained.

At G=1000:

| Evaluation time (s) | Completed hold by that time |
| --- | ---: |
{final_times}

Among the {len(events)} successful controllers, the median first completion time was {np.median(events):.3f} s; this conditional quantity excludes the {100-len(events)} failures. Time to at least 50/100 completed holds was {time_to_half}. Of the successful controllers, {exit_count} later exited the tolerance region before 20 s and {len(events)-exit_count} had no observed later exit. Finite follow-up does not establish permanent stability.

![Success by 20 seconds]({(output/'success_by_20s.png').as_posix()})

![Cumulative completion times]({(output/'cumulative_times.png').as_posix()})

## Fixed method and interpretation

100 independently trained controllers (seeds 20000..20099); one generation = 1000 new interactions followed by one PPO rollout/optimization cycle, with ten epochs inside that cycle. Separate actor/critic gradient clipping is a declared PPO variant. Existing bounded reward, sin/cos input and private running observation normalization were fixed. Training alone used a terminal |x|>=3 m boundary with cost 10; evaluation used the original unbounded rail. No curriculum, teacher or post-hoc reward change.

Nonlinear 1 kg cart and uniform 0.1 kg, 1 m rod; +/-10 N, zero baseline friction, angle from upward vertical. Evaluation start [0,0,pi,0], deterministic matching frozen policy/preprocessing, full 20 s. Hold for 2 s simultaneously within |x|<=0.1 m, |v|<=0.1 m/s, upright wrapped angle<=5 degrees and |omega|<=0.2 rad/s. Float64 RK4 at 2 ms with 20 ms held controls. Monitoring is discrete at 2 ms; exact continuous residence between samples is not proven.

The study measures probability over training randomness, not disturbances. The same 100 seeds are followed across generations, so time/generation cells are correlated; they do not create additional independent models. Intervals are pointwise, not simultaneous confidence bands. A controller without a qualifying hold is right-censored at 20 s, not proven incapable forever. The prior G=100 study and exploratory pilots use different methods and are not pooled or treated as a reward-only causal comparison. Reward comparisons and robustness need separate prospective designs.

## Integrity and reproducibility

See [audit.json]({(output/'audit.json').as_posix()}) for the complete audit, [checkpoint_sources.csv]({(output/'checkpoint_sources.csv').as_posix()}) for canonical source locations/hashes, [individual_events.csv]({(output/'individual_events.csv').as_posix()}) for all 10,100 event records and [endpoint_summary.csv]({(output/'endpoint_summary.csv').as_posix()}) for the endpoint curve with pointwise bounds. Source runs were read without modification. Earlier repeated recovery-boundary evaluations agree exactly and contribute one canonical measurement each. Pauses and repeated unsaved work do not add independent samples.

Figures are available as PNG and PDF. Data/provenance snapshots, all checkpoint/preprocessing/recovery bundles and raw trajectories remain in their source chain. This analysis does not select a best controller, extend training or establish that the method is optimal.
'''
    (output / 'RESULTS.md').write_text(text, encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    output=args.output.resolve()
    assert source.is_relative_to(ROOT/'runs') and output.is_relative_to(ROOT/'analysis')
    output.mkdir(parents=True, exist_ok=False)
    records,g,t,p,lo,hi,manifest=audit(source,output)
    figures(output,records,g,t,p,lo,hi)
    summary=summarize(output,records,g,t,p,lo,hi,manifest)
    write_json(output/'artifact_hashes.json', {file.name:digest(file) for file in output.iterdir() if file.is_file()})
    print(json.dumps({'output':str(output),'summary':summary},indent=2),flush=True)


if __name__=='__main__':
    main()
