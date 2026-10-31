# Reproduce the research results

## Analysis without training

Use Python 3.12. From the repository root:

```console
python -m venv .venv
```

Activate that environment using your operating system's usual command, then:

```console
python -m pip install -r requirements-analysis.txt
python reproduce_analysis.py --output reproduced
```

The output folder must be new. This verifies input checksums, aligns all 100
paired seeds at the 101 evaluated checkpoints, and reproduces cumulative
counts, arm intervals, the final paired comparison, timing summaries and
descriptive transitions. It creates figures and a numerical summary without
training, policy files or Torch. Regenerated figures use a wide layout;
the supplied paper uses portrait figures from the same data.

## Data dictionary

- `data/bounded_events.csv` and `data/quadratic_events.csv`: 10,100 evaluations
  each. `training_seed` identifies the paired controller. `generation` is one
  1,000-interaction PPO rollout/update cycle; measured generations are
  0, 10, ..., 1,000. `success_time` is the first completed two-second hold,
  measured in physical seconds. An empty value is a censored observation at
  20 s, retained in the denominator. It is not a missing trial.
- `event_observed`, `has_succeeded` and `right_censored` describe the first
  event. `observed_time` is the event time or 20 s; `followup_time` is 20 s.
  A later exit is stored separately in `first_post_success_exit_time` and
  does not undo success. `in_region` describes the final state only.
- `data/paired_surface.npz`: seeds [100], generations [101], physical
  reporting times [201], counts [time, generation], and event times
  [seed, generation]. Infinity is the no-event sentinel through 20 s,
  not a claim that future success is impossible. Reporting uses 0.1 s;
  the original event monitor uses 0.002 s.
- `data/paired_seed_results.csv`: final-generation paired outcomes.
  `endpoint_comparison.csv`: descriptive by-20-s counts across training.
  `reported_summary.json`: reported statistics. The exploratory transition
  file is explicitly post hoc descriptive.

The same seeds are followed across checkpoints and paired across rewards.
Those repeated measurements are not additional independent replications.
Only the declared final endpoint receives the primary paired test. The
comparison is a follow-up using an already observed bounded reference arm.

## Supplied checks

From the root, install the test requirements, then run from `source/`:

```console
python -m pip install -r requirements-test.txt
cd source
python -m pytest -q
```

The supplied tests cover nonlinear physics, the Gymnasium environment,
stabilization measurement, and analysis-input consistency.

## Optional full training rerun

This is separate, substantial new computation and is unnecessary for the
event-data analysis above. Install `source/requirements-training-lock.txt`
and the package in `source/`. From `source/`, use a fresh output:

```console
python -m pip install -r requirements-training-lock.txt
python -m pip install -e .
python -m upright_pendulum.train_generations --models 100 --generations 1000 --batch-steps 1000 --minibatch-size 250 --epochs 10 --seed-base 20000 --evaluation-every 10 --confidence-level 0.95 --workers 12 --gradient-clipping independent --training-boundary 3 --boundary-failure-cost 10 --config configs/bounded_boundary_rolling.json --output rerun_bounded
```

Repeat with `configs/quadratic_boundary_rolling_followup.json` and a different
fresh output, such as `rerun_quadratic`. The PPO variant clips actor and critic
gradients separately. Historical source entry points are preserved but need
not be run to reproduce the paper. Exact identity on untested platforms is
not guaranteed by the recorded Windows/CPU environment.

## Paper and video

The editable paper and its supplied figures/equation images are in `paper/`.
To rebuild the PDF, install `paper/requirements-pdf.txt` and run
`python paper/render_manuscript.py`. Rebuilding produces a new PDF and can
restore typesetter metadata; the distributed PDF has been privacy-sanitized.
Changing equation text also requires updating its equation image.

The complete example trace, video-selection record and renderer are in
`media/`. See its README for rendering instructions. The video is a selected
successful illustration and does not estimate cohort success probability.

Full policy/optimizer/recovery archives and all raw control-step traces are
not in this compact repository, apart from the one example trajectory.
They are preserved separately. See [LICENSING.md](LICENSING.md) for the
software MIT grant and the separate pending paper/data/media choices.
