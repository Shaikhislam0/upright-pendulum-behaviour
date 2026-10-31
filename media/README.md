# One successful full evaluation

`successful_simulation.mp4` animates the entire saved trajectory from physical
time 0 to 20 s at 50 frames/s. It includes swing-up, first completed hold, and
the rest of follow-up. The inclusive final sample makes the clip 20.02 s long.
The fixed camera and physical geometry preserve cart travel and rod length.

One trial was selected uniformly from the 120 successful final-generation
arm/seed evaluations (68 bounded, 52 quadratic), with no redraw or screening.
`selection.json` records the random seed, selected trial, event, and hashes.
This is a success-conditioned illustration, not an estimate of success rate.
The paper reports all 100 controllers per arm, including failures.

`successful_trial.csv` is a direct subset of the hash-verified archived trace,
not a new rollout. State samples are 20 ms; the displayed first-hold completion
is the original event from the 2 ms monitor. The video does not infer a new
hold interval from the coarser samples. Force at each noninitial row is the
command applied during the preceding 20 ms control interval. Bottom plots
show the full trace for context; a cursor marks the current physical time.

To render again, install `requirements-video.txt` and run `python render_video.py`.
The encoder dependency includes FFmpeg; no training libraries or model files
are needed. The renderer code is MIT-licensed; licenses for the video,
preview images and recorded trajectory remain pending. See ../LICENSING.md.
