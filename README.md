# Temporal Soft Actor-Critic for Dynamic Navigation

This repository holds the code, configurations and submission tooling behind
the Honours thesis *Temporal Architectures for Deep Reinforcement Learning
Navigation in Dynamic Three-Dimensional Environments* (Adam Cameron, University
of Technology Sydney, 2026), supervised by Dr Dale Grant.

The thesis compares three ways of giving a Soft Actor-Critic (SAC) agent access
to its own history on a moving-platform navigation task: observation stacking,
a Long Short-Term Memory (LSTM) recurrent layer, and a windowed causal
Transformer. Every figure and every statistic in the thesis is produced by code
in this repository from the collected results files, rather than transcribed by
hand.

## Provenance

**This repository is not all original work, and the split matters.**

**Inherited from Dr Dale Grant, used with his permission.** The Unity ML-Agents
environment, including the moving platform, the level types, the sensor suite
and the action model; the Soft Actor-Critic implementation and its training
framework; the baseline observation-stacking scripts; the configuration
skeletons; and approximately 90 per cent of the LSTM variant.

**Built for this thesis.**

- **The Transformer-SAC integration.** A gated Transformer-XL encoder
  integrated into the actor-critic structure with separate actor and critic
  backbones, causal masking, learned positional embeddings and gated residual
  connections.
- **The Unity parameterisation of platform traverse time.** The moving
  platform's traverse time was a fixed scene property and is now an environment
  parameter, which is what makes the duration sweep possible.
- **The verification toolchain.** Configuration generation, batch submission
  with duplicate-submission guards, build-provenance verification from job
  markers, results collation across heterogeneous download layouts, metric
  extraction, statistical recomputation by exact enumeration, and figure
  production.
- **The mechanistic probes.** Seven numbered probe scripts for measuring what a
  trained temporal pathway actually does, each calibrated against deliberately
  broken networks built for the purpose, so that a probe reading of zero is
  known to mean an inert pathway rather than a broken measurement. The five
  distinct measurements the thesis reports are built from them, with
  `probe_00_inspect.py` serving as a structure inspector and
  `probe_05_summary.py` and `probe_06_seed_table.py` aggregating the rest.

The inherited components remain substantially Dr Grant's work. Anyone wishing
to reuse them beyond reading this repository should approach him rather than
treating this repository as the licence.

## Layout

```
temporal-sac-dynamic-navigation/
  Algorithms/                   all Python and the submission bodies
  Environments/
    DRLNavigation/
      Assets/                   the Unity project
      Config/                   one YAML per experimental cell
  Packages/                     the ML-Agents fork this work was built against
  Submissions/                  generated PBS submission scripts and manifests
  ProbeOutputs/                 what each probe printed, kept as a record
  Metrics/                      the metric series behind every reported figure
  requirements.txt              the local Python environment
  requirements_hpc.txt          the cluster Python environment
  run_smoke_tests*.bat          the two local smoke-test runners
```

Two things are deliberately absent. The Unity `Build/` folders are not tracked,
because a compiled player is a build artefact rather than source. The raw
`Results/` tree is not tracked either, because a full run directory carries its
policy checkpoints and the complete set runs to several gigabytes.
`Metrics/` holds what analysis actually needs, which is the metric
series, the seed files and the job markers, and comes to a few megabytes.

Every script written for this work lives in `Algorithms/` and resolves its own
paths from its location, so none of them needs the working directory to be set
before it runs.

## The Tools

Run all of these from Anaconda Prompt with the project's virtual environment
active, from the repository root.

### Preparing and Submitting Runs

| Script | What It Does |
|---|---|
| `generate_runs.py` | Writes the configurations, the submission scripts and the manifest for one sweep |
| `submit_body.sh` | The shared body of every submission script, including the provenance block each job writes |
| `verify_results_provenance.py` | Checks each result directory against the configuration it recorded for itself |

### Collecting and Auditing Results

| Script | What It Does |
|---|---|
| `audit_results.py` | Catalogues every run found under any number of folders and changes nothing. Reads four independent sources of evidence per run and reports any disagreement between them rather than resolving it silently |
| `organise_results.py` | Reorganises the results folder into one canonical copy per cell, quarantining flagged runs by reason. Writes nothing until `--apply` is passed and reverses with `--undo` |

### Measuring What the Trained Networks Do

| Script | What It Does |
|---|---|
| `probe_00_inspect.py` | Inspects a trained actor checkpoint and reports its structure |
| `probe_01_attention.py` | Probes a trained windowed causal Transformer for mask integrity and context use |
| `probe_02_attention_map.py` | Reads the trained Transformer actor's attention distribution directly |
| `probe_03_gate_and_lstm.py` | Finds out why the Transformer's history does not reach its output, and tests the recurrent arm the same way |
| `probe_04_scaling.py` | Separates magnitude from structure and locates the source of the large activations |
| `probe_05_summary.py` | Verifies all three architectures across every checkpoint supplied, in one run |
| `probe_06_seed_table.py` | Settles whether the per-seed differences are about training or about activation structure |

**The probes cannot be run against the collected results in this repository.**
They read `actor_dict_0` policy checkpoints, which are the trained weights,
while the collected results tree holds only the metric series, the seed files
and the job markers. `Metrics/` carries the metric series rather than the
trained weights, for the size reason given above. Running a probe therefore
needs a results tree that still carries its checkpoints, which means a full
cluster download rather than anything in this repository.

What the probes produced is recorded in `ProbeOutputs/` instead, so their
findings can be read without rerunning them.

`probe_02`, `probe_03` and `probe_06` import their loaders from one another and
have to stay together in `Algorithms/`.

### Recomputing the Statistics

| Script | What It Does |
|---|---|
| `analyse_statistics.py` | Recomputes every comparison reported in Chapter 3 and Appendix D by exact enumeration, and prints each recomputed value beside the published one |

```
python Algorithms\analyse_statistics.py --results Metrics
```

That reproduces all eighteen comparisons in the Appendix D contrast table from
the metric series alone, with no stored intermediate. Each row carries the
recomputed difference, the published difference, the exact two-tailed
Mann-Whitney U *p*-value, the published *p*-value, the smallest *p*-value the
design can produce, and Hedges' *g*. A row whose recomputation disagrees with
the published value is marked, so the check is readable without comparing two
documents by eye.

Enumeration rather than approximation is used throughout, since at five seeds
against five there are only 252 rank arrangements and the normal approximation
is unreliable at that size. Ties are half-weighted, which is standard for
Mann-Whitney and is why a tied comparison recomputed with SciPy will not always
agree.

`analyse_statistics.py` holds its `canonical` function identical to the one in
`generate_figures.py` on purpose. A figure and a statistic that disagreed about
which runs form a cell would be worse than either on its own.

### Producing the Figures

| Script | What It Does |
|---|---|
| `generate_figures.py` | Builds Figures 1.1, 2.1, 3.1, 3.2, 3.3, 3.4, 3.8 and 3.9 and prints the numbers behind each one |
| `draw_timeout_figures.py` | Builds Figures 3.5 and 3.6 from the merged comparison groups `organise_results.py` writes under `Results/plotting` |

Figures 2.2 and 2.3 are TikZ pictures written in the thesis source itself, and
Figure 3.7 is four Unity trajectory captures taken inside the editor, so
neither can be produced here.

`Plotting.py` is inherited and draws per-run time series. It produced none of
the figures in the thesis. It carries small local changes from the period when
it was still the only plotting available, which is why it shows as modified
against the fork.

## Reproducing a Figure

```
conda activate drl_nav
cd /d D:\Projects\temporal-sac-dynamic-navigation

python Algorithms\generate_figures.py ^
    --results Metrics ^
    --out "<the thesis repository>\figures"
```

Nothing outside this repository is needed for that command, since
`Metrics/` is tracked.

`Metrics/` carries the run names the cluster wrote, such as
`bl_h_t030_s0_165871`, because that is the scheme `generate_figures.py` parses
into its cell keys. `organise_results.py` relabels runs when it builds
`Results/runs/`, so a bundle collected from that tree carries names like
`bl_h_w4_t3.0s_s0` and every figure comes out empty. Rebuild `Metrics/` from a
cluster download rather than from the organised tree.

Both inputs are optional and a figure whose input is missing is skipped by
name, so the command above builds every figure that needs only results. Add
`--only 3.3` to build one figure, and `--only 2.1 --screenshots "<folder>"` to
redraw the environment captures.

**Figure 2.1 is left out of a default run on purpose.** The version committed
to the thesis carries travel arrows drawn by hand over the Unity captures, and
a default run would replace those with the measured arrows the script draws.

The script prints the numbers behind every figure as it builds them, so a
figure can be checked against the table it belongs with without opening either.

For Figures 3.5 and 3.6:

```
python Algorithms\draw_timeout_figures.py
```

That writes into the thesis repository's `figures` folder by default. Add
`--check` to write to the downloads folder and print the numbers instead, which
verifies a results tree without touching anything that is committed.

Both figure scripts share one palette, defined once in each of them and
mirrored by the colour definitions at the top of the thesis `preamble.tex`,
which the TikZ pictures use. Changing a colour means changing it in all three
places.

## Recorded Probe Output

`ProbeOutputs/` holds the console output of each probe exactly as it was
produced, one file per probe. Every probe prints the checkpoint or tree it read,
the PyTorch version and its random seed in its own header, so each file states
the conditions it was produced under without needing a note alongside it.

These files exist because the checkpoints the probes read are not in this
repository, as explained above. They are the evidence behind the mechanistic
analysis in Section 3.9 of the thesis, and they can be checked against the
tables there without running anything.

## Environment

`requirements.txt` fixes the local environment and `requirements_hpc.txt` fixes
the cluster one. The two differ because the cluster carries PyTorch 1.8.2 while
the local environment carries 1.8.1, which is recorded in Appendix A of the
thesis along with the Unity editor revision and the ML-Agents fork commit the
work was built against.

The figure scripts need NumPy and Matplotlib. Figure 2.1 additionally needs
Pillow and SciPy. The probes need PyTorch.

## Determinism and Provenance

Every job writes a marker recording the configuration it loaded, the
configuration it was asked for, and MD5 digests of all seven Python source
files and of the compiled Unity assembly. Two allocations of the same
configuration to different compute nodes produced bit-identical learned series,
and every run from 13 October 2026 onward carries identical digests across 29
job records. That is what allows a stale run to be separated from a current one
when every other field agrees, and it is why `audit_results.py` reads the
markers rather than the folder names.

## The Thesis

The thesis source is held separately. Appendix A of the thesis gives the
complete configuration listing, Appendix B separates what was inherited from
what was built in more detail than this file does, and Appendix D holds the
supplementary per-seed statistics.
