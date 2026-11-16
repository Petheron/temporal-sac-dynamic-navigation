# Transformer-Based Temporal Sequence Modelling for 3D Navigation

Code to reproduce the experiments from the Honours thesis "Transformer-Based Temporal Sequence Modelling for Dynamic Partially Observable 3D Navigation in Game Environments" by Adam Cameron, supervised by Dr Dale Grant.
The thesis compares three ways of giving a Soft Actor-Critic agent access to its own history on a moving-platform navigation task: observation stacking, an LSTM recurrent layer, and a windowed causal Transformer.
Every figure and every statistic in the thesis is produced by code here from the collected results, rather than transcribed by hand.
See provenance, installation and run procedures below.

## Provenance

This repository is not all original work.

Inherited from Dr Dale Grant, used with his permission:
* The Unity ML-Agents environment, including the moving platform, the level types, the sensor suite and the action model.
* The Soft Actor-Critic implementation and its training framework.
* The baseline observation-stacking scripts and the configuration skeletons.
* Approximately 90 per cent of the LSTM variant.

Built for this thesis:
* The Transformer-SAC integration. A gated Transformer-XL encoder in the actor-critic structure, with separate actor and critic backbones, causal masking, learned positional embeddings and gated residual connections.
* The Unity parameterisation of platform traverse time, which was a fixed scene property and is now an environment parameter. This is what makes the duration sweep possible.
* The verification toolchain: configuration generation, batch submission with duplicate-submission guards, build-provenance verification from job markers, results collation, metric extraction, statistical recomputation and figure production.
* Seven mechanistic probe scripts, each calibrated against deliberately broken networks built for the purpose, so that a probe reading of zero means an inert pathway rather than a broken measurement.

The inherited components remain substantially Dr Grant's work. Anyone wanting to reuse them should approach him rather than treating this repository as the licence.

## Installation

### Anaconda
* Install Anaconda/Miniconda.
* Create a conda environment: `conda create --name drl_nav python=3.8.10`
* Install pytorch first, before anything else. The local environment uses 1.8.1+cu111 and the cluster uses 1.8.2+cu111.
* Install remaining dependencies using `pip install -r requirements.txt`
* The cluster environment is pinned separately in requirements_hpc.txt.

### Unity
* Unity version 2022.2.12f1
* MLAgents package derived from Release 18 branch.
* Install Windows Dedicated Server module to create Windows executables for training.
* Build an .exe into a Build folder.

Appendix A of the thesis records the exact Unity editor revision and the ML-Agents fork commit this work was built against.

## Layout

```
Algorithms/                   all Python and the submission bodies
Environments/DRLNavigation/   the Unity project, and one YAML per experimental cell
Packages/                     the ML-Agents fork this work was built against
Submissions/                  generated PBS submission scripts and manifests
ProbeOutputs/                 what each probe printed, kept as a record
Metrics/                      the metric series behind every reported figure
```

The Unity Build folders and the raw Results tree are not tracked. A full run directory carries its policy checkpoints and the complete set runs to several gigabytes, so Metrics holds what the analysis needs instead, which is the metric series, the seed files and the job markers.

Every script written for this work lives in Algorithms and resolves its own paths from its location, so none of them needs the working directory set before it runs.

## To reproduce a figure

```
conda activate drl_nav
python Algorithms\generate_figures.py --results Metrics --out "<output folder>"
```

That builds Figures 1.1, 2.1, 3.1, 3.2, 3.3, 3.4, 3.8 and 3.9, and prints the numbers behind each one so a figure can be checked against its table without opening either. Add `--only 3.3` to build a single figure.

Figures 3.5 and 3.6 come from a second script, which writes into the thesis figures folder by default. Add `--check` to write to the downloads folder and print the numbers instead.

```
python Algorithms\draw_timeout_figures.py
```

Figures 2.2 and 2.3 are TikZ pictures in the thesis source, and Figure 3.7 is four Unity trajectory captures taken in the editor, so neither can be produced here.

Figure 2.1 is skipped by default. The committed version carries travel arrows drawn by hand over the Unity captures, and a default run would replace them with the measured arrows the script draws. Add `--only 2.1 --screenshots "<folder>"` to redraw it.

Metrics carries the run names the cluster wrote, such as bl_h_t030_s0_165871, because that is the scheme generate_figures.py parses into its cell keys. organise_results.py relabels runs when it builds Results/runs, so a bundle collected from that tree gives empty figures. Rebuild Metrics from a cluster download rather than from the organised tree.

Both figure scripts share one palette, mirrored by the colour definitions at the top of the thesis preamble.tex, which the TikZ pictures use. Changing a colour means changing it in all three places.

## To recompute the statistics

```
python Algorithms\analyse_statistics.py --results Metrics
```

That reproduces all eighteen comparisons in the Appendix C contrast table from the metric series alone, with no stored intermediate. Each row carries the recomputed difference, the published difference, the exact two-tailed Mann-Whitney U p-value, the published p-value, the smallest p-value the design can produce, and Hedges' g. A row whose recomputation disagrees with the published value is marked.

Exact enumeration is used rather than the normal approximation, since at five seeds against five there are only 252 rank arrangements. Ties are half-weighted, which is standard for Mann-Whitney and is why a tied comparison recomputed with SciPy will not always agree.

analyse_statistics.py holds its canonical function identical to the one in generate_figures.py on purpose, so that a figure and a statistic cannot disagree about which runs form a cell.

## The probes

Seven numbered scripts measure what a trained temporal pathway actually does.

* probe_00_inspect.py reports the structure of a trained actor checkpoint.
* probe_01_attention.py checks mask integrity and context use in a trained windowed causal Transformer.
* probe_02_attention_map.py reads the trained Transformer actor's attention distribution directly.
* probe_03_gate_and_lstm.py finds why the Transformer's history does not reach its output, and tests the recurrent arm the same way.
* probe_04_scaling.py separates magnitude from structure and locates the source of the large activations.
* probe_05_summary.py verifies all three architectures across every checkpoint supplied, in one run.
* probe_06_seed_table.py settles whether the per-seed differences are about training or about activation structure.

The probes cannot be run against the results in this repository. They read actor_dict_0 policy checkpoints, and Metrics carries only the metric series, the seed files and the job markers. Running a probe needs a results tree that still has its checkpoints, which means a full cluster download.

What the probes produced is recorded in ProbeOutputs instead, one file per probe. Each file prints the checkpoint or tree it read, the PyTorch version and its random seed in its own header, so it states the conditions it was produced under. These files are the evidence behind Section 3.9 of the thesis.

probe_02, probe_03 and probe_06 import their loaders from one another and have to stay together in Algorithms.

## Other scripts

* generate_runs.py writes the configurations, the submission scripts and the manifest for one sweep.
* submit_body.sh is the shared body of every submission script, including the provenance block each job writes.
* verify_results_provenance.py checks each result directory against the configuration it recorded for itself.
* audit_results.py catalogues every run found under any number of folders and changes nothing. It reads four independent sources of evidence per run and reports any disagreement rather than resolving it silently.
* organise_results.py reorganises the results folder into one canonical copy per cell, quarantining flagged runs by reason. It writes nothing until `--apply` is passed, and reverses with `--undo`.
* Plotting.py is inherited and draws per-run time series. It produced none of the figures in the thesis, and carries small local changes from when it was the only plotting available.

## Determinism

Every job writes a marker recording the configuration it loaded, the configuration it was asked for, and MD5 digests of all seven Python source files and of the compiled Unity assembly. Two allocations of the same configuration to different compute nodes produced bit-identical learned series, and every run from 13 October 2026 onward carries identical digests across 29 job records. That is why audit_results.py reads the markers rather than the folder names.

## The thesis

The thesis source is held separately. Appendix A gives the complete configuration listing, Appendix B separates what was inherited from what was built in more detail than this file does, and Appendix C holds the supplementary per-seed statistics.
