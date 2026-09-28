# Modular Spacecraft Reconfiguration — RL Research Platform

A research platform for studying **mission-driven reconfiguration of modular
spacecraft**. A spacecraft is a set of identical cube modules ("CubeSats") on a
3D lattice that pivot around shared edges to change the overall morphology. The
central research question:

> **Can reinforcement-learning controllers (centralized or decentralized) learn
> to reconfigure a modular spacecraft into mission-appropriate morphologies, and
> how does decentralized control compare with centralized control and classical
> planning heuristics?**

The platform prioritizes a **clean, correct experimental framework** over
high-fidelity mechanical simulation. Physics is abstracted to discrete
graph/state transitions; missions are physically grounded but simplified.

---

## Key ideas

- **Environment owns the spacecraft.** Controllers only *propose* actions. The
  environment validates them, resolves conflicts, applies transitions, and
  computes rewards. Every controller — random, greedy, A\*, centralized GNN,
  decentralized GNN — plugs into the same environment through one interface.

- **Missions = optimize one objective subject to hard constraints** (connectivity,
  no overlap, legal geometry), not arbitrary weighted sums.

- **Centralized vs decentralized is the core comparison.** Both use the *same*
  GNN architecture; they differ only in **what each node observes** (global vs
  local features) and **how far information propagates** (number of
  message-passing layers). This isolates *information locality* as the single
  experimental variable.

- **Everything is reproducible.** Fixed seeds, saved manifests, full per-run
  records, training curves, and replayable move-sequence traces.

---

## Installation

```bash
git clone <repo>
cd Swarm_Config
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[viz,rl,dev]"   # matplotlib/pillow, torch, pytest
```

Requires Python ≥ 3.9. GPU is optional (`--device cuda` for training).

---

## Quick start

```bash
# 1. See the simulator work: legal moves + a hand-driven reconfiguration
python -m scripts.interactive_demo --n 4

# 2. Run classical baselines
python -m scripts.run_baseline --controller greedy --n 6 --mission thermal
python -m scripts.run_baseline --controller astar  --n 4 --mission power

# 3. Train learned controllers
python -m scripts.train_gnn --n 4 6 --missions power thermal comms aperture \
    --n-layers 4 --total-steps 300000 --checkpoint-name gnn_centralized.pt
python -m scripts.train_decentralized --n 4 6 --missions power thermal comms aperture \
    --actor-layers 2 --critic-layers 4 --total-steps 300000 \
    --checkpoint-name gnn_decentralized.pt

# 4. Benchmark everything on shared scenarios
python main.py --missions power thermal comms aperture --n 4 6 8 \
    --config-seeds 0 1 2 --repeats 5 \
    --gnn-checkpoint checkpoints/gnn_centralized.pt \
    --decentralized-checkpoint checkpoints/gnn_decentralized.pt

# 5. Animate the best run of a method on a scenario
python -m scripts.replay results/<id>/traces/power_n4_random_seed0__centralized_gnn.json

# 6. Run the tests
pytest -q
```

---

## The spacecraft model

- **Module**: a cube with a `position` (integer lattice), an `orientation` (one
  of the 24 cube rotations), six `face_capabilities`, and internal state
  (battery, temperature, power generation).
- **Face capabilities**: `SOLAR_PANEL`, `ANTENNA`, `SCIENCE_INSTRUMENT`,
  `RADIATOR`, `STRUCTURAL`, `NONE`.
- **Connectivity**: derived from geometry — two modules are connected iff they
  are face-adjacent. The connection graph is never stored independently.

### Reconfiguration (pivoting-cube model)

An action is **rotate around one of the 12 cube edges, in a + or − direction**.
The rotation resolves to:
- **90°** if there is a supporting cube at the landing position (rolls into the
  adjacent orthogonal cell), or
- **180°** otherwise (wraps over the support to the diagonal cell).

A move is legal only if: the pivot edge has support, the destination is empty,
the swept path is clear, the module stays attached, and the move doesn't
disconnect the structure. All legality is enforced by the environment.

### Simultaneous actions & conflict resolution

All modules propose actions at once. The environment resolves conflicts with a
deterministic **reject-both** rule (no arbitrary priority): if two proposals
collide (same destination, transit conflict, or joint-connectivity violation),
none of the conflicting moves execute.

---

## Missions

Each mission maximizes/minimizes one physically-grounded objective. `sun_direction`
and `earth_direction` are axis-aligned environment parameters.

| Mission | Objective | Physics |
|---------|-----------|---------|
| **power** | maximize | Solar generation ∝ Σ illumination of unoccluded solar faces (dot with sun). |
| **thermal** | minimize | **Coupled per-module steady-state temperature.** Energy balance (min-power dissipation + absorbed solar = radiation + inter-module conduction) solved by Newton iteration; objective = worst-case hot-spot temperature (cruise). |
| **comms** | maximize | **Data rate to Earth.** Antenna faces combine coherently toward earth; `datarate = log2(1 + β·|Σ (antenna·earth)·unoccluded|²)`. |
| **aperture** | maximize | **Interferometric baseline.** Earth-facing science faces are aperture elements; objective = (#elements) + longest pairwise baseline projected ⊥ to earth. |

Physical constants live in `environment/cubesat.py` and are configurable.

---

## Controllers

All implement `Controller.act(observation) -> {module_id: action}`.

| Controller | Learned? | Info access | Notes |
|------------|:--------:|-------------|-------|
| `random` | no | — | legal-action random; stress-tests the sim |
| `greedy_joint` / `greedy_seq` | no | global | picks the immediately best objective-improving joint action |
| `astar` | no | global | search over the configuration graph; optimal for small N, scales poorly |
| `centralized_gnn` | yes | **global** | GNN, many MP layers, full node features |
| `decentralized_gnn` | yes | **local only** | GNN, few MP layers, own+neighbor features; trained via CTDE |

### The GNN

A hand-rolled message-passing network (no external GNN dependency):
```
h_i^0 = encode(node_features_i)
m_ij  = message([h_i, h_j, e_ij]);   m_i = mean_j m_ij
h_i'  = h_i + update([h_i, m_i])         (residual)
```
- **Shared weights across all nodes** → permutation-invariant, handles *any* N,
  no padding, no `N_max`. One checkpoint serves all module counts and missions
  (mission + sun/earth are per-node features).
- **Number of MP layers = information radius.** L layers → an L-hop neighborhood.
  Large L = centralized reasoning; small L = local/decentralized.

### CTDE (decentralized training)

Centralized Training, Decentralized Execution:
- **Actor** (used at train & eval): local node features, few MP layers.
- **Critic** (train only): global node features, many MP layers.
- At execution the critic is discarded; the actor consumes *only*
  `env.all_local_observations()`. Tests assert global state can never leak into
  decentralized action selection.

---

## RL algorithm

**PPO** (clipped objective, GAE) throughout, so the experiment isolates
*architecture and information access* rather than RL algorithm. The decentralized
variant is MAPPO-style (shared actor, centralized critic). Action masking ensures
policies only ever propose legal moves.

---

## Benchmarking & outputs

`main.py` runs every controller on **shared** scenarios (same reset state per
`(mission, N, config_seed)`), so comparisons are fair. Deterministic methods run
once; stochastic methods run `--repeats` times with derived seeds.

Each run saves to `results/<datestring>/`:

```
manifest.json     run configuration + all seeds (reproducibility)
records.json      full per-run metrics (every repeat kept)
records.csv       flat version for plotting/spreadsheets
summary.txt       human-readable mean ± std comparison table
plots/            scaling curves, reward-vs-moves, distance-from-optimum (per mission)
traces/           best run's move sequence per (scenario, method), replayable
```

### Metrics recorded
Mission: initial/final objective, improvement, %improvement, distance-from-best.
Efficiency: steps, total moves, conflicts, invalid actions, total reward.
Compute: wall time, A\* node expansions, plan length.
Run index, and eval seed.

### Regenerate plots without rerunning
```bash
python main.py --replot results/<id>
```

---

## Repository structure

```
environment/
  geometry.py             24 cube rotations, 12 edges, faces, directions
  cubesat.py              module data + physical constants
  configuration.py        canonical hashable configuration (A*-ready)
  graph.py                connectivity derived from geometry
  transitions.py          pivoting-cube legality + 90/180 resolution + orientation
  actions.py              action space (STAY + 24 edge pivots)
  conflict_resolution.py  reject-both simultaneous resolution
  missions.py             power / thermal / comms / aperture + hard constraints
  spacecraft.py           SpacecraftEnv (owns all state)
  make_env.py             factory
  configuration_builder.py

controllers/
  base.py                 Controller interface
  random_controller.py
  gnn_policy.py           hand-rolled message-passing GNN (actor + critic heads)
  centralized_gnn_controller.py
  decentralized_gnn.py    CTDE actor-critic container (local actor + global critic)
  decentralized_gnn_controller.py   local-only execution (leak-proof)

baselines/
  greedy.py               joint + sequential greedy
  astar.py                goal-directed + objective-maximizing search
  astar_controller.py     replays an A* plan step-by-step

rl/
  ppo.py                  PPO clipped update (flat)
  buffer.py               flat rollout buffer + GAE
  encoders.py             per-module feature vector
  action_masking.py       legal-action masks (per-node and flat)
  graph_batch.py          global-graph construction + disjoint batching
  local_graph.py          local-graph construction (own + neighbor only)
  gnn_buffer.py           graph rollout buffer
  gnn_ppo.py              PPO update over graph minibatches
  gnn_trainer.py          centralized GNN trainer
  ctde_buffer.py          CTDE rollout buffer (stores local + global graphs)
  ctde_ppo.py             CTDE PPO update (local actor loss + global critic loss)
  ctde_trainer.py         decentralized (CTDE) trainer

evaluation/
  metrics.py              EpisodeMetrics, run_episode, training-log save/plot
  benchmarking.py         Scenario, RunRecord, registry, run_single
  results_manager.py      dated results dir, manifest, records, summary
  visualization.py        3D rendering, animation, result plots

scripts/
  interactive_demo.py     hand-drive one module per step, then animate
  simple_demo.py          random moves + animation
  audit_pivot.py          numeric + visual audit of a single pivot
  run_baseline.py         run one controller, print metrics, optional animation
  train_gnn.py            train the centralized GNN
  train_decentralized.py  train the decentralized GNN (CTDE)
  replay.py               load a saved trace and animate it

tests/
  test_geometry.py        24 rotations, orientation math
  test_edges.py           12-edge enumeration, rotate_90 helpers
  test_graph.py           connectivity, cut vertices
  test_transitions.py     pivot legality, 90/180, line-end regression
  test_conflict_resolution.py   reject-both semantics
  test_environment.py     determinism, state ownership, local-obs locality
  test_missions.py        objective correctness (power/occlusion)
  test_greedy.py          greedy never worsens; beats random
  test_astar.py           optimal to known target; objective-max
  test_benchmarking.py    shared start state, serialization
  test_gnn_pipeline.py    variable-N, permutation invariance, MP locality
  test_decentralized.py   CTDE structure + LEAK PREVENTION

main.py                   the benchmark entry point
```

---

## The experiments this platform enables

### 1. Core comparison
Centralized GNN vs decentralized GNN vs greedy vs A\* on shared scenarios:
```bash
python main.py --missions power thermal comms aperture --n 4 6 \
    --config-seeds 0 1 2 --repeats 5 \
    --gnn-checkpoint checkpoints/gnn_centralized.pt \
    --decentralized-checkpoint checkpoints/gnn_decentralized.pt
```

### 2. Scaling
Sweep N and watch A\* become infeasible (expansion cap, exploding wall time)
while GNN inference stays cheap:
```bash
python main.py --missions power --n 4 6 8 10 12 16 --config-seeds 0 1 2 \
    --gnn-checkpoint checkpoints/gnn_centralized.pt \
    --decentralized-checkpoint checkpoints/gnn_decentralized.pt
```
Look at `scaling_power_final_objective.png` and per-run `wall_time_s` /
`planning_expansions` in `records.csv`.

### 3. Generalization to unseen N
Train on one size range, evaluate on others — the GNN's shared weights should
transfer:
```bash
python -m scripts.train_gnn --n 6 --missions power --total-steps 400000 \
    --checkpoint-name gnn_n6.pt
python main.py --missions power --n 4 6 8 10 12 --config-seeds 0 1 2 \
    --gnn-checkpoint checkpoints/gnn_n6.pt
```

### 4. Communication radius
Train decentralized policies with different `--actor-layers` (1, 2, 3) and
compare — measures how much locality the decentralized controller needs:
```bash
python -m scripts.train_decentralized --actor-layers 1 --checkpoint-name dec_L1.pt
python -m scripts.train_decentralized --actor-layers 2 --checkpoint-name dec_L2.pt
python -m scripts.train_decentralized --actor-layers 3 --checkpoint-name dec_L3.pt
```

---

## Reproducibility

Every experiment records:
- **Seeds**: environment config seed, per-run eval seeds.
- **Configuration**: full `manifest.json` (missions, N, shapes, repeats,
  controller list, platform/Python version).
- **All runs**: `records.csv`/`records.json` keep every repeat, not just means.
- **Training**: `<checkpoint>_log.csv` and `<checkpoint>_curves.png` beside each
  model.
- **Traces**: the exact move sequence of each best run, replayable bit-for-bit

Deterministic components are tested (`test_environment.py`).

---

## Design principles

- **Physics is abstract; the framework is clean.** The science is *intelligent
  reconfiguration*, not mechanical engineering.
- **Missions are realistic in form** (optimize X subject to hard constraints),
  physically grounded in substance (energy balance, coherent combining,
  interferometric baseline).
- **Centralized vs decentralized is a controlled comparison.** Same architecture;
  the only differences are node-feature scope and MP-layer count. Global state
  cannot leak into decentralized execution (structurally enforced and tested).
- **RL algorithm is held fixed** (PPO/MAPPO). Novelty is the problem and control
  architecture, not a new optimizer.
- **Non-learning baselines are first-class** (greedy, A\*) so RL's value is
  measured, not assumed.
- **Scales from a handful of modules toward large systems** (GNN parameter count
  is independent of N).

---

## Extending the platform

The architecture is built so common extensions require *minimal* change:

- **New mission**: add a `Mission` subclass in `missions.py` + one registry
  entry + one entry in `MISSION_LIST`. Nothing else changes.
- **New controller**: implement `Controller.act(...)`, add one registry entry in
  `build_controller_registry`. It immediately joins the benchmark, plots, and
  traces.
- **New architecture** (e.g. Transformer policy): swap the GNN in the controller;
  the environment, harness, PPO, and metrics are untouched.

### Explicitly out of scope (for now)
Free-flying/detached modules, continuous rigid-body dynamics, orbital mechanics,
magnetic-force modeling, robustness, and continuous action spaces. The abstractions were
chosen so these can be added later without rewriting the core.

---

## Development order (how it was built)

The simulator was built and verified *before* any learning, because an
ambiguous state-transition system would be a larger scientific problem than an
imperfect policy:

1. Geometry (rotations, edges, faces)
2. Reconfiguration (pivot legality, 90/180, connectivity)
3. Mission evaluators
4. Conflict resolution (simultaneous actions)
5. Random controller (sim survives arbitrary actions)
6. Greedy baseline
7. A\* baseline
8. Centralized GNN (PPO)  *(replaced the fixed-N MLP — see note below)*
9. Decentralized GNN (CTDE)
10. Benchmark harness, plots, traces

**Note on the MLP:** an early plan used a flatten-everything MLP as the
centralized baseline. It was dropped because it ties the network to a fixed N,
wastes capacity on padding at large N, and doesn't generalize across sizes. The
centralized controller is instead a GNN (large MP layers), making
centralized/decentralized an architecture-matched comparison.

---

## Testing

```bash
pytest -q                      # full suite
pytest tests/test_transitions.py -q          # movement model
pytest tests/test_decentralized.py -q        # CTDE + leak prevention (critical)
```

The most important tests are in `test_decentralized.py`: they guarantee the
decentralized controller cannot see global state at execution. If those fail,
the centralized-vs-decentralized experiment is invalid.

---

## Caveats

- **Hyperparameters are reasonable defaults, not tuned.** Expect to tune
  learning rate, entropy coefficient, and rollout length per mission for strong
  results. The `ep_obj` in training logs is the signal to watch.
- **CPU is fine for small N; use `--device cuda` for large N or long training.**
- **Changing observation features breaks existing checkpoints** (node dim
  changes). The learned controllers detect this and raise a clear "retrain"
  message; `main.py` then skips that controller gracefully.

---

## License

<add your license here>

## Citation

If this platform supports published work, please cite it as:
```
<add citation once available>
```
