# Shunt

This repository holds the source code of *Shunt: Balancing Compute and KV
Traffic without All-to-All Contention in Disaggregated MoE Serving*, together
with the scripts that run its experiments and turn their logs into the paper's
figures and tables.

In prefill-decode disaggregated serving of a mixture-of-experts (MoE) model
with expert parallelism (EP), the prefill workers of an EP group meet at the
all-to-all (A2A) of every layer. Shunt plans each prefill iteration before it
runs, in three parts:

- **RS** (request scheduling): the proxy places requests on the prefill DP
  workers by their estimated compute, with the LPT rule.
- **EAP** (elastic attention parallelism): a compute straggler lends query
  heads to other GPUs of its node, which compute them and send the results
  back over NVLink.
- **KVLB** (KV traffic load balancing): the A2A has strict priority over KV.
  Each worker's KV on its backend port is limited to what the port can move
  within the iteration's A2A-free compute window. The rest goes to other
  backend ports of the node or to the frontend network.

The repository contains no measured results. What a run measures depends on
the GPUs, the network, drivers, firmware, and software versions. The scripts
only collect the logs of your runs and plot what they contain.

## Contents

```
shunt/          Shunt: planner, engine runtime, proxy, harness, benchmarks, analysis
  shunt/          the Python package
  csrc/           C++ planner cores and the decision-cost benchmark
  configs/        cluster file examples
  experiments/    experiment files, one per group of figures and tables
  tests/          unit tests
vllm/           vLLM (upstream commit 1c607d7b2) with Shunt's hooks
lmcache/        LMCache (upstream commit 140990dc) with the KVLB routing backend
mooncake/       Mooncake (upstream commit b0bda8c) with per-connection traffic classes
```

The first commit of this repository imports the three upstream snapshots
unchanged, and later commits apply Shunt's changes. This command therefore
shows every change to the frameworks:

```bash
git diff $(git rev-list --max-parents=0 HEAD) HEAD -- vllm lmcache mooncake
```

### Changes to the frameworks

File paths are relative to each framework's directory.

vLLM (`vllm/`):

| File | Change |
|---|---|
| `vllm/shunt_integration.py` | Entry points into `shunt.runtime`. Inactive unless `SHUNT_ROLE=prefill`. |
| `vllm/v1/worker/dp_utils.py` | Each DP rank adds its planner inputs to the small all-reduce that vLLM runs across DP ranks before every step, so every rank derives the same group plan. |
| `vllm/v1/worker/gpu_model_runner.py` | Collects the step's requests and planner inputs before that all-reduce. |
| `vllm/v1/worker/gpu_worker.py` | Starts the runtime once the parallel groups exist. Keeps the worker's CUDA device after the KV connector is set up. |
| `vllm/model_executor/models/qwen3_moe.py` | Runs the attention block through elastic attention. |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`, `vllm/model_executor/layers/attention/kv_transfer_utils.py` | Per-layer phase timing: gate, dispatch, experts, combine, and waits for inbound KV. |
| `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` | Reports the KV tokens each request loads to the planner. |

LMCache (`lmcache/`):

| File | Change |
|---|---|
| `lmcache/v1/storage_backend/routing_backend.py` | KVLB routing: one Mooncake client per backend device of the node and one on the frontend device, each with its own traffic class. Every KV chunk goes through the client of the port that the engine's plan picks. |
| `lmcache/v1/storage_backend/shunt_pool.py` | The private transfer pool of each client: GPU memory for backend devices (GPU-direct RDMA), pinned host memory for the frontend. |
| `lmcache/v1/storage_backend/__init__.py`, `lmcache/v1/storage_backend/connector/mooncakestore_connector.py` | Create the routing backend (`shunt_kvlb_enabled`). Let a Mooncake client use a private pool. |
| `lmcache/v1/cache_engine.py` | Layer-wise stores allocate each chunk in the pool of the port it will leave through. |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | Reports KV tokens per request. Adds the experiment-only no-contention mode (`shunt_assume_hit`). |
| `lmcache/v1/gpu_connector/__init__.py` | Uses the worker's current CUDA device (DP ranks are not TP ranks). |

Mooncake (`mooncake/`):

| File | Change |
|---|---|
| `mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp` (and `.h`) | Each transfer engine reads `MC_IB_TC` when it is created, so clients in one process can use different traffic classes. |
| `mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp` (and `.h`), `mooncake-transfer-engine/src/transfer_metadata.cpp`, `mooncake-transfer-engine/include/transfer_metadata.h` | The handshake carries the traffic class of the side that opens a connection, and the other side uses it too. RDMA READ responses, which carry inbound prefix-KV, then travel in the same class as the request. |

## Requirements

The experiment files follow the paper's deployment. Two prefill nodes run one
prefill instance (DP=16, EP=16, TP=1). Two decode nodes each run a decode
instance (DP=8, EP=8, TP=1). Every node has eight GPUs connected by NVLink.
Each GPU uses one port of the backend RDMA network (RoCE), and each node has a
host-attached frontend RDMA NIC. The model is Qwen3-235B-A22B. Another
deployment only needs a different cluster file (see below). Elastic attention
supports Qwen3-MoE models with BF16 or FP16 weights and KV cache, TP=1, and the
FlashAttention backend.

Software: Linux, an NVIDIA driver and CUDA toolkit that match the PyTorch
build, Python 3.10 to 3.13, a C++17 compiler, and the RDMA user libraries
(rdma-core). Building Mooncake needs root for its system packages.

## Installation

Run these steps on every node, with the repository at the same path on all of
them. Everything below runs from the repository root. We use `uv`; plain `pip`
works the same way. Mooncake's build needs two git submodules (pybind11 and
yalantinglibs under `mooncake/extern/`), so clone with `--recursive` or run
`git submodule update --init` first.

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate

# 1. PyTorch, as pinned in vllm/requirements/cuda.txt. Pick the CUDA build that
#    your driver supports (for example cu130 or cu129).
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 2. vLLM. Shunt changes only Python files of vLLM, so the prebuilt kernels of
#    the upstream commit can be used. setuptools_scm cannot read a version
#    outside a vLLM checkout, so the version is given explicitly.
cd vllm
SETUPTOOLS_SCM_PRETEND_VERSION=0.20.1rc1.dev156+g1c607d7b2 \
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=1c607d7b2cd4fb572b919c6053f19d0577203495 \
    uv pip install -e .
cd ..
#    To compile vLLM instead: python use_existing_torch.py;
#    uv pip install -r requirements/build.txt; then
#    SETUPTOOLS_SCM_PRETEND_VERSION=... uv pip install -e . --no-build-isolation

# 3. LMCache (compiles its CUDA kernels against the installed PyTorch).
uv pip install -r lmcache/requirements/build.txt
SETUPTOOLS_SCM_PRETEND_VERSION=0.4.0 uv pip install -e lmcache --no-build-isolation

# 4. Mooncake, from mooncake/ (see mooncake/docs/source/getting_started/build.md).
#    -DWITH_NVIDIA_PEERMEM=OFF registers GPU memory through dma-buf; keep the
#    default if your nodes load nvidia-peermem.
cd mooncake
sudo bash dependencies.sh  # system packages and yalantinglibs
mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON -DWITH_NVIDIA_PEERMEM=OFF
make -j
cd ..
bash scripts/build_wheel.sh              # the Python package, with mooncake_master
uv pip install mooncake-wheel/dist/*.whl
cd ..

# 5. Shunt.
uv pip install -e "shunt[test]"
make -C shunt/csrc         # libshunt.so (planner cores) and scalability_bench
python -m pytest shunt/tests
```

The DeepEP and DBO systems (`+deepep`, `+dbo`) also need DeepEP on the prefill
nodes: see `vllm/tools/ep_kernels/README.md`. The stock Mooncake wheel
(`mooncake-transfer-engine`, or `mooncake-transfer-engine-cuda13` for CUDA 13)
also runs everything, but without the traffic-class changes: all Mooncake
clients of a process share one class, and inbound KV (RDMA READ responses)
travels in the decode side's class.

## Data and model

Download the model to the same path on every node, for example:

```bash
hf download Qwen/Qwen3-235B-A22B --local-dir /models/Qwen3-235B-A22B
```

The experiments replay the public Qwen serving traces from
<https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon>. Put the files
in `traces/` at the repository root on every node:

```
traces/qwen_traceB_blksz_16.jsonl      business subset (all main experiments)
traces/qwen_traceA_blksz_16.jsonl      consumer subset
traces/qwen_thinking_blksz_16.jsonl    reasoning subset
traces/qwen_coder_blksz_16.jsonl       coding subset
```

The trace driver (`shunt.harness.replay`) builds token-id prompts from each
record's block hashes. Records that share leading hashes therefore share
leading tokens, and the trace's prefix reuse becomes real reuse in the
engines. Prompts longer than 16,000 tokens are cut to their first 16,000
tokens (`max_input` in an experiment file, `--max-input` for a single run),
and outputs are capped so that prompt and output fit the engines'
`--max-model-len`.

## Network setup

KV priority relies on the RDMA traffic class (the DSCP is the class shifted
right by two). NCCL marks the A2A with `NCCL_IB_TC`, and Mooncake marks KV with
`MC_IB_TC`. The harness sets them from the cluster file:

- `a2a_tc`: the A2A;
- `kv_borrow_tc`: KV that a worker sends through another GPU's port;
- `kv_tc`: KV on a worker's own port.

The fabric must schedule these classes with strict priority, in that order,
on the NICs and on the switches. On NVIDIA ConnectX NICs, for example,
`mlnx_qos -i <netdev> --trust dscp` classifies packets by DSCP, and the
`--dscp2prio`, `--prio_tc`, and `--tsa` options map the three DSCP values to
strict-priority queues. Systems without KV priority put KV in the A2A's class.

## Compute profile

The planner estimates per-request compute with two linear functions whose
coefficients are profiled on the target GPU. Profile once per GPU type and
model (one GPU; only the model's `config.json` is read, and weights are
random):

```bash
python -m shunt.profiling.compute --model /models/Qwen3-235B-A22B --ep 16 \
    --out profiles/compute.json
```

Set `shunt.compute_profile` in the cluster file to this file. Without a
profile, the planner falls back to nominal FLOP-based coefficients, which are
only meant for testing.

## Cluster file

Copy `shunt/configs/cluster.example.yaml` to `cluster.yaml` and fill in the
hosts, addresses, RDMA devices, traffic classes, and paths. Paths in the file
refer to the nodes, and `workdir` is the repository root there.

The harness runs from the repository root of any machine that reaches all
nodes with passwordless `ssh` (`ssh:` sets the command). That machine needs
the `shunt` package and, for the ORS runs, the trace files, since it computes
the ORS placement itself. On a node, every command runs in a non-login `bash`
after sourcing `activate`, and optionally inside a container (`exec_prefix`,
for example a `docker exec` into the serving container).

To see what a run would start without starting it:

```bash
python -m shunt.harness.run --cluster cluster.yaml --system shunt \
    --load closed:512 --trace traces/qwen_traceB_blksz_16.jsonl \
    --out results/try/shunt --dry-run
```

This prints every configuration file and every role's environment and command
(Mooncake master, decode instances, proxy, prefill instance, trace driver), in
start order. The same commands can be run by hand. All processes need the same
`PYTHONHASHSEED`, because LMCache hashes chunks with Python's `hash`. The
prefill instance needs `--enforce-eager` for elastic attention.

### Trying it on one machine

`shunt/configs/cluster.single-node.yaml` runs a prefill instance on two GPUs
(DP=2, EP=2) and a decode instance on a third, with KV over TCP. It uses a
small Qwen3-MoE checkpoint with random weights:

```bash
python -m shunt.tools.small_model --like Qwen/Qwen3-30B-A3B --out /models/small-qwen3-moe
cp shunt/configs/cluster.single-node.yaml cluster.yaml   # then set the paths in it
python -m shunt.harness.run --cluster cluster.yaml --system shunt --load closed:8 \
    --trace traces/qwen_traceB_blksz_16.jsonl --requests 64 --max-output 4 \
    --drain-s 5 --out results/try/shunt
python -m shunt.analysis.ttft table Shunt=results/try/shunt
```

This exercises placement, the per-iteration plan, elastic attention, KV
routing, logging, and the analysis scripts. It does not exercise RDMA traffic
classes, GPU-direct transfers, DeepEP, or multiple nodes.

## Running the experiments

A run starts one system under one load, replays the trace, and collects every
log. Before and after a run, the harness runs `shunt/shunt/harness/clean.sh` on
every host. It stops all of your vLLM servers, Mooncake masters, proxies,
trace drivers, and RNIC samplers on that host, so do not share the hosts with
other serving jobs of the same user. The harness stops a run when a log shows
an error or stays silent for 180 s.

```bash
python -m shunt.harness.run --list-systems

# one run
python -m shunt.harness.run --cluster cluster.yaml --system shunt \
    --load closed:512 --trace traces/qwen_traceB_blksz_16.jsonl --requests 12000 \
    --out results/closed/shunt

# all runs of an experiment file (finished runs are skipped, so a file can be resumed)
python -m shunt.harness.run --cluster cluster.yaml --plan shunt/experiments/closed.yaml
python -m shunt.harness.run --cluster cluster.yaml --plan shunt/experiments/closed.yaml \
    --only shunt,baseline
```

Loads are `closed:<requests in flight>` or `open:<requests per second>`
(Poisson arrivals, run for `duration` seconds). System names take modifiers:
`+deepep` (DeepEP high-throughput A2A), `+dbo` (DeepEP with dual-batch
overlap), `+nocache` (no prefix reuse), `+tpN`, `+chunkN` (chunked prefill),
and `+thetaX` (the compute-straggler threshold of elastic attention).
`no-contention` is an experiment-only mode for measuring contention: the
prefill instance keeps KV in its own CPU memory (a warm-up pass fills it), and
the decode instances treat every prompt as cached without fetching it, so
their output text is meaningless.

The experiment files in `shunt/experiments/` hold the runs behind the figures
and tables. They use the paper's loads; adjust the concurrency, the rates, and
the request counts to your cluster's capacity.

| File | Runs |
|---|---|
| `motivation.yaml` | Baseline and ORS with per-layer timing, RNIC samples, the no-contention run, chunked prefill, TP=2/4/8, and the other three trace subsets |
| `closed.yaml` | Closed loop: all systems, the ablations, KV-cache-aware dispatch, and the threshold sweep |
| `sweep.yaml` | Open loop: request-rate sweeps of the main systems, the ablations, and the A2A stacks |
| `a2a.yaml` | DeepEP and DBO: Baseline, Shunt, the frontend-only arm, and the no-contention run, with and without per-layer timelines |
| `accuracy.yaml` | Shunt with per-layer timing (NCCL and DBO), for prediction accuracy and KV overflow |
| `workloads.yaml` | The coding subset (open loop) and runs without prefix reuse |

A run directory holds:

- `requests.jsonl`: one line per request from the trace driver (send, first-token, and last-token times);
- `proxy.jsonl`: per-request placement and timestamps from the proxy;
- `shunt/<host>/`: per-rank logs of the prefill engines: `plan` (the group plan), `reqs` (the requests of each rank per iteration), `step` (per-layer phase times, with timing on), `route` and `kvio` (KV bytes per port);
- `bw/<host>.jsonl`: RNIC and interface samples, for runs with `sample_bw`;
- `logs/`: the output of every role; `meta.json`: the run's settings.

Per-layer timing (`timing: 1` in an experiment file, `--timing 1` for a single
run) adds CUDA events around every phase. `timing: 2` also keeps each phase's
start and end offsets. The runs that feed TTFT figures turn timing off.

### Microbenchmarks

These run on one node with eight GPUs. They read only the model's
`config.json` and use random weights.

```bash
# iterations of a measured Baseline run, picked by worker-compute imbalance
python -m shunt.bench.pick_iterations --run results/motivation/baseline-timing \
    --out bench/iterations.json

# elastic attention: attention compute and exposed transfers (Fig. S3)
torchrun --nproc-per-node 8 -m shunt.bench.elastic_overhead \
    --model /models/Qwen3-235B-A22B --iterations bench/iterations.json \
    --profile profiles/compute.json --out bench/eap_overhead.json

# striped ring attention on the same iterations (Table S6); --check compares
# its output with single-GPU attention
torchrun --nproc-per-node 8 -m shunt.bench.ring_attention \
    --model /models/Qwen3-235B-A22B --iterations bench/iterations.json \
    --profile profiles/compute.json --out bench/ring.json

# decision cost of LPT, EAP, and KVLB versus the number of DP workers (Fig. S4);
# --cpu pins the benchmark to one core
python -m shunt.bench.scalability --cpu 11 --out bench/scalability.csv
```

## Figures and tables

Each analysis module has a command line (`python -m shunt.analysis.<module>
--help`). Figures are written as PDF files. Tables are printed and written as
CSV, Markdown, and LaTeX files. One command produces every figure and table
whose inputs exist and lists the ones that are missing:

```bash
python -m shunt.analysis.all --results results --bench bench --traces traces \
    --prefill-host p0 --decode-host d0 \
    --backend-devices mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
    --frontend-device eth0 --out out
```

`--prefill-host` and `--decode-host` name the hosts whose RNIC samples feed
the bandwidth figures, and `--warmup-s` drops the first seconds of every run
from the TTFT statistics. Outputs go to `out/figures` and `out/tables`.

| Figure or table | Experiment: runs | Module |
|---|---|---|
| Fig. 5 (worker-compute imbalance) | motivation: `baseline-timing` | `imbalance plot --kind compute` |
| Fig. 6 (KV imbalance) | motivation: `baseline-timing` | `imbalance plot --kind kv` |
| Fig. 7 (oracle placement) | motivation: `ors-timing` | `imbalance plot --kind all` |
| Fig. 8 (backend RNIC utilization) | motivation: `baseline-timing` (RNIC samples) | `bandwidth` |
| Fig. 9 (contention) | motivation: `baseline`, `no-contention` | `ttft box` |
| Sec. 2.3 (prompt-token split) | motivation: `baseline-timing` | `trace_stats split` |
| Sec. 3.2 (chunked prefill) | motivation: `baseline-timing`, `baseline-chunked` | `imbalance table` |
| Fig. 13a (TTFT) | closed: `baseline`, `ors`, `combo`, `shunt`, `sh-ors` | `ttft box` |
| Fig. 13b (TTFT versus load) | sweep: the same systems, `baseline+dbo`, `shunt+dbo` | `sweep` |
| Table 1 (A2A stacks) | closed: `baseline`, `shunt`; a2a: `baseline+deepep`, `shunt+deepep`, `baseline+dbo`, `shunt+dbo` | `ttft table` |
| Sec. 4.2 (contention under DBO) | a2a: `baseline+dbo`, `no-contention+dbo` | `ttft table` |
| Sec. 4.2 (idle time and A2A occupancy) | motivation: `baseline-timing`; a2a: `baseline+dbo-timing`, `shunt+dbo-timing` | `barrier` |
| Table 2a (coding subset) | workloads: `coder-*` | `sweep` |
| Table 2b (no prefix reuse) | workloads: `nocache-*` | `ttft table` |
| Fig. 14 (ablation) | closed and sweep: `shunt`, `no-rs`, `no-eap`, `no-kvlb` | `ttft box`, `sweep` |
| Fig. 15 (inside KVLB) | closed: `shunt`, `no-prio`, `no-budget`, `no-borrow`, `no-frontend`, `no-kvlb`; sweep: `shunt`, `no-budget`, `no-kvlb` | `ttft box`, `sweep` |
| Table 3 (prediction accuracy) | accuracy: `shunt-timing`, `shunt+dbo-timing` | `accuracy` |
| Sec. 4.3 and S5 (KV overflow) | accuracy: `shunt-timing`, `shunt+dbo-timing` | `overflow` |
| Table S1 (trace subsets) | the four trace files; motivation: `baseline-timing`, `baseline-traceA`, `baseline-thinking`, `baseline-coder` | `trace_stats stats`, `imbalance table` |
| Fig. S1 (input and output lengths) | the business trace file | `trace_stats lengths` |
| Fig. S2 (frontend utilization) | motivation: `baseline-timing` (RNIC samples) | `bandwidth` |
| Table S2 (sweep on the A2A stacks) | sweep: `baseline`, `shunt`, and their `+deepep` and `+dbo` variants | `sweep` |
| Table S3 (sweep of KVLB arms and Combo) | sweep: `shunt`, `no-budget`, `no-kvlb`, `combo`, `shunt-kva` | `sweep` |
| Table S4 (KVLB arms, percentiles) | closed: the KVLB arms and `combo`; a2a: `shunt+dbo`, `no-frontend+dbo` | `ttft table` |
| Table S5 (dispatch policies) | closed: `baseline`, `lpt`, `kva`, `kva-lb`, `shunt`, `shunt-kva`, `shunt-kva-lb` | `dispatch` |
| Fig. S3 (elastic-attention breakdown) | `bench/eap_overhead.json` | `bench overhead` |
| Table S6 (ring attention) | `bench/ring.json`, `bench/eap_overhead.json` | `bench ring` |
| Fig. S4 (decision cost) | `bench/scalability.csv` | `bench scalability` |
| Fig. S5 (straggler threshold) | closed: `shunt-theta*`, `shunt` | `ttft box` |
| Table S7 (TP) | motivation: `baseline-timing`, `baseline-tp2`, `baseline-tp4`, `baseline-tp8` | `imbalance table` |

Figs. 1 to 4 and 10 to 12 are diagrams.

## License

Shunt's code is released under the Apache License 2.0 (`LICENSE`). The
directories `vllm/`, `lmcache/`, and `mooncake/` keep the licenses of their
projects, which are included there. The traces are distributed by their
authors under their own terms.
