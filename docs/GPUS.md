# GPUs: what can mine Vibe Hash, and how well

November 2026, Vibe Hash 0.5.26, PyTorch 2.15.0 (released 2026-10-28). Only an RTX 4070 SUPER and its CPU were
on hand, so every other card here is from documentation, PyTorch's own source, and issue reports (sources at the
end), plus measurements on the 4070 SUPER that stand in for other kinds of hardware where that's possible.

## In plain words

- **NVIDIA RTX 20-series and newer (Turing, Ampere, Ada, Hopper, Blackwell)** mine everything: classic jobs, Evo
  Hash and fine-tunes. This is the tested path. Blackwell (RTX 50, B200) needs a recent driver (570 or newer), and
  `vibehash install-gpu` already picks a build that has it.
- **Older NVIDIA (GTX 10-series, Titan V, V100)** works but is stuck: PyTorch 2.14 is the last version for those
  cards, and NVIDIA's 580 driver series is their last. Evo runs slowly there, because those cards have no bf16 hardware.
- **AMD on Linux** runs the same code as NVIDIA, through PyTorch's ROCm build, which calls the cards "cuda". That
  covers Radeon RX 7000 and 9000, Instinct MI100 to MI355, and Ryzen AI Max. The RX 6000 series works unofficially.
  Nobody has measured how closely AMD agrees with NVIDIA checkers, and AMD's fastest matrix library isn't covered by
  PyTorch's "deterministic" switch. **AMD on Windows** mines on the CPU for now: PyTorch publishes ROCm builds only
  for Linux. AMD now publishes its own Windows builds, but `install-gpu` doesn't use them yet.
- **Intel Arc and Core Ultra graphics** mine classic jobs and fine-tunes, and run Evo in float32. That's slower than
  NVIDIA's bf16, but it agrees with NVIDIA checkers within the margin (measured below, with float32 on CUDA standing in).
- **Apple M-series Macs** work the same way as Intel, in float32. PyTorch 2.15 needs macOS 15 to use the GPU.
- **CPU only** is the reference for classic jobs, and is slow but correct for the rest.
- **The real risk is not "does it run" but "do two honest machines agree".** Classic jobs and fine-tunes compare
  results with tolerances that are hundreds of times wider than the differences between machines. Evo compares
  token choices with a margin of 0.5 logits. The worst honest gaps measured so far are 0.375, between an RTX 4090 and
  a 4070 SUPER, and between a float32 worker and a bf16 checker. That leaves little room, and Turing, AMD, Intel and
  Apple haven't been measured against each other at all.

## What each kind of work asks of the GPU

| | Classic Vibe Hash (MLP) | Evo Hash (rule 2, Qwen3.5) | Fine-tunes (LoRA) |
|---|---|---|---|
| Code | `compute.TorchBackend` | `es_client.runtime2`, `es_hybrid`, `es_fast` | `finetune`, `llm`, `program` |
| Arithmetic | float32 jobs on the GPU; float64 jobs always on the CPU (`compute.py:101`) | **bf16 on CUDA/ROCm, float32 everywhere else** (`es_client.py:387`, `:413`) | float32, frozen weights stored bf16 and cast up in each matmul |
| CUDA graphs | no | yes, one per pass on CUDA/ROCm; eager elsewhere (`es_hybrid.py:149`, `:256`) | no |
| `scaled_dot_product_attention` | no | yes, with a float mask (`es_hybrid.py:210`, `es_fast.py:266`) | no (attention written out) |
| TF32 | must be off | must be off for the float32 recurrent state | must be off (rule 2 rejects TF32: DESIGN.md §8) |
| float64 on the device | never | never in rule 2; rule-1 Evo summed weights in float64 on the device (fixed below) | never (rotary tables on the CPU) |
| Memory | small | Qwen3.5-0.8B: 3.5 GiB peak (bf16, 64-token answers, measured); twice the weights in float32 | 3.5 GiB Qwen3-0.6B, 11.1 GiB Qwen3-4B (measured) |
| How a checker compares | within `tolerance` (1e-4) and 1% of the update | each committed token within `margin` (0.5) of the greedy logit; replay log-probabilities within `lp_tol` (0.05) | rule 2: B, m, v within their tolerances, Adam's first-step sign flips set aside |
| Measured across machines | 4e-9 to 8e-8, CUDA vs CPU | gaps up to 0.375 (below) | 1e-6 of B's size, CUDA vs CPU |

## Compatibility matrix

"Build" is what `vibehash install-gpu` (`gpusetup.py`) installs, with this branch's fixes. "Works" means PyTorch
supports the code path, not that it has been run there. Only the Ada row is tested.

| Family | Cards | Build and driver | Classic | Evo | Fine-tunes | Notes |
|---|---|---|---|---|---|---|
| NVIDIA Pascal, Volta (sm_60, 61, 70) | GTX 10xx, P100, P40, Titan V, V100 | **cu126 only, so PyTorch 2.14 at most**; driver 560.76+ (Windows), 560.28+ (Linux); **R580 is their last driver series** | yes | works, slowly: bf16 emulated, attention on the math kernel; not measured | yes | P100 bf16 batched matmul failed with CUBLAS_STATUS_NOT_SUPPORTED on CUDA 11.3 (open issue, not re-tested) |
| NVIDIA Turing (sm_75) | RTX 20xx, GTX 16xx, T4 | cu130/132/134 with driver 580+, else cu126 | yes | works: bf16 emulated, attention on the math kernel; **margin measurement in progress** (RTX 2060) | yes | no bf16 hardware; GTX 16xx has fp16 NaN history (not used here) |
| NVIDIA Ampere (sm_80, 86, 87) | A100, RTX 30xx, RTX A-series | newest the driver allows | yes | yes: native bf16, memory-efficient attention | yes | TF32 hardware, kept off |
| NVIDIA Ada (sm_89) | RTX 40xx, L4, L40S | newest the driver allows | **tested** | **tested** (4070 SUPER; 4090 answers checked) | **tested** | runs sm_86 code |
| NVIDIA Hopper (sm_90) | H100, H200, GH200 | newest the driver allows | yes | yes | yes | cuDNN attention is preferred here but turned off by deterministic mode, which workers set |
| NVIDIA Blackwell (sm_100, 103, 120, 121) | B200, GB200, B300, RTX 50xx, RTX PRO 6000, DGX Spark | **cu128 or later**, driver 570.65+ (Windows), 570.26+ (Linux) | yes | yes, not measured | yes | sm_103 runs sm_100 code and sm_121 runs sm_120 code; cuBLAS's BF16x9 float32 emulation is opt-in only |
| AMD RDNA3, RDNA4 (Linux) | RX 7600 to 7900 XTX (gfx1100-1102), RX 9060/9070 (gfx1200/1201), AI PRO R9700 | the newest ROCm build ≤ the installed ROCm (rocm10.0/7.14 for PyTorch 2.15; **were missing before this branch**); amdgpu driver | yes | bf16 + HIP graphs, the CUDA path; not measured | yes | attention: AOTriton on gfx1100/1201, the math kernel on gfx1101/1102/1200 (experimental, off by default) |
| AMD RDNA2 (Linux) | RX 6800/6900 (gfx1030); RX 6700/6600 (gfx1031/1032) | as above; **unofficial**; 6700/6600 need `HSA_OVERRIDE_GFX_VERSION=10.3.0` | yes | bf16 without matrix cores (rocBLAS), math attention; slow | yes | officially only the PRO W6800 and V620 |
| AMD CDNA (Linux) | MI100 (gfx908), MI210/250 (gfx90a), MI300X/325X (gfx942), MI350/355 (gfx950) | as above | yes | yes; not measured | yes | MI50/Radeon VII (gfx906) left PyTorch's wheels in 2.15; TF32 hardware on MI300+, kept off |
| AMD Ryzen AI APUs (Linux) | Ryzen AI Max / Strix Halo (gfx1151), Ryzen AI 300 (gfx1150), 780M (gfx1103) | as above; recent kernels for Strix Halo | yes | yes; up to ~50% of RAM as GPU memory by default | yes | gfx1103 compiled in from PyTorch 2.13 |
| AMD on Windows | Radeon RX 6000-9000, Ryzen AI | **none from PyTorch** (its ROCm builds are Linux-only); AMD's own index has Windows PyTorch, not installed by `install-gpu` | CPU | CPU | CPU | WSL2 with ROCm also works (ROCDXG); DirectML is unmaintained (PyTorch 2.4.1, no bf16) |
| Intel Arc A (Alchemist) | A380, A580, A750, A770 | `xpu`; Windows driver 32.0.101.9033+ for 2.15 | float32 yes; float64 on CPU | float32, eager (no graphs): works, slow, not run | yes | **no float64 hardware**; 4 GB single-allocation limit |
| Intel Arc B, Core Ultra | B570, B580, Arc Pro B50/B60; Meteor Lake, Lunar Lake, Arrow Lake H, Panther Lake | `xpu` | yes | float32, eager | yes | has float64 (B580 reports `has_fp64`) |
| Intel Data Center GPU Max | Max 1100, 1550 | `xpu`, Linux only (**was refused before this branch**) | yes | float32, eager | yes | 2 tiles = 2 devices by default (`ZE_FLAT_DEVICE_HIERARCHY`) |
| Apple silicon | M1 to M5 | PyPI build; **macOS 15 for PyTorch 2.15** (14 up to 2.14) | float32 yes; float64 on CPU | float32, eager: works, slow, not run | yes, bf16 weights need macOS 14+ | **no float64 at all**; unified memory |
| CPU | any x86-64 or ARM64 | NumPy; PyPI PyTorch for Evo and fine-tunes | **reference** | float32: correct, too slow for a 5-minute round (estimate) | correct, slow (36 s per 4-step replay of Qwen3-0.6B) | |

## By family

### NVIDIA

**Builds.** PyTorch 2.15 publishes only cu132 and cu134, for Turing (sm_75) and newer with driver 580+. Its
release notes say "Maxwell, Pascal and Volta GPUs and hosts with a CUDA 12.x driver should stay on 2.14". cu126
(Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper; no Blackwell) ends at 2.14.1. The cu128 and cu129 builds lost
Volta in 2.11 (cuDNN 9.15 doesn't support it) and Maxwell/Pascal in 2.8. Blackwell is compiled into cu128, cu129 and
every CUDA 13 build. `install()` takes an index's newest PyTorch, so `install-gpu` before this branch would put a
Linux V100 on cu129, whose newest PyTorch (2.13) has no Volta code: the self-test fails and the worker falls back to
the CPU.
**Fixed:** cu128/cu129 are Turing-and-up in `gpusetup.CUDA_BUILDS`, and cu134 is added.

**bf16 on Pascal, Volta and Turing.** There's no bf16 hardware, but eager PyTorch emulates it. It's slower, and
cuBLAS's `cublasGemmEx` accepts bf16 from compute 5.0. `torch.cuda.is_bf16_supported()` says True there unless asked
`including_emulation=False`. Attention: flash and cuDNN need sm_80+, and any mask rules flash out. The
memory-efficient kernel takes bf16 only from sm_80. So on these cards Evo's bf16 attention runs on the math kernel,
a different rounding from Ampere-and-up's memory-efficient kernel. That is exactly what the RTX 2060 margin
measurement (docs/evo-margin-measurement.md) tests.

**Agreement between NVIDIA cards.** cuBLAS promises bit-identical results "on GPUs with the same architecture and
the same number of SMs" and the same toolkit version, and nothing more. So:
- two RTX 4070 SUPERs on the same PyTorch build agree exactly (measured: every gap 0);
- an RTX 4070 and a 4070 SUPER (46 vs 56 SMs), a laptop and a desktop card of the same name, or an H100 PCIe and
  SXM can round differently;
- the same card on cu126 and cu130 can round differently.

Measured on the 4070 SUPER, one Evo matmul shape (Qwen3.5-0.8B's MLP down-projection, 64×3584 by 3584×1024) comes out
different with `allow_bf16_reduced_precision_reduction` off. That means cuBLAS splits that sum and rounds the pieces
to bf16. cuBLAS picks how to split by shape and card, so another card can split differently. Turning only that flag off moved honest Evo gaps by up to 0.25, close
to the 0.375 between a 4090 and a 4070 SUPER. DeepSeek's code already turns it off (`deepseek.py:88`); Evo's doesn't
(risk 2).

**TF32.** Matmul TF32 is off by default, cuDNN's is on, and the workers turn both off. From PyTorch 2.9 there is a
second set of switches (`torch.backends.cuda.matmul.fp32_precision`). Measured on 2.14.1: if anything sets the new
one to "tf32" after our `allow_tf32 = False`, float32 matmuls silently run in TF32 (errors 1,000 times larger), and
reading `allow_tf32` raises a RuntimeError. **Fixed:** `compute.pin_precision()` writes the settings again before
every classic segment, fine-tune segment, and Evo answer or check.

**Consumer-card problems.** RTX 50-series on a cu126 build fails with "no kernel image is available"
(`install-gpu` avoids it). PyTorch 2.14 turned off two cuDNN convolution engines on sm_120 for illegal memory
accesses. Evo's shared prompt reading does run Qwen3.5's small 1-D convolutions through transformers; whether
those use the disabled engines wasn't checked. Windows kills a GPU kernel that runs over 2 seconds (TDR).
Vibe Hash's kernels are milliseconds, but a display card that's also gaming can stall a round.

### AMD

**Numbering.** After 7.2, ROCm moved to AMD's new build system ("TheRock") and went 7.14, 10.0, 10.1 (10.1.0 on
2026-10-05). PyTorch 2.15 is built for rocm7.14 and rocm10.0, and 2.14 for rocm7.2 and rocm7.14. Every ROCm wheel on
download.pytorch.org is Linux x86-64. `install-gpu` knew builds up to rocm7.2 only. **Fixed:** rocm7.14 and rocm10.0
are in `gpusetup.ROCM_BUILDS` (7.14 sorts after 7.2, as numbers). One thing is unverified: whether ROCm 10.x's new
install layout (`/opt/rocm/core-10.1`) still has the `/opt/rocm/.info/version` file that `gpusetup._rocm_version`
reads. If not, `install-gpu` would say ROCm isn't installed (risk 6).

**Which cards.** ROCm 10.1 officially supports:
- RDNA4 (RX 9060/9070, AI PRO R9700/R9600) and RDNA3 (RX 7600 to 7900 XTX);
- of RDNA2, only the PRO W6800 and V620;
- CDNA1 to CDNA4 (MI100, MI210/250, MI300X/325X, MI350/355);
- Ryzen AI APUs: Max (Strix Halo), 300/400, and the 780M-class Ryzen 200.

PyTorch 2.15's wheels are compiled for gfx908, 90a, 942, 950, 1030, 1100-1103, 1150, 1151, 1200, 1201 and 1250. The
RX 6700 XT and 6600 (gfx1031/1032) aren't among them, so they need `HSA_OVERRIDE_GFX_VERSION=10.3.0` to run gfx1030's
code. PyTorch's maintainers don't support that setup. AMD's own index does have native builds for those cards. MI50
and Radeon VII (gfx906) were dropped from the wheels in 2.15.

**Windows.** PyTorch publishes nothing for AMD on Windows. AMD publishes it on its own index (repo.amd.com):
- ROCm 10.1 with PyTorch 2.14, Python 3.11 to 3.14, Windows 11 25H2, Adrenalin 26.10.41.05;
- for Radeon and Ryzen, with a device extra per card (`torch[device-gfx1100]`).

AMD's notes for the earlier 7.2.1 Windows preview say "No ML training support". Whether that still holds on 10.x
wasn't found. Before this branch, a Windows AMD machine with the HIP SDK installed got builds proposed from
PyTorch's Linux-only index, and then the message "PyTorch has no build for it on Python 3.x". **Fixed:**
`builds_for` says what's actually true. WSL2 has production ROCm support (ROCDXG, Adrenalin 26.2.2 or newer).
torch-directml stopped at PyTorch 2.4.1 and has no bf16.

**The code paths.** PyTorch's ROCm build is the CUDA path: `device="cuda"`, bf16, and `torch.cuda.CUDAGraph`
(HIP graphs), so Evo runs exactly as on NVIDIA. `torch.cuda.is_bf16_supported()` is always True on ROCm. RDNA2 has
no bf16 matrix cores and runs bf16 through rocBLAS, slowly. Attention goes through AOTriton:
- on in PyTorch 2.15 for gfx90a, 942, 950, 1100, 1151 and 1201;
- "experimental" (off unless `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`) on gfx1101-1103, 1150 and 1200;
- absent for gfx1030 and gfx908.

So two AMD cards can run different attention kernels, just as Turing and Ampere do. TF32: since PyTorch 2.10
`allow_tf32` reaches hipBLASLt (MI300 and newer); workers turn it off.

**Determinism.** `use_deterministic_algorithms(True)` turns off atomics on rocBLAS handles only. hipBLASLt is
PyTorch's default matrix library on:
- gfx90a and gfx942;
- gfx1200/1201 from ROCm 6.4, and gfx950 from 7.0;
- gfx1100, 1101 and 1151 from PyTorch 2.14.

The deterministic switch doesn't cover hipBLASLt. AMD's own table marks rocBLAS as graph-safe and hipBLASLt as only
partly so. An issue opened 2026-11-04 reports that graph replay can share hipBLASLt's split-K state with other work,
"causing hangs or wrong results", and Evo replays graphs of bf16 matmuls. AMD itself recommends
`TORCH_BLAS_PREFER_HIPBLASLT=0` for GPU resets during training on the RX 9070 and R9700 (risk 3).

**Problems.**
- Radeon cards on ROCm 10.1 need AMD's out-of-tree amdgpu driver (31.50/31.60) and a listed distribution (Ubuntu
  22.04.5/24.04.5/26.04.1, RHEL 9.8/10.2).
- Multi-GPU systems may need `iommu=pt`.
- A desktop's integrated Radeon can take device 0 (pick with `ROCR_VISIBLE_DEVICES`).
- Strix Halo needs particular kernels (6.18.4+, or Ubuntu's HWE 6.17.0-19).

### Intel (PyTorch XPU)

**Builds and drivers.** PyTorch's `xpu` build supports Arc A and B cards, Arc Pro B-series, the Arc graphics in
Core Ultra (Meteor Lake, Arrow Lake H, Lunar Lake, Panther Lake) on Windows 11, Ubuntu and WSL2, and the Data Center
GPU Max (Ponte Vecchio) on Linux only. Not the Data Center GPU Flex (that was Intel's separate IPEX), and not UHD or
Iris Xe graphics. For 2.15 the Windows driver must be 32.0.101.9033 or newer. Before this branch `install-gpu`
refused anything without "Arc" in its name, which left out the Data Center GPU Max. **Fixed:**
`gpusetup.intel_supported`.

**float64.** Arc A-series has none. A float64 op fails with "Required aspect fp64 is not supported on the device".
Battlemage, Lunar Lake, Meteor Lake and Ponte Vecchio have it, and PyTorch reports it per device as `has_fp64`.
Classic float64 jobs already run on the CPU. Rule-1 Evo summed its weights in float64 on the device, which fails here
and on Apple. **Fixed:** `compute.has_float64`, and `es.Group` does those sums on the CPU, which gives the same float32
bits (IEEE rounding; checked bit-for-bit against CUDA in `tests/test_es.py`).

**Evo.** Runs in float32 (`runtime2`). It's eager, with no graphs: `torch.xpu.XPUGraph` exists since PyTorch 2.11,
but `es_hybrid` only captures graphs on CUDA. bf16 works on XPU but isn't used. Attention goes to oneDNN's kernel,
which takes masks. The flash kernel (Battlemage and Ponte Vecchio, bf16 only) is turned off by deterministic mode.

**TF32.** An Intel GPU's oneDNN matmul uses TF32 when `torch.backends.mkldnn.allow_tf32` is on, and
`set_float32_matmul_precision` doesn't control it (PyTorch source). **Fixed:** `pin_precision` turns it off on builds
that have Intel GPU support. (On other builds setting it only warns, so it's left alone.)

**Problems.** The compute runtime's default limit is 4 GB for one allocation. Qwen3.5-9B's float32 embedding is
248,320 × 4,096 × 4 bytes = 4.07 GB, past it (risk 15). A missing `intel-ocloc` package makes every capability flag
report False. Older Intel guidance says to use the rolling driver stream, not LTS.

### Apple (MPS)

**Builds.** The plain PyPI PyTorch. 2.15 needs macOS 15; on macOS 14 `torch.backends.mps.is_available()` is False,
so the worker silently uses the CPU (risk 12). bf16 tensors need macOS 14 or newer, and fine-tunes keep their frozen
weights in bf16.

**float64.** None at all ("Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support
float64"), and `PYTORCH_ENABLE_MPS_FALLBACK` doesn't help. Handled as for Intel A-series.

**Evo.** float32, eager (MPS has no graph capture). Attention runs on native Metal kernels with a mask. `isin`,
`index_put_` and `repeat_interleave` have MPS kernels. 2.15 raises errors for MPS ops that aren't deterministic; the
ones Evo uses aren't on that list.

**Memory.** Unified. A process may use what Metal recommends (`torch.mps.recommended_max_memory()`), about two
thirds to three quarters of the RAM. Before this branch the worker told the node it had all of it. **Fixed** in
`TorchBackend.memory_mb`.

**Numerics.** No official accuracy statement. Most ops use precise math since macOS 13.4. `PYTORCH_MPS_FAST_MATH`
is opt-in. One open report of wrong results from a very large batched matmul.

### CPU

The NumPy backend is the reference for classic jobs and for arbitration. CPUs don't agree bit for bit either.
OpenBLAS picks kernels by CPU (AVX2, AVX-512, NEON) and splits work by thread count. MKL is reproducible only with
`MKL_CBWR` and fixed threads. That's well inside classic tolerances. PyTorch's `set_float32_matmul_precision("medium")`
makes oneDNN run float32 matmuls in bf16 on CPUs that support it (AVX-512 BF16, AMX), even though PyTorch's docs say
it only affects CUDA. Workers set "highest". Evo and fine-tunes on a CPU are float32 and correct, but slow: a CPU
replay of a 4-step Qwen3-0.6B segment takes 36 s.

## Measured on this computer

RTX 4070 SUPER, PyTorch 2.14.1+cu126, driver 616.92, Windows 11, 2026-11-07. One Evo round of the live job's shape:
- model: Qwen3.5-0.8B (`@2fc0636`), with the base weights and no change applied;
- tasks: the 16-task vibe mix;
- size: 16 antithetic pairs (32 members) × 16 problems, answers up to 512 tokens, round seed `5eed0e7a1c3b2d4f`.

It uses the network's own code: `es_client.runtime2`, `es_fast.answer_members`, `gaps_members` and `score_items`.
"float32" is the model as `runtime2` loads it on Intel, Apple and the CPU, run on CUDA here so it takes minutes. The
arithmetic is float32 either way; only the kernels differ, and kernels matter far less than bf16 versus float32 does.
The answers came from `scripts/evo_margin_xgpu.py answer`. Two small scratch scripts did the float32 answering and
the checks: they load `runtime2(spec, "cpu")` and move it to CUDA, then call `gaps_members` for every member and its
twin swap, and `score_items` on member 0's answers as a replay set. Answering took about a minute, each check about
2 minutes. On real hardware no special script is needed: `scripts/evo_margin_xgpu.py check <a CUDA dump> --device mps`
(or `xpu`, or `cpu`) already runs the float32 path a Mac, Intel card or CPU checker runs.

| Answered in | Checked in | Worst honest gap | Honest members over margin 0.5 | Smallest twin-swap gap | Replay log-probabilities, furthest apart |
|---|---|---|---|---|---|
| bf16 | bf16 (same GPU) | **0** | 0 of 32 | 4.50 | identical |
| bf16 | float32 | 0.20 (80 of 512 answers above 0) | 0 of 32 | 4.46 | 0.013 |
| float32 | float32 | **0** | 0 of 32 | 4.27 | identical |
| float32 | bf16 | **0.375** (27 of 512 above 0; one member over 0.25) | 0 of 32 | 4.25 | 0.009 |
| bf16 | bf16 with split-K bf16 rounding off | 0.25 (56 of 512 above 0) | 0 of 32 | 4.38 | 0.016 |

For comparison, on vibehash-evo-1 (es_local.py's note) RTX 4090 answers checked on the 4070 SUPER had gaps up to
0.375, replay log-probabilities 0.015 apart, and twin-swap cheats 3.5 or more below. 432 of the 512 float32 answers
were token for token the bf16 worker's. Every number is from untrained weights. In a live job a CUDA worker's weights
are bf16(base + change), while a float32 device keeps them exact, a further difference that this didn't include.

## Code risks, ranked

Most harmful first. "Fixed" means fixed on branch `gpu-compat`, with tests; "proposed" means not done.

1. **Evo has 0.125 to spare** (proposed). `margin` is 0.5 (`es_local.py:229`). Honest gaps of 0.375 have been
   measured twice, and Turing, AMD, Intel, Apple and trained weights haven't been measured at all. Evo's arithmetic
   depends on the device kind (bf16 on CUDA/ROCm, float32 elsewhere: `es_client.py:387`, `:413`). It also depends on
   the PyTorch and CUDA version, and on the card's SM count. If any one card kind lands past 0.5, its checkers reject
   honest work. To do, in order:
   - finish the Turing measurement (docs/evo-margin-measurement.md);
   - have one Mac and one Intel Arc check a 4070 SUPER dump with `scripts/evo_margin_xgpu.py check --device mps` or
     `xpu`, and one AMD card (a RunPod MI300X or RX 7900 pod) answer and check;
   - repeat this page's bf16-vs-float32 runs with a trained line (any testnet's saved line);
   - add PyTorch, CUDA/ROCm and dtype to the hardware class, so the census can show version skew. **Done**: workers
     register a number setup beside their class (`hardware.numerics`), checkers send their worst gap, and
     `/network`'s `gaps` counts them by pair of setups (docs/CROSS-HARDWARE.md);
   - pair checkers by setup, with a margin per pair (docs/CROSS-HARDWARE.md, C to E). That matters more once
     Intel, Apple and AMD Evo workers exist.
2. **Evo leaves bf16 split-K rounding on** (proposed, measured). PyTorch's
   `allow_bf16_reduced_precision_reduction` is on by default, and on one GPU it alone moves honest gaps by 0.25 (see
   the measurements). `deepseek.configure_torch` turns it off (`deepseek.py:88`); Evo doesn't. Turning it off should
   shrink the differences between card kinds, because split sums would no longer be rounded to bf16 piece by piece.
   But it changes every worker's bits, so it needs:
   - a new rule, a spec field as `loop_stop` was;
   - first, a two-card-kind measurement with it off: a 4090 pod against the 4070 SUPER.
3. **ROCm's default matrix library on RDNA3/4, MI200/300/350 and Strix Halo isn't deterministic-checked, and has an
   open graph-replay bug** (fixed: `compute.pin_precision` picks rocBLAS on ROCm; the speed it costs isn't measured
   yet). Deterministic mode covers rocBLAS only. hipBLASLt is graph-safe only in part,
   and graph replay can share its split-K state ("hangs or wrong results", reported 2026-11-04). Evo replays graphs
   of bf16 matmuls. Before anyone mines Evo on AMD, have `TorchBackend` pick rocBLAS on ROCm
   (`torch.backends.cuda.preferred_blas_library("cublas")`, or `TORCH_BLAS_PREFER_HIPBLASLT=0` before PyTorch
   starts), and measure the speed it costs on an RDNA3 or MI300 pod.
4. **`install-gpu` put Volta on builds without Volta code** (fixed). In `gpusetup.CUDA_BUILDS`, cu128/cu129 are now
   Turing and up, as they have been since PyTorch 2.11. A V100 used to get cu129, fail its self-test, and mine on the
   CPU.
5. **`install-gpu` didn't know PyTorch 2.15's builds** (fixed): cu134, rocm7.14 and rocm10.0 added.
6. **`install-gpu` may not find ROCm 10.x** (proposed, unverified). `_rocm_version` (`gpusetup.py:106`) reads
   `/opt/rocm/.info/version`, and ROCm 10 installs to `/opt/rocm/core-10.1` with symlinks. Also read
   `/opt/rocm/core-*/.info/version`, or `rocm-sdk version` for pip-installed ROCm, once someone confirms the layout.
7. **TF32 can come back on unseen** (fixed). `compute.pin_precision()` is called from `TorchBackend`
   (`compute.py:35`), `finetune.train_segment`, `es_fast`'s entry points and `es_hybrid._pin_precision`. It never
   reads the flags, because reading them raises once the old and new APIs disagree. It covers Intel GPUs' separate
   TF32 switch too.
8. **Rule-1 Evo crashed on Apple and Intel Arc A** (fixed). `es.Group._stacks` made float64 tensors on the device
   (`es.py:923` and `:927`, before the fix). On those devices the sums now run on the CPU, giving the same bits.
9. **`install-gpu` refused the Data Center GPU Max** (fixed): `gpusetup.intel_supported`.
10. **AMD on Windows got a wrong reason** (fixed): "no build for Python 3.x" became "PyTorch's ROCm builds are
    Linux-only". Proposed next step: install from AMD's Windows index, which needs the card's gfx name before PyTorch
    exists (a name-to-gfx table), and a check that 10.x on Windows trains.
11. **An Evo engine on a GPU other than the current one** (fixed). `cli.pin_gpu` already handled the worker. But
    `scripts/evo_margin_xgpu.py --device cuda:1`, which the margin measurement's instructions tell a second card to
    use, captured empty graphs and decoded forever. Now:
    - the script pins the GPU the way the worker does;
    - `es_fast._same_gpu` refuses a mismatch with a clear error;
    - the script applies the worker's PyTorch settings. It didn't turn on deterministic mode before, and without it
      Hopper and Blackwell pick a different attention kernel.
12. **A Mac on macOS 14 quietly mines on the CPU** with PyTorch 2.15 (fixed). `install-gpu` reads the macOS version
    (`gpusetup.macos_version`) and installs `torch<2.15` on macOS 14, and before 14 says to update macOS.
13. **The GPU self-test only covers the classic path** (proposed). `compute.self_test` (`compute.py:199`) trains a
    float32 MLP. A card can pass it and still fail Evo's bf16 matmuls (the P100 report), graph capture (ROCm), or
    the RX 6700/6600's missing kernels. The fix is a second test on a tiny Qwen3.5, as
    `tests/test_es_hybrid.tiny_hybrid` builds, checking two things:
    - a graph replay equals eager;
    - teacher-forcing the device's own answers gives gaps of exactly 0.

    It belongs in `vibehash devices` and `selftest`.
14. **Pascal, Maxwell and Volta are stuck on PyTorch 2.14** (proposed). Anything that needs a newer PyTorch must
    keep a fallback, and `install-gpu` should tell those owners that 2.14 is the last.
15. **Intel's 4 GB allocation limit** (proposed). Qwen3.5-9B's float32 embedding is over it. Set
    `UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1` before the XPU starts (a community workaround that doesn't always work),
    or keep jobs that size off Intel.
16. **Evo off CUDA is eager and float32** (proposed). It's correct, but probably too slow for a 300 s round on bigger
    models. XPU could capture `XPUGraph`s (2.11+), and XPU and MPS could both run bf16. That changes their rounding
    again, so measure first.
17. **RX 6700/6600 need `HSA_OVERRIDE_GFX_VERSION=10.3.0`** (proposed). When the arch is gfx1031/1032, `vibehash
    devices` could say so, or point to AMD's index, which builds for those cards.
18. **Apple memory was overstated** (fixed): `TorchBackend.memory_mb` reports Metal's recommended working set.
19. **AMD cards were labelled "(CUDA)"** (fixed): `compute.gpu_kind` names PyTorch's ROCm build "ROCm".

## Sources

Read 2026-11-07. "Source" means PyTorch's code on GitHub, read but not run.

**PyTorch builds and releases**
- 2.15.0 release notes: only cu132 and cu134; Turing and newer with driver 580+; "Maxwell, Pascal and Volta GPUs and
  hosts with a CUDA 12.x driver should stay on 2.14"; MPS needs macOS 15; the opt-in `fp32_precision = "bfx9"`.
  https://github.com/pytorch/pytorch/releases/tag/v2.15.0
- 2.11.0: Volta removed from cu128/cu129 (cuDNN 9.15.1); PyPI's default build moves to CUDA 13.0.
  https://github.com/pytorch/pytorch/releases/tag/v2.11.0
- 2.12.0: cu128 removed from the build matrix. https://github.com/pytorch/pytorch/releases/tag/v2.12.0
- 2.8.0: sm_50 to sm_60 dropped from cu128/cu129. https://github.com/pytorch/pytorch/releases/tag/v2.8.0
- 2.14.0: two cuDNN convolution engines turned off on sm_120. https://github.com/pytorch/pytorch/releases/tag/v2.14.0
- Each build's architectures:
  - https://github.com/pytorch/pytorch/blob/release/2.15/.ci/wheel/linux/build_env_setup.py
  - https://github.com/pytorch/pytorch/blob/release/2.15/.ci/wheel/windows/build_env_setup.py
  - https://github.com/pytorch/pytorch/blob/release/2.14/.ci/manywheel/build_env_setup.py
  - https://github.com/pytorch/pytorch/blob/release/2.9/.ci/manywheel/build_cuda.sh
- Which card runs which compiled code (sm_103 on sm_100's, sm_121 on sm_120's), and `is_bf16_supported`:
  https://github.com/pytorch/pytorch/blob/release/2.15/torch/cuda/__init__.py
- The package index: https://download.pytorch.org/whl/torch/

**NVIDIA**
- Minimum drivers per CUDA version: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html and
  https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html
- CUDA 13.0 removed Maxwell, Pascal and Volta:
  https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html
- "The release 580 series will be the last to support GPUs based on the Maxwell, Pascal, and Volta architectures":
  https://forums.developer.nvidia.com/t/unix-graphics-feature-deprecation-schedule/60588
- cuBLAS reproducibility ("the same architecture and the same number of SMs"), `cublasGemmEx` from compute 5.0,
  BF16x9 emulation (opt-in, CC 10.0 and 10.3): https://docs.nvidia.com/cuda/cublas/index.html
- PyTorch dropped its CUBLAS_WORKSPACE_CONFIG requirement in 2.10: https://github.com/pytorch/pytorch/pull/161749
- bf16 emulated before Ampere: https://github.com/pytorch/pytorch/issues/124996 and
  https://github.com/pytorch/pytorch/pull/129288
- P100 bf16 CUBLAS_STATUS_NOT_SUPPORTED: https://github.com/pytorch/pytorch/issues/57773
- Attention kernels by card and dtype, and the mask rule:
  - https://github.com/pytorch/pytorch/blob/release/2.15/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
  - https://github.com/pytorch/pytorch/blob/release/2.15/aten/src/ATen/native/transformers/sdp_utils_cpp.h
  - priority order: https://github.com/pytorch/pytorch/blob/release/2.15/aten/src/ATen/Context.h
- Determinism notes ("Bitwise matching numerics across different SDPA backends are not guaranteed"):
  https://github.com/pytorch/pytorch/blob/release/2.15/docs/source/notes/randomness.md
- The TF32 switches, old and new, and "we do not support to use mix of old and new settings":
  https://github.com/pytorch/pytorch/blob/release/2.15/docs/source/notes/cuda.md
  and https://github.com/pytorch/pytorch/blob/release/2.15/aten/src/ATen/Context.cpp
- CUDA graphs' constraints, and the Windows test skip: the same cuda.md and
  https://github.com/pytorch/pytorch/blob/release/2.15/test/test_cuda.py
- RTX 50-series on old builds:
  https://discuss.pytorch.org/t/nvidia-geforce-rtx-5090-with-cuda-capability-sm-120-is-not-compatible-with-the-current-pytorch-installation/216518
- GTX 16xx fp16 NaNs: https://github.com/pytorch/pytorch/issues/58123
- Windows TDR (2 s): https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys

**AMD**
- ROCm's renumbering: https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html. 10.1's release
  notes: https://rocm.docs.amd.com/en/latest/about/release-notes.html
- Official support, 10.1: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html. 7.2.3:
  https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.3/reference/system-requirements.html
- RDNA4 from 6.4.1: https://rocm.docs.amd.com/en/docs-6.4.1/about/release-notes.html
- PyTorch's ROCm builds per release, and the gfx targets compiled in:
  - https://github.com/pytorch/pytorch/blob/release/2.15/.github/scripts/generate_binary_build_matrix.py
  - https://github.com/pytorch/pytorch/blob/release/2.15/.ci/docker/manywheel/build.sh
  - https://download.pytorch.org/whl/rocm7.2/torch/ (and the other `rocmX.Y` indexes)
- No Windows ROCm builds from PyTorch: https://github.com/pytorch/pytorch/issues/159520
- AMD's Windows PyTorch:
  - https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
  - https://stable.repo.amd.com/rocm/whl-next/
  - 7.2.1's limits: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html
- WSL2: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
- torch-directml: https://pypi.org/project/torch-directml/ and https://github.com/microsoft/DirectML/issues/688
- TheRock's GPU list: https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md
- `HSA_OVERRIDE_GFX_VERSION`:
  - https://github.com/ROCm/rocm-systems/blob/develop/projects/rocr-runtime/libhsakmt/src/topology.c
  - https://github.com/pytorch/pytorch/issues/193668
- Matrix cores and precisions:
  - https://rocm.docs.amd.com/en/latest/reference/precision-support.html
  - https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocwmma/README.md
- Attention through AOTriton:
  - https://github.com/pytorch/pytorch/blob/release/2.15/cmake/External/aotriton.cmake
  - https://github.com/ROCm/aotriton/blob/0.14.2b/v3src/hip-dependent/util.cc
  - and `sdp_utils.cpp`, above
- Determinism and the matrix library:
  - https://github.com/pytorch/pytorch/blob/release/2.15/aten/src/ATen/cuda/CublasHandlePool.cpp
  - https://github.com/pytorch/pytorch/blob/release/2.15/aten/src/ATen/cuda/detail/CUDAHooks.cpp
  - https://github.com/pytorch/pytorch/issues/196164
- HIP graphs:
  - https://github.com/pytorch/pytorch/blob/release/2.15/docs/source/notes/hip.rst
  - https://rocm.docs.amd.com/en/latest/reference/graph-safe-support.html
  - https://github.com/pytorch/pytorch/issues/203474
- TF32 on ROCm: https://github.com/pytorch/pytorch/pull/162998 and
  https://github.com/pytorch/pytorch/blob/release/2.15/docs/source/notes/numerical_accuracy.md
- Strix Halo kernels: https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html. Picking GPUs:
  https://rocm.docs.amd.com/en/latest/reference/system-optimization/gpu-isolation.html

**Intel**
- Supported GPUs and systems: https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html
- Drivers for 2.15: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-15.html
- Device properties (`has_fp64`, `total_memory`): https://docs.pytorch.org/docs/2.15/generated/torch.xpu.get_device_properties.html
- No float64 on Arc A: https://github.com/intel/torch-xpu-ops/issues/628. Which chips have it:
  https://github.com/pytorch/pytorch/issues/147731
- B580 setup, capability flags, `intel-ocloc`: https://github.com/pytorch/pytorch/issues/196074
- Attention and TF32 on XPU (source):
  - https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/mkldnn/xpu/Attention.cpp
  - https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/xpu/sdp_utils.cpp
  - https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/mkldnn/xpu/detail/Matmul.cpp
- XPU graphs arrived in 2.11: https://pytorch.org/blog/pytorch-2-11-release-blog/
- Tiles as devices: https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
- The 4 GB allocation limit: https://github.com/intel/compute-runtime/blob/master/programmers-guide/ALLOCATIONS_GREATER_THAN_4GB.md
  and
  https://discuss.pytorch.org/t/xpu-out-of-memory-error-with-intel-arc-graphics-meteor-lake-despite-sufficient-system-memory-and-reported-xpu-capacity/219812/8

**Apple**
- MPS notes (macOS, float64): https://docs.pytorch.org/docs/2.15/notes/mps.html
- bf16 from macOS 14: https://github.com/pytorch/pytorch/pull/119641
- Environment variables (fallback, fast math, memory watermarks): https://docs.pytorch.org/docs/2.15/mps_environment_variables.html
- Per-process memory: https://docs.pytorch.org/docs/2.15/generated/torch.mps.set_per_process_memory_fraction.html
- Ops that raise in deterministic mode: https://docs.pytorch.org/docs/2.15/generated/torch.use_deterministic_algorithms.html
- Precise versus fast math: https://github.com/pytorch/pytorch/issues/84936. The large-bmm report:
  https://github.com/pytorch/pytorch/issues/111634
- Op coverage (third party): https://qqaatw.dev/pytorch-mps-ops-coverage/

**CPU**
- `set_float32_matmul_precision` and oneDNN on the CPU (source):
  https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/mkldnn/Matmul.cpp and
  https://docs.pytorch.org/docs/2.15/notes/mkldnn.html
- MKL's reproducible mode: https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-2/get-started-with-conditional-num-reproducibility.html
- OpenBLAS's runtime kernel choice: http://www.openmathlib.org/OpenBLAS/docs/runtime_variables/
- PyTorch on reproducibility across platforms: https://docs.pytorch.org/docs/2.15/notes/randomness.html

**This repository**
- DESIGN.md §8: classic and fine-tune measurements, and rule 2.
- `vibehash/es_local.py`: the note above `V2_SETTINGS`, with the 4090/4070 SUPER measurement.
- docs/evo-margin-measurement.md: the cross-GPU margin plan.
- HANDOFF.md: Evo speed and memory on the 4070 SUPER.
