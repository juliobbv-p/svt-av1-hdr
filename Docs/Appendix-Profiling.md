# Appendix: NVIDIA Nsight Systems profiling

SVT-AV1 ships an opt-in profiling integration for **NVIDIA Nsight Systems**,
designed for NVIDIA Grace CPU (aarch64 Neoverse V2) hosts but functional on any
Linux platform with the NVTX3 headers available. The integration covers the
parity matrix of metrics that x86 users typically gather ad-hoc with
`perf record/stat/sched/trace`, unified into one `.nsys-rep` capture.

| Capability                              | Common ad-hoc tool   | This integration                       |
| --                                      | --                   | --                                     |
| CPU sampling / hot functions             | `perf record -g`     | `nsys profile --sample=process-tree`   |
| PMU counters (cycles, IPC, cache, branch)| `perf stat`          | `nsys profile --cpu-core-events=...` (root or `kernel.perf_event_paranoid <= 0`) |
| Thread state timeline                    | `perf sched`         | `nsys profile --cpuctxsw=process-tree` |
| OS runtime trace (futex, syscall, pthread) | `perf trace -s`    | `nsys profile -t osrt`                 |
| Per-stage timing                         | (none, ad-hoc)       | NVTX `:wait` ranges per pipeline stage thread (gap == busy) |
| End-to-end FPS                           | encoder built-in     | unchanged (encoder reports as before)  |
| Per-frame PSNR/SSIM                      | `--enable-stat-report 1` | unchanged                          |

Default is OFF. With NVTX disabled, the encoder bitstream is byte-identical to
a build without the option (verified via md5 between `-DSVT_AV1_NVTX=OFF` and
`-DSVT_AV1_NVTX=ON` builds with the same input).

## Prerequisites

- **NVTX3 headers**: shipped with the CUDA Toolkit (`/usr/local/cuda/include/nvtx3/`)
  or the NVIDIA HPC SDK (`/opt/nvidia/hpc_sdk/Linux_aarch64/<ver>/cuda/include/nvtx3/`).
  CMake searches `CUDA_HOME`, `NVHPC_ROOT`, the standard CUDA install paths, and
  finally `/usr/include`. Set `CUDA_HOME` if your install lives elsewhere.
- **Nsight Systems CLI** (`nsys`): for capture and post-processing. Comes with
  the CUDA Toolkit and the HPC SDK; also available as a standalone download
  from NVIDIA.
- (Optional) **`kernel.perf_event_paranoid <= 2`** for CPU sampling, and `<= 0`
  (or root) for `--cpu-core-events`. On a fresh Grace install, sampling
  typically works out-of-the-box; in containers the sysctl is often locked to
  3 or 4 and only NVTX/OSRT/ctxsw will populate.

## Build

```sh
cd Build/linux
./build.sh release --enable-nvtx
```

Maps to `-DSVT_AV1_NVTX=ON`. The library gains lazy `dlopen`/`dlsym` imports
for the NVTX3 runtime; there is **no hard link dependency** on
`libnvToolsExt`. If the runtime is absent at execution, NVTX calls become
no-ops courtesy of NVTX3's lazy loader.

```sh
ldd Bin/Release/libSvtAv1Enc.so | grep -i nvtx     # expect: empty
nm  Bin/Release/libSvtAv1Enc.so | grep -i nvtx     # expect: empty (NVTX3 funcs are static inline)
nm -D Bin/Release/libSvtAv1Enc.so | grep -E " U dlopen| U dlsym"   # expect: present
```

`SvtAv1EncApp` and every pipeline thread it spawns are named via
`pthread_setname_np` regardless of the build flag. Worker arrays use the form
`svt-me0`...`svt-meN`, `svt-md0`...`svt-mdN`, `svt-cdef0`...`svt-cdefN`, etc.
Singleton stage threads take the kernel function name, truncated by Linux to
the 15-char `TASK_COMM_LEN`:

| Kernel | Thread name |
|---|---|
| `svt_aom_resource_coordination_kernel` | `svt_aom_resourc` |
| `svt_aom_picture_decision_kernel`      | `svt_aom_picture` ⚠ |
| `svt_aom_initial_rate_control_kernel`  | `svt_aom_initial` |
| `svt_aom_picture_manager_kernel`       | `svt_aom_picture` ⚠ |
| `svt_aom_rate_control_kernel`          | `svt_aom_rate_co` |
| `svt_aom_packetization_kernel`         | `svt_aom_packeti` |

⚠ `picture_decision` and `picture_manager` collapse to the same truncated
name. They are still distinct threads — TID disambiguates them in Nsight
timelines, but `top`/`htop`/`ps -L`/`/proc/<pid>/task/*/comm` will show two
threads sharing the `svt_aom_picture` label. The application thread is
`svt-app-main`. All names are visible in `top`, `htop`,
`ps -L -o pid,tid,comm`, `/proc/<pid>/task/*/comm`, and the Nsight Systems
timeline lane labels.

## Standalone runtime

Capture on a Grace host (or any aarch64/x86_64 Linux):

```sh
nsys profile -t osrt,nvtx \
    --sample=process-tree --cpuctxsw=process-tree \
    --output=/tmp/svt --force-overwrite=true \
    ./Bin/Release/SvtAv1EncApp -i in.y4m -b out.ivf --preset 8 --lp 16
```

Add `--cpu-core-events=cycles,instructions,cache-misses,branch-misses` for
Neoverse V2 PMU counters (requires root or `paranoid <= 0`).

Inspect:

```sh
nsys stats --report nvtx_sum   --format=csv --force-export=true /tmp/svt.nsys-rep
nsys stats --report osrt_sum   --format=csv /tmp/svt.nsys-rep
nsys-ui /tmp/svt.nsys-rep   # GUI; per-thread lanes labelled svt-me0, svt-cdef0, ...
```

## Harness runtime (`test/benchmarking/`)

The Python benchmark harness can wrap the encoder with `nsys profile` and add
parity metrics to the per-job CSV alongside BD-rate / VMAF / SSIMULACRA2.

In your config (e.g. `test/benchmarking/configs/test_video_config.yaml`):

```yaml
profiler:
  enabled: true                 # default false; existing configs unaffected
  tool: "nsys"
  command: "nsys profile -t osrt,nvtx --cpuctxsw=process-tree --sample=process-tree --output={profile_path} --force-overwrite=true "
  apply_to: ["svtav1", "svtav1_rtc"]
  post_reports: ["nvtx_sum", "osrt_sum"]

paths:
  profile_dir: "{out_dir}/profiles/{tag}"

binaries:
  linux_aarch64:                # NEW slot: picked automatically on aarch64 Linux
    svtav1_enc: "{root_dir}/bin/linux_aarch64/SvtAv1EncApp"
    # ... mirror the linux_x86_64 entries you need
```

Run as usual:

```sh
./run_comparison.sh configs/test_video_config.yaml
```

The encoded CSV gains three columns next to `encode_time` / `output_size`:

| Column                    | Source                                                  |
| --                        | --                                                      |
| `nsys_report_path`        | path to the per-job `.nsys-rep` (open in `nsys-ui`)     |
| `osrt_total_ms`           | `SUM(end-start)` over `OSRT_API` rows in the SQLite     |
| `cpu_sampling_top1_func`  | top sampled symbol from `SAMPLING_CALLCHAINS` (empty when sampling did not capture — short run, paranoid restriction, etc.) |

## Grace / containers note

Nsight Systems' CPU sampling and PMU counters go through `perf_event_open`,
which respects `kernel.perf_event_paranoid`:

- `paranoid >= 3`: no sampling, no PMU. NVTX, OSRT, and context-switch trace
  still work.
- `paranoid == 2`: per-process sampling allowed. PMU still restricted.
- `paranoid <= 0`: full PMU access (or run nsys as root).

On a bare-metal Grace host you typically already have `paranoid <= 2`. In
shared containers the sysctl is often locked higher; the integration degrades
gracefully — `cpu_sampling_top1_func` ends up empty in the CSV but every other
column populates and the timeline still shows NVTX ranges, OSRT calls, and
thread states.

## Verifying the integration

A 30-frame smoke run is enough to verify the wiring:

```sh
nsys profile -t osrt,nvtx --output=/tmp/smoke --force-overwrite=true \
    ./Bin/Release/SvtAv1EncApp -i sample.y4m -b /tmp/smoke.ivf --preset 11 -n 30
nsys stats --report nvtx_sum --format=csv --force-export=true /tmp/smoke.nsys-rep
# expect a single ":wait" range with Instances >> frame count
```

For the `pthread_setname_np` side, while a longer encode is running:

```sh
ls /proc/$(pidof SvtAv1EncApp)/task/*/comm | xargs -I{} cat {} | sort -u
# expect svt-app-main, 5 distinct svt_aom_* singletons (picdec/picmgr collide), svt-me0..N, svt-md0..N, svt-cdef0..N, ...
```
