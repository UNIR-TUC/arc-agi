"""
Eje D — computational efficiency / silicon utilisation (opt-in acceleration layer).

Implements §9.6 of Docs/Architecture/COMPRESS_ARCHITECTURE_30_06.md on AMD RDNA4
(RX 9070 XT) + ROCm, **without touching the model core**:

  * §9.6.1  Mixed precision BF16 on the RDNA4 Matrix cores, via `torch.autocast`.
  * §9.6.2  Kernel fusion via `torch.compile` (Inductor -> Triton-ROCm).
  * §9.6.3  Host-side silicon utilisation: matmul precision policy, per-worker
            intra-op thread capping, allocator tuning, Inductor cache reuse.

Design constraints
------------------
1. **The model core is untouched.** `arc_compressor.py`, `layers.py`,
   `multitensor_systems.py`, `train.py`, `solution_selection.py` and
   `initializers.py` are not modified. The only contact point is `apply()`,
   which *rebinds the instance attribute* `model.forward` to a wrapped/compiled
   callable. `ARCCompressor` is a plain class (not `nn.Module`), so this works.
2. **Everything is off by default.** An `AccelConfig()` with default values
   reproduces exactly today's behaviour (`is_enabled()` is False).
3. **Never fail a task because of the accelerator.** Compilation problems fall
   back to eager mode with a warning; an unsupported dtype falls back to FP32.

Why plain `torch.autocast` is numerically safe here
---------------------------------------------------
* `layers.channel_layer` (the KL) contains **no matmul** — only exp/mean/sqrt/
  where/randn — so under autocast it stays entirely in FP32. This satisfies the
  §9.6.1 requirement that "KL accumulation must stay in FP32" without having to
  carve out regions by hand.
* `layers.add_residual` ends in `return x + z`, with `x` the FP32 residual
  stream and `z` the output of `affine` (a matmul, hence BF16 under autocast).
  PyTorch type promotion (FP32 + BF16 -> FP32) means **the residual stream stays
  FP32**; only the intermediates are BF16. The same holds for
  `direction_share` (`x_list[d1] + c*affine(...)`) and for the output head
  (`affine(...) + 100*head_weights[1]`). `normalize`'s mean/variance therefore
  keep operating on FP32 tensors.
* `apply()` still casts every forward output back to FP32 defensively, so
  `train.take_step` receives exactly the dtypes it receives today.

Why `dynamic=False`
-------------------
Each worker process (`solve_task.solve_task`) solves **one** task, so the
multitensor shapes are constant across all training steps. Unlike the
`dynamic=True` suggested in §9.6.2 (which assumed a shared process), static
shapes give a single compilation with no dynamic-shape guard overhead.
"""

import os
import warnings
from dataclasses import dataclass, asdict, fields
from typing import Optional

import torch


# ── Configuration ────────────────────────────────────────────────────────────

AMP_CHOICES = ('off', 'bf16', 'fp16')
COMPILE_CHOICES = ('off', 'default', 'reduce-overhead', 'max-autotune')
MATMUL_PRECISION_CHOICES = ('highest', 'high', 'medium')

_AMP_DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16}


@dataclass
class AccelConfig:
    """Serialisable acceleration settings (crosses the `spawn` process boundary).

    Attributes:
        amp (str): 'off' | 'bf16' | 'fp16'. BF16 is recommended on RDNA4: it has
            the same exponent range as FP32, and CompressARC has no loss scaling
            (an FP16 overflow anywhere in the multitensor would wreck the KL).
        compile_mode (str): 'off' | 'default' | 'reduce-overhead' | 'max-autotune'.
            'reduce-overhead' enables HIP graphs — opt-in only, since capturing
            graphs in many concurrent worker processes inflates VRAM.
        matmul_precision (str): argument for `torch.set_float32_matmul_precision`.
            This is the ROCm-correct replacement for
            `torch.backends.cuda.matmul.allow_tf32`, which is a **no-op** on
            RDNA4 (no TF32 hardware) — see §9.2. 'highest' == today's behaviour.
        threads_per_worker (int): `torch.set_num_threads` value per worker
            process, 0 to leave untouched. With 13+ concurrent workers the
            default intra-op thread pools oversubscribe the host CPU.
        alloc_conf (str|None): value for PYTORCH_HIP_ALLOC_CONF /
            PYTORCH_CUDA_ALLOC_CONF, e.g. 'expandable_segments:True'.
        inductor_cache_dir (str|None): persistent TorchInductor cache directory,
            so repeated runs reuse compiled kernels.
        compile_threads (int): TORCHINDUCTOR_COMPILE_THREADS per worker. Phase 1
            no longer compiles (see `for_measurement`), so Phase 2 can afford a
            small pool; a single-threaded Inductor took ~20 min per task.
        memory_planning (bool): enable `torch._inductor.config.memory_planning`,
            which reuses buffers across the fused graph. Measured: compilation
            raised VRAM from ~0.9 GB to ~3.9 GB per task, which cut scheduler
            concurrency from 10 to 3 and ate most of the speedup.
    """

    amp: str = 'off'
    compile_mode: str = 'off'
    matmul_precision: str = 'highest'
    threads_per_worker: int = 0
    alloc_conf: Optional[str] = None
    inductor_cache_dir: Optional[str] = None
    compile_threads: int = 1
    memory_planning: bool = False

    # ── Introspection ───────────────────────────────────────────────────
    def is_enabled(self):
        """True if anything deviates from the untouched baseline."""
        default = AccelConfig()
        return any(getattr(self, f.name) != getattr(default, f.name)
                   for f in fields(self))

    def changes_forward(self):
        """True if `apply()` would wrap or compile `model.forward`."""
        return self.amp != 'off' or self.compile_mode != 'off'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def describe(self):
        """Flat dict for run metadata / logs."""
        d = self.to_dict()
        d['enabled'] = self.is_enabled()
        return d

    def summary(self):
        """One-line human-readable description."""
        if not self.is_enabled():
            return 'accel=off (baseline)'
        return (f'accel amp={self.amp} compile={self.compile_mode} '
                f'matmul={self.matmul_precision} threads={self.threads_per_worker} '
                f'alloc={self.alloc_conf or "-"}')


# Convenience presets so A/B runs are labelled consistently.
#
# NOTE (measured 2026-07-28, 10 training tasks x 300 iterations on RX 9070 XT):
# BF16 autocast is *not* part of `full` any more. Paired per-task timings showed it
# **+2.7 % slower** (8 of 10 tasks worse) with identical VRAM. The workload is
# launch-bound, not FLOP-bound (GPU busy 97 % but mem_busy 3 % and 94 W of 304 W
# TDP), so speeding up arithmetic buys nothing while autocast *adds* one cast
# kernel per `layers.affine` call. `torch.compile`, which reduces the *number* of
# launches, gave 13-26x per iteration in steady state instead.
PRESETS = {
    'baseline': {},
    'bf16': {
        'amp': 'bf16',
        'matmul_precision': 'high',
        'threads_per_worker': 1,
    },
    'compile': {
        'compile_mode': 'default',
        'matmul_precision': 'high',
        'threads_per_worker': 1,
        'inductor_cache_dir': '.inductor_cache',
        'compile_threads': 4,
        'memory_planning': True,
    },
    'full': {
        'compile_mode': 'default',
        'matmul_precision': 'high',
        'threads_per_worker': 1,
        'alloc_conf': 'expandable_segments:True',
        'inductor_cache_dir': '.inductor_cache',
        'compile_threads': 4,
        'memory_planning': True,
    },
}


def config_from_preset(name, **overrides):
    """Build an AccelConfig from a preset name, applying explicit overrides.

    Overrides whose value is None are ignored, so a CLI can pass every flag
    unconditionally and let unspecified ones fall back to the preset.
    """
    if name not in PRESETS:
        raise ValueError(f'unknown accel preset {name!r}; '
                         f'choose from {sorted(PRESETS)}')
    values = dict(PRESETS[name])
    values.update({k: v for k, v in overrides.items() if v is not None})
    return AccelConfig.from_dict(values)


def for_measurement(cfg):
    """Return the config to use for the Phase-1 VRAM measurement run.

    Identical to `cfg` except that **compilation is always disabled**.

    Rationale (measured): Phase 1 runs 2 iterations per task, one task at a time.
    Compiling for that cost ~1200 s of single-threaded Inductor work per task and
    accounted for **92 % of the total wall time** of a 10-task profiling run
    (11 871 s of 12 921 s). Autocast is kept, since it costs nothing to enable
    and does participate in the memory footprint.

    The compiled footprint is recovered afterwards by scaling the measurements
    (see `--compile-memory-factor` in parallel_train.py). The night campaign of
    2026-07-29 showed the scale factor is close to 1: whole-GPU VRAM with 3
    compiled tasks resident was 2.84-2.90 GB, i.e. ~0.95 GB/task against
    ~0.89 GB/task eager. The earlier 3.87 GB figure was an artefact of measuring
    a *compiled* Phase 1, which still held Inductor's autotuning buffers.

    Side benefit: the Phase-1 cache becomes shareable between an eager run and a
    compiled run of the same split, removing a fixed cost from every A/B.
    """
    cfg = AccelConfig.from_dict(cfg)
    if cfg.compile_mode == 'off':
        return cfg
    return AccelConfig.from_dict({**cfg.to_dict(), 'compile_mode': 'off'})


# ── Environment / process configuration ──────────────────────────────────────

def _apply_env(cfg):
    """Set the environment variables that must be visible before the CUDA/HIP
    allocator and TorchInductor initialise. Safe to call more than once."""
    if cfg.alloc_conf:
        # ROCm builds read PYTORCH_HIP_ALLOC_CONF; CUDA builds read the CUDA one.
        # Setting both is harmless and keeps the code portable.
        os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', cfg.alloc_conf)
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', cfg.alloc_conf)
    if cfg.compile_mode != 'off':
        if cfg.inductor_cache_dir:
            os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',
                                  os.path.abspath(cfg.inductor_cache_dir))
        if cfg.compile_threads and cfg.compile_threads > 0:
            os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS',
                                  str(cfg.compile_threads))


def configure_parent(cfg):
    """Called once in the scheduler process, before any worker is spawned, so
    that `multiprocessing.spawn` children inherit the tuned environment."""
    _apply_env(cfg)


def configure_process(cfg):
    """Configure the current (worker) process. Call as early as possible, before
    any GPU tensor is allocated.

    Deliberately performs **no CUDA/HIP calls**: it runs before
    `torch.cuda.set_device(gpu_id)`, and touching `torch.cuda` here would
    initialise the context on the default device instead of the assigned one.
    Device-dependent capability checks live in `apply()`.

    Returns:
        AccelConfig: the effective config.
    """
    cfg = AccelConfig.from_dict(cfg)
    _apply_env(cfg)

    # §9.2: on ROCm/RDNA4 `torch.backends.cuda.matmul.allow_tf32` is a no-op.
    # `set_float32_matmul_precision` is the portable knob that actually lets the
    # backend use reduced-precision accumulation for FP32 matmuls.
    try:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    except Exception as exc:  # pragma: no cover - depends on torch build
        warnings.warn(f'[accel] set_float32_matmul_precision failed: {exc}')

    if cfg.threads_per_worker and cfg.threads_per_worker > 0:
        try:
            torch.set_num_threads(cfg.threads_per_worker)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f'[accel] set_num_threads failed: {exc}')

    if cfg.memory_planning and cfg.compile_mode != 'off':
        # Buffer reuse across the fused graph. Measured to change neither speed
        # nor VRAM here (2864 vs 2842 MB with it off), and it partitions the
        # Inductor cache key, so toggling it forces a full recompile.
        try:
            import torch._inductor.config as inductor_config
            inductor_config.memory_planning = True
        except Exception as exc:  # pragma: no cover - depends on torch build
            warnings.warn(f'[accel] inductor memory_planning unavailable: {exc}')

    return cfg


def _bf16_supported():
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - old/exotic builds
        return False


def backend_info():
    """Backend fingerprint, recorded in run metadata so A/B runs are traceable."""
    info = {
        'torch_version': torch.__version__,
        'hip_version': getattr(torch.version, 'hip', None),
        'cuda_version': getattr(torch.version, 'cuda', None),
        'is_rocm': getattr(torch.version, 'hip', None) is not None,
    }
    try:
        info['gpu_names'] = [torch.cuda.get_device_name(i)
                             for i in range(torch.cuda.device_count())]
    except Exception:  # pragma: no cover
        info['gpu_names'] = []
    return info


def compile_report(cfg):
    """Return torch.compile timings and persistent FX graph cache counters.

    Compilation dominates the cost of this workload (~1200 s per task cold,
    ~250 s warm), so knowing whether it goes into Dynamo tracing (the unrolled
    Python loops of `direction_share`: 64 `affine` calls x 4 layers x ~8
    directional tensors) or into the Inductor scheduler decides which mitigation
    is worth pursuing. Never raises.
    """
    cfg = AccelConfig.from_dict(cfg)
    if cfg.compile_mode == 'off':
        return None
    try:
        from torch._dynamo.utils import compile_times, counters
        inductor = counters.get('inductor', {})
        aot_autograd = counters.get('aot_autograd', {})
        fx_cache_summary = (
            'FXGraph cache: '
            f'hits={inductor.get("fxgraph_cache_hit", 0)} '
            f'misses={inductor.get("fxgraph_cache_miss", 0)} '
            f'bypasses={inductor.get("fxgraph_cache_bypass", 0)}'
        )
        aot_cache_summary = (
            'AOTAutograd cache: '
            f'hits={aot_autograd.get("autograd_cache_hit", 0)} '
            f'misses={aot_autograd.get("autograd_cache_miss", 0)} '
            f'bypasses={aot_autograd.get("autograd_cache_bypass", 0)} '
            f'guard_misses={aot_autograd.get("autograd_cache_guard_miss", 0)}'
        )
        return (f'{fx_cache_summary}\n{aot_cache_summary}\n'
                f'{compile_times(repr="str", aggregate=True)}')
    except Exception:
        return None


# ── Forward-pass wrapping (the only contact point with the model) ────────────

class _EagerFallback:
    """Call a compiled callable, permanently reverting to eager on first error.

    `torch.compile` is lazy: failures surface on the first call, not at
    decoration time. A compilation problem must never kill a task, so the very
    first exception switches this wrapper to the original eager callable for the
    rest of the run.
    """

    def __init__(self, compiled, eager):
        self._compiled = compiled
        self._eager = eager
        self._failed = False

    def __call__(self, *args, **kwargs):
        if self._failed:
            return self._eager(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)
        except Exception as exc:
            self._failed = True
            warnings.warn(f'[accel] torch.compile failed at runtime ({exc}); '
                          f'falling back to eager for the rest of this task.')
            return self._eager(*args, **kwargs)


def _autocast_wrap(forward, dtype):
    """Run `forward` under autocast and cast its outputs back to FP32.

    `ARCCompressor.forward` returns
    `(logits, x_mask, y_mask, KL_amounts, KL_names)`; the first three are
    tensors, `KL_amounts` is a list of tensors and `KL_names` a list of strings.
    """
    def forward_with_autocast():
        with torch.autocast(device_type='cuda', dtype=dtype):
            logits, x_mask, y_mask, KL_amounts, KL_names = forward()
        return (
            logits.float(),
            x_mask.float(),
            y_mask.float(),
            [KL.float() for KL in KL_amounts],
            KL_names,
        )
    return forward_with_autocast


# Compile modes whose outputs live in a HIP/CUDA-graph static pool.
_CUDAGRAPH_MODES = ('reduce-overhead', 'max-autotune')


def _clone_wrap(forward):
    """Copy the forward outputs out of the cudagraph static buffers.

    Under `reduce-overhead` the returned tensors alias a pool that the next
    replay overwrites, while `solution_selection.Logger` retains `logits`,
    `x_mask` and `y_mask` for the whole run. Measured consequence: a
    `reduce-overhead` campaign solved 0/10 tasks at 7.34 steps/s while the same
    tasks compiled normally solved 5/10 (§15.9).
    """
    def forward_with_cloned_outputs():
        logits, x_mask, y_mask, KL_amounts, KL_names = forward()
        return (
            logits.clone(),
            x_mask.clone(),
            y_mask.clone(),
            [KL.clone() for KL in KL_amounts],
            KL_names,
        )
    return forward_with_cloned_outputs


def apply(model, cfg):
    """Rebind `model.forward` with the configured acceleration wrappers.

    Must be called *after* `torch.cuda.set_device(gpu_id)`, since it performs the
    device capability check for BF16.

    Args:
        model (arc_compressor.ARCCompressor): freshly constructed model.
        cfg (AccelConfig|dict|None): acceleration settings.

    Returns:
        model: the same object (mutated in place), for convenience.
    """
    cfg = AccelConfig.from_dict(cfg)
    if not cfg.changes_forward():
        return model

    forward = model.forward  # bound method; captured before rebinding

    if cfg.compile_mode != 'off':
        try:
            compiled = torch.compile(
                forward,
                mode=cfg.compile_mode,
                dynamic=False,   # one task per process => static shapes
                fullgraph=False,  # graph breaks are expected (postprocess_mask)
            )
            forward = _EagerFallback(compiled, model.forward)
        except Exception as exc:
            warnings.warn(f'[accel] torch.compile unavailable ({exc}); '
                          f'running eager.')
        if cfg.compile_mode in _CUDAGRAPH_MODES:
            forward = _clone_wrap(forward)

    dtype = _AMP_DTYPES.get(cfg.amp)
    if dtype is torch.bfloat16 and not _bf16_supported():
        warnings.warn('[accel] BF16 not supported by this device/build — '
                      'falling back to FP32 for this task.')
        dtype = None
    if dtype is not None:
        forward = _autocast_wrap(forward, dtype)

    model.forward = forward
    return model
