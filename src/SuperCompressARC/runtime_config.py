import contextlib

import torch


MIXED_PRECISION_CHOICES = ('off', 'bf16', 'fp16')
COMPILE_FORWARD_CHOICES = ('off', 'default', 'reduce-overhead', 'max-autotune')
MATMUL_PRECISION_CHOICES = ('highest', 'high', 'medium')


def apply_torch_backend_settings(float32_matmul_precision='high'):
    """Apply backend settings shared by sequential and parallel runners."""
    if float32_matmul_precision:
        torch.set_float32_matmul_precision(float32_matmul_precision)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
    cuda_backend = getattr(torch.backends, 'cuda', None)
    cuda_matmul = getattr(cuda_backend, 'matmul', None)
    if cuda_matmul is not None and hasattr(cuda_matmul, 'allow_tf32'):
        cuda_matmul.allow_tf32 = True


def autocast_context(mixed_precision):
    """Return a CUDA autocast context for the requested precision mode."""
    if mixed_precision == 'off':
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if mixed_precision == 'bf16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)


def compile_model_forward(model, compile_forward):
    """Compile ARCCompressor.forward in-place when requested.

    ARCCompressor is not an nn.Module, so compiling the bound forward method is
    the least invasive path: optimizer state, logger side effects, and Python
    solution postprocessing stay outside the compiled region.
    """
    if compile_forward == 'off':
        return False
    if not hasattr(torch, 'compile'):
        return False
    kwargs = {'dynamic': True}
    if compile_forward != 'default':
        kwargs['mode'] = compile_forward
    model.forward = torch.compile(model.forward, **kwargs)
    return True


def runtime_options_dict(mixed_precision='off', compile_forward='off',
                         float32_matmul_precision='high'):
    return {
        'mixed_precision': mixed_precision,
        'compile_forward': compile_forward,
        'float32_matmul_precision': float32_matmul_precision,
    }