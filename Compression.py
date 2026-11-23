import contextlib
import copy
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import threading
import traceback
import warnings
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import psutil


SUPPORTED_BITS = (2, 3, 4, 5, 6, 7, 8)
_MIB = 1024 ** 2

# Candidate clipping ratios tried per output channel (1.0 = plain max-abs scale).
_CLIP_RATIOS = [1.0 - 0.05 * i for i in range(13)]          # 1.00 ... 0.40

_NoSubclass = getattr(torch._C, "DisableTorchFunctionSubclass", None) or torch._C.DisableTorchFunction

# Bit packing
def _bit_layout(bits: int):
    """Return (values_per_group, bytes_per_group) so that a group is byte aligned."""
    g = math.gcd(8, bits)
    return 8 // g, bits // g


def pack_signed_integers(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed two's-complement integers into a compact uint8 tensor.
    Only used at quantization time (not in the inference path)."""
    per_group, bytes_per_group = _bit_layout(bits)
    mask = (1 << bits) - 1
    v = values.detach().reshape(-1).to(torch.int64) & mask
    pad = (-v.numel()) % per_group
    if pad:
        v = F.pad(v, (0, pad))
    v = v.view(-1, per_group)
    shifts = torch.arange(per_group, device=v.device, dtype=torch.int64) * bits
    word = (v << shifts).sum(dim=1)            # bit fields never overlap -> sum == OR
    byte_shifts = torch.arange(bytes_per_group, device=v.device, dtype=torch.int64) * 8
    return ((word.unsqueeze(1) >> byte_shifts) & 0xFF).to(torch.uint8).reshape(-1)

"""Inverse of `pack_signed_integers`.

    Works in chunks of `chunk_groups` byte-aligned groups, so the int64
    temporaries are bounded (~chunk_groups * 8 values * 8 B = 4 MiB) and the
    result is stored as `out_dtype` (int8: 1 byte per value) instead of int64.
    """
def unpack_signed_integers(packed: torch.Tensor, bits: int, num_values: int,
                           out_dtype=torch.int8, chunk_groups: int = 1 << 16) -> torch.Tensor:
    per_group, bytes_per_group = _bit_layout(bits)
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    b = packed.reshape(-1, bytes_per_group)
    n_groups = b.shape[0]
    out = torch.empty(n_groups * per_group, dtype=out_dtype, device=packed.device)
    byte_shifts = torch.arange(bytes_per_group, device=b.device, dtype=torch.int64) * 8
    shifts = torch.arange(per_group, device=b.device, dtype=torch.int64) * bits
    for s in range(0, n_groups, chunk_groups):
        e = min(s + chunk_groups, n_groups)
        word = (b[s:e].to(torch.int64) << byte_shifts).sum(dim=1)
        v = (word.unsqueeze(1) >> shifts) & mask
        out[s * per_group:e * per_group] = ((v ^ sign_bit) - sign_bit).reshape(-1).to(out_dtype)
    return out[:num_values]




"""
# Real int8 activation storage
1.	A subclass of torch.Tensor that really stores a layer's output as int8 and carries the scale with it.
2.	When any torch operation consumes it, __torch_function__ dequantizes it automatically (int8 × scale), so the next layer needs no special handling.
3.	Activation storage is one value per byte even for bit-widths below 8. Only Conv/Linear outputs are held this way, because ReLU6 and residual adds return plain FP32 again.
"""
class QActivation(torch.Tensor):
    """An int8 tensor + per-tensor scale. Any op that consumes it receives the
    dequantized FP tensor; only the int8 storage stays resident in between."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.dtype.__get__:
            return args[0]._out_dtype                      # logical dtype
        if func in _QACT_METADATA_FUNCS:
            with _NoSubclass():
                return func(*args, **kwargs)               # shape/size/... need no dequantization
        args, kwargs = _dequantize_tree((args, kwargs))
        return func(*args, **kwargs)


_QACT_METADATA_FUNCS = (
    torch.Tensor.shape.__get__, torch.Tensor.size, torch.Tensor.dim,
    torch.Tensor.ndimension, torch.Tensor.numel, torch.Tensor.nelement,
    torch.Tensor.device.__get__, torch.Tensor.is_cuda.__get__, torch.Tensor.__len__,
)


def dequantize_activation(x):
    """Return the FP tensor for a QActivation; any other object is returned unchanged."""
    if isinstance(x, QActivation):
        with _NoSubclass():
            q = x.as_subclass(torch.Tensor)
        return q.to(x._out_dtype) * x._qscale
    return x


def _dequantize_tree(obj):
    if isinstance(obj, QActivation):
        return dequantize_activation(obj)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_dequantize_tree(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _dequantize_tree(v) for k, v in obj.items()}
    return obj


def quantize_activation(output: torch.Tensor, scale_value: float, bits: int) -> QActivation:
    """Symmetric quantization of `output` into an int8 container (real storage)."""
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    scale = torch.as_tensor(scale_value, dtype=output.dtype, device=output.device)
    t = output / scale
    t.round_().clamp_(qmin, qmax)
    q = t.to(torch.int8)
    del t
    obj = q.as_subclass(QActivation)
    obj._qscale = scale
    obj._out_dtype = output.dtype
    return obj


"""
# Quantized Version of Conv2d layers so as to accomodate lower bit weights, biases in memory
1.	Replaces nn.Conv2d after weight quantization.
2.	It keeps the weights as bit-packed uint8 (packed_weight) with a bf16 per-output-channel scale, plus the bias and the conv settings (stride, padding, dilation, groups).
3.	In forward it unpacks and dequantizes the weights, then runs a normal F.conv2d. The compute is therefore still FP32; only the storage is compressed.
"""
class QuantizedConv2d(nn.Module):
    """Conv2d whose weight lives in memory as bit-packed integers."""
    def __init__(self, conv: nn.Conv2d, packed, scale, bits, bias):
        super().__init__()
        if conv.padding_mode != "zeros":
            raise NotImplementedError("Only padding_mode='zeros' is supported.")
        self.stride, self.padding = conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups
        self.weight_shape = tuple(conv.weight.shape)
        self.num_values = int(conv.weight.numel())
        self.bits = int(bits)
        self.register_buffer("packed_weight", packed)      # uint8
        self.register_buffer("weight_scale", scale)        # bf16, (out,1,1,1)
        self.register_buffer("bias", bias)                 # fp32 or None

    def dequantize_weight(self, dtype=torch.float32):
        q = unpack_signed_integers(self.packed_weight, self.bits, self.num_values)   # int8
        w = q.view(self.weight_shape).to(dtype)
        del q
        return w.mul_(self.weight_scale.to(dtype))          # in place: no extra FP32 copy

    def forward(self, x):
        x = dequantize_activation(x)
        w = self.dequantize_weight(x.dtype)                 # transient, freed after layer
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def extra_repr(self):
        return f"shape={self.weight_shape}, bits={self.bits}, groups={self.groups}"

"""
# Quantize Linear layers of model so as to accomodate lower bit weights and biases
1.	The same idea as QuantizedConv2d, but for the classifier's nn.Linear.
2.	Packed weights, bf16 scale and bias are stored, and forward dequantizes and calls F.linear.
"""
class QuantizedLinear(nn.Module):
    """Linear layer whose weight lives in memory as bit-packed integers."""
    def __init__(self, linear: nn.Linear, packed, scale, bits, bias):
        super().__init__()
        self.weight_shape = tuple(linear.weight.shape)
        self.num_values = int(linear.weight.numel())
        self.bits = int(bits)
        self.register_buffer("packed_weight", packed)
        self.register_buffer("weight_scale", scale)        # bf16, (out,1)
        self.register_buffer("bias", bias)

    def dequantize_weight(self, dtype=torch.float32):
        q = unpack_signed_integers(self.packed_weight, self.bits, self.num_values)
        w = q.view(self.weight_shape).to(dtype)
        del q
        return w.mul_(self.weight_scale.to(dtype))

    def forward(self, x):
        x = dequantize_activation(x)
        return F.linear(x, self.dequantize_weight(x.dtype), self.bias)

    def extra_repr(self):
        return f"shape={self.weight_shape}, bits={self.bits}"


_FP_LAYERS = (nn.Conv2d, nn.Linear)
_Q_LAYERS = (QuantizedConv2d, QuantizedLinear)
_ALL_LAYERS = _FP_LAYERS + _Q_LAYERS



# Dynamically replaces a specific nested layer inside a PyTorch model
def _set_submodule(model: nn.Module, name: str, new_module: nn.Module):
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new_module)

"""Fold every Conv2d->BatchNorm2d pair (adjacent in a Sequential) into the conv.
    W' = W * gamma/sqrt(var+eps);  b' = beta + (b - mean) * gamma/sqrt(var+eps)
    The BN is replaced by nn.Identity. Returns the names of the folded BN layers.
    (Also works on meta-device models: it only does shape-preserving math.)
"""
def fold_batchnorm(model: nn.Module) -> List[str]:
    folded = []
    for parent_name, parent in list(model.named_modules()):
        if not isinstance(parent, nn.Sequential):
            continue
        keys = list(parent._modules.keys())
        for k0, k1 in zip(keys[:-1], keys[1:]):
            conv, bn = parent._modules[k0], parent._modules[k1]
            if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
                continue
            if (conv.out_channels != bn.num_features or not bn.track_running_stats
                    or bn.running_mean is None):
                continue
            with torch.no_grad():
                factor = (bn.weight if bn.weight is not None
                          else torch.ones_like(bn.running_var)) \
                         / torch.sqrt(bn.running_var + bn.eps)
                beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
                b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
                new_w = conv.weight * factor.view(-1, 1, 1, 1)
                new_b = beta + (b0 - bn.running_mean) * factor
                conv.weight = nn.Parameter(new_w.detach().clone(), requires_grad=False)
                conv.bias = nn.Parameter(new_b.detach().clone(), requires_grad=False)
            parent._modules[k1] = nn.Identity()
            folded.append(f"{parent_name}.{k1}" if parent_name else k1)
    return folded

"""Install forward hooks that replace each listed layer's output by a REAL int8
    `QActivation`. If `stats` is given, per-sample byte counts are recorded once per layer.
"""
def attach_activation_quantizers(model: nn.Module, scales: Dict[str, float], bits: int,
                                 stats: Dict[str, dict] = None):
    handles = []

    def make_hook(name, scale_value):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or isinstance(output, QActivation):
                return output
            if stats is not None and name not in stats:
                n = max(int(output.shape[0]), 1)
                per_sample = output.numel() // n
                stats[name] = {
                    "fp32_bytes_per_sample": per_sample * output.element_size(),
                    "stored_bytes_per_sample": per_sample,                       # int8 container
                    "ideal_packed_bytes_per_sample": math.ceil(per_sample * bits / 8),
                }
            return quantize_activation(output, scale_value, bits)
        return hook

    for name, module in model.named_modules():
        if name in scales:
            handles.append(module.register_forward_hook(make_hook(name, scales[name])))
    return handles



#For Memory measurements
def calculate_memory(model) -> float:
    """Unique tensor storage (params + buffers) of a model in MiB.
    Quantized layers keep uint8 buffers, so this is the REAL resident model size."""
    total_bytes = 0
    visited = set()
    for tensor in model.state_dict().values():
        if not isinstance(tensor, torch.Tensor):
            continue
        try:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key not in visited:
                visited.add(key)
                total_bytes += storage.nbytes()
        except Exception:
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / _MIB


def _malloc_trim():
    """Ask glibc to return freed heap pages to the OS so RSS reflects reality."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass



# To forcefully release Python, PyTorch (GPU), and the C operating system allocator (CPU)
# to immediately release all unused memory back to the OS for accurate RAM/VRAM estimations
def _release_memory(device):
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    _malloc_trim()



""" 
1.	A helper that measures memory for one pipeline stage.
2.	A background thread samples process RSS every 5 ms to get peak RAM, and CUDA peak counters give peak VRAM.
3.	start() and stop() bracket a stage, and stop() returns the stats dictionary.
"""
class _MemoryMeter:
    """RSS sampler (5 ms) + CUDA allocator peak counters for one measured stage."""

    def __init__(self, device, interval=0.005):
        self.device = device
        self.interval = interval
        self.cuda = device.type == "cuda" and torch.cuda.is_available()
        self._proc = psutil.Process(os.getpid())
        self._thread = None

    def start(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._alloc0 = torch.cuda.memory_allocated(self.device) / _MIB
        self._rss0 = self._proc.memory_info().rss / _MIB
        self._peak = self._rss0
        self._stop = threading.Event()

        def sample():
            while not self._stop.wait(self.interval):
                try:
                    cur = self._proc.memory_info().rss / _MIB
                    if cur > self._peak:
                        self._peak = cur
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        self._thread = threading.Thread(target=sample, name="ram-peak-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        self._thread = None
        end = self._proc.memory_info().rss / _MIB
        peak = max(self._peak, end)
        out = {
            "ram_start_mb": self._rss0,
            "ram_end_mb": end,
            "ram_peak_mb": peak,
            "ram_peak_delta_mb": max(0.0, peak - self._rss0),
            "ram_resident_delta_mb": end - self._rss0,
        }
        try:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss      # lifetime OS-exact peak
            out["ram_process_maxrss_mb"] = (r if sys.platform == "darwin" else r * 1024) / _MIB
        except ImportError:
            pass
        if self.cuda:
            torch.cuda.synchronize(self.device)
            out.update({
                "vram_start_allocated_mb": self._alloc0,
                "vram_end_allocated_mb": torch.cuda.memory_allocated(self.device) / _MIB,
                "vram_end_reserved_mb": torch.cuda.memory_reserved(self.device) / _MIB,
                "vram_peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / _MIB,
                "vram_peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / _MIB,
                "vram_total_mb": torch.cuda.get_device_properties(self.device).total_memory / _MIB,
            })
            out["vram_peak_allocated_delta_mb"] = max(0.0, out["vram_peak_allocated_mb"] - self._alloc0)
            out["vram_resident_mb"] = out["vram_end_allocated_mb"] - self._alloc0
        return out


# metric keys that get flattened into scalar W&B metrics per stage (all ram_/vram_ prefixed)
_FLAT_KEYS = ("ram_peak_mb", "ram_peak_delta_mb", "ram_resident_delta_mb", "ram_process_maxrss_mb",
              "vram_peak_allocated_mb", "vram_peak_reserved_mb", "vram_peak_allocated_delta_mb",
              "vram_resident_mb")


def _flatten_stage(stage: str, stats: dict) -> dict:
    """ram_peak_mb + stage '1_fp32_loaded' -> ram_peak_1_fp32_loaded_mb"""
    return {f"{k[:-3]}_{stage}_mb": float(stats[k]) for k in _FLAT_KEYS if k in stats}


def _get_images(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, dict):
        return batch.get("image", batch.get("images"))
    return batch


def _get_labels(batch):
    if isinstance(batch, (list, tuple)) and len(batch) > 1:
        return batch[1]
    if isinstance(batch, dict):
        return batch.get("label", batch.get("labels"))
    raise ValueError("Labels not found in DataLoader batch.")


def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch in loader:
            images = _get_images(batch).to(device)
            labels = _get_labels(batch).to(device)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    if total == 0:
        raise RuntimeError("test_loader produced no samples.")
    return 100.0 * correct / total

"""A DataLoader with persistent_workers=True keeps a live, unpicklable iterator after it
    has been iterated. Return a shallow copy without it (same dataset/settings).
"""
def _pickle_safe_loader(loader):
    if isinstance(loader, torch.utils.data.DataLoader) and getattr(loader, "_iterator", None) is not None:
        loader = copy.copy(loader)
        loader._iterator = None
    return loader


"""Pickle `obj` in the PARENT before spawning. If this fails after a Windows child has
    already started, the child just prints 'EOFError: Ran out of input'. Checking first
    means no child is started and the real reason is reported.
"""
def _require_picklable(obj):
    from multiprocessing.reduction import ForkingPickler
    try:
        ForkingPickler.dumps(obj)
    except Exception as exc:
        raise RuntimeError(f"cannot be pickled for a spawned process "
                           f"({type(exc).__name__}: {exc})") from exc

def _assert_no_meta(model):
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t is not None and t.is_meta:
            raise RuntimeError(f"Tensor '{name}' was never materialised (still on meta device).")


"""Per-output-channel symmetric scale, rounded to bf16 (the stored precision).
    With `clip_search`, a clipping ratio in [0.4, 1.0] minimising squared error is chosen.
"""
def _search_weight_scale(weight: torch.Tensor, bits: int, clip_search: bool) -> torch.Tensor:
    qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    x = weight.detach().to(torch.float32)
    dims = tuple(range(1, x.ndim))
    max_abs = x.abs().amax(dim=dims, keepdim=True)

    def to_stored(s):
        return torch.clamp(s, min=1e-12).to(torch.bfloat16).to(torch.float32)

    if not clip_search:
        return to_stored(max_abs / qmax)
    best_scale = best_err = None
    for ratio in _CLIP_RATIOS:
        cand = to_stored(max_abs * ratio / qmax)
        err = ((torch.clamp(torch.round(x / cand), qmin, qmax) * cand - x) ** 2).sum(dim=dims, keepdim=True)
        if best_err is None:
            best_scale, best_err = cand, err
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, cand, best_scale)
    return best_scale


"""
# Post-training quantizer
1.	The main class: it runs the whole post-training quantization pipeline for one (weight bits, activation bits) pair.
2.	quantize_model() loads the FP32 checkpoint, evaluates it, folds BatchNorm, packs the weights per layer, calibrates activation ranges, and evaluates the quantized model.
3.	It then saves the compressed artifact, measures reloading it in a fresh process, writes the CSV/JSON reports, prints the summary and logs to W&B.
4.	The first and last layers stay at 8 bits (sensitive_layer_bits).
5.	Every stage is wrapped in _stage() so RAM/VRAM is recorded per stage.
"""
class PostTrainingQuantizer:
    calculate_memory = staticmethod(calculate_memory)
    get_images = staticmethod(_get_images)
    get_labels = staticmethod(_get_labels)

    def __init__(
        self,
        model,
        model_path,
        file_name,
        weight_quant_bits,
        activation_quant_bits,
        calibration_loader,
        test_loader,
        calibration_batches=100,
        output_name=None,
        wandb_project=None,
        wandb_entity=None,
        run_name=None,
        fold_bn=True,
        weight_clip_search=True,
        sensitive_layer_bits=8,
        activation_calibration="percentile",   # "percentile" | "minmax"
        activation_percentile=99.99,
        verify_packing=True,
        wandb_mode=None,                        # None -> respect WANDB_MODE env
        builder=None,                           # picklable () -> (model, calib_loader, test_loader)
        measure_artifact_load=True,
    ):
        if weight_quant_bits not in SUPPORTED_BITS:
            raise ValueError(f"weight_quant_bits must be one of {SUPPORTED_BITS}")
        if activation_quant_bits not in SUPPORTED_BITS:
            raise ValueError(f"activation_quant_bits must be one of {SUPPORTED_BITS}")
        if activation_calibration not in ("percentile", "minmax"):
            raise ValueError("activation_calibration must be 'percentile' or 'minmax'")

        self.model_template = copy.deepcopy(model).to("meta") #Create a model template in memory only its architecture without its weights
        self.model = None
        self.model_path = model_path
        self.file_name = file_name
        self.weight_quant_bits = int(weight_quant_bits)
        self.activation_quant_bits = int(activation_quant_bits)
        self.calibration_loader = calibration_loader
        self.test_loader = test_loader
        self.calibration_batches = int(calibration_batches)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_mode = wandb_mode
        self.run_name = run_name
        self.fold_bn = fold_bn
        self.weight_clip_search = weight_clip_search
        self.sensitive_layer_bits = sensitive_layer_bits
        self.activation_calibration = activation_calibration
        self.activation_percentile = float(activation_percentile)
        self.verify_packing = verify_packing
        self.builder = builder
        self.measure_artifact_load = measure_artifact_load

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mem_prefix = "vram" if self.device.type == "cuda" else "ram"
        self.backend = (
            f"{self.device.type.upper()} real-quantized weights (bit-packed) + real int8 activations"
        )
        self.activation_stats: Dict[str, dict] = {}
        self.activation_scales: Dict[str, float] = {}
        self.activation_measure: Dict[str, dict] = {}
        self.activation_hook_handles = []
        self.layer_report: List[dict] = []
        self.weight_payload: Dict[str, dict] = {}
        self.storage_report: Dict[str, float] = {}
        self.folded_bn_names: List[str] = []
        self.quantized_weight_state_keys = set()
        self.memory_stages: Dict[str, dict] = {}
        self.last_report = {}
        self._ram_baseline_mb = 0.0
        self._vram_baseline_mb = 0.0
        self.output_name = output_name or (
            f"Quantized_W{self.weight_quant_bits}A{self.activation_quant_bits}_PTQ.pth"
        )
        self.checkpoint_path = os.path.join(model_path, file_name)

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(self.checkpoint_path)
        if self.calibration_loader is None or self.test_loader is None:
            raise ValueError("calibration_loader and test_loader are required.")
        if self.calibration_batches < 1:
            raise ValueError("calibration_batches must be at least 1.")
        os.makedirs(self.model_path, exist_ok=True)

        print("\nQuantization Device Configuration")
        print("----------------------------------")
        print(f"Selected device: {self.device}")
        if self.device.type == "cuda":
            torch.zeros(1, device=self.device)        # create the CUDA context BEFORE any baseline
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA runtime reported by PyTorch: {torch.version.cuda}")
        else:
            print("CUDA GPU unavailable; using CPU.")

    # Some basic helper functions
    @staticmethod
    def _tensor_bytes(tensor):
        return int(tensor.numel() * tensor.element_size())

    @contextlib.contextmanager
    def _stage(self, name):
        """Measure one stage. Resident numbers are ABSOLUTE vs. the pre-load baseline."""
        meter = _MemoryMeter(self.device)
        meter.start()
        try:
            yield
        finally:
            stats = meter.stop()
            stats["ram_resident_delta_mb"] = stats["ram_end_mb"] - self._ram_baseline_mb
            if "vram_end_allocated_mb" in stats:
                stats["vram_resident_mb"] = stats["vram_end_allocated_mb"] - self._vram_baseline_mb
            self.memory_stages[name] = stats

    def _release_memory(self):
        _release_memory(self.device)

    def load_model(self):
        self.model = copy.deepcopy(self.model_template) #Create a deepcopy/clone of model architecture without weights inside the RAM(memory)
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, nn.Module):
            self.model = checkpoint.cpu()
        else:
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
            if not isinstance(checkpoint, dict):
                raise TypeError("Checkpoint must contain a model or state_dict.")
            checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
            self.model.load_state_dict(checkpoint, strict=True, assign=True)
        del checkpoint
        _assert_no_meta(self.model)
        self.model = self.model.to(self.device).eval()
        self._release_memory()
        return self.model

    # For weights
    def _fold_batchnorm(self):
        self.folded_bn_names = fold_batchnorm(self.model)
        print(f"Folded {len(self.folded_bn_names)} BatchNorm layers into their convolutions.")

    '''Quantize Each Layer By making layers for compact weights instead of F32 '''
    def _quantize_layer_weights(self):
        """Replace every Conv2d/Linear by a packed-integer layer (real compression)."""
        self.weight_payload = {}
        report = []
        layers = [(n, m) for n, m in self.model.named_modules()  #make list of layers aling with their names
                  if n and isinstance(m, _FP_LAYERS) and m.weight is not None]
        if not layers:
            raise RuntimeError("No Conv2d or Linear layers were found to quantize.")

        sensitive = set() #Hold first and last layer as compression them will leads to drastic redution in model accuracy
        if self.sensitive_layer_bits:
            sensitive = {layers[0][0], layers[-1][0]}      # first conv, last linear

        quantized_keys = set()
        for name, module in layers:
            is_sensitive = name in sensitive
            bits = (max(self.weight_quant_bits, self.sensitive_layer_bits)
                    if is_sensitive else self.weight_quant_bits)
            qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))

            #Normalize weigths
            weight = module.weight.detach().clone()
            scale = _search_weight_scale(weight, bits, self.weight_clip_search).to(weight.device)
            q_values = torch.clamp(torch.round(weight.float() / scale), qmin, qmax)
            packed = pack_signed_integers(q_values, bits)   #Bit Packing
            scale_bf16 = scale.to(torch.bfloat16)           # stored precision (exact)

            if self.verify_packing:
                restored = unpack_signed_integers(packed, bits, weight.numel())
                if not torch.equal(restored, q_values.reshape(-1).to(torch.int8)):
                    raise RuntimeError(f"Bit-packing round trip failed for layer {name}")
                del restored

            bias = module.bias.detach().clone() if module.bias is not None else None
            cls = QuantizedConv2d if isinstance(module, nn.Conv2d) else QuantizedLinear
            _set_submodule(self.model, name, cls(module, packed, scale_bf16, bits, bias))

            quantized_keys.update({f"{name}.packed_weight", f"{name}.weight_scale"})
            self.weight_payload[name] = {
                "packed_values": packed,           # same storage as the module buffer
                "scale": scale_bf16,
                "shape": list(weight.shape),
                "bits": bits,
                "num_values": int(weight.numel()),
                "layer_type": module.__class__.__name__,
            }
            report.append({
                "layer": name,
                "layer_type": module.__class__.__name__,
                "weight_shape": list(weight.shape),
                "weight_elements": int(weight.numel()),
                "weight_bits": bits,
                "weight_fp32_bytes": int(weight.numel() * 4),
                "weight_packed_bytes": int(packed.numel()),
                "weight_scale_bytes": self._tensor_bytes(scale_bf16),
                "status": ("weight quantized (per-output-channel)"
                           + (" - sensitive-layer exception" if is_sensitive else "")),
            })
            del weight, q_values

        self.quantized_weight_state_keys = quantized_keys
        self.layer_report = report
        # `layers`/`module` still reference the original FP32 modules: drop them first.
        del layers, module
        self._release_memory()


    #Finding activation ranges which are calibrated based on already weight quantized models
    def _calibrate_activation_ranges(self):
        """Collect per-layer activation clip values on the (weight-quantized) model."""
        self.activation_stats = {}
        handles = []
        use_pct = self.activation_calibration == "percentile"   #To ignore high outliers

        def make_hook(layer_name):              #Hook function for foward hook which is a callback funciton after each layer finishes it's work.
            def hook(_module, _inputs, output):
                if not isinstance(output, torch.Tensor) or output.numel() == 0:
                    return
                flat = output.detach().float().abs().reshape(-1)
                if use_pct:
                    if flat.numel() > 1_000_000:                # torch.quantile size limit
                        flat = flat[:: flat.numel() // 1_000_000 + 1]
                    cur_clip = float(torch.quantile(flat, self.activation_percentile / 100.0).item())
                else:
                    cur_clip = float(flat.max().item())
                entry = self.activation_stats.setdefault(
                    layer_name, {"clip_sum": 0.0, "clip_max": 0.0, "observed_batches": 0})
                entry["clip_sum"] += cur_clip
                entry["clip_max"] = max(entry["clip_max"], cur_clip)
                entry["observed_batches"] += 1
            return hook

        for name, module in self.model.named_modules():
            if name and isinstance(module, _ALL_LAYERS):
                handles.append(module.register_forward_hook(make_hook(name)))  #Register hook with each layer

        self.model.eval()
        try:
            with torch.inference_mode():
                for index, batch in enumerate(self.calibration_loader):
                    if index >= self.calibration_batches:
                        break
                    self.model(_get_images(batch).to(self.device))
        finally:
            for handle in handles:
                handle.remove()

        if not self.activation_stats:
            raise RuntimeError("No Conv2d/Linear activations were captured during calibration.")

        qmax = (1 << (self.activation_quant_bits - 1)) - 1   #Gets the maximum positive integer for the activation bit-width.
        self.activation_scales = {}
        for name, stats in self.activation_stats.items():
            clip = (stats["clip_sum"] / max(stats["observed_batches"], 1)
                    if use_pct else stats["clip_max"])
            self.activation_scales[name] = max(clip / qmax, 1e-12)


# It attaches persistent forward hooks to model layers so that during inference,
# every layer output activation is dynamically quantized into low-bit integers on the fly.

    def _install_activation_quantization_hooks(self):
        self._remove_activation_hooks()
        self.activation_measure = {}
        self.activation_hook_handles = attach_activation_quantizers(
            self.model, self.activation_scales, self.activation_quant_bits,
            stats=self.activation_measure)

    def _remove_activation_hooks(self):
        for handle in self.activation_hook_handles:
            handle.remove()
        self.activation_hook_handles = []

    #To evaluate model accuracy by running it over test dataset
    def evaluate_model(self, model=None):
        return _evaluate(self.model if model is None else model, self.test_loader, self.device)

    # -------------------------- reports and storage accounting --------------------------
    def _build_layer_report(self):
        report_by_name = {item["layer"]: item for item in self.layer_report}
        rows = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            if isinstance(module, _Q_LAYERS):
                row = dict(report_by_name[name])
                meas = self.activation_measure.get(name, {})
                row["activation_bits"] = self.activation_quant_bits if name in self.activation_scales else 32
                row["activation_status"] = "real int8 container (QActivation)"
                row["activation_scale"] = self.activation_scales.get(name, 0.0)
                row["activation_fp32_bytes_per_sample"] = meas.get("fp32_bytes_per_sample", 0)
                row["activation_stored_bytes_per_sample"] = meas.get("stored_bytes_per_sample", 0)
                row["activation_ideal_packed_bytes_per_sample"] = meas.get("ideal_packed_bytes_per_sample", 0)
                rows.append(row)
            elif name in self.folded_bn_names:
                rows.append({
                    "layer": name, "layer_type": "BatchNorm2d", "weight_shape": [],
                    "weight_elements": 0, "weight_bits": "N/A",
                    "status": "folded into preceding conv (no memory kept)",
                    "activation_bits": "N/A", "activation_status": "n/a",
                })
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                rows.append({
                    "layer": name, "layer_type": module.__class__.__name__,
                    "weight_shape": [], "weight_elements": 0, "weight_bits": 32,
                    "status": "left FP32 (not folded)",
                    "activation_bits": 32, "activation_status": "not quantized at this module",
                })
            elif len(list(module.children())) == 0:
                rows.append({
                    "layer": name, "layer_type": module.__class__.__name__,
                    "weight_shape": [], "weight_elements": 0, "weight_bits": "N/A",
                    "status": "no quantizable weights",
                    "activation_bits": 32, "activation_status": "not quantized by this implementation",
                })
        return rows

    #Calculate Storage required
    def _calculate_storage(self, original_tensor_mb, quantized_tensor_mb):
        """Resident-memory metrics come from `calculate_memory`; artifact-payload numbers
        (incl. JSON metadata) are kept only as clearly named artifact estimates."""
        packed_weight_bytes = sum(int(i["packed_values"].numel()) for i in self.weight_payload.values())
        weight_scale_bytes = sum(self._tensor_bytes(i["scale"]) for i in self.weight_payload.values())

        fp32_state_bytes = 0
        bias_bytes = 0
        for key, tensor in self.model.state_dict().items():
            if key not in self.quantized_weight_state_keys:
                fp32_state_bytes += self._tensor_bytes(tensor)
                if key.endswith(".bias"):
                    bias_bytes += self._tensor_bytes(tensor)

        activation_scale_bytes = 4 * len(self.activation_scales)
        metadata = {
            "format": "mobile-net-ptq-packed-v3",
            "weight_bits": self.weight_quant_bits,
            "activation_bits": self.activation_quant_bits,
            "sensitive_layer_bits": self.sensitive_layer_bits,
            "bn_folded": bool(self.folded_bn_names),
            "weight_quantization": "symmetric per-output-channel (bf16 scale)",
            "activation_quantization": "symmetric per-layer, int8 container at runtime",
            "zero_point": "implicit 0 (not stored)",
            "layers": {
                name: {"shape": i["shape"], "bits": i["bits"],
                       "num_values": i["num_values"], "layer_type": i["layer_type"]}
                for name, i in self.weight_payload.items()
            },
            "activation_scales": self.activation_scales,
        }
        metadata_bytes = len(json.dumps(metadata, separators=(",", ":")).encode("utf-8"))
        artifact_payload_bytes = (packed_weight_bytes + weight_scale_bytes + fp32_state_bytes
                                  + activation_scale_bytes + metadata_bytes)

        act_fp32 = sum(m["fp32_bytes_per_sample"] for m in self.activation_measure.values())
        act_stored = sum(m["stored_bytes_per_sample"] for m in self.activation_measure.values())
        act_ideal = sum(m["ideal_packed_bytes_per_sample"] for m in self.activation_measure.values())

        eligible_original_bytes = sum(int(i["num_values"]) * 4 for i in self.weight_payload.values())
        p = self.mem_prefix
        self.storage_report = {
            # ---- headline (resident, measured) ----
            "original_model_size_mb": float(original_tensor_mb),
            "model_size_mb": float(quantized_tensor_mb),
            "compression_ratio": float(original_tensor_mb / max(quantized_tensor_mb, 1e-12)),
            f"{p}_model_fp32_tensor_mb": float(original_tensor_mb),
            f"{p}_model_quantized_tensor_mb": float(quantized_tensor_mb),
            "weight_compression_ratio": float(
                eligible_original_bytes / max(packed_weight_bytes + weight_scale_bytes, 1)),
            # ---- component byte counts ----
            "packed_quantized_weight_bytes": int(packed_weight_bytes),
            "weight_scale_bytes": int(weight_scale_bytes),
            "bias_bytes": int(bias_bytes),
            "unquantized_fp32_state_bytes": int(fp32_state_bytes),
            "activation_scale_bytes": int(activation_scale_bytes),
            "metadata_bytes_not_resident": int(metadata_bytes),
            "artifact_estimated_payload_mb": artifact_payload_bytes / _MIB,
            # ---- activations (int8 container is real; ideal packed is theoretical) ----
            "activation_fp32_bytes_per_sample_total": int(act_fp32),
            "activation_int8_stored_bytes_per_sample_total": int(act_stored),
            "activation_ideal_bitpacked_bytes_per_sample_total": int(act_ideal),
            "activation_compression_ratio_runtime_int8": float(act_fp32 / max(act_stored, 1)),
            "activation_compression_ratio_ideal_bitpacked": float(act_fp32 / max(act_ideal, 1)),
            "bn_layers_folded": len(self.folded_bn_names),
        }
        return metadata

    """Save packed weights + remaining FP32 state (loadable with load_compressed_artifact)."""
    def _save_compressed_artifact(self, metadata):
        fp32_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in self.model.state_dict().items()
            if key not in self.quantized_weight_state_keys
        }
        artifact = {
            "format": "mobile-net-ptq-packed-v3",
            "metadata": metadata,
            "packed_weights": {k: {**v, "packed_values": v["packed_values"].cpu(),
                                   "scale": v["scale"].cpu()}
                               for k, v in self.weight_payload.items()},
            "fp32_state": fp32_state,
            "storage_report": self.storage_report,
        }
        output_path = os.path.join(self.model_path, self.output_name)
        torch.save(artifact, output_path)
        artifact_bytes = os.path.getsize(output_path)
        self.storage_report["disk_quantized_artifact_bytes"] = artifact_bytes
        self.storage_report["disk_quantized_artifact_mb"] = artifact_bytes / _MIB
        print(f"Compressed artifact saved: {output_path}")
        print(f"Serialized artifact size (disk, NOT the headline metric): "
              f"{artifact_bytes / _MIB:.4f} MB")
        return output_path

    """Rebuild the real-quantized model (packed weights in memory) from an artifact.
            Never materialises FP32 Conv/Linear weights: the skeleton lives on the meta device.
            Returns (model, activation_scales, activation_bits). Attach activation quantizers
            with `attach_activation_quantizers(model, scales, bits)`.
    """
    @staticmethod
    def load_compressed_artifact(model_template, artifact_path, device="cpu"):
        device = torch.device(device)
        artifact = torch.load(artifact_path, map_location=device, weights_only=True)
        meta = artifact["metadata"]
        model = copy.deepcopy(model_template).to("meta")
        if meta.get("bn_folded"):
            fold_batchnorm(model)                         # structure only; values come from artifact
        state = artifact["fp32_state"]
        for name, item in artifact["packed_weights"].items():
            old = model.get_submodule(name)
            cls = QuantizedConv2d if isinstance(old, nn.Conv2d) else QuantizedLinear
            bias = state.get(f"{name}.bias")
            _set_submodule(model, name, cls(old, item["packed_values"].to(device),
                                            item["scale"].to(device), item["bits"],
                                            bias.to(device) if bias is not None else None))
        packed_biases = {f"{n}.bias" for n in artifact["packed_weights"]}
        own = model.state_dict()
        rest = {k: v.to(device) for k, v in state.items() if k in own and k not in packed_biases}
        model.load_state_dict(rest, strict=False, assign=True)
        _assert_no_meta(model)
        del artifact, state, rest
        return model.eval(), meta["activation_scales"], meta["activation_bits"]

    # ------------- fresh-process measurement of "load the compressed model" -------------
    def _measure_artifact_load_metrics(self, artifact_path):
        if not self.measure_artifact_load:
            return {}
        source = self.builder if self.builder is not None else (
            self.model_template, _pickle_safe_loader(self.test_loader))
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        proc = ctx.Process(target=_artifact_load_worker,
                           args=(result_queue, source, artifact_path, str(self.device)))
        result = None
        try:
            _require_picklable((source, artifact_path))
            proc.start()
        except Exception as exc:
            warnings.warn(f"Could not start artifact-load subprocess ({exc}); measuring in-process "
                          f"(RSS baseline will be polluted).")
            result = _artifact_load_measure(source, artifact_path, str(self.device))
            result["artifact_load_isolated"] = 0
        else:
            while result is None:
                try:
                    result = result_queue.get(timeout=1.0)
                except queue_mod.Empty:
                    if not proc.is_alive():
                        try:
                            result = result_queue.get(timeout=1.0)
                        except queue_mod.Empty:
                            result = {"error": "artifact-load subprocess died without a result"}
            proc.join()
        if "error" in result:
            warnings.warn(f"Artifact-load measurement failed:\n{result['error']}")
            return {}
        if abs(result["artifact_quantized_accuracy"] - self.storage_report["_quantized_accuracy"]) > 1e-6:
            warnings.warn("Accuracy after loading the artifact differs from the in-process accuracy!")
        return result

    #Save reports for further analysis
    def _save_reports(self, original_accuracy, quantized_accuracy, artifact_path):
        tag = f"W{self.weight_quant_bits}A{self.activation_quant_bits}"
        layer_csv = os.path.join(self.model_path, f"LayerReport_{tag}.csv")
        if self.layer_report:
            fields = sorted({key for row in self.layer_report for key in row})
            with open(layer_csv, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.layer_report)

        result = {
            "weight_quant_bits": self.weight_quant_bits,
            "activation_quant_bits": self.activation_quant_bits,
            "device": str(self.device),
            "backend": self.backend,
            "original_test_accuracy": original_accuracy,
            "quantized_accuracy": quantized_accuracy,
            "accuracy_drop": original_accuracy - quantized_accuracy,
            "artifact_path": artifact_path,
            **{k: v for k, v in self.storage_report.items() if not k.startswith("_")},
        }
        json_path = os.path.join(self.model_path, f"CompressionReport_{tag}.json")
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        self.last_report = result
        return result

    """One W&B run per configuration. Config + metric names match the
        assignment's parallel-coordinates plot (weight_quant_bits,
        activation_quant_bits, compression_ratio, model_size_mb, quantized_accuracy).
        All other memory metrics are prefixed ram_ / vram_, disk metrics disk_."""
    def _log_to_wandb(self, report):
        if not self.wandb_project:
            return
        try:
            import wandb
        except ImportError:
            warnings.warn("wandb is not installed; skipping W&B logging.")
            return

        config = {
            "weight_quant_bits": self.weight_quant_bits,
            "activation_quant_bits": self.activation_quant_bits,
            "calibration_batches": self.calibration_batches,
            "activation_calibration": self.activation_calibration,
            "activation_percentile": self.activation_percentile,
            "fold_bn": self.fold_bn,
            "weight_clip_search": self.weight_clip_search,
            "sensitive_layer_bits": self.sensitive_layer_bits,
            "device": str(self.device),
            "quantization_method": "real in-memory symmetric per-channel weights (bit-packed) "
                                   "+ real int8 per-layer symmetric activations",
        }
        metrics = {k: v for k, v in report.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}

        try:
            run = wandb.init(
                project=self.wandb_project, entity=self.wandb_entity,
                name=self.run_name or f"W{self.weight_quant_bits}A{self.activation_quant_bits}",
                config=config, reinit=True, mode=self.wandb_mode,
            )
        except Exception as exc:                                   # offline / not logged in
            warnings.warn(f"wandb.init failed ({exc}); skipping W&B logging.")
            return

        try:
            wandb.log(metrics)
            run.summary.update(metrics)             # summary values feed Parallel Coordinates

            layer_cols = ["layer", "layer_type", "weight_shape", "weight_bits", "weight_fp32_bytes",
                          "weight_packed_bytes", "weight_scale_bytes", "activation_bits",
                          "activation_scale", "activation_fp32_bytes_per_sample",
                          "activation_stored_bytes_per_sample",
                          "activation_ideal_packed_bytes_per_sample", "status"]
            text_cols = {"layer", "layer_type", "weight_shape", "status"}
            layer_table = wandb.Table(columns=layer_cols)
            for row in self.layer_report:
                if "weight_packed_bytes" not in row:
                    continue
                layer_table.add_data(*[
                    (str(row.get(c, "")) if c in text_cols else row.get(c, 0)) for c in layer_cols
                ])

            sr = self.storage_report
            storage_table = wandb.Table(columns=["component", "bytes"], data=[
                ["packed weights", sr["packed_quantized_weight_bytes"]],
                ["weight scales (bf16)", sr["weight_scale_bytes"]],
                ["biases / other FP32 state", sr["unquantized_fp32_state_bytes"]],
            ])

            stage_keys = sorted({k for v in self.memory_stages.values() for k in v})
            #This table gives peak run usuage for every 6 stage of Run.
            mem_table = wandb.Table(
                columns=["stage"] + stage_keys,
                data=[[stage] + [float(vals.get(k, 0.0)) for k in stage_keys]
                      for stage, vals in self.memory_stages.items()],
            )
            wandb.log({
                "tables/layer_report": layer_table,
                "tables/memory_stages": mem_table,
                "plots/resident_storage_breakdown": wandb.plot.bar(
                    storage_table, "component", "bytes", title="Resident model storage breakdown"),
                "plots/weight_bytes_per_layer": wandb.plot.bar(
                    wandb.Table(columns=["layer", "packed_bytes"], data=[
                        [r["layer"], r["weight_packed_bytes"]] for r in self.layer_report
                        if "weight_packed_bytes" in r]),
                    "layer", "packed_bytes", title="Packed weight bytes per layer"),
            })
        finally:
            run.finish()

    #Quantize model to lower weight bits and activaion bits
    def quantize_model(self):
        print("\nStarting configurable post-training quantization")
        print(f"Configuration: W{self.weight_quant_bits}A{self.activation_quant_bits}")
        print(f"Execution device: {self.device}")

        self.memory_stages = {}
        disk_original_checkpoint_mb = os.path.getsize(self.checkpoint_path) / _MIB

        # Absolute baselines, taken ONCE, before any FP32 weight exists in this process.
        self._release_memory()
        self._ram_baseline_mb = psutil.Process(os.getpid()).memory_info().rss / _MIB
        if self.device.type == "cuda":
            self._vram_baseline_mb = torch.cuda.memory_allocated(self.device) / _MIB

        #Load and Calulate Uncompressed (F32) model's Size in MB and evaluate it's Accuracy
        with self._stage("1_fp32_loaded"):
            self.load_model()
        original_tensor_mb = calculate_memory(self.model)

        with self._stage("2_fp32_evaluated"):
            original_accuracy = self.evaluate_model(self.model)

        folded_accuracy = original_accuracy
        ''' Without folding, you would have to quantize the Conv output to low-bit integers (e.g., INT8),
                convert back to float to run BatchNorm, and quantize again for the next layer. 
                Every extra quantization step introduces rounding noise that degrades model accuracy.'''
        with self._stage("3_bn_folded"):
            if self.fold_bn:
                self._fold_batchnorm()
                folded_accuracy = self.evaluate_model(self.model)
                print(f"FP32 accuracy after BN folding: {folded_accuracy:.2f}% (must match original)")

        with self._stage("4_weights_packed"):
            self._quantize_layer_weights()
        quantized_tensor_mb = calculate_memory(self.model)

        with self._stage("5_activations_calibrated"):    #Finding activation ranges which are calibrated based on already weight quantized models
            self._calibrate_activation_ranges()

        with self._stage("6_compressed_evaluated"):
            self._install_activation_quantization_hooks()
            quantized_accuracy = self.evaluate_model(self.model)

        self.layer_report = self._build_layer_report()
        metadata = self._calculate_storage(original_tensor_mb, quantized_tensor_mb)
        self.storage_report["_quantized_accuracy"] = quantized_accuracy   # internal, not reported
        artifact_path = self._save_compressed_artifact(metadata)

        for stage, stats in self.memory_stages.items():
            self.storage_report.update(_flatten_stage(stage, stats))
        self.storage_report.update({
            "fp32_accuracy_after_bn_fold": float(folded_accuracy),
            "disk_original_checkpoint_mb": float(disk_original_checkpoint_mb),
            "ram_process_baseline_mb": float(self._ram_baseline_mb),
        })
        if self.device.type == "cuda":
            self.storage_report["vram_baseline_allocated_mb"] = float(self._vram_baseline_mb)
            self.storage_report["vram_total_mb"] = float(
                self.memory_stages["6_compressed_evaluated"]["vram_total_mb"])

        # Fresh process: load ONLY the compressed artifact and evaluate it.
        self.storage_report.update(self._measure_artifact_load_metrics(artifact_path))

        report = self._save_reports(original_accuracy, quantized_accuracy, artifact_path)
        self._print_summary(report)
        self._log_to_wandb(report)
        return self.model

    """Print Summary stats on to the console"""
    def _print_summary(self, report):
        p = self.mem_prefix
        g = report.get
        print("\n" + "=" * 66)
        print(f"COMPRESSION RESULTS: W{self.weight_quant_bits}A{self.activation_quant_bits}")
        print("=" * 66)
        print(f"Execution device                       : {self.device}")
        print(f"Original test accuracy                 : {report['original_test_accuracy']:.2f}%")
        print(f"Quantized test accuracy                : {report['quantized_accuracy']:.2f}%")
        print(f"Accuracy drop                          : {report['accuracy_drop']:.2f}%")
        print(f"FP32 model, resident tensors ({p})     : {report['original_model_size_mb']:.4f} MB")
        print(f"Quantized model, resident tensors ({p}) : {report['model_size_mb']:.4f} MB")
        print(f"In-memory compression ratio            : {report['compression_ratio']:.3f}x")
        print(f"Weight CR (Conv/Linear weights only)   : {report['weight_compression_ratio']:.3f}x")
        for stage in ("1_fp32_loaded", "4_weights_packed", "6_compressed_evaluated"):
            if f"ram_peak_{stage}_mb" in report:
                line = (f"[{stage}] RAM peak {report[f'ram_peak_{stage}_mb']:.1f} MB "
                        f"(delta {report[f'ram_peak_delta_{stage}_mb']:.1f})")
                if f"vram_peak_allocated_{stage}_mb" in report:
                    line += (f" | VRAM peak alloc {report[f'vram_peak_allocated_{stage}_mb']:.2f} MB, "
                             f"resident {report[f'vram_resident_{stage}_mb']:.2f} MB")
                print(line)
        if "ram_peak_artifact_load_mb" in report:
            line = (f"[artifact load, fresh process] RAM peak {report['ram_peak_artifact_load_mb']:.1f} MB "
                    f"(delta {report['ram_peak_delta_artifact_load_mb']:.1f})")
            if "vram_peak_allocated_artifact_load_mb" in report:
                line += f" | VRAM peak alloc {report['vram_peak_allocated_artifact_load_mb']:.2f} MB"
            print(line)
        print(f"Activation CR runtime (int8 container) : {report['activation_compression_ratio_runtime_int8']:.3f}x")
        print(f"Activation CR ideal bit-packed (theory): {report['activation_compression_ratio_ideal_bitpacked']:.3f}x")
        print(f"Disk: original checkpoint / artifact   : {report['disk_original_checkpoint_mb']:.4f} / "
              f"{report['disk_quantized_artifact_mb']:.4f} MB (not used for the ratio)")
        print("=" * 66)

        print("\nW&B LOGGED METRICS")
        print("-" * 66)

        for key, value in report.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"{key}: {value}")

# Subprocess workers (module level so they are picklable under "spawn")
def _artifact_load_measure(source, artifact_path, device_str) -> dict:
    """Load ONLY the compressed artifact and evaluate it; return ram_/vram_ metrics."""
    device = torch.device(device_str)
    if callable(source):
        model, _calib, test_loader = source()
    else:
        model, test_loader = source
    template = model.to("meta")                 # drop any FP32 weights before measuring
    del model
    if device.type == "cuda":
        torch.zeros(1, device=device)           # CUDA context exists before the baseline
    _release_memory(device)

    meter = _MemoryMeter(device)
    meter.start()
    q_model, scales, act_bits = PostTrainingQuantizer.load_compressed_artifact(
        template, artifact_path, device)
    handles = attach_activation_quantizers(q_model, scales, act_bits)
    load = meter.stop()
    tensor_mb = calculate_memory(q_model)

    meter.start()
    accuracy = _evaluate(q_model, test_loader, device)
    ev = meter.stop()
    for h in handles:
        h.remove()

    p = "vram" if device.type == "cuda" else "ram"
    out = {"artifact_quantized_accuracy": float(accuracy),
           f"{p}_model_artifact_tensor_mb": float(tensor_mb),
           "artifact_load_isolated": 1}
    out.update(_flatten_stage("artifact_load", load))
    out.update(_flatten_stage("artifact_eval", ev))
    return out


def _artifact_load_worker(result_queue, source, artifact_path, device_str):
    try:
        result_queue.put(_artifact_load_measure(source, artifact_path, device_str))
    except BaseException:
        result_queue.put({"error": traceback.format_exc()})


def _quantize_worker(result_queue, source, w_bits, a_bits, init_kwargs):
    try:
        if callable(source):
            model, calib, test = source()
        else:
            model, calib, test = source
        quantizer = PostTrainingQuantizer(
            model=model, weight_quant_bits=w_bits, activation_quant_bits=a_bits,
            calibration_loader=calib, test_loader=test,
            builder=source if callable(source) else None, **init_kwargs)
        quantizer.quantize_model()
        result_queue.put({"ok": True, "report": quantizer.last_report})
    except BaseException:
        result_queue.put({"ok": False, "error": traceback.format_exc()})


def _run_quantizer_isolated(source, w_bits, a_bits, init_kwargs):
    """Run one configuration in a fresh process. Returns its report, or None if the
    process could not be started (e.g. unpicklable loaders)."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_quantize_worker,
                       args=(result_queue, source, w_bits, a_bits, init_kwargs))
    try:
        _require_picklable((source, init_kwargs))
        proc.start()
    except Exception as exc:
        warnings.warn(f"Could not start quantization subprocess: {exc}")
        return None
    result = None
    while result is None:
        try:
            result = result_queue.get(timeout=1.0)
        except queue_mod.Empty:
            if not proc.is_alive():
                try:
                    result = result_queue.get(timeout=1.0)
                except queue_mod.Empty:
                    result = {"ok": False, "error": "quantization subprocess died without a result"}
    proc.join()
    if not result["ok"]:
        raise RuntimeError(f"W{w_bits}A{a_bits} run failed in subprocess:\n{result['error']}")
    return result["report"]
