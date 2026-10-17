"""Fast diffusion-model loader.

ComfyUI normally loads a safetensors state dict into CPU memory and then copies
every tensor into freshly-created model parameters.  For native/default model
weights that second copy is unnecessary: the model can take ownership of the
already-loaded tensors.  This node uses that path only when no dtype conversion
or quantization override is requested, and delegates all other cases to the
core loader so its precision and compatibility behavior stay unchanged.
"""

from __future__ import annotations

import logging
import resource
import time

import torch
import comfy.model_management as model_management
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths
from comfy.utils import ProgressBar


def _runtime_memory():
    """Return cheap process/device memory diagnostics for loader timing logs."""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    parts = [f"rss_max={rss_mb:.0f}MB"]
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        parts.append(
            "cuda{} allocated={:.0f}MB reserved={:.0f}MB".format(
                device,
                torch.cuda.memory_allocated(device) / (1024 * 1024),
                torch.cuda.memory_reserved(device) / (1024 * 1024),
            )
        )
    return ", ".join(parts)


def _direct_load_device(model_path, model_options, zero_copy):
    """Return a GPU target for the native safetensors path when safe.

    Native/default weights can be assigned without conversion.  In high-VRAM
    mode the offload device is the same GPU, so loading safetensors tensors
    there avoids first moving an empty model to the GPU and then replacing
    those parameters with CPU tensors.
    """
    if not zero_copy or model_options.get("dtype") is not None:
        return None
    if not model_path.lower().endswith((".safetensors", ".sft")):
        return None

    load_device = model_options.get("load_device", model_management.get_torch_device())
    offload_device = model_options.get(
        "offload_device", model_management.unet_offload_device()
    )
    if model_management.is_device_cpu(load_device) or model_management.is_device_cpu(offload_device):
        return None
    if torch.device(load_device) != torch.device(offload_device):
        return None
    return load_device


def _estimate_construction_operations(state_dict):
    """Estimate how many parameterized modules the model constructor will create."""
    module_names = set()
    for key in state_dict:
        if "." in key:
            module_names.add(key.rsplit(".", 1)[0])
    return max(1, len(module_names))


def _progress_operations(base_operations, progress, state_dict):
    """Wrap Comfy's operation classes so construction can report real activity.

    Comfy's model constructors create their Linear/Conv/Norm layers through an
    operations class.  Wrapping those classes keeps the core model untouched,
    while turning each layer construction into a progress update.
    """
    expected = _estimate_construction_operations(state_dict)
    state = {"created": 0, "last_value": 10, "last_log_value": 10, "started": time.perf_counter()}

    class ProgressOperations(base_operations):
        pass

    def make_progress_operation(operation):
        class ProgressOperation(operation):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                state["created"] += 1
                value = min(
                    89,
                    10 + int(80 * state["created"] / expected),
                )
                if value != state["last_value"]:
                    state["last_value"] = value
                    progress.update_absolute(value)
                if value >= state["last_log_value"]:
                    logging.info(
                        "[Load Model] construction progress=%d%% layers=%d/%d elapsed=%.1fs %s",
                        value,
                        state["created"],
                        expected,
                        time.perf_counter() - state["started"],
                        _runtime_memory(),
                    )
                    state["last_log_value"] = value + 5

        ProgressOperation.__name__ = operation.__name__
        return ProgressOperation

    for name in dir(base_operations):
        operation = getattr(base_operations, name)
        try:
            is_module = isinstance(operation, type) and issubclass(
                operation, torch.nn.Module
            )
        except TypeError:
            is_module = False
        if is_module:
            setattr(ProgressOperations, name, make_progress_operation(operation))

    return ProgressOperations


def _load_diffusion_model_state_dict_fast(
    state_dict,
    model_options=None,
    metadata=None,
    disable_dynamic=False,
    zero_copy=False,
    progress=None,
):
    """Core diffusion loader with an opt-in assign=True weight path.

    This mirrors ComfyUI's current ``load_diffusion_model_state_dict`` while
    making the assignment choice explicit.  Keeping the normal ModelPatcher
    class is intentional: only construction is optimized; runtime model
    management keeps its usual non-dynamic behavior.
    """
    model_options = {} if model_options is None else dict(model_options)
    direct_load_device = model_options.get("_direct_load_device")
    total_started = time.perf_counter()
    logging.info(
        "[Load Model] state-dict pipeline begin keys=%d %s",
        len(state_dict),
        _runtime_memory(),
    )
    dtype = model_options.get("dtype", None)
    custom_operations = model_options.get("custom_operations", None)

    if custom_operations is None:
        state_dict, metadata = comfy.utils.convert_old_quants(
            state_dict, "", metadata=metadata
        )

    diffusion_model_prefix = comfy.sd.model_detection.unet_prefix_from_state_dict(
        state_dict
    )
    stripped_state_dict = comfy.utils.state_dict_prefix_replace(
        state_dict, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(stripped_state_dict) > 0:
        state_dict = stripped_state_dict
        if custom_operations is None:
            state_dict, metadata = comfy.utils.convert_old_quants(
                state_dict, "", metadata=metadata
            )

    if progress is not None:
        progress.update_absolute(2)
    logging.info(
        "[Load Model] state-dict normalization complete keys=%d elapsed=%.1fs %s",
        len(state_dict),
        time.perf_counter() - total_started,
        _runtime_memory(),
    )

    parameters = comfy.utils.calculate_parameters(state_dict)
    weight_dtype = comfy.utils.weight_dtype(state_dict)
    load_device = model_options.get(
        "load_device", model_management.get_torch_device()
    )
    model_config = comfy.sd.model_detection.model_config_from_unet(
        state_dict, "", metadata=metadata
    )

    if model_config is not None:
        new_state_dict = state_dict
    else:
        new_state_dict = comfy.sd.model_detection.convert_diffusers_mmdit(
            state_dict, ""
        )
        if new_state_dict is not None:
            model_config = comfy.sd.model_detection.model_config_from_unet(
                new_state_dict, ""
            )
            if model_config is None:
                return None
        else:
            model_config = comfy.sd.model_detection.model_config_from_diffusers_unet(
                state_dict
            )
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)
            new_state_dict = {}
            for key in diffusers_keys:
                if key in state_dict:
                    new_state_dict[diffusers_keys[key]] = state_dict.pop(key)
                else:
                    logging.warning("{} {}".format(diffusers_keys[key], key))

    offload_device = model_options.get(
        "offload_device", model_management.unet_offload_device()
    )
    supported_dtypes = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=supported_dtypes,
            weight_dtype=weight_dtype,
        )
    else:
        unet_dtype = dtype

    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(
            None, load_device, model_config.supported_inference_dtypes
        )
    else:
        manual_cast_dtype = model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes
        )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    if custom_operations is not None:
        model_config.custom_operations = custom_operations
    elif progress is not None:
        # BaseModel normally selects these operations internally.  Select the
        # same implementation here, then wrap its layer classes so the long
        # constructor phase reports progress instead of looking frozen.
        base_operations = comfy.ops.pick_operations(
            model_config.unet_config.get("dtype", None),
            model_config.manual_cast_dtype,
            fp8_optimizations=model_config.optimizations.get("fp8", False),
            model_config=model_config,
        )
        model_config.custom_operations = _progress_operations(
            base_operations, progress, new_state_dict
        )

    logging.info(
        "[Load Model] config ready class=%s params=%d state_keys=%d dtype=%s manual_cast=%s load_device=%s offload_device=%s assign_candidate=%s direct_load_device=%s %s",
        type(model_config).__name__,
        parameters,
        len(new_state_dict),
        unet_dtype,
        manual_cast_dtype,
        load_device,
        offload_device,
        bool(zero_copy and dtype is None and custom_operations is None and model_config.quant_config is None),
        direct_load_device,
        _runtime_memory(),
    )

    if progress is not None:
        progress.update_absolute(10)
    construction_started = time.perf_counter()
    logging.info("[Load Model] BEGIN model construction %s", _runtime_memory())
    model = model_config.get_model(new_state_dict, "")
    logging.info(
        "[Load Model] END model construction elapsed=%.1fs model=%s %s",
        time.perf_counter() - construction_started,
        type(model).__name__,
        _runtime_memory(),
    )
    if progress is not None:
        progress.update_absolute(90)
    patcher_started = time.perf_counter()
    logging.info("[Load Model] BEGIN patcher construction %s", _runtime_memory())
    model_patcher_type = (
        comfy.model_patcher.ModelPatcher
        if disable_dynamic
        else comfy.model_patcher.CoreModelPatcher
    )
    model_patcher = model_patcher_type(
        model, load_device=load_device, offload_device=offload_device
    )
    logging.info(
        "[Load Model] END patcher construction elapsed=%.1fs %s",
        time.perf_counter() - patcher_started,
        _runtime_memory(),
    )
    move_started = time.perf_counter()
    logging.info("[Load Model] BEGIN model.to(%s) %s", offload_device, _runtime_memory())
    if not model_management.is_device_cpu(offload_device) and direct_load_device is None:
        model.to(offload_device)
    elif direct_load_device is not None:
        logging.info("[Load Model] skip model.to(%s); weights are already resident on target device", offload_device)
    logging.info(
        "[Load Model] END model.to(%s) elapsed=%.1fs %s",
        offload_device,
        time.perf_counter() - move_started,
        _runtime_memory(),
    )
    if progress is not None:
        progress.update_absolute(94)

    # assign=True makes parameters reference the already-loaded tensors rather
    # than copying ~30GB of weights a second time.  It is used only by the
    # native/default path; converted and quantized paths retain core behavior.
    assign = bool(
        zero_copy
        and dtype is None
        and custom_operations is None
        and model_config.quant_config is None
    )
    weights_started = time.perf_counter()
    logging.info(
        "[Load Model] BEGIN load_model_weights assign=%s keys=%d %s",
        assign,
        len(new_state_dict),
        _runtime_memory(),
    )
    model.load_model_weights(new_state_dict, "", assign=assign)
    logging.info(
        "[Load Model] END load_model_weights elapsed=%.1fs %s",
        time.perf_counter() - weights_started,
        _runtime_memory(),
    )
    if progress is not None:
        progress.update_absolute(99)
    left_over = state_dict.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: %s", left_over)
    logging.info(
        "[Load Model] state-dict pipeline complete elapsed=%.1fs %s",
        time.perf_counter() - total_started,
        _runtime_memory(),
    )
    return model_patcher


def _load_diffusion_model_fast(
    model_path,
    model_options=None,
    zero_copy=False,
    progress=None,
):
    model_options = {} if model_options is None else dict(model_options)
    direct_load_device = _direct_load_device(model_path, model_options, zero_copy)
    if direct_load_device is not None:
        model_options["_direct_load_device"] = direct_load_device
    file_started = time.perf_counter()
    logging.info(
        "[Load Model] BEGIN load_torch_file path=%s device=%s %s",
        model_path,
        direct_load_device or "cpu",
        _runtime_memory(),
    )
    state_dict, metadata = comfy.utils.load_torch_file(
        model_path,
        device=direct_load_device,
        return_metadata=True,
    )
    logging.info(
        "[Load Model] END load_torch_file elapsed=%.1fs keys=%d %s",
        time.perf_counter() - file_started,
        len(state_dict),
        _runtime_memory(),
    )
    if progress is not None:
        progress.update_absolute(1)
    model = _load_diffusion_model_state_dict_fast(
        state_dict,
        model_options=model_options,
        metadata=metadata,
        zero_copy=zero_copy,
        progress=progress,
    )
    if model is None:
        raise RuntimeError(f"ERROR: Could not detect model type of: {model_path}")
    model.cached_patcher_init = (
        _load_diffusion_model_fast,
        (model_path, model_options, zero_copy),
    )
    if progress is not None:
        progress.update_absolute(100)
    return model


class LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "The diffusion model to load."},
                ),
                "weight_dtype": (
                    [
                        "default",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Loads a diffusion model with a zero-copy native-weight path. "
        "Explicit FP8 weight choices use ComfyUI's standard conversion path."
    )

    def load(
        self,
        model_name,
        weight_dtype,
    ):
        dtype_map = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        model_options = {}
        if weight_dtype in dtype_map:
            model_options["dtype"] = dtype_map[weight_dtype]
            logging.info("Setting %s weight dtype to %s", model_name, model_options["dtype"])
        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        progress = ProgressBar(100)
        progress.update_absolute(0)
        model = _load_diffusion_model_fast(
            model_path,
            model_options=model_options,
            zero_copy=weight_dtype == "default",
            progress=progress,
        )
        return (model,)
