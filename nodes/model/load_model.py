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

import torch
import comfy.model_management as model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils
import folder_paths


def _load_diffusion_model_state_dict_fast(
    state_dict,
    model_options=None,
    metadata=None,
    disable_dynamic=False,
    zero_copy=False,
):
    """Core diffusion loader with an opt-in assign=True weight path.

    This mirrors ComfyUI's current ``load_diffusion_model_state_dict`` while
    making the assignment choice explicit.  Keeping the normal ModelPatcher
    class is intentional: only construction is optimized; runtime model
    management keeps its usual non-dynamic behavior.
    """
    model_options = {} if model_options is None else dict(model_options)
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

    if custom_operations is not None:
        model_config.custom_operations = custom_operations
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_state_dict, "")
    model_patcher_type = (
        comfy.model_patcher.ModelPatcher
        if disable_dynamic
        else comfy.model_patcher.CoreModelPatcher
    )
    model_patcher = model_patcher_type(
        model, load_device=load_device, offload_device=offload_device
    )
    if not model_management.is_device_cpu(offload_device):
        model.to(offload_device)

    # assign=True makes parameters reference the already-loaded tensors rather
    # than copying ~30GB of weights a second time.  It is used only by the
    # native/default path; converted and quantized paths retain core behavior.
    assign = bool(
        zero_copy
        and dtype is None
        and custom_operations is None
        and model_config.quant_config is None
    )
    model.load_model_weights(new_state_dict, "", assign=assign)
    left_over = state_dict.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: %s", left_over)
    return model_patcher


def _load_diffusion_model_fast(
    model_path,
    model_options=None,
    zero_copy=False,
):
    model_options = {} if model_options is None else dict(model_options)
    state_dict, metadata = comfy.utils.load_torch_file(
        model_path, return_metadata=True
    )
    model = _load_diffusion_model_state_dict_fast(
        state_dict,
        model_options=model_options,
        metadata=metadata,
        zero_copy=zero_copy,
    )
    if model is None:
        raise RuntimeError(f"ERROR: Could not detect model type of: {model_path}")
    model.cached_patcher_init = (
        _load_diffusion_model_fast,
        (model_path, model_options, zero_copy),
    )
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
        model = _load_diffusion_model_fast(
            model_path,
            model_options=model_options,
            zero_copy=weight_dtype == "default",
        )
        return (model,)
