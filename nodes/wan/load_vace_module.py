import torch
import folder_paths
import comfy.sd
import comfy.utils

_DTYPE_MAP = {
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e4m3fn_fast": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


class LoadVACEModuleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "The base WanVideo diffusion model."},
                ),
                "vace_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "The WanVideo VACE module to merge into the model."},
                ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_vace"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Loads a WanVideo diffusion model and merges a VACE module into it, "
        "producing a single MODEL output compatible with the rest of the pipeline."
    )

    def load_vace(self, model_name, vace_name, weight_dtype):
        model_options = {}
        if weight_dtype in _DTYPE_MAP:
            model_options["dtype"] = _DTYPE_MAP[weight_dtype]
        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["fp8_optimizations"] = True

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        vace_path = folder_paths.get_full_path_or_raise("diffusion_models", vace_name)

        sd = comfy.utils.load_torch_file(model_path)
        vace_sd = comfy.utils.load_torch_file(vace_path)
        sd.update(vace_sd)
        del vace_sd

        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options)
        return (model,)
