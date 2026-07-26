import importlib
import importlib.util
import json
import re
import shutil
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

COMFY_ROOT = Path("/data/comfy/ComfyUI")
if str(COMFY_ROOT) not in sys.path and COMFY_ROOT.exists():
    sys.path.insert(0, str(COMFY_ROOT))

import comfy.model_management as model_management

try:
    import folder_paths

    COMFY_OUTPUT_DIR = Path(folder_paths.get_output_directory())
    MODELS_DIR = Path(folder_paths.models_dir) / "hybrid_mamomask"
except Exception:
    COMFY_OUTPUT_DIR = COMFY_ROOT / "output"
    MODELS_DIR = COMFY_ROOT / "models" / "hybrid_mamomask"

CURRENT_DIR = Path(__file__).resolve().parent
_SPACE_REPO_ID = "Chetanaa/T2M"
_SPACE_FILES = [
    "common/*",
    "models/*",
    "models/**/*",
    "utils/*",
    "checkpoints/t2m/**/*",
]
_SPACE_ROOT_FILES = [
    "requirements.txt",
]
_LOCAL_FBX_LIB = CURRENT_DIR / "lib" / "ms_fbx_export.py"
_LOCAL_TEMPLATE_FBX = CURRENT_DIR / "assets" / "boy_Rigging_smplx_tex.fbx"

_BONES_22 = [
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (9, 13), (13, 16), (16, 18), (18, 20),
]


def _sanitize_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return text or "motion"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _ensure_runtime_deps() -> None:
    missing = []
    for module_name in ("clip", "mambapy", "einops", "ftfy", "regex"):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "HybridMaMoMask is missing Python dependencies: "
            f"{', '.join(missing)}. Install the updated link-comfy-nodes requirements first."
        )


def _ensure_model_repo(model_root: Path) -> None:
    from huggingface_hub import snapshot_download

    model_root.mkdir(parents=True, exist_ok=True)
    required = [
        model_root / "models" / "mask_transformer" / "transformer.py",
        model_root / "models" / "vq" / "model.py",
        model_root / "common" / "quaternion.py",
        model_root / "utils" / "motion_process.py",
        model_root / "checkpoints" / "t2m" / "hybrid_mamomask_v1" / "model" / "net_best_fid.tar",
        model_root / "checkpoints" / "t2m" / "res_transformer_v1" / "model" / "net_best_fid.tar",
        model_root / "checkpoints" / "t2m" / "length_estimator" / "model" / "finest.tar",
        model_root / "checkpoints" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "model" / "net_best_fid.tar",
        model_root / "checkpoints" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "meta" / "mean.npy",
        model_root / "checkpoints" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "meta" / "std.npy",
    ]
    if all(path.exists() for path in required):
        return

    print(f"[HybridMaMoMask] Downloading model repo from Hugging Face Space {_SPACE_REPO_ID}")
    snapshot_download(
        repo_id=_SPACE_REPO_ID,
        repo_type="space",
        local_dir=str(model_root),
        allow_patterns=_SPACE_ROOT_FILES + _SPACE_FILES,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("[HybridMaMoMask] Model download complete.")


@contextmanager
def _mounted_space_packages(repo_root: Path):
    package_roots = {
        "models": repo_root / "models",
        "utils": repo_root / "utils",
        "common": repo_root / "common",
    }
    previous = {
        name: {k: v for k, v in list(sys.modules.items()) if k == name or k.startswith(f"{name}.")}
        for name in package_roots
    }
    try:
        for name, root in package_roots.items():
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(root)]
            pkg.__file__ = str(root / "__init__.py")
            sys.modules[name] = pkg
        yield
    finally:
        for name, modules in previous.items():
            for key in [k for k in list(sys.modules) if k == name or k.startswith(f"{name}.")]:
                if key not in modules:
                    del sys.modules[key]
            for key, value in modules.items():
                sys.modules[key] = value
            if not modules and name in sys.modules:
                del sys.modules[name]


@dataclass
class HybridMaMoMaskModelWrapper:
    device: torch.device
    repo_root: Path
    vq_model: torch.nn.Module
    transformer: torch.nn.Module
    residual_transformer: torch.nn.Module
    length_estimator: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray


@dataclass
class HybridMaMoMaskData:
    text: str
    seed: int
    fps: int
    motion_263: np.ndarray
    motion_263_denorm: np.ndarray
    xyz: np.ndarray


class HybridMaMoMaskLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "offload_to_cpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("HYBRID_MAMOMASK_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Motion/T2M"

    def load(self, offload_to_cpu: bool = False):
        _ensure_runtime_deps()
        _ensure_model_repo(MODELS_DIR)

        device = torch.device("cpu") if offload_to_cpu else (
            model_management.get_torch_device() if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"[HybridMaMoMask] Loading models from {MODELS_DIR} on {device}")

        with _mounted_space_packages(MODELS_DIR):
            get_opt = importlib.import_module("utils.get_opt").get_opt
            model_mod = importlib.import_module("models.vq.model")
            transformer_mod = importlib.import_module("models.mask_transformer.transformer")

            checkpoints_dir = MODELS_DIR / "checkpoints"
            dataset_name = "t2m"
            dim_pose = 263

            model_opt = get_opt(
                str(checkpoints_dir / dataset_name / "hybrid_mamomask_v1" / "opt.txt"),
                device=device,
            )
            vq_opt = get_opt(
                str(checkpoints_dir / dataset_name / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "opt.txt"),
                device=device,
            )
            vq_opt.dim_pose = dim_pose
            res_opt = get_opt(
                str(checkpoints_dir / dataset_name / "res_transformer_v1" / "opt.txt"),
                device=device,
            )

            vq_model = model_mod.RVQVAE(
                vq_opt,
                vq_opt.dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm,
            )
            vq_ckpt = torch.load(
                checkpoints_dir / dataset_name / vq_opt.name / "model" / "net_best_fid.tar",
                map_location="cpu",
            )
            vq_model.load_state_dict(vq_ckpt["vq_model"] if "vq_model" in vq_ckpt else vq_ckpt["net"])

            model_opt.num_tokens = vq_opt.nb_code
            model_opt.num_quantizers = vq_opt.num_quantizers
            model_opt.code_dim = vq_opt.code_dim
            res_opt.num_quantizers = vq_opt.num_quantizers
            res_opt.num_tokens = vq_opt.nb_code

            residual_transformer = transformer_mod.ResidualTransformer(
                code_dim=vq_opt.code_dim,
                cond_mode="text",
                latent_dim=res_opt.latent_dim,
                ff_size=res_opt.ff_size,
                num_layers=res_opt.n_layers,
                num_heads=res_opt.n_heads,
                dropout=res_opt.dropout,
                clip_dim=512,
                shared_codebook=vq_opt.shared_codebook,
                cond_drop_prob=res_opt.cond_drop_prob,
                share_weight=res_opt.share_weight,
                clip_version="ViT-B/32",
                opt=res_opt,
            )
            res_ckpt = torch.load(
                checkpoints_dir / dataset_name / res_opt.name / "model" / "net_best_fid.tar",
                map_location="cpu",
            )
            residual_transformer.load_state_dict(res_ckpt["res_transformer"], strict=False)

            transformer = transformer_mod.MaskTransformer(
                code_dim=model_opt.code_dim,
                cond_mode="text",
                latent_dim=model_opt.latent_dim,
                ff_size=model_opt.ff_size,
                num_layers=model_opt.n_layers,
                num_heads=model_opt.n_heads,
                dropout=model_opt.dropout,
                clip_dim=512,
                cond_drop_prob=model_opt.cond_drop_prob,
                clip_version="ViT-B/32",
                opt=model_opt,
            )
            trans_ckpt = torch.load(
                checkpoints_dir / dataset_name / model_opt.name / "model" / "net_best_fid.tar",
                map_location="cpu",
            )
            trans_key = "t2m_transformer" if "t2m_transformer" in trans_ckpt else "trans"
            transformer.load_state_dict(trans_ckpt[trans_key], strict=False)

            length_estimator = model_mod.LengthEstimator(512, 50)
            len_ckpt = torch.load(
                checkpoints_dir / dataset_name / "length_estimator" / "model" / "finest.tar",
                map_location="cpu",
            )
            length_estimator.load_state_dict(len_ckpt["estimator"])

        for module in (vq_model, transformer, residual_transformer, length_estimator):
            module.eval().to(device)

        mean = np.load(
            MODELS_DIR / "checkpoints" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "meta" / "mean.npy"
        )
        std = np.load(
            MODELS_DIR / "checkpoints" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "meta" / "std.npy"
        )

        print("[HybridMaMoMask] All models loaded.")
        return (
            HybridMaMoMaskModelWrapper(
                device=device,
                repo_root=MODELS_DIR,
                vq_model=vq_model,
                transformer=transformer,
                residual_transformer=residual_transformer,
                length_estimator=length_estimator,
                mean=mean,
                std=std,
            ),
        )


class HybridMaMoMaskGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HYBRID_MAMOMASK_MODEL",),
                "text": ("STRING", {"default": "A person walks forward and turns around.", "multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF}),
            },
            "optional": {
                "duration_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 9.8, "step": 0.2, "tooltip": "0 uses the model's length estimator."},
                ),
                "timesteps": ("INT", {"default": 18, "min": 1, "max": 64}),
                "cond_scale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "res_cond_scale": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "topk_filter_thres": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gsample": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYBRID_MAMOMASK_DATA",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "generate"
    CATEGORY = "Motion/T2M"

    def generate(
        self,
        model,
        text,
        seed,
        duration_seconds=0.0,
        timesteps=18,
        cond_scale=4.0,
        res_cond_scale=5.0,
        temperature=1.0,
        topk_filter_thres=0.9,
        gsample=False,
    ):
        text = text.strip()
        if not text:
            raise RuntimeError("Text prompt is required.")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        with _mounted_space_packages(model.repo_root):
            motion_process = importlib.import_module("utils.motion_process")

            prompt_list = [text]
            with torch.no_grad():
                if duration_seconds <= 0:
                    text_embedding = model.transformer.encode_text(prompt_list)
                    pred_dis = model.length_estimator(text_embedding)
                    probs = F.softmax(pred_dis, dim=-1)
                    token_lens = Categorical(probs).sample()
                else:
                    motion_frames = max(4, min(int(duration_seconds * 20), 196))
                    token_lens = torch.LongTensor([motion_frames // 4])

                token_lens = token_lens.to(model.device).long()
                motion_len = int((token_lens * 4)[0].item())

                mids = model.transformer.generate(
                    prompt_list,
                    token_lens,
                    timesteps=timesteps,
                    cond_scale=cond_scale,
                    temperature=temperature,
                    topk_filter_thres=topk_filter_thres,
                    gsample=gsample,
                )
                mids = model.residual_transformer.generate(
                    mids,
                    prompt_list,
                    token_lens,
                    temperature=temperature,
                    cond_scale=res_cond_scale,
                )
                pred_motions = model.vq_model.forward_decoder(mids).detach().cpu().numpy().astype(np.float32)
                motion_denorm = (pred_motions * model.std + model.mean).astype(np.float32)
                xyz = motion_process.recover_from_ric(
                    torch.from_numpy(motion_denorm[0, :motion_len]).float(), 22
                ).numpy().astype(np.float32)

        motion_263 = pred_motions[0, :motion_len]
        motion_263_denorm = motion_denorm[0, :motion_len]
        print(f"[HybridMaMoMask] Generated motion for: {text}")
        return (HybridMaMoMaskData(text, seed, 20, motion_263, motion_263_denorm, xyz),)


class HybridMaMoMaskPreviewAnimation:
    BONES = _BONES_22

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"motion": ("HYBRID_MAMOMASK_DATA",)}}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview"
    CATEGORY = "Motion/T2M"
    OUTPUT_NODE = True

    def preview(self, motion: HybridMaMoMaskData):
        num_frames, num_joints, _ = motion.xyz.shape
        motion_json = json.dumps(
            {
                "xyz": motion.xyz.flatten().tolist(),
                "num_frames": num_frames,
                "num_joints": num_joints,
                "fps": motion.fps,
                "text": motion.text,
                "bones": self.BONES,
            }
        )
        return {"ui": {"motion_data": [motion_json]}, "result": ()}


def _get_rot_matrices_from_motion(motion_263_denorm: np.ndarray, repo_root: Path):
    with _mounted_space_packages(repo_root):
        motion_process = importlib.import_module("utils.motion_process")
        quaternion = importlib.import_module("common.quaternion")

        data = torch.from_numpy(motion_263_denorm).float().unsqueeze(0)
        root_rot_quat, root_pos = motion_process.recover_root_rot_pos(data)
        root_cont6d = quaternion.quaternion_to_cont6d(root_rot_quat)
        joints_num = 22
        start_idx = 1 + 2 + 1 + (joints_num - 1) * 3
        end_idx = start_idx + (joints_num - 1) * 6
        cont6d_params = data[..., start_idx:end_idx]
        cont6d_params = torch.cat([root_cont6d, cont6d_params], dim=-1).view(-1, joints_num, 6)
        rot_mats = quaternion.cont6d_to_matrix(cont6d_params).cpu().numpy().astype(np.float32)
        root_trans = root_pos.squeeze(0).cpu().numpy().astype(np.float32)
        return rot_mats, root_trans


class HybridMaMoMaskExportFBX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("HYBRID_MAMOMASK_DATA",),
                "model": ("HYBRID_MAMOMASK_MODEL",),
                "output_dir": ("STRING", {"default": "hybrid_mamomask"}),
                "filename_prefix": ("STRING", {"default": "motion"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "export"
    CATEGORY = "Motion/T2M"
    OUTPUT_NODE = True

    def export(self, motion, model, output_dir, filename_prefix):
        if not _LOCAL_FBX_LIB.exists() or not _LOCAL_TEMPLATE_FBX.exists():
            raise RuntimeError(
                "HybridMaMoMask FBX export assets were not found in link-comfy-nodes."
            )

        ms_fbx_export = _load_module_from_path("hybrid_mamomask_ms_fbx_export", _LOCAL_FBX_LIB)
        output_root = COMFY_OUTPUT_DIR / output_dir
        output_root.mkdir(parents=True, exist_ok=True)

        stem = f"{_sanitize_filename(filename_prefix)}{time.strftime('%m%d%y%H%M', time.localtime())}"
        fbx_path = output_root / f"{stem}.fbx"

        rot_matrices, root_trans = _get_rot_matrices_from_motion(motion.motion_263_denorm, model.repo_root)
        ok = ms_fbx_export.write_fbx_with_character(
            template_fbx_path=str(_LOCAL_TEMPLATE_FBX),
            rot_matrices=rot_matrices,
            translations=root_trans,
            save_path=str(fbx_path),
            fps=float(motion.fps),
            scale=100.0,
        )
        if not ok:
            raise RuntimeError(f"FBX export failed: {fbx_path}")

        for sidecar in (Path(str(fbx_path) + ".png"), fbx_path.parent / (fbx_path.stem + ".fbm")):
            try:
                if sidecar.is_dir():
                    shutil.rmtree(sidecar, ignore_errors=True)
                elif sidecar.exists():
                    sidecar.unlink()
            except OSError:
                pass

        fbx_filename = fbx_path.name
        fbx_subfolder = str(fbx_path.parent.relative_to(COMFY_OUTPUT_DIR)).replace("\\", "/")
        download_url = f"/view?filename={fbx_filename}&subfolder={fbx_subfolder}&type=output"
        print(f"[HybridMaMoMask] Exported FBX: {fbx_path}")
        return {
            "ui": {
                "text": [f'<a href="{download_url}" download="{fbx_filename}">Download: {fbx_filename}</a>']
            },
            "result": (str(fbx_path),),
        }
