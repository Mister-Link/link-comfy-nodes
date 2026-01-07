from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

try:
    from comfy.utils import ProgressBar
except Exception:  # pragma: no cover - ComfyUI provides this dsaat runtime.
    ProgressBar = None

UPLOAD_URL = "https://photoai.imglarger.com/api/PhoAi/Upload"
STATUS_URL = "https://bgeraser.com/"
STATUS_ACTION = "1af975bb141cc30518bca9d55ab31f1f992fec60"
STATUS_STATE = "%5B%22%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2C%22%2F%22%2C%22refresh%22%5D%7D%2Cnull%2Cnull%2Ctrue%5D"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "bgeraser"


def build_headers() -> tuple[dict, dict, dict]:
    upload_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:146.0) Gecko/20100101 Firefox/146.0",
        "Origin": "https://bgeraser.com",
        "Referer": "https://bgeraser.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
    }
    download_headers = {
        "User-Agent": upload_headers["User-Agent"],
        "Referer": upload_headers["Referer"],
        "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8",
        "Accept-Language": upload_headers["Accept-Language"],
        "Accept-Encoding": upload_headers["Accept-Encoding"],
    }
    status_headers = {
        "User-Agent": upload_headers["User-Agent"],
        "Accept": "text/x-component",
        "Accept-Language": upload_headers["Accept-Language"],
        "Accept-Encoding": upload_headers["Accept-Encoding"],
        "Referer": upload_headers["Referer"],
        "Origin": upload_headers["Origin"],
        "Content-Type": "text/plain;charset=UTF-8",
        "Next-Action": STATUS_ACTION,
        "Next-Router-State-Tree": STATUS_STATE,
    }
    return upload_headers, status_headers, download_headers


def _image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    img_255 = (image * 255.0).clip(0, 255).astype(np.uint8)
    if img_255.ndim == 2:
        pil_img = Image.fromarray(img_255, mode="L").convert("RGB")
    elif img_255.shape[2] == 4:
        rgba = Image.fromarray(img_255, mode="RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        pil_img = background
    else:
        pil_img = Image.fromarray(img_255[:, :, :3], mode="RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _upload_image(
    image_bytes: bytes,
    name: str,
    session: requests.Session,
    headers: dict,
) -> str:
    data = {"type": "4", "mattValue": "0"}

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            files = {"file": (name, io.BytesIO(image_bytes), "image/jpeg")}
            resp = session.post(
                UPLOAD_URL, headers=headers, files=files, data=data, timeout=30
            )
            resp.raise_for_status()
            break
        except requests.HTTPError as exc:
            last_exc = exc
            if attempt == 2:
                raise
            time.sleep(1.0)
    if last_exc:
        _ = last_exc

    payload = resp.json()
    if payload.get("code") != 200:
        raise RuntimeError(f"Upload failed: {payload}")

    return payload["data"]["code"]


def _poll_and_download(
    pending: Dict[str, Tuple[List[int], str]],
    session: requests.Session,
    status_headers: dict,
    download_headers: dict,
    results: List[Optional[np.ndarray]],
    progress_bar: Optional[object],
    request_timeout: float,
    completed_counter: List[int],
) -> None:
    check_interval = 3.0
    max_checks = 120
    total = len(results)
    failure_counts: Dict[str, int] = {}
    for _ in range(max_checks):
        if not pending:
            return
        codes = list(pending.keys())
        status_body = json.dumps([{"type": 4, "codes": codes}], separators=(",", ":"))
        try:
            status = session.post(
                STATUS_URL,
                headers=status_headers,
                data=status_body,
                timeout=request_timeout,
            )
            status.raise_for_status()
        except requests.RequestException as exc:
            print(f"BgEraser: status check failed ({exc}); retrying...")
            time.sleep(check_interval)
            continue
        data_line = next(
            (line for line in status.text.splitlines() if line.startswith("1:")), None
        )
        if not data_line:
            time.sleep(check_interval)
            continue
        status_payload = json.loads(data_line[2:])
        if status_payload.get("code") != 200:
            raise RuntimeError(f"Status failed: {status_payload}")
        status_data = status_payload.get("data", {})
        urls = status_data.get("downloadUrls", {})
        if isinstance(urls, dict) and urls:
            for code, url in urls.items():
                if code not in pending:
                    continue
                indices, cache_key = pending.pop(code)
                try:
                    download = session.get(
                        url,
                        headers=download_headers,
                        timeout=request_timeout,
                        stream=True,
                    )
                    download.raise_for_status()
                    img = Image.open(io.BytesIO(download.content)).convert("RGBA")
                except requests.RequestException as exc:
                    failures = failure_counts.get(code, 0) + 1
                    failure_counts[code] = failures
                    if failures >= 2:
                        raise RuntimeError(
                            f"BgEraser: download failed twice for {code}: {exc}"
                        ) from exc
                    pending[code] = (indices, cache_key)
                    print(f"BgEraser: download failed ({exc}); retrying...")
                    continue
                result_array = np.asarray(img, dtype=np.float32) / 255.0
                for idx in indices:
                    results[idx] = result_array
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_path = CACHE_DIR / f"{cache_key}.png"
                    img.save(cache_path, format="PNG")
                except Exception as exc:
                    print(f"BgEraser: cache write failed ({exc})")
                completed_counter[0] += len(indices)
                completed = completed_counter[0]
                print(f"BgEraser: downloaded {completed}/{total}")
                if progress_bar is not None:
                    progress_bar.update(len(indices))
        time.sleep(check_interval)

    if pending:
        remaining = ", ".join(sorted(pending.keys()))
        raise RuntimeError(f"Timed out waiting for: {remaining}")


class BulkBackgroundRemoverBgEraserNode:
    """Remove backgrounds from a batch of images using bgeraser.com."""

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "alpha")
    FUNCTION = "remove_background"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def remove_background(self, images: torch.Tensor, unique_id: Optional[str] = None):
        frames = images.detach().cpu().float()
        if frames.ndim != 4:
            raise ValueError("Expected images with shape (N, H, W, C)")

        session = requests.Session()
        upload_headers, status_headers, download_headers = build_headers()

        pending: Dict[str, Tuple[List[int], str]] = {}
        pending_by_key: Dict[str, str] = {}
        results: List[Optional[np.ndarray]] = [None] * frames.shape[0]
        total_steps = frames.shape[0] * 2
        progress_bar = (
            ProgressBar(total_steps, node_id=unique_id) if ProgressBar else None
        )
        completed_counter = [0]

        for idx, img_data in enumerate(frames.numpy()):
            image_bytes = _image_to_jpeg_bytes(img_data)
            cache_key = hashlib.sha256(image_bytes).hexdigest()
            cache_path = CACHE_DIR / f"{cache_key}.png"
            if cache_path.exists():
                try:
                    cached = Image.open(cache_path).convert("RGBA")
                    result_array = np.asarray(cached, dtype=np.float32) / 255.0
                    results[idx] = result_array
                    completed_counter[0] += 1
                    if progress_bar is not None:
                        progress_bar.update(2)
                    print(f"BgEraser: cache hit {completed_counter[0]}/{len(results)}")
                    continue
                except Exception as exc:
                    print(f"BgEraser: cache read failed ({exc}); reprocessing...")
            if cache_key in pending_by_key:
                code = pending_by_key[cache_key]
                pending[code][0].append(idx)
                if progress_bar is not None:
                    progress_bar.update(1)
                continue
            code = _upload_image(
                image_bytes,
                name=f"frame_{idx}.jpg",
                session=session,
                headers=upload_headers,
            )
            pending[code] = ([idx], cache_key)
            pending_by_key[cache_key] = code
            if progress_bar is not None:
                progress_bar.update(1)

        _poll_and_download(
            pending,
            session=session,
            status_headers=status_headers,
            download_headers=download_headers,
            results=results,
            progress_bar=progress_bar,
            request_timeout=60.0,
            completed_counter=completed_counter,
        )

        if any(result is None for result in results):
            raise RuntimeError("Missing output for one or more images.")

        first_shape = results[0].shape
        if any(result.shape != first_shape for result in results):
            raise RuntimeError("Output images have mismatched dimensions.")

        stacked = torch.from_numpy(np.stack(results))
        if stacked.shape[-1] == 4:
            images_out = stacked[..., :3]
            alpha_out = stacked[..., 3]
        else:
            images_out = stacked[..., :3]
            alpha_out = torch.ones(
                (stacked.shape[0], stacked.shape[1], stacked.shape[2]),
                dtype=stacked.dtype,
            )
        return (images_out, alpha_out)
