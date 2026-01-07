from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import requests
import torch
from PIL import Image


class ProgressBarProtocol(Protocol):
    """Protocol for ComfyUI ProgressBar."""

    def update(self, value: int) -> None: ...


try:
    from comfy.utils import ProgressBar  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - ComfyUI provides this at runtime.
    ProgressBar = None

UPLOAD_URL = "https://photoai.imglarger.com/api/PhoAi/Upload"
STATUS_URL = "https://bgeraser.com/"
STATUS_ACTION = "1af975bb141cc30518bca9d55ab31f1f992fec60"
STATUS_STATE = "%5B%22%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2C%22%2F%22%2C%22refresh%22%5D%7D%2Cnull%2Cnull%2Ctrue%5D"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "bgeraser"
BATCH_SIZE = 20
RATE_LIMIT_SLEEP = 12.0


def build_headers() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
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


def _image_to_jpeg_bytes(
    image: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> bytes:
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
    headers: dict[str, str],
    max_attempts: int = 5,
    rate_limit_sleep: float = RATE_LIMIT_SLEEP,
) -> str:
    data = {"type": "4", "mattValue": "0"}

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            files = {"file": (name, io.BytesIO(image_bytes), "image/jpeg")}
            resp = session.post(
                UPLOAD_URL, headers=headers, files=files, data=data, timeout=30
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                raise
            time.sleep(1.0)
            continue

        payload = resp.json()
        if payload.get("code") == 200:
            return payload["data"]["code"]
        if payload.get("code") == 999 and attempt < max_attempts - 1:
            time.sleep(rate_limit_sleep)
            continue
        raise RuntimeError(f"Upload failed: {payload}")

    if last_exc:
        _ = last_exc
    raise RuntimeError("Upload failed: exhausted retries.")


def _poll_and_download(
    pending: dict[str, tuple[list[int], str]],
    session: requests.Session,
    status_headers: dict[str, str],
    download_headers: dict[str, str],
    results: list[np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None],
    progress_bar: ProgressBarProtocol | None,
    request_timeout: float,
    completed_counter: list[int],
) -> None:
    check_interval = 3.0
    max_checks = 120
    total = len(results)
    failure_counts: dict[str, int] = {}
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
            print(f"BgEraser: status check failed ({exc}); retrying...")  # noqa: T201
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
                    print(f"BgEraser: download failed ({exc}); retrying...")  # noqa: T201
                    continue
                result_array = np.asarray(img, dtype=np.float32) / 255.0
                for idx in indices:
                    results[idx] = result_array
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_path = CACHE_DIR / f"{cache_key}.png"
                    img.save(cache_path, format="PNG")
                except Exception as exc:
                    print(f"BgEraser: cache write failed ({exc})")  # noqa: T201
                completed_counter[0] += len(indices)
                completed = completed_counter[0]
                print(f"BgEraser: downloaded {completed}/{total}")  # noqa: T201
                if progress_bar is not None:
                    progress_bar.update(len(indices))
        time.sleep(check_interval)

    if pending:
        remaining = ", ".join(sorted(pending.keys()))
        raise RuntimeError(f"Timed out waiting for: {remaining}")


class BulkBackgroundRemoverBgEraserNode:
    """Remove backgrounds from a batch of images using bgeraser.com."""

    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("images", "alpha")
    FUNCTION: str = "remove_background"
    CATEGORY: str = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def remove_background(self, images: torch.Tensor, unique_id: str | None = None):
        frames = images.detach().cpu().float()
        if frames.ndim != 4:
            raise ValueError("Expected images with shape (N, H, W, C)")

        session = requests.Session()
        upload_headers, status_headers, download_headers = build_headers()

        pending: dict[str, tuple[list[int], str]] = {}
        pending_by_key: dict[str, str] = {}
        results: list[np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None] = [
            None
        ] * frames.shape[0]
        total_steps = frames.shape[0] * 2
        progress_bar = (
            ProgressBar(total_steps, node_id=unique_id) if ProgressBar else None
        )
        completed_counter = [0]

        def flush_pending() -> None:
            if not pending:
                return
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
            pending.clear()
            pending_by_key.clear()

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
                    print(f"BgEraser: cache hit {completed_counter[0]}/{len(results)}")  # noqa: T201
                    continue
                except Exception as exc:
                    print(f"BgEraser: cache read failed ({exc}); reprocessing...")  # noqa: T201
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
            if len(pending) >= BATCH_SIZE:
                flush_pending()

        flush_pending()

        if any(result is None for result in results):
            raise RuntimeError("Missing output for one or more images.")

        # Type narrowing: after checking no results are None, we can safely cast
        non_null_results: list[np.ndarray[tuple[int, ...], np.dtype[np.floating]]] = [
            r for r in results if r is not None
        ]

        first_shape = non_null_results[0].shape
        if any(result.shape != first_shape for result in non_null_results):
            raise RuntimeError("Output images have mismatched dimensions.")

        stacked = torch.from_numpy(np.stack(non_null_results))
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
