import asyncio
from decimal import Decimal, InvalidOperation
import functools
import glob
import json
import os
import re
import subprocess
import time

from aiohttp import web

from server import PromptServer

WEB_DIRECTORY = "web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

active_downloads = {}
download_queue = []
current_download = None
event_loop = None

# The image's boot-time tiering job downloads hot models outside this node's
# in-memory queue.  Keep a tiny rate sample here so the node can surface that
# real work instead of showing a misleading idle Download button.
_prefetch_samples = {}


class BandwidthThrottler:
    """Token bucket for bandwidth limiting."""

    def __init__(self, max_bytes_per_sec):
        self.max_bytes_per_sec = max_bytes_per_sec
        self.tokens = max_bytes_per_sec if max_bytes_per_sec else 0
        self.last_refill = time.time()

    def throttle(self, bytes_count):
        if not self.max_bytes_per_sec:
            return

        self.tokens = min(
            self.tokens + (elapsed * self.max_bytes_per_sec), self.max_bytes_per_sec * 2
        )
        self.last_refill = now

        if self.tokens < bytes_count:
            sleep_time = (bytes_count - self.tokens) / self.max_bytes_per_sec
            time.sleep(sleep_time)
            self.tokens = 0
            self.last_refill = time.time()
        else:
            self.tokens -= bytes_count


class ModelManager:
    """Handles model.json operations and file system interactions."""

    @staticmethod
    def models_json_path():
        home = os.path.expanduser("~")
        return os.path.join(home, ".config", "comfy", "models.json")

    @staticmethod
    def load_models():
        path = ModelManager.models_json_path()
        folder = os.path.dirname(path)

        if not os.path.isdir(folder):
            return None, "missing_dir"
        if not os.path.exists(path):
            return None, "missing_file"

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                return None, "invalid_format"
            return data, None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def add_model(filename, model_path, size="", url=""):
        path = ModelManager.models_json_path()

        if not os.path.exists(path):
            raise FileNotFoundError("models.json not found")

        with open(path, "r", encoding="utf-8") as f:
            models_data = json.load(f) if os.path.getsize(path) > 0 else []

        if not isinstance(models_data, list):
            models_data = []

        if any(m.get("filename", "").lower() == filename.lower() for m in models_data):
            raise ValueError("Model already exists")

        new_model = {"filename": filename, "path": model_path}
        if size:
            new_model["size"] = size
        if url:
            new_model["url"] = url

        models_data.append(new_model)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(models_data, f, indent=2)


class WorkflowAnalyzer:
    """Analyzes workflows and finds model file references."""

    _MODEL_EXTENSIONS = (".safetensors", ".gguf")

    @staticmethod
    def extract_safetensors(workflow):
        """Recursively find all model file references in workflow."""
        found = set()

        def traverse(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    traverse(v)
            elif isinstance(obj, list):
                for v in obj:
                    traverse(v)
            elif isinstance(obj, str) and obj.strip().lower().endswith(
                WorkflowAnalyzer._MODEL_EXTENSIONS
            ):
                found.add(os.path.basename(obj.strip()))

        for node in workflow.get("nodes", []):
            traverse(node)

        return found

    @staticmethod
    def match_models(model_names):
        """Match found model names to models.json entries."""
        models_data, error = ModelManager.load_models()
        if error:
            return [], error

        home = os.path.expanduser("~")
        results = []

        for model_name in sorted(model_names):
            matched = next(
                (
                    m
                    for m in models_data
                    if m.get("filename", "").lower() == model_name.lower()
                ),
                None,
            )

            if matched:
                model_path = matched.get("path") or matched.get("type", "unknown")
                full_path = os.path.join(
                    home, "ComfyUI", "models", model_path, model_name
                )
                results.append(
                    {
                        "name": model_name,
                        "type": model_path,
                        "size": matched.get("size", "unknown"),
                        "url": matched.get("url", ""),
                        "available": True,
                        "exists": os.path.exists(full_path),
                    }
                )
            else:
                results.append(
                    {
                        "name": model_name,
                        "type": "unknown",
                        "size": None,
                        "available": False,
                        "exists": False,
                    }
                )

        return results, None


class Downloader:
    """Handles model downloads with bandwidth limiting."""

    _UNIT_MULTIPLIERS = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
    }

    @staticmethod
    def convert_hf_url(url):
        """Convert HuggingFace web URL to direct resolve URL."""
        if "/blob/" in url:
            base, file_path = url.split("/blob/", 1)
            return f"{base}/resolve/{file_path}"
        return url

    @staticmethod
    def extract_filename_from_url(url):
        """Extract the original filename from a URL."""
        path = url.split("?")[0]
        return path.split("/")[-1]

    @staticmethod
    def size_to_bytes(value):
        """Parse catalog sizes such as '32.8 GB'; return 0 when unknown."""
        if not value:
            return 0
        match = re.match(r"^\s*([\d.]+)\s*([KMGT]?i?B)\s*$", str(value), re.I)
        if not match:
            return 0
        try:
            return int(
                Decimal(match.group(1))
                * Downloader._UNIT_MULTIPLIERS.get(match.group(2), 0)
            )
        except (InvalidOperation, ValueError):
            return 0

    @staticmethod
    def run_download(url, target_dir, download_id, filename, max_speed_mbps, loop,
                     object_key=None, expected_bytes=0):
        """Execute download with aria2c (HF/direct URL) or the `r2` CLI (R2 object)."""
        global current_download
        os.makedirs(target_dir, exist_ok=True)

        try:
            if object_key:
                Downloader._download_with_r2(
                    object_key, target_dir, download_id, filename, expected_bytes
                )
            else:
                Downloader._download_with_aria2c(
                    url, target_dir, download_id, filename, max_speed_mbps
                )
        except Exception as e:
            if download_id in active_downloads:
                active_downloads[download_id]["status"] = "failed"
                active_downloads[download_id]["error"] = str(e)
        finally:
            current_download = None
            if loop:
                loop.call_soon_threadsafe(DownloadQueue.process_next_sync, loop)

    @staticmethod
    def _download_with_aria2c(url, target_dir, download_id, filename, max_speed_mbps):
        """Download using aria2c and stream progress from its console output."""
        direct_url = Downloader.convert_hf_url(url)
        cmd = [
            "aria2c",
            "--enable-color=false",
            "--summary-interval=1",
            "--console-log-level=warn",
            "--show-console-readout=true",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--continue=true",
            "--file-allocation=none",
            "-x",
            "16",
            "-s",
            "16",
            "-k",
            "1M",
            "-d",
            target_dir,
            "-o",
            filename,
            direct_url,
        ]

        if max_speed_mbps:
            cmd.extend(["--max-download-limit", f"{max_speed_mbps}M"])

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        for line in iter(process.stdout.readline, ""):
            if download_id not in active_downloads:
                continue

            percent = Downloader._parse_aria2c_progress(line)
            if percent is not None:
                active_downloads[download_id]["progress"] = f"{percent:.0f}"

        returncode = process.wait()
        final_path = os.path.join(target_dir, filename)
        success = (
            returncode == 0
            and os.path.exists(final_path)
            and os.path.getsize(final_path) > 0
        )

        Downloader._finalize(download_id, filename, target_dir, success)

    @staticmethod
    def _download_with_r2(object_key, target_dir, download_id, filename, expected_bytes):
        """Download a model straight from the R2 bucket via the `r2` CLI instead of HTTP."""
        final_path = os.path.join(target_dir, filename)
        dest_tmp = final_path + ".part"

        process = subprocess.Popen(
            ["r2", "get", object_key, dest_tmp],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        previous_size = 0
        previous_time = time.monotonic()
        # AWS CLI uses a randomized suffix while multipart-copying to the requested
        # .part path.  Account for both names to expose real byte progress.
        while process.poll() is None:
            if download_id not in active_downloads:
                process.terminate()
                return
            partials = glob.glob(dest_tmp + "*")
            downloaded = max((os.path.getsize(path) for path in partials), default=0)
            now = time.monotonic()
            elapsed = now - previous_time
            if elapsed > 0:
                speed = max(0.0, (downloaded - previous_size) / elapsed)
                active_downloads[download_id]["speed_bytes_per_second"] = speed
            previous_size, previous_time = downloaded, now
            active_downloads[download_id]["bytes_downloaded"] = downloaded
            active_downloads[download_id]["bytes_total"] = expected_bytes
            if expected_bytes:
                active_downloads[download_id]["progress"] = f"{min(99.9, downloaded * 100 / expected_bytes):.1f}"
                speed = active_downloads[download_id].get("speed_bytes_per_second", 0)
                if speed > 0:
                    active_downloads[download_id]["eta_seconds"] = max(0, (expected_bytes - downloaded) / speed)
            else:
                active_downloads[download_id]["progress"] = "5"
            time.sleep(1)

        _, stderr = process.communicate()
        success = (
            process.returncode == 0
            and os.path.exists(dest_tmp)
            and os.path.getsize(dest_tmp) > 0
        )
        if success:
            os.replace(dest_tmp, final_path)
        elif stderr:
            print(f"r2 download failed for {object_key}: {stderr}")

        Downloader._finalize(download_id, filename, target_dir, success)

    @staticmethod
    def _parse_aria2c_progress(line):
        """Parse aria2c progress lines and return percent as float."""
        if "(" in line and "%)" in line:
            match = re.search(r"\((\d+)%\)", line)
            if match:
                return float(match.group(1))

        match = re.search(r"\s([\d.]+)([KMGTP]?i?B)/([\d.]+)([KMGTP]?i?B)\(", line)
        if not match:
            return None

        downloaded = Downloader._to_bytes(match.group(1), match.group(2))
        total = Downloader._to_bytes(match.group(3), match.group(4))
        if total <= 0:
            return None

        return min(100.0, max(0.0, (downloaded / total) * 100))

    @staticmethod
    def _to_bytes(value, unit):
        """Convert size strings like 6.3GiB to bytes."""
        multiplier = Downloader._UNIT_MULTIPLIERS.get(unit)
        if multiplier is None:
            return 0.0
        return float(value) * multiplier

    @staticmethod
    def _finalize(download_id, filename, target_dir, success):
        """Mark download as completed or failed."""
        global current_download
        if download_id in active_downloads:
            if success:
                active_downloads[download_id]["status"] = "completed"
                active_downloads[download_id]["progress"] = "100"
            else:
                active_downloads[download_id]["status"] = "failed"
        current_download = None


class DownloadQueue:
    """Manages download queue to prevent parallel downloads."""

    @staticmethod
    async def add_to_queue(
        url, target_dir, download_id, filename, max_speed_mbps, loop, object_key=None,
        expected_bytes=0
    ):
        """Add download to queue."""
        global current_download

        download_queue.append(
            {
                "url": url,
                "target_dir": target_dir,
                "download_id": download_id,
                "filename": filename,
                "max_speed_mbps": max_speed_mbps,
                "object_key": object_key,
                "expected_bytes": expected_bytes,
            }
        )

        active_downloads[download_id]["status"] = "pending"

        if current_download is None:
            DownloadQueue.process_next_sync(loop)

    @staticmethod
    def process_next_sync(loop):
        """Process next download in queue."""
        global current_download

        if current_download is not None or not download_queue:
            return

        dl = download_queue.pop(0)
        current_download = dl["download_id"]

        active_downloads[dl["download_id"]]["status"] = "downloading"
        active_downloads[dl["download_id"]]["progress"] = "0"

        loop.run_in_executor(
            None,
            functools.partial(
                Downloader.run_download,
                dl["url"],
                dl["target_dir"],
                dl["download_id"],
                dl["filename"],
                dl["max_speed_mbps"],
                loop,
                object_key=dl.get("object_key"),
                expected_bytes=dl.get("expected_bytes", 0),
            ),
        )


# Routes


@PromptServer.instance.routes.get("/workflow_checker/list_models")
async def list_models(request):
    models_data, error = ModelManager.load_models()
    path = ModelManager.models_json_path()

    if error:
        return web.json_response(
            {"ok": False, "reason": error, "path": path}, status=404
        )

    return web.json_response({"ok": True, "path": path, "data": models_data})


@PromptServer.instance.routes.post("/workflow_checker/analyze")
async def analyze_models(request):
    models_data, error = ModelManager.load_models()
    path = ModelManager.models_json_path()

    if error:
        return web.json_response(
            {"ok": False, "reason": error, "path": path}, status=404
        )

    try:
        body = await request.json()
        workflow = body.get("workflow", {}) or {}

        found = WorkflowAnalyzer.extract_safetensors(workflow)
        results, error = WorkflowAnalyzer.match_models(found)

        if error:
            return web.json_response({"ok": False, "reason": error}, status=400)

        return web.json_response({"ok": True, "models": results, "path": path})

    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


@PromptServer.instance.routes.post("/workflow_checker/add_model")
async def add_model(request):
    try:
        body = await request.json()
        ModelManager.add_model(
            body.get("filename"),
            body.get("path"),
            body.get("size", ""),
            body.get("url", ""),
        )
        return web.json_response({"ok": True})
    except FileNotFoundError:
        return web.json_response(
            {"ok": False, "error": "models.json not found"}, status=404
        )
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


def _r2_object_exists(object_key):
    """Synchronous; call via loop.run_in_executor, never directly from a route.

    The `r2` CLI's `ls` always appends a trailing slash to whatever prefix it's
    given (directory-listing semantics), so `r2 ls path/file.ext` silently
    queries the non-existent "directory" path/file.ext/ instead of checking
    the file itself. List the parent directory and match the basename instead.
    """
    if "/" not in object_key:
        return False
    parent, basename = object_key.rsplit("/", 1)
    try:
        # The `r2` CLI re-verifies the Cloudflare token via a live API call on
        # every single invocation (no credential caching) -- ~10-11s observed
        # even for a trivial `ls`, so this needs real headroom above that or
        # every check spuriously times out and silently falls back to HF.
        result = subprocess.run(
            ["r2", "ls", parent + "/"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return any(basename == line.split()[-1] for line in result.stdout.splitlines() if line.strip())


def _prefetch_status():
    """Report tiering's staged downloads in the same shape as node downloads."""
    home = os.path.expanduser("~")
    manifest_path = os.path.join(home, ".cache", "hot-models-manifest.json")
    stage_root = os.path.join(home, ".cache", "r2-model-stage")
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

    now = time.monotonic()
    downloads = {}
    for entry in manifest:
        filename = entry.get("filename")
        model_path = entry.get("path")
        if not filename or not model_path:
            continue
        final_path = os.path.join(home, "ComfyUI", "models", model_path, filename)
        download_id = f"prefetch:{filename}"
        expected = Downloader.size_to_bytes(entry.get("size"))
        if os.path.exists(final_path):
            downloads[download_id] = {
                "status": "completed", "progress": "100", "filename": filename,
                "bytes_downloaded": os.path.getsize(final_path), "bytes_total": expected,
                "source": "hot-model-prefetch",
            }
            continue

        partials = glob.glob(os.path.join(stage_root, model_path, filename + ".part*"))
        if not partials:
            downloads[download_id] = {
                "status": "pending", "progress": "0", "filename": filename,
                "bytes_downloaded": 0, "bytes_total": expected, "source": "hot-model-prefetch",
            }
            continue

        downloaded = max(os.path.getsize(path) for path in partials)
        old_time, old_size = _prefetch_samples.get(download_id, (now, downloaded))
        elapsed = now - old_time
        speed = max(0.0, (downloaded - old_size) / elapsed) if elapsed > 0 else 0.0
        _prefetch_samples[download_id] = (now, downloaded)
        downloads[download_id] = {
            "status": "downloading",
            "progress": f"{min(99.9, downloaded * 100 / expected):.1f}" if expected else "0",
            "filename": filename,
            "bytes_downloaded": downloaded,
            "bytes_total": expected,
            "speed_bytes_per_second": speed,
            "eta_seconds": max(0, (expected - downloaded) / speed) if expected and speed else None,
            "source": "hot-model-prefetch",
        }
    return downloads


@PromptServer.instance.routes.post("/workflow_checker/download_model")
async def download_model(request):
    try:
        body = await request.json()
        url = body.get("url")
        path = body.get("path")
        filename = body.get("filename")
        max_speed_mbps = body.get("max_speed_mbps")
        supplied_size = body.get("size", "")

        if not path or not filename:
            return web.json_response(
                {"ok": False, "error": "path and filename required"}, status=400
            )

        # models.json (this static catalog) never actually carries a "source"
        # field -- it's just filename/path/size/url -- so a check that only
        # trusted entry.get("source") == "r2" here would never fire; R2's hot
        # set is dynamic (tier-models, elsewhere) and this catalog was never
        # kept in sync with it. Ask R2 directly whether the object exists
        # right now instead of trusting stale/absent metadata -- if it's
        # there, it's cheaper and faster to pull from than HF regardless of
        # what this catalog entry happens to say.
        models_data, _ = ModelManager.load_models()
        entry = next(
            (
                m
                for m in (models_data or [])
                if m.get("filename", "").lower() == filename.lower()
            ),
            None,
        )

        object_key = None
        candidate_key = (entry.get("object_key") if entry else None) or (
            f"{(entry.get('path') if entry else path).strip('/')}/{filename}"
        )
        loop = asyncio.get_event_loop()
        if await loop.run_in_executor(None, functools.partial(_r2_object_exists, candidate_key)):
            object_key = candidate_key

        if not object_key and not url:
            return web.json_response(
                {"ok": False, "error": "URL and path required"}, status=400
            )

        download_id = f"{filename}_{id(asyncio.current_task())}"
        home = os.path.expanduser("~")
        target_dir = os.path.join(home, "ComfyUI", "models", path)

        active_downloads[download_id] = {
            "status": "pending",
            "progress": "0",
            "filename": filename,
            "bytes_downloaded": 0,
            "bytes_total": Downloader.size_to_bytes(
                supplied_size or (entry.get("size", "") if entry else "")
            ),
        }

        await DownloadQueue.add_to_queue(
            url, target_dir, download_id, filename, max_speed_mbps, loop,
            object_key=object_key,
            expected_bytes=active_downloads[download_id]["bytes_total"],
        )

        return web.json_response({"ok": True, "download_id": download_id})

    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


@PromptServer.instance.routes.get("/workflow_checker/download_status/{download_id}")
async def download_status(request):
    download_id = request.match_info["download_id"]

    if download_id not in active_downloads:
        return web.json_response(
            {"ok": False, "error": "Download not found"}, status=404
        )

    dl = active_downloads[download_id]
    return web.json_response(
        {
            "ok": True,
            "status": dl["status"],
            "progress": dl.get("progress", "0"),
            "filename": dl.get("filename", ""),
            "error": dl.get("error"),
        }
    )


@PromptServer.instance.routes.get("/workflow_checker/queue_status")
async def queue_status(request):
    """Return current queue state and all active downloads."""
    downloads = {}

    for download_id, info in active_downloads.items():
        filename = info.get("filename", "")
        downloads[download_id] = {
            "status": info.get("status"),
            "progress": info.get("progress", "0"),
            "filename": filename,
            "bytes_downloaded": info.get("bytes_downloaded", 0),
            "bytes_total": info.get("bytes_total", 0),
            "speed_bytes_per_second": info.get("speed_bytes_per_second", 0),
            "eta_seconds": info.get("eta_seconds"),
        }

    # A click-initiated download owns the matching model's display state while it
    # exists; otherwise display the autonomous hot-model prefetch state.
    for download_id, info in _prefetch_status().items():
        if not any(existing.get("filename") == info["filename"] for existing in downloads.values()):
            downloads[download_id] = info

    return web.json_response(
        {
            "ok": True,
            "queue": {
                "current_download_id": current_download,
                "downloads": downloads,
                "queued_count": len(download_queue),
            },
        }
    )
