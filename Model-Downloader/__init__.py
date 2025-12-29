import asyncio
import json
import os
import pty
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from aiohttp import web

from server import PromptServer

WEB_DIRECTORY = str(Path(__file__).parent.parent.joinpath("web", "model_downloader"))
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

active_downloads = {}
download_queue = []
current_download = None
event_loop = None


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
    """Analyzes workflows and finds safetensors models."""

    @staticmethod
    def extract_safetensors(workflow):
        """Recursively find all .safetensors references in workflow."""
        found = set()

        def traverse(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    traverse(v)
            elif isinstance(obj, list):
                for v in obj:
                    traverse(v)
            elif (
                isinstance(obj, str)
                and ".safetensors" in obj
                and obj.strip().endswith(".safetensors")
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

    _HF_HOSTS = ("huggingface.co", "hf.co")

    @staticmethod
    def _is_huggingface_url(url):
        try:
            host = urlparse(url).hostname or ""
        except ValueError:
            return False
        return host.endswith(Downloader._HF_HOSTS)

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
    def run_download(url, target_dir, download_id, filename, max_speed_mbps, loop):
        """Execute download with hfdl for HuggingFace, aria2c otherwise."""
        global current_download
        os.makedirs(target_dir, exist_ok=True)

        try:
            if Downloader._is_huggingface_url(url):
                Downloader._download_with_hfdl(url, target_dir, download_id, filename)
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
    def _download_with_hfdl(url, target_dir, download_id, filename):
        """Download using hfdl for HuggingFace URLs."""
        direct_url = Downloader.convert_hf_url(url)
        cmd = [
            "/usr/local/bin/hfdl",
            direct_url,
            "--directory",
            target_dir,
            "--filename",
            filename,
        ]

        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, text=False
        )
        os.close(slave_fd)
        os.set_blocking(master_fd, False)

        buffer = ""
        last_percent = None
        while True:
            if download_id not in active_downloads:
                continue

            try:
                data = os.read(master_fd, 1024)
            except BlockingIOError:
                data = b""

            if not data:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
                continue

            text = data.decode(errors="ignore")
            buffer += text
            if len(buffer) > 1000:
                buffer = buffer[-1000:]

            match = re.search(r"(\d{1,3})%", buffer)
            if match:
                percent = min(100, max(0, int(match.group(1))))
                if percent != last_percent:
                    active_downloads[download_id]["progress"] = str(percent)
                    last_percent = percent

        os.close(master_fd)

        returncode = process.wait()
        final_path = os.path.join(target_dir, filename)
        file_exists = os.path.exists(final_path) and os.path.getsize(final_path) > 0
        success = returncode == 0 or file_exists

        Downloader._finalize(download_id, filename, target_dir, success)

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
        url, target_dir, download_id, filename, max_speed_mbps, loop
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
            Downloader.run_download,
            dl["url"],
            dl["target_dir"],
            dl["download_id"],
            dl["filename"],
            dl["max_speed_mbps"],
            loop,
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


@PromptServer.instance.routes.post("/workflow_checker/download_model")
async def download_model(request):
    try:
        body = await request.json()
        url = body.get("url")
        path = body.get("path")
        filename = body.get("filename")
        max_speed_mbps = body.get("max_speed_mbps")

        if not url or not path:
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
        }

        loop = asyncio.get_event_loop()
        await DownloadQueue.add_to_queue(
            url, target_dir, download_id, filename, max_speed_mbps, loop
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
    home = os.path.expanduser("~")

    for download_id, info in active_downloads.items():
        filename = info.get("filename", "")
        status = info.get("status")

        # If status is "failed", check if file actually exists and mark as completed
        if status == "failed":
            # Try to find the file by checking the download metadata
            # The download_id format is: filename_{task_id}
            # We need to determine the target directory
            models_data, _ = ModelManager.load_models()
            if models_data:
                matched = next(
                    (
                        m
                        for m in models_data
                        if m.get("filename", "").lower() == filename.lower()
                    ),
                    None,
                )
                if matched:
                    model_path = matched.get("path") or matched.get("type", "unknown")
                    target_dir = os.path.join(home, "ComfyUI", "models", model_path)
                    final_path = os.path.join(target_dir, filename)

                    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                        # File exists! Change status to completed
                        active_downloads[download_id]["status"] = "completed"
                        active_downloads[download_id]["progress"] = "100"
                        status = "completed"

        downloads[download_id] = {
            "status": status,
            "progress": info.get("progress", "0"),
            "filename": filename,
        }

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
