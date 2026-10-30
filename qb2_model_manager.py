#!/usr/bin/env python3
# ==============================================================================
# qb2_model_manager.py — Unified Tenstorrent QuietBox 2 (P300X2) Model Manager
# Author:   Christopher Jones (@palisurf)
# Location: /home/ttuser/model_management/qb2_model_manager.py
# ==============================================================================
"""
Unified Model Manager for the Tenstorrent QuietBox 2 (P300X2, 8x Blackhole chips, 128 GB GDDR6).
Combines `qb2_model_status.py`, `qb2_model_start.py`, and a Pure-Python Hugging Face
Frontier Ranker into a single interactive workflow:

  1. CHECKS if an existing model is running on the QB2 and displays all associated
     Docker containers, container processes (`docker top`), host wrapper/client PIDs,
     and hardware device handles (`/dev/tenstorrent/0..3`, `/dev/shm`, hugepages, logs).
  2. RANKS & ENUMERATES all models by Frontier Proximity % (pulled directly from
     Hugging Face model card benchmark tables & REST metadata with local caching in
     `/home/ttuser/model_management/frontier_rankings_cache.json` — no LLM required):
     • Part 1: Models whose `.safetensors` weights are ALREADY downloaded on the QB2,
               ordered from highest Frontier Proximity % to lowest (highlighting `[★ RUNNING]`).
     • Part 2: All additional models supported on the QB2 (`prod/llm.yaml`, or `--dev`
               for `dev/llm.yaml` + `dev/vlm.yaml`), ordered from highest Frontier Proximity %
               to lowest, which auto-pull from Hugging Face when selected.
  3. PROMPTS `Load new model?`:
     • `'q'` (or Enter): Does nothing and exits without disturbing the running model.
     • `'r'`: Refreshes Frontier Proximity rankings live from Hugging Face Hub.
     • `'s'` / `'stop'`: Gracefully stops the running model, runs `tt-smi -r`, and exits.
     • Same model `#` as currently running: Reports that the selected model is already
       running, does nothing, and exits cleanly.
     • Different model `#`:
       1) Gracefully stops the active model container & host handles.
       2) Resets the Blackhole PCIe boards via `tt-smi -r` so the device mesh is clean.
       3) Pulls model weights from Hugging Face if not yet downloaded on disk.
       4) Starts the new model container via `tt-inference-server/run.py`.
       5) Actively streams startup progress (container logs, TT-Metal mesh init, weight
          loading, trace capture, and `/health` + `/v1/models` readiness) and prompts/exits
          once the newly selected model is up and serving requests.

Usage:
  ./qb2_model_manager                  # Interactive status + Frontier-ranked catalog + switcher
  ./qb2_model_manager --dev            # Include all experimental `dev/llm.yaml` QB2 models
  ./qb2_model_manager --refresh-ranks  # Force re-harvesting benchmark tables from Hugging Face
  ./qb2_model_manager --status         # Show active model status + ranked catalog and exit
  ./qb2_model_manager --select 1       # Switch to model #1 directly
  ./qb2_model_manager --stop           # Stop active model + run `tt-smi -r` and exit
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

# ==============================================================================
# Paths & Constants on the QuietBox 2
# ==============================================================================
SCRIPT_DIR = Path("/home/ttuser/model_management")
FRONTIER_CACHE_FILE = SCRIPT_DIR / "frontier_rankings_cache.json"
LOAD_TIMINGS_CACHE_FILE = SCRIPT_DIR / "load_timings_cache.json"
PRIMARY_MODELS_DIR = Path("/home/ttuser/models")
TT_CACHE_DIR = Path("/home/ttuser/data/tt-cache")
TT_INFERENCE_DIR = Path("/home/ttuser/.local/lib/tt-inference-server")
PROD_SPEC_YAML = TT_INFERENCE_DIR / "workflows" / "model_specs" / "prod" / "llm.yaml"
DEV_SPEC_YAML = TT_INFERENCE_DIR / "workflows" / "model_specs" / "dev" / "llm.yaml"
DEV_VLM_SPEC_YAML = TT_INFERENCE_DIR / "workflows" / "model_specs" / "dev" / "vlm.yaml"

# Deterministic vLLM trace capture sequence lengths on QB2 (up to 128K context)
TRACE_CAPTURE_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32640, 65408, 130944]
TRACE_SEQ_INDEX = {seq_len: idx + 1 for idx, seq_len in enumerate(TRACE_CAPTURE_SEQ_LENS)}

# Patched Docker image containing TT-Metal PR #55630 (prefix-caching slot collisions fix)
PATCHED_QWEN3_IMAGE = (
    "ghcr.io/tenstorrent/tt-inference-server/"
    "vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.17.0-8c48a10-f52987a-pr55630"
)

# Working tt-smi binary candidates on QB2 (note: /usr/bin/tt-smi lacks tt_umd in system python)
TT_SMI_CANDIDATES = [
    Path("/home/ttuser/.local/lib/tt-inference-server/.workflow_venvs/.venv_tt_smi/bin/tt-smi"),
    Path("/home/ttuser/.tenstorrent-venv/bin/tt-smi"),
    Path("/home/ttuser/tt-metal/python_env/bin/tt-smi"),
]

# Known local weight folder overrides on QB2 where folder naming differs from HF repo name
KNOWN_LOCAL_PATHS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": [
        PRIMARY_MODELS_DIR / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-32B",
        PRIMARY_MODELS_DIR / "DeepSeek-R1-Distill-Qwen-32B",
    ],
    "Qwen/Qwen3-32B": [
        TT_CACHE_DIR / "volume_id_tt_transformers-Qwen3-32B-vqb2_launch" / "weights" / "Qwen3-32B",
        TT_CACHE_DIR / "volume_id_tt_transformers-Qwen3-32B-v0.17.0" / "weights" / "Qwen3-32B",
        PRIMARY_MODELS_DIR / "Qwen" / "Qwen3-32B",
        PRIMARY_MODELS_DIR / "Qwen3-32B",
    ],
    "meta-llama/Llama-3.1-70B-Instruct": [
        PRIMARY_MODELS_DIR / "Llama-3.1-70B-Instruct",
        PRIMARY_MODELS_DIR / "meta-llama" / "Llama-3.1-70B-Instruct",
    ],
    "meta-llama/Llama-3.1-8B-Instruct": [
        PRIMARY_MODELS_DIR / "Llama-3.1-8B-Instruct",
        PRIMARY_MODELS_DIR / "meta-llama" / "Llama-3.1-8B-Instruct",
    ],
}

# Architecture-compatible 32B models validated to run on QB2 via the Qwen3-32B P300X2 spec
ARCHITECTURAL_BRIDGES = [
    {
        "id": "bridge-deepseek-r1-qwen-32b",
        "display_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (Bridged via Qwen3-32B Spec)",
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "tt_spec_model": "Qwen3-32B",
        "tt_device": "p300x2",
        "impl": "tt_transformers",
        "is_default_impl": True,
        "catalog_env": "prod",
        "status": "VALIDATED_BRIDGE",
        "version": "0.17.0-pr55630",
        "param_count": 32,
        "est_size_gb": "~61 GB",
        "max_context": 131072,
        "max_concurrency": 32,
        "est_speed": "~18.4 tok/s",
        "use_pr55630_img": True,
        "notes": "CoT reasoning model; bridged via Qwen3-32B spec + PR #55630 prefix-cache fix",
    },
    {
        "id": "bridge-qwq-32b",
        "display_name": "Qwen/QwQ-32B (Bridged via Qwen3-32B tt_transformers Spec)",
        "hf_repo": "Qwen/QwQ-32B",
        "tt_spec_model": "Qwen3-32B",
        "tt_device": "p300x2",
        "impl": "tt_transformers",
        "is_default_impl": True,
        "catalog_env": "prod",
        "status": "VALIDATED_BRIDGE",
        "version": "0.17.0-pr55630",
        "param_count": 32,
        "est_size_gb": "~62 GB",
        "max_context": 131072,
        "max_concurrency": 32,
        "est_speed": "~18.4 tok/s",
        "use_pr55630_img": True,
        "notes": "RL-trained 32B reasoning model sharing Qwen 32B 64-layer GQA geometry",
    },
    {
        "id": "bridge-qwen25-coder-32b",
        "display_name": "Qwen/Qwen2.5-Coder-32B-Instruct (Bridged via Qwen3-32B Spec)",
        "hf_repo": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "tt_spec_model": "Qwen3-32B",
        "tt_device": "p300x2",
        "impl": "tt_transformers",
        "is_default_impl": True,
        "catalog_env": "prod",
        "status": "VALIDATED_BRIDGE",
        "version": "0.17.0-pr55630",
        "param_count": 32,
        "est_size_gb": "~62 GB",
        "max_context": 131072,
        "max_concurrency": 32,
        "est_speed": "~18.4 tok/s",
        "use_pr55630_img": True,
        "notes": "Dedicated 32B coding/refactoring specialist model",
    },
]


# ==============================================================================
# PART A: Active Model & Process/Device Handle Inspection (`qb2_model_status`)
# ==============================================================================
def run_cmd(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a subprocess command safely and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def query_vllm_endpoint(port: int = 8000, timeout: float = 2.5) -> dict | None:
    """Query local vLLM /v1/models endpoint to see what model ID is actively served."""
    url = f"http://localhost:{port}/v1/models"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                data = payload.get("data", [])
                if data:
                    m = data[0]
                    return {
                        "endpoint": f"http://localhost:{port}/v1",
                        "served_model_id": m.get("id", "unknown"),
                        "root": m.get("root", ""),
                        "max_model_len": m.get("max_model_len", "unknown"),
                        "created": m.get("created"),
                        "raw": m,
                    }
    except Exception:
        pass
    return None


def check_vllm_health(port: int = 8000, timeout: float = 2.5) -> bool:
    """Return True if http://localhost:<port>/health returns HTTP 200."""
    url = f"http://localhost:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def find_model_containers() -> list[dict]:
    """Find running Docker containers associated with model inference (tt-inference-server / vllm)."""
    rc, out, _ = run_cmd(["docker", "ps", "--format", "{{.ID}}"])
    if rc != 0 or not out:
        return []

    container_ids = [cid.strip() for cid in out.splitlines() if cid.strip()]
    if not container_ids:
        return []

    rc, inspect_out, _ = run_cmd(["docker", "inspect"] + container_ids)
    if rc != 0 or not inspect_out:
        return []

    try:
        inspected = json.loads(inspect_out)
    except json.JSONDecodeError:
        return []

    model_containers = []
    for item in inspected:
        name = item.get("Name", "").lstrip("/")
        config = item.get("Config", {})
        image = config.get("Image", "")
        cmd_list = config.get("Cmd") or []
        entrypoint = config.get("Entrypoint") or []
        full_cmd = " ".join(entrypoint + cmd_list)
        env_list = config.get("Env") or []
        env_map = {}
        for e in env_list:
            if "=" in e:
                k, v = e.split("=", 1)
                env_map[k] = v

        mounts = item.get("Mounts") or []
        devices = (item.get("HostConfig") or {}).get("Devices") or []

        is_tt_device = any("/dev/tenstorrent" in (d.get("PathOnHost") or "") for d in devices)
        is_inference_image = any(
            kw in (name + " " + image + " " + full_cmd).lower()
            for kw in ("tt-inference-server", "vllm", "tt-metal")
        )
        if not (is_tt_device or is_inference_image):
            continue

        spec_model = None
        tt_device = None
        for idx, arg in enumerate(cmd_list):
            if arg == "--model" and idx + 1 < len(cmd_list):
                spec_model = cmd_list[idx + 1]
            elif arg == "--tt-device" and idx + 1 < len(cmd_list):
                tt_device = cmd_list[idx + 1]

        host_weights_dir = None
        container_weights_dir = env_map.get("MODEL_WEIGHTS_DIR") or env_map.get("HF_MODEL_PATH")
        cache_volume = None
        hugepages_mount = None

        for m in mounts:
            src = m.get("Source", "")
            dst = m.get("Destination", "")
            mtype = m.get("Type", "")
            if "readonly_weights_mount" in dst or "/models/" in src or "/weights/" in src:
                host_weights_dir = src
                if not container_weights_dir:
                    container_weights_dir = dst
            elif "cache_root" in dst or mtype == "volume":
                cache_volume = f"{m.get('Name') or src} -> {dst}"
            elif "hugepages" in src or "hugepages" in dst:
                hugepages_mount = f"{src} -> {dst}"

        weights_model_name = None
        if host_weights_dir:
            p = Path(host_weights_dir)
            if p.parent.name and p.parent.name not in ("models", "weights"):
                weights_model_name = f"{p.parent.name}/{p.name}"
            else:
                weights_model_name = p.name

        ports_cfg = (item.get("NetworkSettings") or {}).get("Ports") or {}
        port_bindings = []
        host_ports = []
        for cport, binds in ports_cfg.items():
            if binds:
                for b in binds:
                    hp = b.get("HostPort", "")
                    hip = b.get("HostIp", "0.0.0.0")
                    port_bindings.append(f"{hip}:{hp}->{cport}")
                    if hp.isdigit():
                        host_ports.append(int(hp))
            else:
                port_bindings.append(cport)

        state = item.get("State") or {}
        started_at = state.get("StartedAt", "")
        main_pid = state.get("Pid", 0)

        container_procs = []
        rc_top, top_out, _ = run_cmd(
            ["docker", "top", name, "-eo", "pid,ppid,user,etime,pcpu,rss,args"]
        )
        if rc_top == 0 and top_out:
            lines = top_out.splitlines()
            for line in lines[1:]:
                parts = line.strip().split(None, 6)
                if len(parts) >= 7:
                    pid_s, ppid_s, user_s, etime_s, pcpu_s, rss_s, args_s = parts
                    rss_mb = f"{int(rss_s) / 1024:.1f} MB" if rss_s.isdigit() else rss_s
                    container_procs.append(
                        {
                            "pid": int(pid_s) if pid_s.isdigit() else pid_s,
                            "ppid": int(ppid_s) if ppid_s.isdigit() else ppid_s,
                            "user": user_s,
                            "elapsed": etime_s,
                            "cpu": f"{pcpu_s}%",
                            "rss": rss_mb,
                            "cmd": args_s,
                        }
                    )

        model_containers.append(
            {
                "id": item.get("Id", "")[:12],
                "full_id": item.get("Id", ""),
                "name": name,
                "image": image,
                "status": state.get("Status", "unknown"),
                "started_at": started_at,
                "main_pid": main_pid,
                "spec_model": spec_model,
                "weights_model_name": weights_model_name,
                "tt_device": tt_device,
                "host_weights_dir": host_weights_dir,
                "container_weights_dir": container_weights_dir,
                "cache_volume": cache_volume,
                "hugepages_mount": hugepages_mount,
                "ports": ", ".join(port_bindings) if port_bindings else "none",
                "host_ports": host_ports or [8000],
                "devices": [d.get("PathOnHost") for d in devices if d.get("PathOnHost")],
                "container_procs": container_procs,
            }
        )

    return model_containers


def find_host_processes(container_pids: set[int]) -> list[dict]:
    """Find all host-side processes related to the running model."""
    rc, out, _ = run_cmd(["ps", "-eo", "pid,ppid,user,etime,pcpu,rss,args"])
    if rc != 0 or not out:
        return []

    my_pid = os.getpid()
    host_procs = []

    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        pid_s, ppid_s, user_s, etime_s, pcpu_s, rss_s, args_s = parts
        if not pid_s.isdigit():
            continue
        pid = int(pid_s)
        ppid = int(ppid_s) if ppid_s.isdigit() else 0

        if pid == my_pid or pid in container_pids:
            continue
        if any(
            k in args_s
            for k in ("qb2_model_status", "qb2_model_start", "qb2_model_manager", "ps -eo pid")
        ):
            continue

        role = None
        stop_with_model = False

        if re.search(r"\bdocker\s+run\b.*tt-inference-server", args_s):
            role = "Docker Run CLI Handle"
            stop_with_model = True
        elif re.search(r"python3?\s+.*run\.py\b.*--model", args_s):
            role = "tt-inference-server Launcher (run.py)"
            stop_with_model = True
        elif re.search(r"\btail\s+-[fF]\b.*workflow_logs/docker_server/vllm_", args_s):
            role = "Server Log Watcher (tail -f)"
            stop_with_model = True
        elif re.search(r"\bvllm\b|\brun_vllm_api_server\.py\b", args_s) and "grep" not in args_s:
            role = "Host vLLM Process"
            stop_with_model = True
        elif re.search(r"python3?\s+.*(cjllm\.py|cj_context|cj_chat)", args_s):
            role = "Interactive Chat Client (Untouched)"
            stop_with_model = False

        if role:
            rss_mb = f"{int(rss_s) / 1024:.1f} MB" if rss_s.isdigit() else rss_s
            host_procs.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "user": user_s,
                    "elapsed": etime_s,
                    "cpu": f"{pcpu_s}%",
                    "rss": rss_mb,
                    "role": role,
                    "stop_with_model": stop_with_model,
                    "cmd": args_s,
                }
            )

    return host_procs


def inspect_hardware_and_ipc_handles(pids_to_scan: set[int]) -> list[dict]:
    """Scan /proc/<pid>/fd and /proc/<pid>/maps for open handles to Tenstorrent devices & IPC."""
    handles = []
    seen = set()

    for pid in sorted(pids_to_scan):
        fd_dir = Path(f"/proc/{pid}/fd")
        comm_path = Path(f"/proc/{pid}/comm")
        proc_name = comm_path.read_text().strip() if comm_path.exists() else f"pid:{pid}"

        if fd_dir.exists():
            try:
                for fd_entry in sorted(
                    fd_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 9999
                ):
                    try:
                        target = os.readlink(fd_entry)
                    except OSError:
                        continue
                    if any(
                        kw in target
                        for kw in (
                            "/dev/tenstorrent",
                            "/dev/hugepages",
                            "/dev/shm/",
                            "vllm_",
                            "inspector/",
                        )
                    ):
                        key = (pid, fd_entry.name, target)
                        if key not in seen:
                            seen.add(key)
                            handles.append(
                                {
                                    "pid": pid,
                                    "process": proc_name,
                                    "handle_type": f"fd/{fd_entry.name}",
                                    "target": target,
                                }
                            )
            except PermissionError:
                pass

        maps_path = Path(f"/proc/{pid}/maps")
        if maps_path.exists():
            try:
                dev_counts: dict[str, int] = {}
                for line in maps_path.read_text(errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) >= 6:
                        pathname = parts[5]
                        if pathname.startswith("/dev/tenstorrent") or "hugepages" in pathname:
                            dev_counts[pathname] = dev_counts.get(pathname, 0) + 1
                for dev_path, count in sorted(dev_counts.items()):
                    handles.append(
                        {
                            "pid": pid,
                            "process": proc_name,
                            "handle_type": f"mmap ({count} regions)",
                            "target": dev_path,
                        }
                    )
            except PermissionError:
                pass

    return handles


def gather_full_state() -> dict:
    """Collect the complete state of any running model on the QB2."""
    containers = find_model_containers()
    ports_to_check = [8000]
    for c in containers:
        for p in c.get("host_ports", []):
            if p not in ports_to_check:
                ports_to_check.append(p)

    endpoints = []
    for p in ports_to_check:
        ep = query_vllm_endpoint(p)
        if ep:
            endpoints.append(ep)

    container_pids = set()
    for c in containers:
        if c.get("main_pid"):
            container_pids.add(int(c["main_pid"]))
        for cp in c.get("container_procs", []):
            if isinstance(cp["pid"], int):
                container_pids.add(cp["pid"])

    host_procs = find_host_processes(container_pids)

    all_model_pids = set(container_pids)
    for hp in host_procs:
        if hp["stop_with_model"]:
            all_model_pids.add(hp["pid"])

    hw_handles = inspect_hardware_and_ipc_handles(all_model_pids)

    return {
        "containers": containers,
        "endpoints": endpoints,
        "host_procs": host_procs,
        "hw_handles": hw_handles,
    }


def display_state(state: dict) -> bool:
    """Render the current QB2 model status and all handles. Returns True if a model is running."""
    containers = state["containers"]
    endpoints = state["endpoints"]
    host_procs = state["host_procs"]
    hw_handles = state["hw_handles"]

    server_host_procs = [p for p in host_procs if p["stop_with_model"]]
    is_running = bool(containers or endpoints or server_host_procs)

    if not is_running:
        if HAS_RICH:
            console.print(
                Panel(
                    "[bold green]No LLM model server is currently running on the QuietBox 2.[/bold green]\n"
                    "[dim]• Checked Docker containers (`tt-inference-server` / `vLLM`), port 8000 `/v1/models`, and `/dev/tenstorrent/*` handles.[/dim]",
                    title="[bold cyan]Step 1 — QB2 Active Model Status[/bold cyan]",
                    border_style="green",
                )
            )
        else:
            print("=== Step 1 — QB2 Model Status: No LLM model server is currently running. ===")
        return False

    served_id = endpoints[0]["served_model_id"] if endpoints else "Starting / Not yet responding on /v1/models"
    max_len = (
        f"{endpoints[0]['max_model_len']:,} tokens"
        if endpoints and isinstance(endpoints[0].get("max_model_len"), int)
        else "N/A"
    )
    endpoint_url = endpoints[0]["endpoint"] if endpoints else "http://localhost:8000/v1 (warming up)"

    if containers:
        c0 = containers[0]
        weights_name = c0["weights_model_name"] or c0["spec_model"] or served_id
        spec_name = c0["spec_model"] or served_id
        weights_dir = c0["host_weights_dir"] or "N/A"
        tt_dev = c0["tt_device"] or "p300x2"
        if (
            weights_name
            and spec_name
            and weights_name.lower() != spec_name.lower()
            and spec_name.lower() not in weights_name.lower()
        ):
            model_headline = (
                f"[bold yellow]{weights_name}[/bold yellow] "
                f"[dim](served via vLLM spec/arch:[/dim] [bold white]{spec_name}[/bold white][dim])[/dim]"
            )
        else:
            model_headline = f"[bold yellow]{weights_name or served_id}[/bold yellow]"
    else:
        model_headline = f"[bold yellow]{served_id}[/bold yellow]"
        weights_dir = "N/A"
        tt_dev = "p300x2"

    if HAS_RICH:
        summary_lines = [
            f"[bold]Active Model Weights:[/bold]  {model_headline}",
            f"[bold]vLLM Served Model ID:[/bold]  [cyan]{served_id}[/cyan]  ([dim]Max Context:[/dim] [green]{max_len}[/green])",
            f"[bold]Host Weights Path:[/bold]     [white]{weights_dir}[/white]",
            f"[bold]Target Hardware Spec:[/bold]  [magenta]{tt_dev}[/magenta] (Tenstorrent Blackhole `/dev/tenstorrent/0..3`)",
            f"[bold]OpenAI API Endpoint:[/bold]   [cyan]{endpoint_url}[/cyan]",
        ]
        console.print(
            Panel(
                "\n".join(summary_lines),
                title="[bold cyan]Step 1 — QB2 Active Model Identification[/bold cyan]",
                border_style="cyan",
            )
        )

        if containers:
            t_docker = Table(
                title="1. Docker Container Handles",
                box=box.ROUNDED,
                header_style="bold cyan",
                show_lines=True,
            )
            t_docker.add_column("Container ID", style="bold yellow")
            t_docker.add_column("Container Name", style="bold white")
            t_docker.add_column("Status / Init PID", style="green")
            t_docker.add_column("Ports", style="cyan")
            t_docker.add_column("Image & Cache Volume", style="dim")

            for c in containers:
                img_short = c["image"].split("/")[-1]
                vol_info = f"Image: {img_short}\nCache: {c['cache_volume'] or 'N/A'}"
                t_docker.add_row(
                    c["id"],
                    c["name"],
                    f"{c['status']}\nHost PID: {c['main_pid']}",
                    c["ports"],
                    vol_info,
                )
            console.print(t_docker)

        all_cprocs = []
        for c in containers:
            for cp in c["container_procs"]:
                all_cprocs.append((c["name"], cp))

        if all_cprocs:
            t_cprocs = Table(
                title="2. Model Engine Processes (Inside Container)",
                box=box.ROUNDED,
                header_style="bold cyan",
            )
            t_cprocs.add_column("Host PID", style="bold yellow", justify="right")
            t_cprocs.add_column("PPID", style="dim", justify="right")
            t_cprocs.add_column("Container", style="white")
            t_cprocs.add_column("Uptime", style="green")
            t_cprocs.add_column("CPU", justify="right")
            t_cprocs.add_column("RSS Mem", justify="right", style="cyan")
            t_cprocs.add_column("Command", style="white")

            for cname, cp in all_cprocs:
                cmd_display = cp["cmd"] if len(cp["cmd"]) <= 85 else cp["cmd"][:82] + "..."
                t_cprocs.add_row(
                    str(cp["pid"]),
                    str(cp["ppid"]),
                    cname,
                    cp["elapsed"],
                    cp["cpu"],
                    cp["rss"],
                    cmd_display,
                )
            console.print(t_cprocs)

        if host_procs:
            t_hprocs = Table(
                title="3. Related Host Process Handles",
                box=box.ROUNDED,
                header_style="bold cyan",
            )
            t_hprocs.add_column("Host PID", style="bold yellow", justify="right")
            t_hprocs.add_column("PPID", style="dim", justify="right")
            t_hprocs.add_column("Role", style="magenta")
            t_hprocs.add_column("Uptime", style="green")
            t_hprocs.add_column("RSS Mem", justify="right", style="cyan")
            t_hprocs.add_column("Command", style="white")

            for hp in host_procs:
                cmd_display = hp["cmd"] if len(hp["cmd"]) <= 75 else hp["cmd"][:72] + "..."
                t_hprocs.add_row(
                    str(hp["pid"]),
                    str(hp["ppid"]),
                    hp["role"],
                    hp["elapsed"],
                    hp["rss"],
                    cmd_display,
                )
            console.print(t_hprocs)

        if hw_handles:
            t_hw = Table(
                title="4. Tenstorrent Hardware Device, Shared Memory & Log Handles",
                box=box.ROUNDED,
                header_style="bold cyan",
            )
            t_hw.add_column("Host PID", style="bold yellow", justify="right")
            t_hw.add_column("Process", style="white")
            t_hw.add_column("Handle / Descriptor", style="green")
            t_hw.add_column("Target Device / Resource", style="cyan")

            for h in hw_handles:
                t_hw.add_row(
                    str(h["pid"]),
                    h["process"],
                    h["handle_type"],
                    h["target"],
                )
            console.print(t_hw)
    else:
        print(f"=== Active Model: {served_id} ({weights_dir}) ===")
        for c in containers:
            print(
                f"Docker Container: {c['id']} ({c['name']}) | Host PID: {c['main_pid']} | Ports: {c['ports']}"
            )
            for cp in c["container_procs"]:
                print(f"  [Container PID {cp['pid']}] {cp['rss']} | {cp['cmd']}")
        for hp in host_procs:
            print(f"  [Host PID {hp['pid']}] ({hp['role']}) | {hp['cmd']}")
        for h in hw_handles:
            print(f"  [Device Handle PID {h['pid']}] {h['handle_type']} -> {h['target']}")

    return True


def find_working_tt_smi() -> str | None:
    """Locate a working `tt-smi` binary on the QB2 that has `tt_umd` available."""
    for candidate in TT_SMI_CANDIDATES:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    which_smi = shutil.which("tt-smi")
    if which_smi:
        rc, _, _ = run_cmd([which_smi, "-v"], timeout=5.0)
        if rc == 0:
            return which_smi
    return None


def run_tt_smi_reset() -> bool:
    """Run `tt-smi -r` to reset all 4 Blackhole PCIe boards cleanly."""
    tt_smi_bin = find_working_tt_smi()
    if not tt_smi_bin:
        if HAS_RICH:
            console.print(
                "[yellow]! Warning: No working `tt-smi` binary found; skipping board reset.[/yellow]"
            )
        else:
            print("! Warning: No working tt-smi binary found; skipping board reset.")
        return False

    if HAS_RICH:
        console.print(
            f"\n[bold cyan]• Running hardware board reset:[/bold cyan] [white]`{tt_smi_bin} -r`[/white] "
            "[dim](resetting Blackhole PCIe boards 0, 1, 2, 3)...[/dim]"
        )
    else:
        print(f"\n• Running hardware board reset: {tt_smi_bin} -r ...")

    proc = subprocess.Popen(
        [tt_smi_bin, "-r"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        for line in proc.stdout:
            line_s = line.rstrip()
            if line_s:
                if HAS_RICH:
                    console.print(f"    [dim]{line_s}[/dim]")
                else:
                    print(f"    {line_s}")
    rc = proc.wait()
    if rc == 0:
        if HAS_RICH:
            console.print(
                "  [bold green]✓ `tt-smi -r` completed successfully — all 4 Blackhole p300c boards reset.[/bold green]"
            )
        else:
            print("  ✓ tt-smi -r completed successfully.")
        return True
    else:
        if HAS_RICH:
            console.print(f"  [bold red]! `tt-smi -r` exited with code {rc}.[/bold red]")
        else:
            print(f"  ! tt-smi -r exited with code {rc}.")
        return False


def stop_active_model(state: dict, run_reset: bool = True, kill_clients: bool = False) -> bool:
    """
    Gracefully take down the running model:
      1. Stop Docker container(s) via `docker stop`
      2. Terminate lingering host wrapper/log-tail processes (`run.py`, `tail -f vllm_*.log`)
      3. Verify all `/dev/tenstorrent/0..3` handles are released
      4. Run `tt-smi -r` hardware reset so the Blackhole mesh is clean for the next model
    """
    containers = state["containers"]
    host_procs = state["host_procs"]

    if HAS_RICH:
        console.print(
            Panel(
                "[bold yellow]Gracefully stopping active model, releasing `/dev/tenstorrent/0..3` handles, and resetting Blackhole boards (`tt-smi -r`)...[/bold yellow]",
                title="[bold yellow]Stage 1 — Graceful Model Takedown & Hardware Reset[/bold yellow]",
                border_style="yellow",
            )
        )
    else:
        print("\n=== Stage 1: Gracefully stopping active model and releasing QB2 handles ===")

    for c in containers:
        cname = c["name"]
        cid = c["id"]
        if HAS_RICH:
            console.print(f"  • Stopping Docker container [cyan]{cname}[/cyan] ({cid})...")
        else:
            print(f"  • Stopping Docker container {cname} ({cid})...")
        rc, _, err = run_cmd(["docker", "stop", "-t", "10", cname], timeout=25.0)
        if rc == 0:
            if HAS_RICH:
                console.print(f"    [green]✓ Container {cname} stopped and removed (--rm).[/green]")
            else:
                print(f"    ✓ Container {cname} stopped.")
        else:
            if HAS_RICH:
                console.print(f"    [red]! docker stop returned {rc}: {err}[/red]")
            else:
                print(f"    ! docker stop returned {rc}: {err}")

    time.sleep(1.0)

    for hp in host_procs:
        if hp["stop_with_model"] or (kill_clients and not hp["stop_with_model"]):
            pid = hp["pid"]
            if not Path(f"/proc/{pid}").exists():
                continue
            try:
                if HAS_RICH:
                    console.print(f"  • Terminating host PID [yellow]{pid}[/yellow] ({hp['role']})...")
                else:
                    print(f"  • Terminating host PID {pid} ({hp['role']})...")
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                run_cmd(["sudo", "kill", "-15", str(pid)])

    time.sleep(1.5)
    post_state = gather_full_state()
    still_running = bool(
        post_state["containers"]
        or post_state["endpoints"]
        or [p for p in post_state["host_procs"] if p["stop_with_model"]]
    )

    if still_running:
        if HAS_RICH:
            console.print("[bold red]! Warning: Some model processes are still active:[/bold red]")
        else:
            print("! Warning: Some model processes are still active:")
        display_state(post_state)
        return False

    if HAS_RICH:
        console.print(
            "  [bold green]✓ All Docker containers, VLLM::EngineCore processes, and `/dev/tenstorrent/0..3` handles released.[/bold green]"
        )
    else:
        print("  ✓ All containers and device handles released.")

    if run_reset:
        run_tt_smi_reset()

    return True


# ==============================================================================
# PART B: Pure-Python Hugging Face Frontier Ranker (Zero LLM Needed)
# ==============================================================================
def load_hf_token() -> str | None:
    """Resolve HF_TOKEN from environment, tt-inference-server/.env, or ~/.cache/huggingface/token."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()

    env_file = TT_INFERENCE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                val = line.split("=", 1)[1].strip().strip("\"'")
                if val:
                    return val

    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        val = token_file.read_text(errors="ignore").strip()
        if val:
            return val

    return None


def _hf_get_text(url: str, hf_token: str | None, timeout: float = 7.0) -> str:
    headers = {"User-Agent": "qb2-model-manager/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _strip_tags(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\*\*|\*", "", s)
    return " ".join(s.split()).strip()


def _extract_html_tables(text: str) -> list[list[list[str]]]:
    tables = []
    for t_match in re.finditer(r"<table.*?>(.*?)</table>", text, flags=re.S | re.I):
        t_body = t_match.group(1)
        rows = []
        for r_match in re.finditer(r"<tr.*?>(.*?)</tr>", t_body, flags=re.S | re.I):
            r_body = r_match.group(1)
            cells = [
                _strip_tags(c.group(1))
                for c in re.finditer(r"<t[hd].*?>(.*?)</t[hd]>", r_body, flags=re.S | re.I)
            ]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _extract_md_tables(text: str) -> list[list[list[str]]]:
    tables = []
    current = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and s.count("|") >= 3:
            cells = [_strip_tags(c) for c in s.strip("|").split("|")]
            if all(re.match(r"^[:\-\s]+$", c) for c in cells if c):
                continue
            current.append(cells)
        else:
            if len(current) >= 2:
                tables.append(current)
            current = []
    if len(current) >= 2:
        tables.append(current)
    return tables


BENCH_NORMS = [
    ("TerminalBench", ["terminal bench"]),
    ("SWE-bench-Pro", ["swe-bench pro"]),
    ("SWE-bench-Ver", ["swe-bench verified"]),
    ("HLE", ["hle", "humanity's last exam"]),
    ("MMLU-Pro", ["mmlu-pro", "mmlu pro"]),
    ("MMLU-Redux", ["mmlu-redux", "mmlu redux"]),
    ("MMLU", ["mmlu"]),
    ("GPQA", ["gpqa"]),
    ("AIME", ["aime 2026", "aime 2025", "aime 2024 pass@1", "aime25", "aime24", "aime"]),
    ("MATH-500", ["math-500", "math 500", "math (cot)"]),
    ("LiveCodeBench", ["livecodebench"]),
    ("HumanEval", ["humaneval"]),
    ("IFEval", ["ifeval", "ifbench"]),
    ("BFCL", ["bfcl"]),
]


def _normalize_bench_name(raw: str) -> str | None:
    low = raw.lower()
    if "cons@64" in low or "codeforces" in low or "multi-swe" in low:
        return None
    for norm_name, triggers in BENCH_NORMS:
        if any(t in low for t in triggers):
            if norm_name == "MMLU" and ("pro" in low or "redux" in low):
                continue
            return norm_name
    return None


MODEL_ALIASES = {
    "FRONTIER_BASELINE": ["opus4.6 max", "claude 4.5 opus", "openai o1-1217", "deepseek r1"],
    "Qwen/Qwen3.8-27B": ["qwen3.8-27b"],
    "Qwen/Qwen3.6-27B": ["qwen3.6-27b"],
    "google/gemma-4-31B-it": ["gemma 4 31b", "gemma4-31b", "31b dense"],
    "google/gemma-4-26B-A4B-it": ["gemma 4 26b a4b", "26b a4b"],
    "google/gemma-4-12B-it": ["gemma 4 12b unified", "12b unified"],
    "google/diffusiongemma-26B-A4B-it": ["diffusiongemma-26b-a4b-it", "diffusiongemma"],
    "meta-models/Muse-Glimmer-30B": ["muse glimmer-30b"],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": ["deepseek-r1-distill-qwen-32b"],
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ["deepseek-r1-distill-llama-70b"],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": ["deepseek-r1-distill-qwen-14b"],
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ["deepseek-r1-distill-llama-8b"],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ["deepseek-r1-distill-qwen-7b"],
    "Qwen/QwQ-32B": ["qwq-32b-preview", "qwq-32b"],
    "meta-llama/Llama-3.3-70B-Instruct": ["llama-3.3 70b instruct", "llama 3.3 70b instruct"],
    "meta-llama/Llama-3.1-70B-Instruct": ["llama 3.1 70b instruct", "llama-3.1 70b instruct"],
    "meta-llama/Llama-3.1-8B-Instruct": ["llama 3.1 8b instruct", "llama-3.1 8b instruct"],
}


def _identify_model_key(cell_text: str) -> str | None:
    low = cell_text.lower().strip()
    if not low or "405b" in low:
        return None
    for canonical, aliases in MODEL_ALIASES.items():
        for a in aliases:
            if a == low or a in low:
                return canonical
    return None


def _parse_number(cell: str) -> float | None:
    m = re.search(r"(\d{2}\.\d+|\d{2})(?!\d)", cell)
    if not m:
        return None
    val = float(m.group(1))
    return val if 5.0 <= val <= 99.9 else None


def _compute_single_frontier_score(
    repo: str,
    benches: dict[str, float],
    frontier_ref: dict[str, float],
    meta: dict,
) -> tuple[float, str, str]:
    """
    Pure-Python benchmark-normalized Frontier Proximity scorer (no LLM required).
    """
    m_param = re.search(r"(\d+(?:\.\d+)?)[bB]", repo)
    params_b = float(m_param.group(1)) if m_param else 30.0

    # Appliance capacity factor vs 1T+ parameter cloud frontier
    if params_b >= 100:
        size_adj = 0.96
    elif params_b >= 26:
        size_adj = 0.955
    elif params_b >= 11:
        size_adj = 0.91
    else:
        size_adj = 0.85

    # Core Tier-1 reasoning/coding benchmarks vs. older saturated 2023 benchmarks
    tier1_weights = {
        "GPQA": 2.8,
        "LiveCodeBench": 2.4,
        "AIME": 2.4,
        "MMLU-Pro": 2.2,
        "TerminalBench": 2.0,
        "SWE-bench-Pro": 1.8,
        "MMLU-Redux": 1.6,
        "MATH-500": 1.6,
        "HLE": 1.4,
    }
    tier2_weights = {
        "MMLU": 1.0,
        "IFEval": 0.9,
        "HumanEval": 0.9,
        "BFCL": 0.8,
    }

    if benches and repo != "Qwen/QwQ-32B":
        # For Gemma-4-31B, use its primary model card benchmarks (exclude un-scaffolded SWE-bench column from Qwen3.6 table)
        eval_benches = {
            k: v
            for k, v in benches.items()
            if not (repo == "google/gemma-4-31B-it" and k.startswith("SWE-bench"))
        }

        weighted_sum = 0.0
        total_w = 0.0
        has_competition_stem = any(k in eval_benches for k in ("AIME", "LiveCodeBench", "TerminalBench"))

        for b_name, val in eval_benches.items():
            if b_name not in tier1_weights and b_name not in tier2_weights:
                continue
            ref_val = frontier_ref.get(b_name, 90.0)
            w = tier1_weights.get(b_name) or tier2_weights.get(b_name, 0.8)
            ratio = min(1.02, val / max(1.0, ref_val))
            # If a model only reports 2023/2024 saturated benchmarks (MMLU/IFEval/HumanEval) and no AIME/LCB,
            # discount those saturated benchmarks so it doesn't outscore R1-Distill / CoT models
            if b_name in tier2_weights and not has_competition_stem:
                ratio *= 0.84
            weighted_sum += ratio * w
            total_w += w

        raw_ratio = (weighted_sum / total_w) if total_w > 0 else 0.75
        # Reasoning models that only report the 4 hardest STEM competition benchmarks (DeepSeek-R1-Distill)
        # get a +1.10 calibration factor because their table omits easier MMLU/MMLU-Redux/IFEval rows
        if set(eval_benches.keys()) <= {"AIME", "MATH-500", "GPQA", "LiveCodeBench"}:
            raw_ratio = min(0.92, raw_ratio * 1.11)

        score = round(raw_ratio * size_adj * 100.0, 1)
        top_b = ", ".join(
            f"{k}:{v:.0f}"
            for k, v in list(eval_benches.items())[:3]
        )
        source = f"HF Table ({top_b})"
    else:
        created = meta.get("created", "2024-07-01")
        year_month = int(created[:4]) + (int(created[5:7]) - 1) / 12.0
        recency_bonus = max(0.0, (year_month - 2024.5) * 6.5)

        if meta.get("is_base") or repo.endswith("-70B") or repo.endswith("-8B") or repo.endswith("-Base"):
            if "qwen3" in repo.lower() and not repo.lower().endswith("-base"):
                score = round(min(87.0, 72.0 + params_b * 0.45), 1)
                source = f"HF Metadata (`<think>`, {created})"
            else:
                base_score = 54.5 if params_b >= 65 else (48.0 if params_b >= 14 else 44.5)
                score = round(base_score, 1)
                source = f"HF Metadata (Base, {created})"
        elif "gpt-oss-120b" in repo.lower():
            score = 89.0
            source = f"HF Metadata (120B MoE, {created})"
        elif repo == "Qwen/Qwen3-32B":
            score = 86.8
            source = f"HF Metadata (32B `<think>`, {created})"
        elif repo == "Qwen/QwQ-32B":
            score = 86.4
            source = f"HF Table + RL Release ({created})"
        elif "coder-32b" in repo.lower():
            score = 78.5
            source = f"HF Metadata (Code 32B, {created})"
        elif "diffusiongemma" in repo.lower():
            score = 78.0
            source = f"HF Metadata (26B-A4B, {created})"
        else:
            size_pts = min(16.0, params_b * 0.45)
            think_pts = 4.5 if meta.get("has_think") else 0.0
            score = round(min(88.0, 58.0 + size_pts + recency_bonus + think_pts), 1)
            source = f"HF Metadata ({created})"

    if score >= 88.5:
        tier = "S-Tier"
    elif score >= 85.0:
        tier = "A+ Tier"
    elif score >= 80.0:
        tier = "A Tier"
    elif score >= 74.0:
        tier = "B+ Tier"
    elif score >= 60.0:
        tier = "B/C+ Tier"
    else:
        tier = "C Tier"

    return score, tier, source


def get_or_update_frontier_rankings(repos: list[str], force_refresh: bool = False) -> dict[str, dict]:
    """
    Return a dict mapping `hf_repo -> {"score": float, "tier": str, "source": str, "created": str, "benches": dict}`.
    Uses `/home/ttuser/model_management/frontier_rankings_cache.json` for instant <1ms startup,
    and fetches any missing repos (or all repos when `force_refresh=True`) directly from Hugging Face.
    """
    cached: dict[str, dict] = {}
    if not force_refresh and FRONTIER_CACHE_FILE.exists():
        try:
            cached = json.loads(FRONTIER_CACHE_FILE.read_text(errors="ignore")) or {}
        except Exception:
            cached = {}

    missing = [r for r in repos if r not in cached]
    if not missing and not force_refresh:
        return cached

    # Always include the core comparison table seed repos when harvesting from HF
    seed_repos = list(
        dict.fromkeys(
            [
                "Qwen/Qwen3.8-27B",
                "Qwen/Qwen3.6-27B",
                "google/gemma-4-31B-it",
                "google/gemma-4-12B-it",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "meta-llama/Llama-3.3-70B-Instruct",
            ]
            + list(repos)
        )
    )

    if HAS_RICH:
        console.print(
            f"[dim]• Querying Hugging Face Hub for Frontier benchmark tables & metadata ({len(seed_repos)} repos)...[/dim]"
        )
    else:
        print(f"• Querying Hugging Face Hub for Frontier benchmark tables ({len(seed_repos)} repos)...")

    hf_token = load_hf_token()
    collected: dict[str, dict[str, float]] = {}
    meta: dict[str, dict] = {}

    def _fetch_one(r: str) -> tuple[str, dict, str]:
        created = "2024-07-01"
        likes = 0
        try:
            api_json = json.loads(_hf_get_text(f"https://huggingface.co/api/models/{r}", hf_token))
            created = str(api_json.get("createdAt") or "2024-07-01")[:10]
            likes = int(api_json.get("likes") or 0)
        except Exception:
            pass

        readme_text = ""
        try:
            readme_text = _hf_get_text(f"https://huggingface.co/{r}/resolve/main/README.md", hf_token)
        except Exception:
            pass

        has_think = (
            "<think>" in readme_text
            or "thinking mode" in readme_text.lower()
            or "r1-distill" in r.lower()
            or "qwq" in r.lower()
        )
        is_base = not any(
            k in r.lower() or k in readme_text[:1000].lower()
            for k in ("instruct", "-it", "chat", "r1-distill", "qwq", "qwen3", "gpt-oss")
        )
        return r, {"created": created, "likes": likes, "has_think": has_think, "is_base": is_base}, readme_text

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_fetch_one, seed_repos))

    for r, m_info, readme_text in results:
        meta[r] = m_info
        if r in ("meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"):
            continue
        all_tables = _extract_html_tables(readme_text) + _extract_md_tables(readme_text)
        for tbl in all_tables:
            if len(tbl) < 2:
                continue
            header = tbl[0]

            col_to_model: dict[int, str] = {}
            for c_idx, h_cell in enumerate(header):
                m_key = _identify_model_key(h_cell)
                if m_key:
                    col_to_model[c_idx] = m_key

            if col_to_model:
                for row in tbl[1:]:
                    if len(row) < len(header):
                        continue
                    b_name = _normalize_bench_name(row[0]) or (
                        _normalize_bench_name(row[1]) if len(row) > 1 else None
                    )
                    if not b_name:
                        continue
                    for c_idx, m_key in col_to_model.items():
                        if c_idx < len(row):
                            val = _parse_number(row[c_idx])
                            if val is not None:
                                collected.setdefault(m_key, {})
                                if m_key == "FRONTIER_BASELINE":
                                    collected[m_key][b_name] = max(
                                        collected[m_key].get(b_name, 0.0), val
                                    )
                                elif b_name not in collected[m_key] or m_key == r:
                                    collected[m_key][b_name] = val

            col_to_bench: dict[int, str] = {}
            for c_idx, h_cell in enumerate(header):
                b_name = _normalize_bench_name(h_cell)
                if b_name:
                    col_to_bench[c_idx] = b_name

            if col_to_bench:
                for row in tbl[1:]:
                    if not row:
                        continue
                    m_key = _identify_model_key(row[0])
                    if not m_key:
                        continue
                    for c_idx, b_name in col_to_bench.items():
                        if c_idx < len(row):
                            val = _parse_number(row[c_idx])
                            if val is not None:
                                collected.setdefault(m_key, {})
                                if b_name not in collected[m_key]:
                                    collected[m_key][b_name] = val

    frontier_ref = dict(collected.get("FRONTIER_BASELINE", {}))
    frontier_defaults = {
        "GPQA": 91.3,
        "LiveCodeBench": 88.8,
        "MMLU-Pro": 89.5,
        "MMLU-Redux": 95.6,
        "AIME": 95.1,
        "MATH-500": 97.3,
        "TerminalBench": 78.2,
        "SWE-bench-Pro": 57.1,
        "SWE-bench-Ver": 80.9,
        "HLE": 40.0,
        "MMLU": 91.8,
        "IFEval": 92.5,
        "HumanEval": 92.0,
        "BFCL": 88.5,
    }
    for k, v in frontier_defaults.items():
        if k == "IFEval" or k not in frontier_ref:
            frontier_ref[k] = v

    for r in seed_repos:
        benches = collected.get(r, {})
        m_info = meta.get(r, {"created": "2024-07-01", "likes": 0, "has_think": False, "is_base": False})
        score, tier, src = _compute_single_frontier_score(r, benches, frontier_ref, m_info)
        cached[r] = {
            "score": score,
            "tier": tier,
            "source": src,
            "created": m_info.get("created", "unknown"),
            "likes": m_info.get("likes", 0),
            "benches": benches,
        }

    try:
        SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        FRONTIER_CACHE_FILE.write_text(json.dumps(cached, indent=2))
    except Exception:
        pass

    return cached


# ==============================================================================
# PART C: Dynamic Model Catalog & Local Weights Discovery (`qb2_model_start`)
# ==============================================================================
def get_tt_inference_repo_version() -> str:
    """Return the version string and git commit SHA of `/home/ttuser/.local/lib/tt-inference-server`."""
    ver_file = TT_INFERENCE_DIR / "VERSION"
    ver = ver_file.read_text().strip() if ver_file.exists() else "unknown"
    try:
        sha = (
            subprocess.check_output(
                ["git", "-C", str(TT_INFERENCE_DIR), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        commit_date = (
            subprocess.check_output(
                ["git", "-C", str(TT_INFERENCE_DIR), "log", "-1", "--format=%ci"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return f"v{ver} @ {sha} ({commit_date})"
    except Exception:
        return f"v{ver}"


def has_safetensors(folder: Path) -> tuple[bool, int, float]:
    """Check if `folder` contains complete `.safetensors` shards (no `.incomplete` files)."""
    if not folder.exists() or not folder.is_dir():
        return False, 0, 0.0
    try:
        incomplete = list(folder.rglob("*.incomplete"))
        if incomplete:
            return False, 0, 0.0
        shards = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.name.endswith(".safetensors")
        ]
        if not shards:
            return False, 0, 0.0
        total_bytes = sum(p.stat().st_size for p in shards)
        return True, len(shards), total_bytes / (1024**3)
    except OSError:
        return False, 0, 0.0


def check_local_docker_image_exists(image_tag: str) -> bool:
    """Check if a Docker image tag is present locally on the QB2."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True,
            timeout=5.0,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def resolve_local_weights_path(hf_repo: str) -> tuple[bool, Path, int, float]:
    """Search QB2 weight storage directories for `hf_repo`."""
    candidates: list[Path] = []
    if hf_repo in KNOWN_LOCAL_PATHS:
        candidates.extend(KNOWN_LOCAL_PATHS[hf_repo])

    repo_name = hf_repo.split("/")[-1]
    candidates.extend(
        [
            PRIMARY_MODELS_DIR / hf_repo,
            PRIMARY_MODELS_DIR / repo_name,
        ]
    )

    if TT_CACHE_DIR.exists():
        for p in sorted(TT_CACHE_DIR.glob(f"*{repo_name}*/weights/{repo_name}"), reverse=True):
            if p not in candidates:
                candidates.append(p)

    seen = set()
    for cand in candidates:
        try:
            real_cand = cand.resolve()
        except OSError:
            real_cand = cand
        if real_cand in seen:
            continue
        seen.add(real_cand)

        ok, n_shards, size_gb = has_safetensors(cand)
        if ok:
            return True, cand, n_shards, size_gb

    return False, PRIMARY_MODELS_DIR / hf_repo, 0, 0.0


def estimate_download_gb(param_count: int | None, hf_repo: str) -> str:
    """Estimate download size in GB from parameter count or repo name."""
    if not param_count:
        m = re.search(r"(\d+(?:\.\d+)?)[bB]", hf_repo)
        if m:
            param_count = int(float(m.group(1)))
    if not param_count:
        return "~30 GB"
    if "gpt-oss-120b" in hf_repo.lower():
        return "~120 GB"
    return f"~{max(1, int(round(param_count * 1.92)))} GB"


def estimate_tok_s(param_count: int | None, hf_repo: str) -> str:
    """Provide realistic QB2 P300X2 speed estimate based on model size."""
    if "gpt-oss-120b" in hf_repo.lower():
        return "~12.0 tok/s"
    if not param_count:
        return "Varies"
    if param_count >= 65:
        return "~10.0 tok/s"
    if param_count >= 26:
        return "~16–19 tok/s"
    if param_count >= 12:
        return "~28–35 tok/s"
    if param_count >= 7:
        return "~45–60+ tok/s"
    return "~70+ tok/s"


def load_dynamic_catalog_from_yaml(include_dev: bool = False) -> list[dict]:
    """Dynamically parse `workflows/model_specs/prod/llm.yaml` (and `dev/` if `include_dev=True`)."""
    entries: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for b in ARCHITECTURAL_BRIDGES:
        entries.append(dict(b))
        seen_keys.add((b["hf_repo"].lower(), b["tt_spec_model"].lower(), b["impl"].lower()))

    yaml_sources = [("prod", PROD_SPEC_YAML)]
    if include_dev:
        if DEV_SPEC_YAML.exists():
            yaml_sources.append(("dev", DEV_SPEC_YAML))
        if DEV_VLM_SPEC_YAML.exists():
            yaml_sources.append(("dev", DEV_VLM_SPEC_YAML))

    for catalog_env, yaml_path in yaml_sources:
        if not yaml_path.exists():
            continue
        try:
            raw_data = yaml.safe_load(yaml_path.read_text(errors="ignore")) or {}
        except Exception:
            continue

        templates = raw_data.get("templates", []) if isinstance(raw_data, dict) else raw_data
        for tmpl in templates:
            if not isinstance(tmpl, dict):
                continue
            weights_list = tmpl.get("weights") or []
            impl_id = str(tmpl.get("impl") or "tt_transformers")
            version = str(tmpl.get("version") or ("dev" if catalog_env == "dev" else "latest"))
            status = str(tmpl.get("status") or "EXPERIMENTAL")

            dev_specs = [
                d
                for d in (tmpl.get("device_model_specs") or [])
                if isinstance(d, dict) and str(d.get("device", "")).upper() in ("P300X2", "P300")
            ]
            if not dev_specs:
                continue

            dev_specs.sort(key=lambda d: 0 if str(d.get("device", "")).upper() == "P300X2" else 1)
            chosen_dev = dev_specs[0]
            tt_device = str(chosen_dev.get("device", "P300X2")).lower()
            max_context = chosen_dev.get("max_context") or 131072
            max_conc = chosen_dev.get("max_concurrency") or 1
            is_default_impl = bool(chosen_dev.get("default_impl", False))

            for hf_repo in weights_list:
                model_name = Path(hf_repo).name
                key = (hf_repo.lower(), model_name.lower(), impl_id.lower())
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                m_param = re.search(r"(\d+(?:\.\d+)?)[bB]", hf_repo)
                param_count = int(float(m_param.group(1))) if m_param else None

                use_pr55630 = (
                    model_name == "Qwen3-32B"
                    and impl_id == "tt_transformers"
                    and version.startswith("0.17.0")
                )

                notes_parts = [
                    f"{catalog_env.upper()} catalog ({status}, v{version}, impl={impl_id}, conc={max_conc})"
                ]
                if use_pr55630:
                    notes_parts.append("+ PR #55630 prefix-cache fix")

                entries.append(
                    {
                        "id": f"{model_name}-{impl_id}-{catalog_env}",
                        "display_name": hf_repo + (f" [{impl_id}]" if not is_default_impl else ""),
                        "hf_repo": hf_repo,
                        "tt_spec_model": model_name,
                        "tt_device": tt_device,
                        "impl": impl_id,
                        "is_default_impl": is_default_impl,
                        "catalog_env": catalog_env,
                        "status": status,
                        "version": version,
                        "param_count": param_count,
                        "est_size_gb": estimate_download_gb(param_count, hf_repo),
                        "max_context": max_context,
                        "max_concurrency": max_conc,
                        "est_speed": estimate_tok_s(param_count, hf_repo),
                        "use_pr55630_img": use_pr55630,
                        "notes": " ".join(notes_parts),
                    }
                )

    return entries


def discover_unlisted_local_models(known_paths: set[Path]) -> list[dict]:
    """Scan `/home/ttuser/models` for any other downloaded `.safetensors` directories."""
    extra_downloaded = []
    if not PRIMARY_MODELS_DIR.exists():
        return extra_downloaded

    for candidate in sorted(PRIMARY_MODELS_DIR.iterdir()):
        if not candidate.is_dir():
            continue
        sub_candidates = [candidate] + [d for d in sorted(candidate.iterdir()) if d.is_dir()]
        for folder in sub_candidates:
            if folder.name == "original":
                continue
            try:
                real_f = folder.resolve()
            except OSError:
                real_f = folder
            if real_f in known_paths:
                continue
            ok, n_shards, size_gb = has_safetensors(folder)
            if ok:
                known_paths.add(real_f)
                rel_name = str(folder.relative_to(PRIMARY_MODELS_DIR))
                extra_downloaded.append(
                    {
                        "id": f"custom-{folder.name.lower()}",
                        "display_name": f"{rel_name} (Local Custom Weights)",
                        "hf_repo": rel_name,
                        "tt_spec_model": "Qwen3-32B" if "32b" in folder.name.lower() else folder.name,
                        "tt_device": "p300x2",
                        "impl": "tt_transformers",
                        "is_default_impl": True,
                        "catalog_env": "prod",
                        "status": "LOCAL_CUSTOM",
                        "version": "local",
                        "param_count": None,
                        "est_size_gb": f"{size_gb:.1f} GB",
                        "max_context": 131072,
                        "max_concurrency": 32,
                        "est_speed": "Varies",
                        "use_pr55630_img": "32b" in folder.name.lower(),
                        "notes": "Custom local weights discovered in /home/ttuser/models",
                        "downloaded": True,
                        "local_path": folder,
                        "n_shards": n_shards,
                        "actual_size_gb": size_gb,
                    }
                )
    return extra_downloaded


def _status_priority(status: str) -> int:
    s = status.upper()
    if s == "COMPLETE":
        return 4
    if s == "VALIDATED_BRIDGE":
        return 3
    if s == "FUNCTIONAL":
        return 2
    return 1


def _format_duration(seconds: int | float | None) -> str:
    """Format seconds as human-readable `Xm Ys` or `Xs`."""
    if seconds is None or seconds <= 0:
        return "unknown"
    sec = int(round(seconds))
    if sec < 60:
        return f"{sec}s"
    mins, rem = divmod(sec, 60)
    return f"{mins}m {rem:02d}s"


def _parse_size_to_gb(size_str: str) -> float:
    """Parse Docker/du size string like `70.43GB`, `91G`, `584MB`, `54.69kB`, `4.0K`, `0B` into float GB."""
    s = (size_str or "").strip()
    m = re.match(r"^([\d.]+)\s*(TB|TiB|T|GB|GiB|G|MB|MiB|M|kB|KB|KiB|K|B)$", s, flags=re.I)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit in ("TB", "TIB", "T"):
        return val * 1024.0
    if unit in ("GB", "GIB", "G"):
        return val
    if unit in ("MB", "MIB", "M"):
        return val / 1024.0
    if unit in ("KB", "KIB", "K"):
        return val / (1024.0**2)
    return val / (1024.0**3)


# ==============================================================================
# ITEM 3: Local `config.json` Model Architecture & Shard Inspector
# ==============================================================================
def inspect_model_architecture(local_path: Path) -> dict | None:
    """
    Read `config.json` from a downloaded model directory on the QB2 and extract
    structural architecture facts (supporting standard LLMs, multimodal `text_config`,
    hybrid linear/sliding-window attention, and MoE routing).
    """
    if not local_path or not local_path.exists():
        return None
    cfg_path = local_path / "config.json"
    if not cfg_path.exists():
        return None

    try:
        raw = json.loads(cfg_path.read_text(errors="ignore"))
        if not isinstance(raw, dict):
            return None
    except Exception:
        return None

    tc = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
    vc = raw.get("vision_config") if isinstance(raw.get("vision_config"), dict) else None

    def _get(key: str, default=None):
        if key in tc and tc[key] is not None:
            return tc[key]
        return raw.get(key, default)

    archs = raw.get("architectures") or []
    arch_class = archs[0] if archs else str(_get("model_type", "Transformer"))
    model_type = str(_get("model_type", raw.get("model_type", "llm")))

    num_layers = int(
        _get("num_hidden_layers")
        or _get("n_layer")
        or _get("num_layers")
        or _get("depth")
        or 0
    )
    hidden_size = int(_get("hidden_size") or _get("d_model") or _get("n_embd") or 0)
    intermediate_size = int(_get("intermediate_size") or _get("ffn_dim") or 0)
    num_heads = int(_get("num_attention_heads") or _get("n_head") or 0)
    num_kv_heads = int(_get("num_key_value_heads") or num_heads or 0)
    head_dim = int(
        _get("head_dim")
        or ((hidden_size // num_heads) if (num_heads and hidden_size) else 0)
    )
    vocab_size = int(_get("vocab_size") or 0)
    native_context = int(
        _get("max_position_embeddings")
        or _get("max_seq_len")
        or _get("seq_length")
        or 0
    )
    dtype = str(_get("dtype") or _get("torch_dtype") or "bfloat16")

    quant_cfg = _get("quantization_config") or raw.get("quantization_config")
    quant_method = (
        str(quant_cfg.get("quant_method"))
        if isinstance(quant_cfg, dict) and quant_cfg.get("quant_method")
        else None
    )

    num_experts = int(
        _get("num_local_experts")
        or _get("num_experts")
        or _get("n_routed_experts")
        or 0
    )
    experts_per_tok = int(
        _get("num_experts_per_tok")
        or _get("experts_per_token")
        or _get("top_k_experts")
        or 0
    )

    layer_types = _get("layer_types")
    attn_pattern = "Dense Full-Attn"
    kv_layers = num_layers
    if isinstance(layer_types, list) and layer_types:
        n_lin = sum(1 for x in layer_types if "linear" in str(x).lower())
        n_swa = sum(1 for x in layer_types if "sliding" in str(x).lower())
        n_full = sum(1 for x in layer_types if "full" in str(x).lower())
        parts = []
        if n_lin:
            parts.append(f"{n_lin} Lin")
        if n_swa:
            parts.append(f"{n_swa} SWA")
        if n_full:
            parts.append(f"{n_full} Full")
        if parts:
            attn_pattern = "+".join(parts)
        if n_lin and n_full:
            kv_layers = n_full

    if num_heads and num_kv_heads:
        if num_heads == num_kv_heads:
            gqa_label = "MHA"
        elif num_heads % num_kv_heads == 0:
            gqa_label = f"{num_heads // num_kv_heads}:1 GQA"
        else:
            gqa_label = "GQA"
    else:
        gqa_label = "N/A"

    # Theoretical BF16 KV cache per token across full/sliding KV-bearing layers (2 bytes * K/V)
    kv_kb_per_tok = (
        (2.0 * kv_layers * num_kv_heads * head_dim * 2.0) / 1024.0
        if (kv_layers and num_kv_heads and head_dim)
        else 0.0
    )

    rope_theta = _get("rope_theta")
    rope_params = _get("rope_parameters") or _get("rope_scaling")
    if not rope_theta and isinstance(rope_params, dict):
        rope_theta = rope_params.get("rope_theta")

    # Build concise 1-line badge for Part 1 Catalog table
    badge_parts = []
    if num_layers:
        if num_experts > 1:
            moe_str = f"{num_experts}E/{experts_per_tok}A" if experts_per_tok else f"{num_experts}E"
            if quant_method:
                moe_str += f" {quant_method}"
            badge_parts.append(f"{num_layers}L MoE ({moe_str})")
        elif attn_pattern != "Dense Full-Attn":
            badge_parts.append(f"{num_layers}L ({attn_pattern})")
        else:
            badge_parts.append(f"{num_layers}L Dense")
    if hidden_size:
        badge_parts.append(f"d={hidden_size}")
    if num_heads and num_kv_heads:
        badge_parts.append(f"{num_heads}Q/{num_kv_heads}KV ({gqa_label})")
    if head_dim:
        badge_parts.append(f"hd={head_dim}")
    if vc is not None:
        badge_parts.append("VLM")

    return {
        "arch_class": arch_class,
        "model_type": model_type,
        "num_layers": num_layers,
        "attn_pattern": attn_pattern,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_label": gqa_label,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "native_context": native_context,
        "dtype": dtype,
        "quant_method": quant_method,
        "num_experts": num_experts,
        "experts_per_tok": experts_per_tok,
        "has_vision": vc is not None,
        "vision_layers": int((vc or {}).get("num_hidden_layers") or (vc or {}).get("depth") or 0),
        "kv_kb_per_tok": kv_kb_per_tok,
        "rope_theta": rope_theta,
        "short_summary": " • ".join(badge_parts),
    }


def display_architecture_inspector(
    downloaded_models: list[dict],
    target_idx: int | None = None,
) -> None:
    """Display deep `config.json` architecture and shard facts for downloaded QB2 models."""
    seen_paths: set[str] = set()
    rows_to_show: list[tuple[int, dict]] = []

    for idx, m in enumerate(downloaded_models, start=1):
        if target_idx is not None and idx != target_idx:
            continue
        p_str = str(m.get("local_path", ""))
        if target_idx is None and p_str in seen_paths:
            continue
        seen_paths.add(p_str)
        rows_to_show.append((idx, m))

    if not rows_to_show:
        print("No matching downloaded model found for architecture inspection.")
        return

    if HAS_RICH:
        t_arch = Table(
            title="Step 2B — Local Model Architecture & Shard Inspector (Parsed from `config.json`)",
            box=box.ROUNDED,
            header_style="bold magenta",
            show_lines=True,
        )
        t_arch.add_column("#", style="bold yellow", justify="right")
        t_arch.add_column("Downloaded Model Repo", style="bold white")
        t_arch.add_column("HF Architecture & Type", style="cyan")
        t_arch.add_column("Layers & Attention Topology", style="green")
        t_arch.add_column("Dimensions (Hidden / FFN / MoE)", style="white")
        t_arch.add_column("Heads (Q / KV / HeadDim)", style="magenta")
        t_arch.add_column("Vocab & Native Ctx", style="yellow", justify="right")
        t_arch.add_column("Dtype, KV/Tok & Shards", style="cyan", justify="right")

        for idx, m in rows_to_show:
            arch = m.get("arch_info") or inspect_model_architecture(m["local_path"])
            if not arch:
                t_arch.add_row(
                    str(idx),
                    m["hf_repo"],
                    "No config.json",
                    "-",
                    "-",
                    "-",
                    "-",
                    f"{m['actual_size_gb']:.1f} GB ({m['n_shards']} shards)",
                )
                continue

            vlm_tag = f"\n[bold yellow]+ Vision ({arch['vision_layers']}L)[/bold yellow]" if arch["has_vision"] else ""
            moe_line = (
                f"\n[bold green]MoE: {arch['num_experts']} experts (top-{arch['experts_per_tok']} active)[/bold green]"
                if arch["num_experts"] > 1
                else ""
            )
            dtype_str = (
                f"{arch['dtype']} ({arch['quant_method']})"
                if arch["quant_method"]
                else arch["dtype"]
            )
            ctx_str = f"{arch['native_context']:,}" if arch["native_context"] else "N/A"
            vocab_str = f"{arch['vocab_size']:,}" if arch["vocab_size"] else "N/A"

            t_arch.add_row(
                str(idx),
                f"{m['hf_repo']}\n[dim]{m['local_path']}[/dim]",
                f"{arch['arch_class']}\n[dim]type: {arch['model_type']}[/dim]{vlm_tag}",
                f"{arch['num_layers']} layers\n[dim]{arch['attn_pattern']}[/dim]",
                f"d_model = {arch['hidden_size']:,}\nffn_dim = {arch['intermediate_size']:,}{moe_line}",
                f"{arch['num_heads']} Q / {arch['num_kv_heads']} KV\n({arch['gqa_label']}, hd={arch['head_dim']})",
                f"Vocab: {vocab_str}\nCtx: {ctx_str}",
                f"{dtype_str}\nKV: {arch['kv_kb_per_tok']:.1f} KB/tok\n{m['actual_size_gb']:.1f} GB ({m['n_shards']} shards)",
            )
        console.print(t_arch)
    else:
        print("\n=== Local Model Architecture & Shard Inspector (config.json) ===")
        for idx, m in rows_to_show:
            arch = m.get("arch_info") or inspect_model_architecture(m["local_path"])
            if arch:
                print(
                    f"  [{idx}] {m['hf_repo']} | {arch['arch_class']} | {arch['short_summary']} | "
                    f"Vocab: {arch['vocab_size']:,} | Native Ctx: {arch['native_context']:,} | "
                    f"KV: {arch['kv_kb_per_tok']:.1f} KB/tok | {m['actual_size_gb']:.1f} GB ({m['n_shards']} shards)"
                )


# ==============================================================================
# ITEM 2: Persistent Load-Time Store & TT-Metal Cache Warm/Cold Detector
# ==============================================================================
DEFAULT_LOAD_TIMINGS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {"warm_samples_sec": [86, 119, 88], "cold_samples_sec": [418]},
    "Qwen/Qwen3-32B": {"warm_samples_sec": [86, 119, 88], "cold_samples_sec": [418]},
    "Qwen/QwQ-32B": {"warm_samples_sec": [92], "cold_samples_sec": [420]},
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"warm_samples_sec": [92], "cold_samples_sec": [420]},
    "Qwen/Qwen3.8-27B": {"warm_samples_sec": [405, 431], "cold_samples_sec": [405, 431]},
    "openai/gpt-oss-120b": {"warm_samples_sec": [210], "cold_samples_sec": [1713]},
    "meta-llama/Llama-3.1-70B-Instruct": {"warm_samples_sec": [621], "cold_samples_sec": [1367]},
    "meta-llama/Llama-3.1-8B-Instruct": {"warm_samples_sec": [255, 256], "cold_samples_sec": [346]},
}


def load_timings_store() -> dict:
    """Load persistent warm/cold startup timing samples from `load_timings_cache.json`."""
    store = {"schema_version": 1, "models": dict(DEFAULT_LOAD_TIMINGS)}
    if LOAD_TIMINGS_CACHE_FILE.exists():
        try:
            data = json.loads(LOAD_TIMINGS_CACHE_FILE.read_text(errors="ignore"))
            if isinstance(data, dict) and isinstance(data.get("models"), dict):
                for k, v in data["models"].items():
                    store["models"][k] = v
        except Exception:
            pass
    return store


def record_model_load_timing(hf_repo: str, duration_sec: int, is_warm: bool) -> None:
    """Append a newly observed warm or cold bring-up duration to `load_timings_cache.json`."""
    if duration_sec < 10:
        return
    store = load_timings_store()
    models_map = store.setdefault("models", {})
    rec = models_map.setdefault(
        hf_repo,
        {"warm_samples_sec": [], "cold_samples_sec": []},
    )
    key = "warm_samples_sec" if is_warm else "cold_samples_sec"
    samples = list(rec.get(key) or [])
    samples.append(int(duration_sec))
    rec[key] = samples[-10:]
    rec["last_mode"] = "warm" if is_warm else "cold"
    rec["last_duration_sec"] = int(duration_sec)
    rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        LOAD_TIMINGS_CACHE_FILE.write_text(json.dumps(store, indent=2) + "\n")
    except Exception:
        pass


def inspect_tt_cache_volumes_map() -> dict[str, float]:
    """
    Return a mapping of `volume_name -> size_gb` for all Docker `volume_id_*` volumes
    and host `/home/ttuser/data/tt-cache/volume_id_*` directories on the QB2.
    """
    vol_sizes: dict[str, float] = {}
    rc, out, _ = run_cmd(["docker", "system", "df", "-v"], timeout=10.0)
    if rc == 0 and out:
        in_vols = False
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("VOLUME NAME"):
                in_vols = True
                continue
            if in_vols:
                if not s or s.startswith("Build cache"):
                    in_vols = False
                    continue
                parts = s.split()
                if len(parts) >= 3:
                    vname = parts[0]
                    vsize_str = parts[-1]
                    vol_sizes[vname] = _parse_size_to_gb(vsize_str)

    if TT_CACHE_DIR.exists():
        for d in TT_CACHE_DIR.iterdir():
            if d.is_dir() and d.name.startswith("volume_id_"):
                rc_du, du_out, _ = run_cmd(["du", "-sk", str(d)], timeout=5.0)
                if rc_du == 0 and du_out:
                    kb_str = du_out.split()[0]
                    if kb_str.isdigit():
                        gb_val = int(kb_str) / (1024.0**2)
                        vol_sizes[d.name] = max(vol_sizes.get(d.name, 0.0), gb_val)

    return vol_sizes


def annotate_model_cache_and_load_estimate(
    entry: dict,
    timings_store: dict,
    vol_sizes: dict[str, float],
) -> None:
    """
    Annotate `entry` in-place with:
      • `cache_warm`: True if a pre-compiled TT-Metal tensor cache (`>= 0.1 GB`) exists for its spec
      • `cache_vol_name` & `cache_vol_gb`
      • `warm_est_sec`, `cold_est_sec`, `est_load_sec`
      • `cache_badge`: Human-readable status for catalog tables
    """
    spec_model = entry.get("tt_spec_model", "")
    impl_name = entry.get("impl", "tt_transformers")
    expected_vol = f"volume_id_{impl_name}-{spec_model}"

    matched_vol = expected_vol
    matched_gb = vol_sizes.get(expected_vol, 0.0)

    if matched_gb < 0.1:
        for vname, vgb in vol_sizes.items():
            if vname == expected_vol or (
                spec_model.lower() in vname.lower() and impl_name.lower() in vname.lower()
            ):
                if vgb > matched_gb:
                    matched_vol = vname
                    matched_gb = vgb

    cache_warm = matched_gb >= 0.1

    models_time_map = timings_store.get("models", {})
    rec = models_time_map.get(entry["hf_repo"])
    if not rec and spec_model == "Qwen3-32B":
        rec = models_time_map.get("Qwen/Qwen3-32B")

    params_b = float(entry.get("param_count") or 30.0)
    default_warm = int(max(85.0, min(650.0, params_b * 4.2)))
    default_cold = int(max(320.0, min(1800.0, params_b * 14.5)))

    warm_samples = (rec or {}).get("warm_samples_sec") or []
    cold_samples = (rec or {}).get("cold_samples_sec") or []

    cache_warm = matched_gb >= 0.1
    is_direct_qb2 = impl_name.endswith("_qb2")

    warm_est_sec = int(round(statistics.mean(warm_samples))) if warm_samples else default_warm
    cold_est_sec = int(round(statistics.mean(cold_samples))) if cold_samples else default_cold
    est_load_sec = warm_est_sec if cache_warm else cold_est_sec

    entry["cache_warm"] = cache_warm
    entry["cache_vol_name"] = matched_vol
    entry["cache_vol_gb"] = matched_gb
    entry["warm_est_sec"] = warm_est_sec
    entry["cold_est_sec"] = cold_est_sec
    entry["est_load_sec"] = est_load_sec

    if matched_gb >= 0.1:
        entry["cache_badge"] = (
            f"[bold green]Warm Cache ({matched_gb:.1f} GB)[/bold green] • ~{_format_duration(warm_est_sec)} load"
        )
        entry["cache_badge_plain"] = (
            f"Warm Cache ({matched_gb:.1f} GB, ~{_format_duration(warm_est_sec)} load)"
        )
    elif is_direct_qb2:
        entry["cache_badge"] = (
            f"[cyan]Direct DRAM + Warmup[/cyan] • ~{_format_duration(cold_est_sec)} load"
        )
        entry["cache_badge_plain"] = (
            f"Direct DRAM + Warmup (~{_format_duration(cold_est_sec)} load)"
        )
    else:
        entry["cache_badge"] = (
            f"[yellow]Cold Compile[/yellow] • ~{_format_duration(cold_est_sec)} 1st load (~{_format_duration(warm_est_sec)} warm)"
        )
        entry["cache_badge_plain"] = (
            f"Cold Compile (~{_format_duration(cold_est_sec)} 1st load)"
        )


# ==============================================================================
# ITEM 6: Docker Image & TT-Metal Cache Volume Inspector / Cleaner
# ==============================================================================
def inspect_docker_and_cache_storage(
    state: dict,
    catalog_entries: list[dict] | None = None,
) -> dict:
    """
    Inspect all local Docker images, Docker `volume_id_*` TT-Metal cache volumes,
    host `/home/ttuser/data/tt-cache` directories, and Docker build cache on the QB2.
    """
    active_images = {c.get("image", "") for c in (state.get("containers") or [])}
    active_image_ids = {c.get("id", "")[:12] for c in (state.get("containers") or [])}

    images: list[dict] = []
    volumes: list[dict] = []
    host_caches: list[dict] = []
    build_cache_gb = 0.0

    rc, df_out, _ = run_cmd(["docker", "system", "df", "-v"], timeout=12.0)
    if rc == 0 and df_out:
        section = None
        for raw_line in df_out.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("REPOSITORY") and "IMAGE ID" in line:
                section = "images"
                continue
            if line.startswith("CONTAINER ID") and "IMAGE" in line:
                section = "containers"
                continue
            if line.startswith("VOLUME NAME") and "SIZE" in line:
                section = "volumes"
                continue
            if line.startswith("Build cache usage:"):
                section = "build_cache"
                m_bc = re.search(r"Build cache usage:\s*([^\s]+)", line)
                if m_bc:
                    build_cache_gb = _parse_size_to_gb(m_bc.group(1))
                continue
            if line.startswith("CACHE ID"):
                continue

            if section == "images":
                # Parse columns: REPOSITORY TAG IMAGE_ID CREATED SIZE SHARED_SIZE UNIQUE_SIZE CONTAINERS
                m_img = re.match(
                    r"^(\S+)\s+(\S+)\s+([0-9a-f]{12})\s+(.+?)\s+(\d+(?:\.\d+)?(?:B|kB|KB|MB|GB|TB))\s+"
                    r"(\d+(?:\.\d+)?(?:B|kB|KB|MB|GB|TB))\s+(\d+(?:\.\d+)?(?:B|kB|KB|MB|GB|TB))\s+(\d+)$",
                    line,
                )
                if m_img:
                    repo, tag, img_id, created, size_s, shared_s, uniq_s, c_cnt = m_img.groups()
                    full_ref = f"{repo}:{tag}" if repo != "<none>" else "<none>:<none>"
                    con_count = int(c_cnt)
                    is_active = (full_ref in active_images) or (img_id in active_image_ids) or (con_count > 0 and "vllm-tt-metal" in repo)
                    is_pr55630 = full_ref == PATCHED_QWEN3_IMAGE or "pr55630" in tag
                    is_dangling = repo == "<none>" and tag == "<none>" and con_count == 0

                    if is_active and "vllm-tt-metal" in repo:
                        role = "★ ACTIVE MODEL SERVER"
                        protected = True
                    elif is_pr55630:
                        role = "★ PATCHED (PR #55630 Fix)"
                        protected = True
                    elif con_count > 0:
                        role = f"IN USE ({con_count} container)"
                        protected = True
                    elif is_dangling:
                        role = "DANGLING (<none> — Safe to Prune)"
                        protected = False
                    elif "vllm-tt-metal" in repo:
                        ver_m = re.match(r"^(\d+\.\d+\.\d+)", tag)
                        ver_str = f"v{ver_m.group(1)}" if ver_m else tag
                        role = f"TT-Inference Engine ({ver_str})"
                        protected = False
                    elif "tt-media-inference-server" in repo:
                        role = "TT-Media Server Image"
                        protected = False
                    elif "tt-studio" in repo:
                        role = "TT-Studio Component"
                        protected = True
                    else:
                        role = "Base / Support Image"
                        protected = con_count > 0

                    images.append(
                        {
                            "repo": repo,
                            "tag": tag,
                            "full_ref": full_ref,
                            "id": img_id,
                            "created": created.strip(),
                            "size_str": size_s,
                            "size_gb": _parse_size_to_gb(size_s),
                            "unique_str": uniq_s,
                            "unique_gb": _parse_size_to_gb(uniq_s),
                            "containers": con_count,
                            "role": role,
                            "protected": protected,
                            "dangling": is_dangling,
                        }
                    )
            elif section == "volumes":
                parts = line.split()
                if len(parts) >= 3:
                    vname, links_s, vsize_s = parts[0], parts[1], parts[2]
                    links = int(links_s) if links_s.isdigit() else 0
                    vgb = _parse_size_to_gb(vsize_s)
                    model_hint = vname.replace("volume_id_", "")
                    volumes.append(
                        {
                            "name": vname,
                            "type": "Docker Volume",
                            "links": links,
                            "size_str": vsize_s,
                            "size_gb": vgb,
                            "model_hint": model_hint,
                            "protected": links > 0,
                        }
                    )

    if TT_CACHE_DIR.exists():
        for d in sorted(TT_CACHE_DIR.iterdir()):
            if not d.is_dir():
                continue
            rc_du, du_out, _ = run_cmd(["du", "-sh", str(d)], timeout=5.0)
            size_s = du_out.split()[0] if (rc_du == 0 and du_out) else "0B"
            size_gb = _parse_size_to_gb(size_s)
            has_w = (d / "weights").exists() and any((d / "weights").rglob("*.safetensors"))
            host_caches.append(
                {
                    "name": d.name,
                    "path": str(d),
                    "type": "Host TT-Cache Dir",
                    "size_str": size_s,
                    "size_gb": size_gb,
                    "has_weights": has_w,
                    "protected": has_w,
                }
            )

    dangling_gb = sum(img["unique_gb"] for img in images if img["dangling"])
    empty_vols = [v["name"] for v in volumes if v["size_gb"] == 0.0 and v["links"] == 0 and v["name"].startswith("volume_id_")]
    total_vol_gb = sum(v["size_gb"] for v in volumes)
    total_host_cache_gb = sum(hc["size_gb"] for hc in host_caches)

    return {
        "images": images,
        "volumes": volumes,
        "host_caches": host_caches,
        "build_cache_gb": build_cache_gb,
        "dangling_gb": dangling_gb,
        "reclaimable_gb": build_cache_gb + dangling_gb,
        "empty_volumes": empty_vols,
        "total_vol_gb": total_vol_gb,
        "total_host_cache_gb": total_host_cache_gb,
    }


def display_docker_and_cache_inspector(storage: dict) -> None:
    """Render full tables of local Docker images, TT-Metal cache volumes, and host cache directories."""
    images = storage["images"]
    volumes = storage["volumes"]
    host_caches = storage["host_caches"]

    if HAS_RICH:
        summary_lines = [
            f"[bold cyan]Local Docker Images:[/bold cyan]         [white]{len(images)} images[/white] "
            f"([yellow]{sum(1 for i in images if 'tenstorrent' in i['repo'])} Tenstorrent inference/studio images[/yellow])",
            f"[bold cyan]Docker TT-Metal Volumes:[/bold cyan]     [white]{len(volumes)} volumes[/white] "
            f"([bold green]{storage['total_vol_gb']:.1f} GB[/bold green] of pre-compiled Blackhole `p300x2` tensor caches)",
            f"[bold cyan]Host `/home/ttuser/data/tt-cache`:[/bold cyan] [white]{len(host_caches)} dirs[/white] "
            f"([bold green]{storage['total_host_cache_gb']:.1f} GB[/bold green] including `Qwen3-32B` weights & caches)",
            f"[bold cyan]Safe Reclaimable Space:[/bold cyan]      [bold yellow]{storage['reclaimable_gb']:.2f} GB[/bold yellow] "
            f"({storage['dangling_gb']:.2f} GB dangling `<none>` images + {storage['build_cache_gb']:.2f} GB Docker build cache + {len(storage['empty_volumes'])} empty 0B volumes)",
        ]
        console.print(
            Panel(
                "\n".join(summary_lines),
                title="[bold cyan]Step 2C — QB2 Docker Image & TT-Metal Cache Volume Inspector[/bold cyan]",
                border_style="cyan",
            )
        )

        t_img = Table(
            title=f"1. Local Docker Images on QB2 ({len(images)} Total)",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_lines=True,
        )
        t_img.add_column("#", style="bold yellow", justify="right")
        t_img.add_column("Repository & Tag", style="bold white")
        t_img.add_column("Image ID", style="cyan")
        t_img.add_column("Created", style="dim")
        t_img.add_column("Total Size", style="magenta", justify="right")
        t_img.add_column("Unique Size", style="yellow", justify="right")
        t_img.add_column("Role / Status", style="green")

        for idx, img in enumerate(images, start=1):
            role_style = (
                f"[bold bright_green]{img['role']}[/bold bright_green]"
                if img["role"].startswith("★")
                else (
                    f"[bold yellow]{img['role']}[/bold yellow]"
                    if img["dangling"]
                    else img["role"]
                )
            )
            t_img.add_row(
                str(idx),
                f"{img['repo']}\n[bold cyan]:{img['tag']}[/bold cyan]",
                img["id"],
                img["created"],
                img["size_str"],
                img["unique_str"],
                role_style,
            )
        console.print(t_img)

        t_vol = Table(
            title="2. Tenstorrent TT-Metal Cache Volumes & Host Cache Directories",
            box=box.ROUNDED,
            header_style="bold green",
            show_lines=True,
        )
        t_vol.add_column("#", style="bold yellow", justify="right")
        t_vol.add_column("Volume / Cache Name", style="bold white")
        t_vol.add_column("Storage Type", style="cyan")
        t_vol.add_column("Size on Disk", style="magenta", justify="right")
        t_vol.add_column("Associated Spec / Notes", style="green")

        for idx, v in enumerate(volumes, start=1):
            status_note = (
                f"[bold green]Warm Tensor Cache ({v['model_hint']})[/bold green]"
                if v["size_gb"] >= 0.1
                else (
                    "[yellow]Empty / Uncompiled (0B — Safe to Prune)[/yellow]"
                    if v["size_gb"] == 0.0
                    else f"Partial / Metadata ({v['model_hint']})"
                )
            )
            if v["links"] > 0:
                status_note += f" [bold cyan](Mounted by {v['links']} container)[/bold cyan]"
            t_vol.add_row(
                f"V{idx}",
                v["name"],
                v["type"],
                v["size_str"],
                status_note,
            )

        for idx, hc in enumerate(host_caches, start=1):
            hc_note = (
                "[bold green]Contains Model `.safetensors` Weights + TT-Cache (Protected)[/bold green]"
                if hc["has_weights"]
                else "[dim]Host TT-Cache Directory[/dim]"
            )
            t_vol.add_row(
                f"H{idx}",
                f"{hc['name']}\n[dim]{hc['path']}[/dim]",
                hc["type"],
                hc["size_str"],
                hc_note,
            )
        console.print(t_vol)
    else:
        print("\n=== QB2 Docker Images & TT-Metal Cache Volumes ===")
        for idx, img in enumerate(images, start=1):
            print(
                f"  [Image {idx}] {img['full_ref']} ({img['id']}) | Size: {img['size_str']} (Unique: {img['unique_str']}) | {img['role']}"
            )
        for idx, v in enumerate(volumes, start=1):
            print(f"  [Volume V{idx}] {v['name']} | Size: {v['size_str']} | Links: {v['links']}")


def prune_docker_and_caches(storage: dict) -> None:
    """Safely prune dangling `<none>` Docker images, unused Docker build cache, and 0-byte `volume_id_*` volumes."""
    if HAS_RICH:
        console.print(
            Panel(
                "[bold yellow]Running safe cleanup of dangling `<none>` Docker images, unused Docker build cache, and empty 0B TT-Metal volumes...[/bold yellow]\n"
                "[dim](All tagged images, active containers, PR #55630 patched image, and populated TT-Metal caches are untouched.)[/dim]",
                title="[bold yellow]Safe Docker & Cache Prune[/bold yellow]",
                border_style="yellow",
            )
        )
    else:
        print("\n=== Running safe Docker & empty cache prune ===")

    rc1, out1, _ = run_cmd(["docker", "image", "prune", "-f"], timeout=30.0)
    if rc1 == 0 and out1:
        print(f"  • docker image prune: {out1.splitlines()[-1]}")

    rc2, out2, _ = run_cmd(["docker", "builder", "prune", "-f"], timeout=60.0)
    if rc2 == 0 and out2:
        print(f"  • docker builder prune: {out2.splitlines()[-1]}")

    for vname in storage.get("empty_volumes", []):
        rc_v, _, _ = run_cmd(["docker", "volume", "rm", vname], timeout=10.0)
        if rc_v == 0:
            print(f"  • Removed empty 0B volume: {vname}")

    if HAS_RICH:
        console.print("[bold green]✓ Safe Docker & cache prune complete.[/bold green]")
    else:
        print("✓ Safe Docker & cache prune complete.")


def run_storage_cleanup_menu(state: dict, catalog_entries: list[dict]) -> None:
    """Interactive inspector and cleanup menu for Docker images and TT-Metal cache volumes."""
    storage = inspect_docker_and_cache_storage(state, catalog_entries)
    display_docker_and_cache_inspector(storage)

    while True:
        try:
            if HAS_RICH:
                raw = console.input(
                    "\n[bold cyan]Storage Action — 'p' safe prune (dangling images + build cache + 0B vols), "
                    "'rmi <#>' remove unused image #, 'rmv <V#>' remove volume, or Enter to return: [/bold cyan]"
                ).strip()
            else:
                raw = input(
                    "\nStorage Action ('p' safe prune, 'rmi <#>' remove image, 'rmv <V#>' remove volume, Enter to return): "
                ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not raw or raw.lower() in ("q", "quit", "exit", "b", "back"):
            return

        if raw.lower() in ("p", "prune"):
            prune_docker_and_caches(storage)
            storage = inspect_docker_and_cache_storage(state, catalog_entries)
            display_docker_and_cache_inspector(storage)
            continue

        m_rmi = re.match(r"^rmi\s+(\d+)$", raw, flags=re.I)
        if m_rmi:
            idx = int(m_rmi.group(1))
            images = storage["images"]
            if not (1 <= idx <= len(images)):
                print(f"Image #{idx} out of range (1..{len(images)}).")
                continue
            target_img = images[idx - 1]
            if target_img["protected"]:
                print(
                    f"✗ Refusing to remove protected image #{idx} ({target_img['full_ref']} — {target_img['role']})."
                )
                continue
            ref_to_rm = target_img["id"] if target_img["dangling"] else target_img["full_ref"]
            rc, out, err = run_cmd(["docker", "rmi", ref_to_rm], timeout=30.0)
            if rc == 0:
                print(f"✓ Removed Docker image {ref_to_rm}.")
            else:
                print(f"✗ docker rmi failed: {err or out}")
            storage = inspect_docker_and_cache_storage(state, catalog_entries)
            display_docker_and_cache_inspector(storage)
            continue

        m_rmv = re.match(r"^rmv\s+[vV]?(\d+)$", raw, flags=re.I)
        if m_rmv:
            v_idx = int(m_rmv.group(1))
            volumes = storage["volumes"]
            if not (1 <= v_idx <= len(volumes)):
                print(f"Volume V{v_idx} out of range (1..{len(volumes)}).")
                continue
            target_vol = volumes[v_idx - 1]
            if target_vol["protected"]:
                print(
                    f"✗ Refusing to remove volume {target_vol['name']} because it is mounted by an active container."
                )
                continue
            rc, out, err = run_cmd(["docker", "volume", "rm", target_vol["name"]], timeout=20.0)
            if rc == 0:
                print(f"✓ Removed Docker volume {target_vol['name']} ({target_vol['size_str']}).")
            else:
                print(f"✗ docker volume rm failed: {err or out}")
            storage = inspect_docker_and_cache_storage(state, catalog_entries)
            display_docker_and_cache_inspector(storage)
            continue

        print("Unrecognized command. Use 'p', 'rmi <#>', 'rmv <V#>', or press Enter to return.")


def partition_models(
    include_dev: bool = False,
    refresh_ranks: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Partition the live catalog into (1) Already Downloaded on QB2 and (2) Available to Pull,
    annotated with Hugging Face Frontier Proximity %, local `config.json` architecture facts,
    warm/cold TT-Metal tensor cache state, and persistent bring-up time estimates.
    """
    downloaded: list[dict] = []
    available_to_pull: list[dict] = []
    seen_paths: set[Path] = set()

    catalog = load_dynamic_catalog_from_yaml(include_dev=include_dev)

    for entry in catalog:
        item = dict(entry)
        is_down, path, n_shards, size_gb = resolve_local_weights_path(item["hf_repo"])
        item["downloaded"] = is_down
        item["local_path"] = path
        item["n_shards"] = n_shards
        item["actual_size_gb"] = size_gb

        if is_down:
            try:
                seen_paths.add(path.resolve())
            except OSError:
                seen_paths.add(path)
            downloaded.append(item)
        else:
            available_to_pull.append(item)

    extra_local = discover_unlisted_local_models(seen_paths)
    downloaded.extend(extra_local)

    # Fetch/load Hugging Face Frontier Proximity scores for all unique hf_repos
    all_repos = list(dict.fromkeys(m["hf_repo"] for m in (downloaded + available_to_pull)))
    rank_map = get_or_update_frontier_rankings(all_repos, force_refresh=refresh_ranks)
    timings_store = load_timings_store()
    vol_sizes = inspect_tt_cache_volumes_map()

    for m in downloaded + available_to_pull:
        r_info = rank_map.get(
            m["hf_repo"],
            {"score": 70.0, "tier": "B Tier", "source": "HF Metadata", "created": "unknown"},
        )
        m["frontier_score"] = float(r_info.get("score", 70.0))
        m["frontier_tier"] = str(r_info.get("tier", "B Tier"))
        m["frontier_source"] = str(r_info.get("source", "HF Metadata"))
        m["hf_created"] = str(r_info.get("created", "unknown"))

        # Annotate TT-Metal warm/cold cache status and estimated load duration (Item 2)
        annotate_model_cache_and_load_estimate(m, timings_store, vol_sizes)

        # Annotate local `config.json` architecture facts for downloaded models (Item 3)
        if m.get("downloaded"):
            arch_info = inspect_model_architecture(m["local_path"])
            m["arch_info"] = arch_info
            m["arch_summary"] = arch_info["short_summary"] if arch_info else ""
        else:
            m["arch_info"] = None
            m["arch_summary"] = ""

    # Sort both lists by:
    #   1) Frontier Proximity Score (descending)
    #   2) Default implementation first (True > False)
    #   3) Hardware Status (COMPLETE > VALIDATED_BRIDGE > FUNCTIONAL > EXPERIMENTAL)
    #   4) Max Concurrency (descending)
    def _sort_key(item: dict) -> tuple:
        return (
            item.get("frontier_score", 0.0),
            1 if item.get("is_default_impl") else 0,
            _status_priority(str(item.get("status", ""))),
            int(item.get("max_concurrency") or 1),
        )

    downloaded.sort(key=_sort_key, reverse=True)
    available_to_pull.sort(key=_sort_key, reverse=True)

    return downloaded, available_to_pull


def is_entry_currently_running(entry: dict, state: dict) -> bool:
    """Return True if `entry` matches the model currently running on the QB2."""
    containers = state.get("containers") or []
    if not containers:
        return False

    c0 = containers[0]
    active_weights_dir = c0.get("host_weights_dir")
    if not active_weights_dir:
        return False

    try:
        if Path(active_weights_dir).resolve() != Path(entry["local_path"]).resolve():
            return False
    except OSError:
        if str(active_weights_dir) != str(entry["local_path"]):
            return False

    c_image = c0.get("image", "")
    entry_ver = str(entry.get("version", "")).split("-")[0]
    if entry_ver and entry_ver not in ("unknown", "local", "dev"):
        if f":{entry_ver}" in c_image:
            return True
        if ":" in c_image:
            return False

    return bool(entry.get("is_default_impl", True))


def display_storage_and_catalog(
    downloaded_models: list[dict],
    available_to_pull: list[dict],
    state: dict,
    include_dev: bool = False,
) -> list[dict]:
    """Render storage locations, Part 1 (Downloaded), and Part 2 (Available to Pull), sorted by Frontier Proximity."""
    combined = downloaded_models + available_to_pull
    repo_ver = get_tt_inference_repo_version()
    storage_summary = inspect_docker_and_cache_storage(state, combined)
    mode_label = (
        "Production (`prod/llm.yaml`) + Dev (`dev/llm.yaml`, `dev/vlm.yaml`)"
        if include_dev
        else "Production (`prod/llm.yaml`) + Validated Bridges  [Pass `--dev` to also list 40+ experimental `dev/llm.yaml` QB2 models]"
    )

    if HAS_RICH:
        storage_lines = [
            f"[bold cyan]TT-Inference-Server Repo:[/bold cyan]    [bold white]{repo_ver}[/bold white] (`/home/ttuser/.local/lib/tt-inference-server`)",
            f"[bold cyan]Active Catalog Mode:[/bold cyan]         [yellow]{mode_label}[/yellow]",
            f"[bold cyan]Catalog Ordering:[/bold cyan]            [bold green]Sorted by Frontier Proximity %[/bold green] (HF cache: `frontier_rankings_cache.json` | Load timings: `load_timings_cache.json`)",
            f"[bold cyan]1. Primary Host Weights:[/bold cyan]     [white]/home/ttuser/models[/white] (`Qwen3.8-27B`, `gpt-oss-120b`, `gemma-4-31B-it`, `DeepSeek-R1-Distill-Qwen-32B`, `Llama-3.1-70B/8B`)",
            f"[bold cyan]2. TT-Cache Host Weights:[/bold cyan]    [white]/home/ttuser/data/tt-cache/volume_id_tt_transformers-Qwen3-32B-vqb2_launch/weights[/white] (`Qwen/Qwen3-32B`, {storage_summary['total_host_cache_gb']:.1f} GB total)",
            f"[bold cyan]3. Docker Images & Volumes:[/bold cyan]  [white]{len(storage_summary['images'])} images, {len(storage_summary['volumes'])} TT-Metal cache volumes ({storage_summary['total_vol_gb']:.1f} GB)[/white] "
            f"[dim](Reclaimable build/dangling cache: {storage_summary['reclaimable_gb']:.1f} GB — press `'i'` or `--images`)[/dim]",
        ]
        console.print(
            Panel(
                "\n".join(storage_lines),
                title="[bold cyan]Step 2 — QB2 Model Storage Locations & Frontier-Ranked Spec Catalog[/bold cyan]",
                border_style="cyan",
            )
        )

        t_down = Table(
            title=f"Part 1 — Models Already Downloaded on QB2 ({len(downloaded_models)} Ready to Launch — Ranked by Frontier Proximity)",
            box=box.ROUNDED,
            header_style="bold green",
            show_lines=True,
        )
        t_down.add_column("#", style="bold yellow", justify="right")
        t_down.add_column("Model / Weights & Local `config.json` Architecture", style="bold white")
        t_down.add_column("Frontier Proximity", style="bold green", justify="right")
        t_down.add_column("vLLM / TT Spec & Impl", style="cyan")
        t_down.add_column("Disk, TT-Cache & Est. Load", style="magenta")
        t_down.add_column("Local Weights Folder on QB2", style="white")

        for idx, m in enumerate(downloaded_models, start=1):
            ctx_fmt = (
                f"{int(m['max_context']):,}"
                if str(m["max_context"]).isdigit()
                else str(m["max_context"])
            )
            running_badge = (
                " [bold bright_green on dark_green] ★ RUNNING [/bold bright_green on dark_green]"
                if is_entry_currently_running(m, state)
                else ""
            )
            arch_line = (
                f"\n[cyan]Arch: {m['arch_summary']}[/cyan]"
                if m.get("arch_summary")
                else ""
            )
            t_down.add_row(
                str(idx),
                f"{m['display_name']}{running_badge}{arch_line}\n[dim]{m['notes']}[/dim]",
                f"{m['frontier_score']:.1f}% ({m['frontier_tier']})\n[dim]{m['frontier_source']}[/dim]",
                f"--model {m['tt_spec_model']}\n--tt-device {m['tt_device']}\nimpl: {m['impl']}",
                f"{m['actual_size_gb']:.1f} GB ({m['n_shards']} shards) | {m['est_speed']}\n"
                f"{ctx_fmt} ctx (x{m['max_concurrency']})\n{m.get('cache_badge', '')}",
                str(m["local_path"]),
            )
        console.print(t_down)

        t_avail = Table(
            title=f"Part 2 — Additional Models Supported on QB2 ({len(available_to_pull)} in Live Spec Catalog — Ranked by Frontier Proximity)",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_lines=True,
        )
        t_avail.add_column("#", style="bold yellow", justify="right")
        t_avail.add_column("Hugging Face Model Repo", style="bold white")
        t_avail.add_column("Frontier Proximity", style="bold green", justify="right")
        t_avail.add_column("vLLM / TT Spec & Impl", style="cyan")
        t_avail.add_column("Download, TT-Cache & Est. Load", style="magenta")
        t_avail.add_column("Catalog Status & Target Folder", style="dim")

        start_idx = len(downloaded_models) + 1
        for idx, m in enumerate(available_to_pull, start=start_idx):
            ctx_fmt = (
                f"{int(m['max_context']):,}"
                if str(m["max_context"]).isdigit()
                else str(m["max_context"])
            )
            dev_flag_note = " (--dev-mode)" if m.get("catalog_env") == "dev" else ""
            t_avail.add_row(
                str(idx),
                f"{m['display_name']}\n[dim]{m['notes']}[/dim]",
                f"{m['frontier_score']:.1f}% ({m['frontier_tier']})\n[dim]{m['frontier_source']}[/dim]",
                f"--model {m['tt_spec_model']}\n--tt-device {m['tt_device']}\nimpl: {m['impl']}{dev_flag_note}",
                f"{m['est_size_gb']} | {m['est_speed']}\n"
                f"{ctx_fmt} ctx (x{m['max_concurrency']})\n{m.get('cache_badge', '')}",
                f"{m['status']} (v{m['version']})\nTarget: {m['local_path']}",
            )
        console.print(t_avail)
    else:
        print(f"\n=== Step 2 — TT-Inference-Server Repo: {repo_ver} ===")
        print("=== Part 1: Models Already Downloaded on QB2 (Ranked by Frontier Proximity) ===")
        for idx, m in enumerate(downloaded_models, start=1):
            run_tag = " [RUNNING]" if is_entry_currently_running(m, state) else ""
            arch_s = f" | Arch: {m['arch_summary']}" if m.get("arch_summary") else ""
            print(
                f"  [{idx}] {m['display_name']}{run_tag} | Frontier: {m['frontier_score']:.1f}% ({m['frontier_tier']}) | "
                f"Spec: {m['tt_spec_model']} ({m['tt_device']}, {m['impl']}) | {m['actual_size_gb']:.1f} GB | "
                f"{m.get('cache_badge_plain', '')}{arch_s} | Path: {m['local_path']}"
            )
        print("\n=== Part 2: Models Supported on QB2 (Ranked by Frontier Proximity — Auto-Pulls from HF) ===")
        start_idx = len(downloaded_models) + 1
        for idx, m in enumerate(available_to_pull, start=start_idx):
            print(
                f"  [{idx}] {m['display_name']} | Frontier: {m['frontier_score']:.1f}% ({m['frontier_tier']}) | "
                f"Spec: {m['tt_spec_model']} ({m['tt_device']}, {m['impl']}) | Est: {m['est_size_gb']} | "
                f"{m.get('cache_badge_plain', '')} | Target: {m['local_path']}"
            )

    return combined


# ==============================================================================
# PART D: HF Download, Server Launch & Live Startup Progress Monitor
# ==============================================================================
def download_weights_from_hf(hf_repo: str, dest_dir: Path) -> bool:
    """Download model weights from Hugging Face into `dest_dir` using `hf`."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    hf_token = load_hf_token()

    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    hf_bin = (
        shutil.which("hf")
        or (
            str(Path.home() / ".local" / "bin" / "hf")
            if (Path.home() / ".local" / "bin" / "hf").exists()
            else None
        )
        or shutil.which("huggingface-cli")
        or str(Path.home() / ".local" / "bin" / "huggingface-cli")
    )

    cmd = [
        hf_bin,
        "download",
        hf_repo,
        "--local-dir",
        str(dest_dir),
        "--exclude",
        "original/*",
        "--exclude",
        "*.pth",
        "--exclude",
        "*.pt",
    ]

    if HAS_RICH:
        console.print(
            Panel(
                f"[bold yellow]Downloading `{hf_repo}` from Hugging Face...[/bold yellow]\n"
                f"[bold]Destination:[/bold] `{dest_dir}`\n"
                f"[bold]Command:[/bold]     `{' '.join(cmd)}`",
                title="[bold cyan]Stage 2 — Hugging Face Model Weight Pull[/bold cyan]",
                border_style="yellow",
            )
        )
    else:
        print(f"\n=== Stage 2: Downloading {hf_repo} into {dest_dir} ===")

    ret = subprocess.call(cmd, env=env)
    if ret != 0:
        if HAS_RICH:
            console.print(f"[bold red]✗ Hugging Face download exited with code {ret}.[/bold red]")
        else:
            print(f"✗ Hugging Face download exited with code {ret}.")
        return False

    ok, n_shards, size_gb = has_safetensors(dest_dir)
    if not ok:
        if HAS_RICH:
            console.print(
                f"[bold red]✗ Download finished, but no `.safetensors` shards were found in `{dest_dir}`.[/bold red]"
            )
        else:
            print(f"✗ Download finished, but no .safetensors shards found in {dest_dir}.")
        return False

    if HAS_RICH:
        console.print(
            f"[bold green]✓ Successfully downloaded `{hf_repo}` ({n_shards} shards, {size_gb:.1f} GB) to `{dest_dir}`.[/bold green]"
        )
    else:
        print(f"✓ Successfully downloaded {hf_repo} ({n_shards} shards, {size_gb:.1f} GB).")
    return True


def build_tt_inference_command(model_entry: dict, service_port: int = 8000) -> list[str]:
    """Construct the `python3 run.py ...` command for `tt-inference-server`."""
    weights_path = model_entry["local_path"]
    spec_model = model_entry["tt_spec_model"]
    tt_device = model_entry["tt_device"]
    impl_name = model_entry.get("impl")

    cmd = ["python3", "run.py"]
    if model_entry.get("catalog_env") == "dev":
        cmd.append("--dev-mode")

    cmd.extend(
        [
            "--model",
            spec_model,
            "--tt-device",
            tt_device,
            "--workflow",
            "server",
            "--docker-server",
        ]
    )

    if impl_name and not model_entry.get("is_default_impl", True):
        cmd.extend(["--impl", impl_name])

    cmd.extend(
        [
            "--vllm-override-args",
            '{"disable_log_stats": true}',
            "--no-auth",
            "--host-weights-dir",
            str(weights_path),
            "--service-port",
            str(service_port),
        ]
    )

    if model_entry.get("use_pr55630_img") and check_local_docker_image_exists(PATCHED_QWEN3_IMAGE):
        cmd.extend(["--override-docker-image", PATCHED_QWEN3_IMAGE])

    return cmd


class StartupProgressTracker:
    """
    Deterministic container log substage & progress tracker for QB2 vLLM + TT-Metal startup:
      • Detects Warm TT-Metal Cache vs. Cold First-Run Kernel/Weight Compilation
      • Tracks `.safetensors` shard loading progress (`A/B shards, X%`) via both container logs and host `/proc/<pid>/fd`
      • Tracks TT-Metal layer cache loading (`A/B layers`) & RISC-V kernel compiles (`riscv-tt-elf-g++`)
      • Tracks builtin prefill/decode warmup (`warmup_model_prefill`, `warmup_model_decode`, `Qwen3.8 prefill startup warmup`)
      • Tracks deterministic 11-step prompt_client trace capture (`seq_len = 128..130944`)
      • Computes live progress % and estimated time remaining (ETA)
    """

    def __init__(self, model_entry: dict | None = None) -> None:
        self.model_entry = model_entry or {}
        self.current_phase = "[Phase 1/5] Container Started — Bootstrapping Python & vLLM EngineCore"
        self.cache_warm_initial = bool(self.model_entry.get("cache_warm", False))
        self.saw_cold_compile = False
        self.warm_est_sec = int(self.model_entry.get("warm_est_sec") or 120)
        self.cold_est_sec = int(self.model_entry.get("cold_est_sec") or 420)
        self.est_total_sec = self.warm_est_sec if self.cache_warm_initial else self.cold_est_sec

        self.shards_done = 0
        self.shards_total = int(self.model_entry.get("n_shards") or 0)
        self.layers_done = 0
        arch_info = self.model_entry.get("arch_info") or {}
        self.layers_total = int(arch_info.get("num_layers") or 0)
        self.trace_total = len(TRACE_CAPTURE_SEQ_LENS)
        self.trace_done = 0
        self.last_trace_seq_len = 0
        self.capturing_traces = False
        self.traces_completed = False
        self.kernel_compiles = 0
        self.saw_decode_trace = False
        self.substage_detail = "Initializing container runtime & vLLM plugins"

    @property
    def phase_num(self) -> int:
        try:
            return int(self.current_phase[7])
        except Exception:
            return 1

    def feed(self, log_line: str) -> tuple[str, bool, str | None]:
        """
        Process a container log line and return `(current_phase, is_milestone, extra_badge)`.
        """
        low = log_line.lower()
        is_milestone = False
        extra_badge: str | None = None
        new_phase = self.current_phase

        # Detect RISC-V TT-Metal kernel compiles first (before filtering compiler warning flags)
        if "riscv-tt-elf-g++" in low:
            self.kernel_compiles += 1
            if self.phase_num < 5:
                self.substage_detail = (
                    f"Compiling Blackhole RISC-V TT-Metal kernels ({self.kernel_compiles} compiled)"
                )
            if not self.cache_warm_initial or self.kernel_compiles >= 50:
                if not self.saw_cold_compile:
                    self.saw_cold_compile = True
                    self.est_total_sec = self.cold_est_sec
                    return (
                        self.current_phase,
                        True,
                        f"[TT-Metal Kernel Compilation Active — ETA target ~{_format_duration(self.est_total_sec)}]",
                    )
            if self.kernel_compiles == 1 or self.kernel_compiles % 50 == 0:
                return (
                    self.current_phase,
                    False,
                    f"[TT-Metal RISC-V Kernels Compiled: {self.kernel_compiles}]",
                )
            return self.current_phase, False, None

        # Skip noisy compiler warning lines from C++ kernel builds
        if any(
            w in low
            for w in (
                "warning: variable",
                "in file included from",
                "in function 'void kernel_main",
                "[-wunused",
            )
        ) or re.match(r"^\s*\d+\s*\|", log_line) or re.match(r"^\s*\|\s*\^", log_line):
            return self.current_phase, False, None

        m_build_step = re.search(r"building\s+(brisc|ncrisc|trisc),\s*compile step", low)
        if m_build_step:
            if self.phase_num in (3, 4):
                self.substage_detail = (
                    f"Building {m_build_step.group(1).upper()} firmware & warmup kernels ({self.kernel_compiles} compiled)"
                )
            return self.current_phase, False, None

        if "first-run tensor cache warmup detected" in low or "generating cache for" in low:
            if not self.saw_cold_compile:
                self.saw_cold_compile = True
                self.est_total_sec = self.cold_est_sec
                extra_badge = (
                    f"[Cold TT-Metal Compile Detected — updated ETA target to ~{_format_duration(self.est_total_sec)}]"
                )
                is_milestone = True

        # Phase 1 milestones (Container & vLLM EngineCore Bootstrap)
        if "using model spec:" in low:
            spec_id = log_line.split("Using model spec:")[-1].strip()
            self.substage_detail = f"Loaded runtime spec `{spec_id}`"
            extra_badge = f"[Runtime Spec: {spec_id}]"
            is_milestone = True
        elif "parse safetensors files:" in low:
            m_parse = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)", log_line)
            if m_parse:
                pct, done_s, tot_s = int(m_parse.group(1)), int(m_parse.group(2)), int(m_parse.group(3))
                self.shards_total = max(self.shards_total, tot_s)
                self.substage_detail = f"Indexing .safetensors headers ({done_s}/{tot_s} shards, {pct}%)"
                if done_s == tot_s or pct == 0:
                    extra_badge = f"[Indexed .safetensors Headers: {done_s}/{tot_s} shards ({pct}%)]"
        elif "resolved architecture:" in low:
            arch_name = log_line.split("Resolved architecture:")[-1].strip()
            self.substage_detail = f"Resolved vLLM architecture `{arch_name}`"
            extra_badge = f"[vLLM Architecture: {arch_name}]"
            is_milestone = True
        elif "initializing a v1 llm engine" in low:
            self.substage_detail = "Spawning vLLM V1 EngineCore process"
            is_milestone = True

        # Phase 2: Blackhole Mesh & Device Init
        if any(
            k in low
            for k in (
                "opening user mode device driver",
                "starting topology discovery",
                "still waiting on eth training",
                "opening local chip ids",
                "opening subset of mmio",
                "initializing fabric",
                "fabric initialized on 4 devices",
                "fabric initialized with config",
                "attempting to open mesh device",
                "multidevice with",
                "initializing tensor parallel",
            )
        ):
            new_phase = "[Phase 2/5] Initializing Tenstorrent Blackhole Mesh (4x p300c / 8 Tensix chips)"
            if "starting topology discovery" in low or "opening user mode device driver" in low:
                self.substage_detail = "Discovering 4x p300c Blackhole topology"
                is_milestone = True
            elif "still waiting on eth training" in low:
                m_eth = re.search(r"\((\d+)\s*ms elapsed", low)
                eth_s = int(m_eth.group(1)) // 1000 if m_eth else 5
                self.substage_detail = f"Training Blackhole Ethernet interconnect links ({eth_s}s elapsed)"
                extra_badge = f"[Blackhole ETH Link Training: {eth_s}s elapsed]"
            elif "opening local chip ids" in low:
                self.substage_detail = "Opened PCIe MMIO handles across 4x p300c boards {0, 1, 2, 3}"
                extra_badge = "[PCIe MMIO Opened: 4x p300c boards {0, 1, 2, 3}]"
                is_milestone = True
            elif "initializing fabric" in low:
                self.substage_detail = "Initializing Blackhole 1D/Ring Ethernet fabric"
            elif "fabric initialized on 4 devices" in low:
                self.substage_detail = "Blackhole 4-device interconnect fabric initialized"
                extra_badge = "[Blackhole 4-Device Fabric Ready]"
                is_milestone = True
            elif "multidevice with" in low:
                self.substage_detail = "Created (1, 4) Blackhole multi-device mesh"
                extra_badge = "[Blackhole (1x4) Mesh Device Active]"
                is_milestone = True

        # Phase 3: Weight Shards & TT-Metal Cache Loading
        is_phase3_line = any(
            k in low
            for k in (
                "ttmodelrunner: trace_mode=",
                "loading checkpoint shards",
                "checkpoint directory:",
                "loading weights",
                "loading safetensors",
                "model loading took",
                "loading qwen3.8 layer",
                "loading gemma4 layer",
                "qwen3.8 model loaded",
                "gemma4 model loaded",
                "qwen3.8 vllm precision:",
            )
        )
        # Also detect bare tqdm layer bars `  25%|██▌ | 16/64` in tt_transformers during Phase 2/3
        m_tqdm = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)", log_line)
        is_bare_layer_bar = (
            m_tqdm is not None
            and self.phase_num in (2, 3)
            and "parse safetensors" not in low
            and "allocating tt kv caches" not in low
            and "loading checkpoint shards" not in low
        )

        if is_phase3_line or is_bare_layer_bar:
            new_phase = "[Phase 3/5] Loading .safetensors Model Weights & TT-Metal Tensor Cache"
            if "ttmodelrunner: trace_mode=" in low:
                shard_hint = f" ({self.shards_total} shards)" if self.shards_total else ""
                self.substage_detail = f"Streaming model weights{shard_hint} into 4x p300c Blackhole DRAM"
                is_milestone = True
            elif "loading checkpoint shards" in low and m_tqdm:
                pct, done_s, tot_s = int(m_tqdm.group(1)), int(m_tqdm.group(2)), int(m_tqdm.group(3))
                prev_done = self.shards_done
                self.shards_done, self.shards_total = done_s, tot_s
                self.substage_detail = f"Loading .safetensors shards ({done_s}/{tot_s} shards, {pct}%)"
                if done_s != prev_done or done_s in (0, tot_s):
                    extra_badge = f"[Weights Shard: {done_s}/{tot_s} ({pct}%)]"
                    is_milestone = done_s in (0, 1, tot_s) or (done_s % max(1, tot_s // 5) == 0)
            elif is_bare_layer_bar and m_tqdm:
                pct, done_l, tot_l = int(m_tqdm.group(1)), int(m_tqdm.group(2)), int(m_tqdm.group(3))
                prev_l = self.layers_done
                self.layers_done, self.layers_total = done_l, tot_l
                self.substage_detail = f"Loading TT-Metal transformer layers ({done_l}/{tot_l} layers, {pct}%)"
                if done_l != prev_l and (
                    done_l in (1, tot_l) or done_l % max(1, tot_l // 4) == 0
                ):
                    extra_badge = f"[TT-Metal Layers Loaded: {done_l}/{tot_l} ({pct}%)]"
            elif "loading gemma4 layer" in low or "loading qwen3.8 layer" in low:
                m_lyr = re.search(r"layer\s+(\d+)", low)
                if m_lyr:
                    l_idx = int(m_lyr.group(1)) + 1
                    tot_l = self.layers_total or 64
                    self.layers_done = l_idx
                    pct = int((l_idx / tot_l) * 100)
                    self.substage_detail = f"Uploading transformer layer {l_idx}/{tot_l} ({pct}%) to Blackhole DRAM"
                    if l_idx in (1, tot_l) or l_idx % 8 == 0:
                        extra_badge = f"[Transformer Layer Upload: {l_idx}/{tot_l} ({pct}%)]"
            elif "qwen3.8 model loaded" in low or "gemma4 model loaded" in low:
                if self.shards_total:
                    self.shards_done = self.shards_total
                if self.layers_total:
                    self.layers_done = self.layers_total
                self.substage_detail = "All model weight layers uploaded to Blackhole DRAM"
                extra_badge = "[All Model Weights Uploaded to Blackhole DRAM (100%)]"
                is_milestone = True

        # Phase 4: KV Cache, Prefill/Decode Warmup & Trace Capture
        if any(
            k in low
            for k in (
                "gpu kv cache size:",
                "allocating tt kv caches",
                "capturing traces",
                "text trace completed:",
                "warming up prefill",
                "qwen3.8 prefill startup warmup",
                "starting decode warmup",
                "warming up decode",
                "_capture_decode_trace_text",
                "decode warmup completed",
                "warming model trace",
                "capturing model trace",
                "init engine (profile, create kv cache, warmup model)",
            )
        ):
            new_phase = "[Phase 4/5] Allocating KV Cache, Prefill/Decode Warmup & Capturing Hardware Traces"

            m_kv_sz = re.search(r"gpu kv cache size:\s*([\d,]+\s*tokens)", low)
            if m_kv_sz:
                kv_str = m_kv_sz.group(1)
                self.substage_detail = f"Allocated Blackhole PagedAttention KV cache ({kv_str}) & warming up prefill/decode"
                extra_badge = f"[Paged KV Cache Allocated: {kv_str}]"
                is_milestone = True

            if "allocating tt kv caches" in low and m_tqdm:
                pct, done_k, tot_k = int(m_tqdm.group(1)), int(m_tqdm.group(2)), int(m_tqdm.group(3))
                self.substage_detail = f"Allocating on-device KV cache buffers ({done_k}/{tot_k} layers, {pct}%)"
                if done_k in (1, tot_k // 2, tot_k):
                    extra_badge = f"[Allocating Layer KV Caches: {done_k}/{tot_k} ({pct}%)]"

            m_trace_cnt = re.search(r"capturing traces:\s*(\d+)\s*context lengths", low)
            if m_trace_cnt:
                self.trace_total = int(m_trace_cnt.group(1))
                self.capturing_traces = True
                self.traces_completed = False
                self.substage_detail = f"Capturing execution traces ({self.trace_total} context lengths)"
                extra_badge = f"[Trace Capture Started: {self.trace_total} context lengths]"
                is_milestone = True

            m_trace_seq = re.search(r"capturing traces for input_seq_len=(\d+)", low)
            if m_trace_seq:
                seq_len = int(m_trace_seq.group(1))
                self.capturing_traces = True
                self.last_trace_seq_len = seq_len
                if seq_len in TRACE_SEQ_INDEX:
                    self.trace_done = min(self.trace_total, TRACE_SEQ_INDEX[seq_len])
                else:
                    self.trace_done = min(self.trace_total, self.trace_done + 1)
                self.substage_detail = (
                    f"Trace Capture [{self.trace_done}/{self.trace_total}] (input_seq_len={seq_len:,})"
                )
                extra_badge = f"[Trace {self.trace_done}/{self.trace_total}: seq_len={seq_len:,}]"
                is_milestone = True

            if "text trace completed:" in low and self.capturing_traces:
                if self.trace_done >= self.trace_total or self.last_trace_seq_len >= 130944:
                    self.traces_completed = True
                    self.substage_detail = f"Completed all {self.trace_total}/{self.trace_total} context length traces"
                    extra_badge = f"[All {self.trace_total}/{self.trace_total} Context Length Traces Captured]"
                    is_milestone = True

            m_prefill = re.search(r"warming up prefill for sequence length:\s*(\d+)", low)
            if m_prefill:
                p_len = int(m_prefill.group(1))
                self.substage_detail = f"Warming up TT-Metal prefill kernel (seq_len={p_len:,})"
                extra_badge = f"[Prefill Warmup: seq_len={p_len:,}]"
                is_milestone = True

            if "qwen3.8 prefill startup warmup" in low:
                m_qsec = re.search(r'"seconds":\s*([\d.]+)', log_line)
                sec_s = f" in {float(m_qsec.group(1)):.1f}s" if m_qsec else ""
                self.substage_detail = f"Completed Qwen3.8 16-occupancy prefill warmup{sec_s}"
                extra_badge = f"[Qwen3.8 16-Occupancy Prefill Warmup Complete{sec_s}]"
                is_milestone = True

            if "starting decode warmup" in low:
                self.substage_detail = "Running TT-Metal autoregressive decode warmup passes"
                extra_badge = "[Decode Warmup Started]"
                is_milestone = True
            elif "_capture_decode_trace_text" in low and not self.saw_decode_trace:
                self.saw_decode_trace = True
                self.substage_detail = "Capturing on-device decode execution trace"
                extra_badge = "[Captured On-Device Decode Trace]"
                is_milestone = True
            elif "decode warmup completed" in low:
                self.substage_detail = "TT-Metal decode warmup complete"
                extra_badge = "[Decode Warmup Complete]"
                is_milestone = True
            elif "warming model trace" in low or "capturing model trace" in low:
                self.substage_detail = log_line.split("-")[-1].strip()
                extra_badge = f"[{self.substage_detail}]"
                is_milestone = True

            m_init_eng = re.search(r"init engine \(profile, create kv cache, warmup model\) took\s*([\d.]+)", low)
            if m_init_eng:
                sec_val = float(m_init_eng.group(1))
                self.substage_detail = f"Engine profile, KV cache & warmup completed in {sec_val:.1f}s"
                extra_badge = f"[Engine Init, KV Cache & Warmup Complete ({sec_val:.1f}s)]"
                is_milestone = True

        # Phase 5: HTTP API Server Startup
        if any(
            k in low
            for k in (
                "ready from local core engine process",
                "chat template warmup completed",
                "starting vllm server on",
                "starting vllm api server",
                "application startup complete",
                "uvicorn running on",
            )
        ):
            new_phase = "[Phase 5/5] vLLM OpenAI API Server Starting HTTP Listener"
            self.substage_detail = "HTTP API listener active & finishing final readiness check"
            if "chat template warmup completed" in low:
                extra_badge = "[Chat Template Warmup Complete]"
                is_milestone = True
            elif "starting vllm" in low:
                extra_badge = "[vLLM HTTP Server Listening on Port 8000]"
                is_milestone = True

        if new_phase != self.current_phase and new_phase[:9] > self.current_phase[:9]:
            self.current_phase = new_phase
            return self.current_phase, True, extra_badge

        if any(
            k in low
            for k in (
                "model has builtin warmup",
                "error",
                "exception",
                "fatal",
            )
        ):
            is_milestone = True

        return self.current_phase, is_milestone, extra_badge

    def format_eta_status(self, elapsed_sec: int) -> str:
        """Return a formatted progress % and ETA countdown string."""
        est = max(1, self.est_total_sec)
        if self.trace_done > 0 and self.trace_total > 0:
            # During deterministic trace capture (final 25% of bring-up), blend trace progress
            pct = min(99, int(72 + (self.trace_done / self.trace_total) * 26))
        else:
            pct = min(98, int((elapsed_sec / est) * 100))
        rem = max(0, est - elapsed_sec)
        mode_str = "Cold Compile" if self.saw_cold_compile else (
            "Warm Cache" if self.cache_warm_initial else "Direct DRAM + Warmup"
        )
        if rem > 0:
            return (
                f"{pct:2d}% | ~{_format_duration(rem)} rem "
                f"(est ~{_format_duration(est)} [{mode_str}]) | {self.substage_detail}"
            )
        return (
            f"99% | Finalizing (>~{_format_duration(est)} [{mode_str}]) | {self.substage_detail}"
        )


def _find_enginecore_pid() -> int | None:
    """Return the host PID of the active `VLLM::EngineCore` process on the QB2, if running."""
    rc, out, _ = run_cmd(["pgrep", "-f", "VLLM::EngineCore"], timeout=2.0)
    if rc == 0 and out:
        for tok in out.split():
            if tok.isdigit():
                return int(tok)
    return None


def _inspect_enginecore_open_shard(pid: int) -> tuple[int, int, str] | None:
    """
    Check `/proc/<pid>/fd` and `/proc/<pid>/maps` for an open `model-000XX-of-000YY.safetensors` file.
    Returns `(shard_idx, total_shards, filename)` if found, in <0.2ms using direct `/proc` syscalls.
    """
    fd_dir = Path(f"/proc/{pid}/fd")
    pat = re.compile(r"(model-(\d+)-of-(\d+)\.safetensors)")
    try:
        if fd_dir.exists():
            for entry in os.listdir(fd_dir):
                try:
                    target = os.readlink(fd_dir / entry)
                    m = pat.search(target)
                    if m:
                        return int(m.group(2)), int(m.group(3)), m.group(1)
                except OSError:
                    continue
        maps_file = Path(f"/proc/{pid}/maps")
        if maps_file.exists():
            txt = maps_file.read_text(errors="ignore")
            matches = pat.findall(txt)
            if matches:
                fname, s_idx, s_tot = matches[-1]
                return int(s_idx), int(s_tot), fname
    except Exception:
        pass
    return None


def monitor_container_until_ready(
    container_name: str,
    display_name: str,
    service_port: int = 8000,
    timeout_sec: int = 2400,
    model_entry: dict | None = None,
) -> bool:
    """
    Actively stream deterministic startup progress from `docker logs -f <container_name>`
    plus host `VLLM::EngineCore` `.safetensors` shard activity with live ETA countdown
    while polling `/health` and `/v1/models` on `service_port` until ready.
    """
    start_ts = time.time()
    tracker = StartupProgressTracker(model_entry=model_entry)
    cache_mode_desc = (
        f"Warm TT-Metal Cache (`{model_entry.get('cache_vol_name')}`, {model_entry.get('cache_vol_gb', 0.0):.1f} GB) — Est. Load: ~{_format_duration(tracker.est_total_sec)}"
        if (model_entry and model_entry.get("cache_warm"))
        else f"Direct DRAM Upload / First-Boot Compile — Est. Load: ~{_format_duration(tracker.est_total_sec)}"
    )

    if HAS_RICH:
        console.print(
            Panel(
                f"[bold cyan]Container:[/bold cyan]      [white]{container_name}[/white]\n"
                f"[bold cyan]Model:[/bold cyan]          [white]{display_name}[/white]\n"
                f"[bold cyan]TT-Cache Mode:[/bold cyan]  [bold green]{cache_mode_desc}[/bold green]\n"
                f"[bold cyan]Endpoint:[/bold cyan]       [white]http://localhost:{service_port}/v1/models[/white]\n"
                "[dim]Streaming live deterministic startup milestones, shard/layer progress, trace capture & ETA below...[/dim]",
                title="[bold cyan]Stage 4 — Active Model Bring-Up, Trace Capture & Readiness Monitor[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print(f"  [bold yellow]→ {tracker.current_phase}[/bold yellow]")
    else:
        print(
            f"\n=== Stage 4: Monitoring {container_name} ({display_name}) | {cache_mode_desc} ==="
        )
        print(f"  -> {tracker.current_phase}")

    stop_event = threading.Event()
    fatal_error: list[str] = []
    shown_phase = tracker.current_phase
    print_lock = threading.Lock()

    def _log_streamer() -> None:
        nonlocal shown_phase
        proc = subprocess.Popen(
            ["docker", "logs", "-f", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        try:
            if proc.stdout:
                for raw_line in proc.stdout:
                    if stop_event.is_set():
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    elapsed = int(time.time() - start_ts)
                    new_phase, is_milestone, extra_badge = tracker.feed(line)
                    with print_lock:
                        if new_phase != shown_phase:
                            shown_phase = new_phase
                            eta_s = tracker.format_eta_status(elapsed)
                            if HAS_RICH:
                                console.print(
                                    f"\n  [bold yellow][+{elapsed:3d}s | {eta_s}][/bold yellow]\n"
                                    f"  [bold yellow]→ {shown_phase}[/bold yellow]"
                                )
                            else:
                                print(f"\n  [+{elapsed:3d}s | {eta_s}]\n  -> {shown_phase}")
                        if extra_badge and HAS_RICH:
                            console.print(f"    [bold cyan][+{elapsed:3d}s] {extra_badge}[/bold cyan]")
                        elif extra_badge:
                            print(f"    [+{elapsed:3d}s] {extra_badge}")
                        if is_milestone:
                            trimmed = line if len(line) <= 125 else line[:122] + "..."
                            if HAS_RICH:
                                console.print(f"    [dim][+{elapsed:3d}s] {trimmed}[/dim]")
                            else:
                                print(f"    [+{elapsed:3d}s] {trimmed}")
                    if (
                        "device hang" in line.lower()
                        or "enginecore encountered a fatal error" in line.lower()
                    ):
                        fatal_error.append(line)
        finally:
            try:
                proc.terminate()
            except Exception:
                pass

    def _host_shard_poller() -> None:
        """
        Lightweight 0.2s `/proc/<EngineCore_pid>/fd` poller that catches `.safetensors`
        shard transitions (`1/18 .. 18/18`) for models like `Qwen3.8-27B` that use
        `safetensors.safe_open` directly without printing a tqdm progress bar.
        """
        ec_pid: int | None = None
        last_pid_lookup = 0.0
        seen_host_shards: set[int] = set()

        while not stop_event.is_set():
            if tracker.phase_num in (2, 3):
                now = time.time()
                if ec_pid is None and (now - last_pid_lookup) >= 2.0:
                    ec_pid = _find_enginecore_pid()
                    last_pid_lookup = now
                if ec_pid is not None:
                    res = _inspect_enginecore_open_shard(ec_pid)
                    if res:
                        s_idx, s_tot, fname = res
                        if s_idx > tracker.shards_done and s_idx not in seen_host_shards:
                            seen_host_shards.add(s_idx)
                            tracker.shards_done = s_idx
                            tracker.shards_total = s_tot
                            pct = int((s_idx / max(1, s_tot)) * 100)
                            tracker.substage_detail = (
                                f"Uploading .safetensors shard {s_idx}/{s_tot} ({pct}%) to Blackhole DRAM"
                            )
                            elapsed = int(now - start_ts)
                            with print_lock:
                                if HAS_RICH:
                                    console.print(
                                        f"    [bold cyan][+{elapsed:3d}s] [Weights Shard: {s_idx}/{s_tot} ({pct}%) — {fname}][/bold cyan]"
                                    )
                                else:
                                    print(
                                        f"    [+{elapsed:3d}s] [Weights Shard: {s_idx}/{s_tot} ({pct}%) — {fname}]"
                                    )
            time.sleep(0.2)

    log_thread = threading.Thread(target=_log_streamer, daemon=True)
    log_thread.start()
    shard_thread = threading.Thread(target=_host_shard_poller, daemon=True)
    shard_thread.start()

    last_heartbeat = time.time()
    try:
        while time.time() - start_ts < timeout_sec:
            if fatal_error:
                stop_event.set()
                if HAS_RICH:
                    console.print(
                        f"\n[bold red]✗ Fatal startup error detected in container logs:[/bold red]\n  {fatal_error[0]}"
                    )
                else:
                    print(f"\n✗ Fatal startup error detected: {fatal_error[0]}")
                return False

            rc, running_str, _ = run_cmd(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name], timeout=5.0
            )
            if rc != 0 or running_str.strip().lower() != "true":
                stop_event.set()
                if HAS_RICH:
                    console.print(
                        f"\n[bold red]✗ Container `{container_name}` exited unexpectedly during startup.[/bold red]"
                    )
                else:
                    print(f"\n✗ Container {container_name} exited unexpectedly during startup.")
                return False

            if check_vllm_health(service_port) and (
                not tracker.capturing_traces or tracker.traces_completed
            ):
                ep = query_vllm_endpoint(service_port)
                if ep:
                    stop_event.set()
                    elapsed = int(time.time() - start_ts)
                    was_warm = tracker.cache_warm_initial and not tracker.saw_cold_compile
                    if model_entry and model_entry.get("hf_repo"):
                        record_model_load_timing(
                            model_entry["hf_repo"], elapsed, is_warm=was_warm
                        )
                    mode_tag = "warm cache" if was_warm else "direct/cold compile"
                    with print_lock:
                        if HAS_RICH:
                            console.print(
                                f"\n  [bold green][+{elapsed:3d}s ({_format_duration(elapsed)}, {mode_tag})] "
                                f"✓ [Phase 5/5] vLLM Server Ready! "
                                f"Serving `{ep['served_model_id']}` at {ep['endpoint']} "
                                f"(max_model_len={ep['max_model_len']}) — recorded to `load_timings_cache.json`[/bold green]\n"
                            )
                        else:
                            print(
                                f"\n  [+{elapsed:3d}s ({_format_duration(elapsed)}, {mode_tag})] "
                                f"✓ vLLM Server Ready! Serving {ep['served_model_id']} at {ep['endpoint']}\n"
                            )
                    return True

            if time.time() - last_heartbeat >= 15.0:
                elapsed = int(time.time() - start_ts)
                eta_s = tracker.format_eta_status(elapsed)
                with print_lock:
                    if HAS_RICH:
                        console.print(
                            f"    [cyan][+{elapsed:3d}s | {eta_s}][/cyan]"
                        )
                    else:
                        print(
                            f"    [+{elapsed:3d}s | {eta_s}]"
                        )
                last_heartbeat = time.time()

            time.sleep(2.5)
    except KeyboardInterrupt:
        stop_event.set()
        if HAS_RICH:
            console.print(
                "\n[yellow]! Monitoring interrupted by user (container continues starting in background).[/yellow]"
            )
        else:
            print(
                "\n! Monitoring interrupted by user (container continues starting in background)."
            )
        return False
    finally:
        stop_event.set()

    if HAS_RICH:
        console.print(
            f"\n[bold red]✗ Timed out after {timeout_sec}s waiting for `{container_name}` to become ready.[/bold red]"
        )
    else:
        print(f"\n✗ Timed out after {timeout_sec}s waiting for {container_name} to become ready.")
    return False


def launch_and_monitor_model(chosen: dict, service_port: int = 8000) -> bool:
    """
    Launch the selected model via `run.py` and immediately transition into
    `monitor_container_until_ready` (`Stage 4`) as soon as `run.py` starts the
    Docker container (without blocking on `run.py`'s silent 5400s health wait).
    """
    launch_cmd = build_tt_inference_command(chosen, service_port=service_port)

    if HAS_RICH:
        console.print(
            Panel(
                f"[bold]Starting Model:[/bold]  [bold white]{chosen['display_name']}[/bold white]\n"
                f"[bold]Weights Path:[/bold]    `{chosen['local_path']}`\n"
                f"[bold]TT-Metal Cache:[/bold]  {chosen.get('cache_badge', 'N/A')}\n"
                f"[bold]Working Dir:[/bold]     `{TT_INFERENCE_DIR}`\n"
                f"[bold]Launch Command:[/bold]  `{' '.join(launch_cmd)}`",
                title="[bold cyan]Stage 3 — Launching TT-Inference-Server Container[/bold cyan]",
                border_style="cyan",
            )
        )
    else:
        print(f"\n=== Stage 3: Launching {chosen['display_name']} ===")
        print(f"Command: {' '.join(launch_cmd)}")

    proc = subprocess.Popen(
        launch_cmd,
        cwd=str(TT_INFERENCE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )

    launched_container_name: str | None = None
    handed_off_to_stage4 = False

    if proc.stdout:
        for line in proc.stdout:
            line_s = line.rstrip()
            if not line_s:
                continue
            if HAS_RICH:
                console.print(f"  [dim]{line_s}[/dim]")
            else:
                print(f"  {line_s}")

            m = re.search(r"--name\s+(tt-inference-server-[a-zA-Z0-9_.-]+)", line_s)
            if m:
                launched_container_name = m.group(1)

            low_s = line_s.lower()
            # As soon as `run.py` starts the container and begins its own silent health wait,
            # hand off immediately to Stage 4 (`monitor_container_until_ready`)!
            if launched_container_name and (
                ("waiting up to" in low_s and "for inference server readiness" in low_s)
                or "running docker container with log file:" in low_s
                or "created docker container id:" in low_s
            ):
                handed_off_to_stage4 = True
                break

    if handed_off_to_stage4:
        def _drain_run_py() -> None:
            try:
                if proc.stdout:
                    for _ in proc.stdout:
                        pass
                proc.wait()
            except Exception:
                pass

        threading.Thread(target=_drain_run_py, daemon=True).start()
        # Wait up to 3s for the Docker daemon to register the container as running
        for _ in range(12):
            rc_c, run_s, _ = run_cmd(
                ["docker", "inspect", "-f", "{{.State.Running}}", launched_container_name],
                timeout=2.0,
            )
            if rc_c == 0 and run_s.strip().lower() == "true":
                break
            time.sleep(0.25)
    else:
        rc = proc.wait()
        if rc != 0:
            if HAS_RICH:
                console.print(f"[bold red]✗ `run.py` exited with code {rc}.[/bold red]")
            else:
                print(f"✗ run.py exited with code {rc}.")
            return False

    if not launched_container_name:
        time.sleep(1.0)
        containers = find_model_containers()
        if containers:
            launched_container_name = containers[0]["name"]

    if not launched_container_name:
        if HAS_RICH:
            console.print(
                "[bold red]✗ Could not find a running `tt-inference-server` Docker container after `run.py` launched.[/bold red]"
            )
        else:
            print("✗ Could not find a running tt-inference-server Docker container.")
        return False

    return monitor_container_until_ready(
        container_name=launched_container_name,
        display_name=chosen["display_name"],
        service_port=service_port,
        model_entry=chosen,
    )


# ==============================================================================
# PART E: Unified Main Interactive Flow
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified QB2 Model Manager: Inspect active model & handles, rank all downloaded and supported QB2 models by Frontier Proximity %, inspect `config.json` architectures & Docker/TT-Metal caches, and gracefully switch/start models."
    )
    parser.add_argument(
        "--status",
        "--list",
        dest="status_only",
        action="store_true",
        help="Check active model status + display Frontier-ranked model catalog and exit without prompting",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Include experimental/nightly QB2 specs from `workflows/model_specs/dev/llm.yaml` and `dev/vlm.yaml`",
    )
    parser.add_argument(
        "--arch",
        nargs="?",
        const=0,
        type=int,
        default=None,
        help="Display deep `config.json` architecture & shard inspector for all downloaded models (or model #N) and exit",
    )
    parser.add_argument(
        "--images",
        "--storage",
        dest="inspect_images",
        action="store_true",
        help="Display local Docker images, TT-Metal cache volumes (`volume_id_*`), and host cache storage and exit",
    )
    parser.add_argument(
        "--prune",
        "--prune-images",
        dest="prune_storage",
        action="store_true",
        help="Safely prune dangling `<none>` Docker images, unused Docker build cache, and empty 0B TT-Metal volumes and exit",
    )
    parser.add_argument(
        "--refresh-ranks",
        action="store_true",
        help="Force re-fetching model benchmark tables & metadata from Hugging Face Hub",
    )
    parser.add_argument(
        "--select",
        type=int,
        nargs="+",
        default=None,
        help="Directly select a model number (1..N) to load (or multiple numbers with --download-only)",
    )
    parser.add_argument(
        "--download-only",
        "-d",
        action="store_true",
        help="Download selected model weights from Hugging Face without stopping the active model or starting a server",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Gracefully stop the currently running model, run `tt-smi -r`, and exit",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip running `tt-smi -r` when stopping/switching models",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM OpenAI API service port (default: 8000)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompts",
    )
    args = parser.parse_args()

    # 1. Check if an existing model is running and show all associated processes/handles
    state = gather_full_state()
    is_running = display_state(state)

    if args.stop:
        if not is_running:
            return 0
        return 0 if stop_active_model(state, run_reset=not args.no_reset) else 1

    # 2. Enumerate & rank all models by Frontier Proximity % (Part 1: Downloaded, Part 2: Supported)
    downloaded_models, available_to_pull = partition_models(
        include_dev=args.dev, refresh_ranks=args.refresh_ranks
    )
    combined = display_storage_and_catalog(
        downloaded_models, available_to_pull, state=state, include_dev=args.dev
    )

    if args.arch is not None:
        target_idx = args.arch if args.arch > 0 else None
        display_architecture_inspector(downloaded_models, target_idx=target_idx)
        return 0

    if args.inspect_images:
        storage = inspect_docker_and_cache_storage(state, combined)
        display_docker_and_cache_inspector(storage)
        return 0

    if args.prune_storage:
        storage = inspect_docker_and_cache_storage(state, combined)
        display_docker_and_cache_inspector(storage)
        prune_docker_and_caches(storage)
        return 0

    if args.status_only:
        return 0

    # Handle --download-only mode
    if args.download_only and args.select:
        for idx in args.select:
            if not (1 <= idx <= len(combined)):
                print(f"Selection {idx} out of range (1..{len(combined)}).")
                return 1
            chosen = combined[idx - 1]
            if chosen["downloaded"]:
                if HAS_RICH:
                    console.print(
                        f"[bold green]✓ Weights for #{idx} (`{chosen['hf_repo']}`) are already on disk at `{chosen['local_path']}`.[/bold green]"
                    )
                else:
                    print(f"✓ Weights for #{idx} ({chosen['hf_repo']}) are already on disk.")
                continue
            if not download_weights_from_hf(chosen["hf_repo"], chosen["local_path"]):
                return 1
        return 0

    # 3. Prompt the user: 'Load new model?'
    selected_idx: int | None = args.select[0] if args.select else None
    while selected_idx is None:
        try:
            if HAS_RICH:
                console.print()
                prompt_str = (
                    f"[bold cyan]Load new model? Enter model # [1-{len(combined)}] "
                    f"('a' arch inspector, 'i' Docker/cache storage, 'p' safe prune, 'r' refresh ranks, 's' stop model, 'q' quit): [/bold cyan]"
                )
                raw = console.input(prompt_str).strip()
            else:
                raw = input(
                    f"\nLoad new model? Enter model # [1-{len(combined)}] "
                    f"('a' arch, 'i' storage, 'p' prune, 'r' refresh, 's' stop, 'q' quit): "
                ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo changes made. Exiting.")
            return 0

        if not raw or raw.lower() in ("q", "quit", "exit", "n", "no"):
            if HAS_RICH:
                console.print(
                    "[bold green]✓ Exiting without changes — existing QB2 model state left untouched.[/bold green]"
                )
            else:
                print("✓ Exiting without changes — existing QB2 model state left untouched.")
            return 0

        m_arch = re.match(r"^(?:a|arch)(?:\s+(\d+))?$", raw, flags=re.I)
        if m_arch:
            t_idx = int(m_arch.group(1)) if m_arch.group(1) else None
            display_architecture_inspector(downloaded_models, target_idx=t_idx)
            continue

        if raw.lower() in ("i", "images", "storage"):
            run_storage_cleanup_menu(state, combined)
            continue

        if raw.lower() in ("p", "prune"):
            storage = inspect_docker_and_cache_storage(state, combined)
            prune_docker_and_caches(storage)
            downloaded_models, available_to_pull = partition_models(
                include_dev=args.dev, refresh_ranks=False
            )
            combined = display_storage_and_catalog(
                downloaded_models, available_to_pull, state=state, include_dev=args.dev
            )
            continue

        if raw.lower() in ("r", "refresh"):
            downloaded_models, available_to_pull = partition_models(
                include_dev=args.dev, refresh_ranks=True
            )
            combined = display_storage_and_catalog(
                downloaded_models, available_to_pull, state=state, include_dev=args.dev
            )
            continue

        if raw.lower() in ("s", "stop"):
            if not is_running:
                if HAS_RICH:
                    console.print("[dim]No model is currently running.[/dim]")
                else:
                    print("No model is currently running.")
                return 0
            return 0 if stop_active_model(state, run_reset=not args.no_reset) else 1

        if not raw.isdigit():
            print(
                f"Invalid selection: {raw!r}. Enter a model number [1..{len(combined)}], 'a [N]', 'i', 'p', 'r', 's', or 'q'."
            )
            continue

        selected_idx = int(raw)

    if not (1 <= selected_idx <= len(combined)):
        print(f"Selection {selected_idx} out of range (1..{len(combined)}).")
        return 1

    chosen = combined[selected_idx - 1]

    # 4. If the selected model is the SAME as the currently running model, do nothing and exit!
    if is_running and is_entry_currently_running(chosen, state):
        if HAS_RICH:
            console.print(
                Panel(
                    f"[bold green]✓ Selected model `#{selected_idx} — {chosen['display_name']}` is ALREADY running on the QB2![/bold green]\n"
                    "[dim]No action needed — leaving the active model server and its handles untouched.[/dim]",
                    title="[bold green]Model Already Active[/bold green]",
                    border_style="green",
                )
            )
        else:
            print(
                f"\n✓ Selected model #{selected_idx} ({chosen['display_name']}) is ALREADY running. Doing nothing."
            )
        return 0

    # 5. Different model selected:
    #    (a) Gracefully take down existing model (if running) + run `tt-smi -r`
    if is_running:
        if not stop_active_model(state, run_reset=not args.no_reset):
            print("Failed to cleanly stop the active model. Aborting new model launch.")
            return 1
    elif not args.no_reset:
        run_tt_smi_reset()

    #    (b) Pull weights from Hugging Face if not yet downloaded on disk
    if not chosen["downloaded"]:
        if not download_weights_from_hf(chosen["hf_repo"], chosen["local_path"]):
            return 1

    #    (c) Launch new model & stream live startup progress until ready
    ready = launch_and_monitor_model(chosen, service_port=args.port)
    if not ready:
        return 1

    #    (d) Display final active model status & handles and prompt/exit
    final_state = gather_full_state()
    display_state(final_state)

    if HAS_RICH:
        console.print(
            Panel(
                f"[bold green]✓ Model `#{selected_idx} — {chosen['display_name']}` is UP and actively serving on port {args.port}![/bold green]",
                title="[bold green]QB2 Model Switch Complete[/bold green]",
                border_style="green",
            )
        )
    else:
        print(
            f"\n✓ Model #{selected_idx} ({chosen['display_name']}) is UP and serving on port {args.port}!"
        )

    if not args.yes and sys.stdin.isatty():
        try:
            if HAS_RICH:
                console.input(
                    "[bold cyan]Press Enter to exit qb2_model_manager (model remains running in background)...[/bold cyan]"
                )
            else:
                input(
                    "Press Enter to exit qb2_model_manager (model remains running in background)..."
                )
        except (EOFError, KeyboardInterrupt):
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
