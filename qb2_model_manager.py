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
PRIMARY_MODELS_DIR = Path("/home/ttuser/models")
TT_CACHE_DIR = Path("/home/ttuser/data/tt-cache")
TT_INFERENCE_DIR = Path("/home/ttuser/.local/lib/tt-inference-server")
PROD_SPEC_YAML = TT_INFERENCE_DIR / "workflows" / "model_specs" / "prod" / "llm.yaml"
DEV_SPEC_YAML = TT_INFERENCE_DIR / "workflows" / "model_specs" / "dev" / "llm.yaml"
DEV_VLM_SPEC_YAML = TT_INFERENCE_DIR / "workflows" / "model_specs" / "dev" / "vlm.yaml"

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


def partition_models(
    include_dev: bool = False,
    refresh_ranks: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Partition the live catalog into (1) Already Downloaded on QB2 and (2) Available to Pull,
    annotated and sorted highest-to-lowest by Hugging Face Frontier Proximity %.
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

    for m in downloaded + available_to_pull:
        r_info = rank_map.get(
            m["hf_repo"],
            {"score": 70.0, "tier": "B Tier", "source": "HF Metadata", "created": "unknown"},
        )
        m["frontier_score"] = float(r_info.get("score", 70.0))
        m["frontier_tier"] = str(r_info.get("tier", "B Tier"))
        m["frontier_source"] = str(r_info.get("source", "HF Metadata"))
        m["hf_created"] = str(r_info.get("created", "unknown"))

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
    mode_label = (
        "Production (`prod/llm.yaml`) + Dev (`dev/llm.yaml`, `dev/vlm.yaml`)"
        if include_dev
        else "Production (`prod/llm.yaml`) + Validated Bridges  [Pass `--dev` to also list 40+ experimental `dev/llm.yaml` QB2 models]"
    )

    if HAS_RICH:
        storage_lines = [
            f"[bold cyan]TT-Inference-Server Repo:[/bold cyan]    [bold white]{repo_ver}[/bold white] (`/home/ttuser/.local/lib/tt-inference-server`)",
            f"[bold cyan]Active Catalog Mode:[/bold cyan]         [yellow]{mode_label}[/yellow]",
            f"[bold cyan]Catalog Ordering:[/bold cyan]            [bold green]Sorted by Frontier Proximity %[/bold green] (Extracted from Hugging Face model cards & metadata; cache: `frontier_rankings_cache.json`)",
            f"[bold cyan]1. Primary Host Weights:[/bold cyan]     [white]/home/ttuser/models[/white] (`Qwen3.8-27B`, `gemma-4-31B-it`, `DeepSeek-R1-Distill-Qwen-32B`, `Llama-3.1-70B/8B`)",
            f"[bold cyan]2. TT-Cache Host Weights:[/bold cyan]    [white]/home/ttuser/data/tt-cache/volume_id_tt_transformers-Qwen3-32B-vqb2_launch/weights[/white] (`Qwen/Qwen3-32B`)",
            f"[bold cyan]3. Docker TT-Metal Volumes:[/bold cyan]  [white]/var/lib/docker/volumes/volume_id_tt_transformers-*[/white] (Pre-compiled Blackhole `p300x2` tensor caches)",
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
        t_down.add_column("Model / Weights", style="bold white")
        t_down.add_column("Frontier Proximity", style="bold green", justify="right")
        t_down.add_column("vLLM / TT Spec & Impl", style="cyan")
        t_down.add_column("Disk & Speed / Ctx", style="magenta")
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
            t_down.add_row(
                str(idx),
                f"{m['display_name']}{running_badge}\n[dim]{m['notes']}[/dim]",
                f"{m['frontier_score']:.1f}% ({m['frontier_tier']})\n[dim]{m['frontier_source']}[/dim]",
                f"--model {m['tt_spec_model']}\n--tt-device {m['tt_device']}\nimpl: {m['impl']}",
                f"{m['actual_size_gb']:.1f} GB ({m['n_shards']} shards)\n{m['est_speed']} | {ctx_fmt} ctx (x{m['max_concurrency']})",
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
        t_avail.add_column("Download & Speed / Ctx", style="magenta")
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
                f"{m['est_size_gb']}\n{m['est_speed']} | {ctx_fmt} ctx (x{m['max_concurrency']})",
                f"{m['status']} (v{m['version']})\nTarget: {m['local_path']}",
            )
        console.print(t_avail)
    else:
        print(f"\n=== Step 2 — TT-Inference-Server Repo: {repo_ver} ===")
        print("=== Part 1: Models Already Downloaded on QB2 (Ranked by Frontier Proximity) ===")
        for idx, m in enumerate(downloaded_models, start=1):
            run_tag = " [RUNNING]" if is_entry_currently_running(m, state) else ""
            print(
                f"  [{idx}] {m['display_name']}{run_tag} | Frontier: {m['frontier_score']:.1f}% ({m['frontier_tier']}) | "
                f"Spec: {m['tt_spec_model']} ({m['tt_device']}, {m['impl']}) | {m['actual_size_gb']:.1f} GB | Path: {m['local_path']}"
            )
        print("\n=== Part 2: Models Supported on QB2 (Ranked by Frontier Proximity — Auto-Pulls from HF) ===")
        start_idx = len(downloaded_models) + 1
        for idx, m in enumerate(available_to_pull, start=start_idx):
            print(
                f"  [{idx}] {m['display_name']} | Frontier: {m['frontier_score']:.1f}% ({m['frontier_tier']}) | "
                f"Spec: {m['tt_spec_model']} ({m['tt_device']}, {m['impl']}) | Est: {m['est_size_gb']} | Target: {m['local_path']}"
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


def classify_startup_phase(log_line: str, current_phase: str) -> tuple[str, bool]:
    """Map a container log line to a human-readable startup milestone phase."""
    low = log_line.lower()
    is_milestone = False

    if any(
        k in low
        for k in (
            "opening device",
            "chip_ids",
            "tt-metal",
            "fabric",
            "mesh_device",
            "initializing tensor parallel",
        )
    ):
        new_phase = "[Phase 2/5] Initializing Tenstorrent Blackhole Mesh (4x p300c / 8 Tensix chips)"
        if new_phase != current_phase:
            return new_phase, True
    if any(
        k in low
        for k in ("loading weights", "safetensors", "model loading took", "loading model")
    ):
        new_phase = "[Phase 3/5] Loading .safetensors Model Weights & Compiling TT-Metal Kernels"
        if new_phase != current_phase:
            return new_phase, True
    if any(
        k in low
        for k in ("capturing", "trace", "warming up", "warmup", "kv cache", "allocated")
    ):
        new_phase = "[Phase 4/5] Allocating KV Cache & Capturing Prefill/Decode Hardware Traces"
        if new_phase != current_phase:
            return new_phase, True
    if any(
        k in low
        for k in ("application startup complete", "uvicorn running on", "starting vllm api server")
    ):
        new_phase = "[Phase 5/5] vLLM OpenAI API Server Starting HTTP Listener"
        if new_phase != current_phase:
            return new_phase, True

    if any(
        k in low
        for k in (
            "initializing",
            "loading weights",
            "model loading",
            "kv cache",
            "trace",
            "warmup",
            "uvicorn running",
            "application startup complete",
            "error",
            "exception",
            "fatal",
        )
    ):
        is_milestone = True

    return current_phase, is_milestone


def monitor_container_until_ready(
    container_name: str,
    display_name: str,
    service_port: int = 8000,
    timeout_sec: int = 1800,
) -> bool:
    """
    Actively stream startup progress from `docker logs -f <container_name>` while polling
    `/health` and `/v1/models` on `service_port` until the model is ready to serve requests.
    """
    start_ts = time.time()
    current_phase = "[Phase 1/5] Container Started — Bootstrapping Python & vLLM EngineCore"

    if HAS_RICH:
        console.print(
            Panel(
                f"[bold cyan]Container:[/bold cyan] [white]{container_name}[/white]\n"
                f"[bold cyan]Model:[/bold cyan]     [white]{display_name}[/white]\n"
                f"[bold cyan]Endpoint:[/bold cyan]  [white]http://localhost:{service_port}/v1/models[/white]\n"
                "[dim]Streaming live container startup milestones below until HTTP readiness (`200 OK`)...[/dim]",
                title="[bold cyan]Stage 4 — Active Model Bring-Up & Readiness Monitor[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print(f"  [bold yellow]→ {current_phase}[/bold yellow]")
    else:
        print(
            f"\n=== Stage 4: Monitoring {container_name} ({display_name}) until ready on port {service_port} ==="
        )
        print(f"  -> {current_phase}")

    stop_event = threading.Event()
    fatal_error: list[str] = []

    def _log_streamer() -> None:
        nonlocal current_phase
        proc = subprocess.Popen(
            ["docker", "logs", "-f", "--tail", "25", container_name],
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
                    new_phase, is_milestone = classify_startup_phase(line, current_phase)
                    if new_phase != current_phase:
                        current_phase = new_phase
                        if HAS_RICH:
                            console.print(
                                f"\n  [bold yellow][+{elapsed:3d}s] → {current_phase}[/bold yellow]"
                            )
                        else:
                            print(f"\n  [+{elapsed:3d}s] -> {current_phase}")
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

    log_thread = threading.Thread(target=_log_streamer, daemon=True)
    log_thread.start()

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

            if check_vllm_health(service_port):
                ep = query_vllm_endpoint(service_port)
                if ep:
                    stop_event.set()
                    elapsed = int(time.time() - start_ts)
                    if HAS_RICH:
                        console.print(
                            f"\n  [bold green][+{elapsed:3d}s] ✓ [Phase 5/5] vLLM Server Ready! "
                            f"Serving `{ep['served_model_id']}` at {ep['endpoint']} (max_model_len={ep['max_model_len']})[/bold green]\n"
                        )
                    else:
                        print(
                            f"\n  [+{elapsed:3d}s] ✓ vLLM Server Ready! Serving {ep['served_model_id']} at {ep['endpoint']}\n"
                        )
                    return True

            if time.time() - last_heartbeat >= 15.0:
                elapsed = int(time.time() - start_ts)
                if HAS_RICH:
                    console.print(
                        f"    [cyan][+{elapsed:3d}s] Still warming up ({current_phase})... waiting for HTTP 200 on port {service_port}[/cyan]"
                    )
                else:
                    print(
                        f"    [+{elapsed:3d}s] Still warming up ({current_phase})... waiting for HTTP 200 on port {service_port}"
                    )
                last_heartbeat = time.time()

            time.sleep(3.0)
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
    """Launch the selected model via `run.py` and actively monitor container startup until ready."""
    launch_cmd = build_tt_inference_command(chosen, service_port=service_port)

    if HAS_RICH:
        console.print(
            Panel(
                f"[bold]Starting Model:[/bold]  [bold white]{chosen['display_name']}[/bold white]\n"
                f"[bold]Weights Path:[/bold]    `{chosen['local_path']}`\n"
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
    if proc.stdout:
        for line in proc.stdout:
            line_s = line.rstrip()
            if line_s:
                if HAS_RICH:
                    console.print(f"  [dim]{line_s}[/dim]")
                else:
                    print(f"  {line_s}")
                m = re.search(r"--name\s+(tt-inference-server-[a-zA-Z0-9_.-]+)", line_s)
                if m:
                    launched_container_name = m.group(1)

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
                "[bold red]✗ Could not find a running `tt-inference-server` Docker container after `run.py` completed.[/bold red]"
            )
        else:
            print("✗ Could not find a running tt-inference-server Docker container.")
        return False

    return monitor_container_until_ready(
        container_name=launched_container_name,
        display_name=chosen["display_name"],
        service_port=service_port,
    )


# ==============================================================================
# PART E: Unified Main Interactive Flow
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified QB2 Model Manager: Inspect active model & handles, rank all downloaded and supported QB2 models by Frontier Proximity %, and gracefully switch/start models."
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
                    f"[bold cyan]Load new model? Enter model number [1-{len(combined)}] "
                    f"('r' to refresh HF ranks, 's' to stop active model, 'q' to quit): [/bold cyan]"
                )
                raw = console.input(prompt_str).strip()
            else:
                raw = input(
                    f"\nLoad new model? Enter model number [1-{len(combined)}] "
                    f"('r' to refresh HF ranks, 's' to stop active model, 'q' to quit): "
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
                f"Invalid selection: {raw!r}. Please enter a model number between 1 and {len(combined)}, 'r', 's', or 'q'."
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
