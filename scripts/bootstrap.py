#!/usr/bin/env python3
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Initialize upstream sources and apply the tracked G05_A1Z adaptations."""
import argparse
from pathlib import Path
import json
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = "a1z-data-collect"
PINNED = ("GalaxeaVLA",)
LEADER = f"{COLLECTOR}/a1z-t"
SDK = f"{LEADER}/galaxea_a1z"
MAIN_REF = "refs/remotes/origin/main"
LEADER_MAIN = MAIN_REF


def git(*args, check=True):
    return subprocess.run(["git", "-c", "fetch.recurseSubmodules=false",
                           "-c", "submodule.recurse=false", "-C", str(ROOT), *args], check=check,
                          text=True, capture_output=True)


def require_clean(folder):
    if git("-C", folder, "status", "--porcelain", "--ignore-submodules=all").stdout.strip():
        raise RuntimeError(f"Local edits found in {folder}; preserve them before updating")


def resolve_main(folder, label):
    """Fetch the tracked main branch of a component and return its commit."""
    fetched = git("-C", folder, "fetch", "--no-tags", "origin",
                  f"+refs/heads/main:{MAIN_REF}", check=False)
    if fetched.returncode:
        raise RuntimeError(f"Cannot fetch {label} main; retry with --tested.\n{fetched.stderr.strip()}")
    return git("-C", folder, "rev-parse", MAIN_REF).stdout.strip()


COLLECTOR_PATCH = "patches/a1z-data-collect-integration.patch"
COLLECTOR_URL = "https://github.com/userguide-galaxea/a1z-data-collect.git"
LEADER_URL = "https://github.com/userguide-galaxea/a1z-t.git"
SDK_URL = "https://github.com/userguide-galaxea/galaxea_a1z.git"


def ensure_clone(path, url):
    """Clone a first-party dependency once; these are plain clones, not submodules."""
    if (ROOT / path / ".git").exists():
        return False
    git("clone", "--branch", "main", url, str(path))
    return True


def collector_patch_applied():
    return git("-C", COLLECTOR, "apply", "--reverse", "--check", str(ROOT / COLLECTOR_PATCH),
               check=False).returncode == 0


def init_collector(versions, tested):
    """Follow a1z-data-collect main, or fall back to the tested commit."""
    ensure_clone(COLLECTOR, COLLECTOR_URL)
    if tested:
        revision = versions[COLLECTOR]
        if git("-C", COLLECTOR, "cat-file", "-e", revision + "^{commit}", check=False).returncode:
            git("-C", COLLECTOR, "fetch", "--no-tags", "origin", revision)
    else:
        revision = resolve_main(COLLECTOR, "A1Z data-collect")
    if git("-C", COLLECTOR, "rev-parse", "HEAD").stdout.strip() != revision:
        # The integration patch is reapplied below, so undo it before switching.
        patch_was_applied = collector_patch_applied()
        if patch_was_applied:
            git("-C", COLLECTOR, "apply", "--reverse", str(ROOT / COLLECTOR_PATCH))
        try:
            require_clean(COLLECTOR)
        except RuntimeError:
            # Restore the workspace we touched before reporting the conflict.
            if patch_was_applied:
                git("-C", COLLECTOR, "apply", str(ROOT / COLLECTOR_PATCH))
            raise
        git("-C", COLLECTOR, "checkout", "--detach", revision)
    mode = "tested commit" if tested else "origin/main"
    print(f"A1Z data-collect ({mode}): {revision}")
    return revision


def init_leader(versions, tested):
    ensure_clone(LEADER, LEADER_URL)
    if tested:
        revision = versions[LEADER]
        if git("-C", LEADER, "cat-file", "-e", revision + "^{commit}", check=False).returncode:
            git("-C", LEADER, "fetch", "--no-tags", "origin", revision)
    else:
        fetched = git("-C", LEADER, "fetch", "--no-tags", "origin",
                      f"+refs/heads/main:{LEADER_MAIN}", check=False)
        if fetched.returncode:
            raise RuntimeError(f"Cannot fetch A1Z-T main; retry with --tested-a1z-t.\n{fetched.stderr.strip()}")
        revision = git("-C", LEADER, "rev-parse", LEADER_MAIN).stdout.strip()
    if git("-C", LEADER, "rev-parse", "HEAD").stdout.strip() != revision:
        require_clean(LEADER)
        git("-C", LEADER, "checkout", "--detach", revision)
    mode = "tested commit" if tested else "origin/main"
    print(f"A1Z-T ({mode}): {revision}")
    return revision


def init_sdk(versions, tested):
    """Clone and follow galaxea_a1z main, or fall back to the tested commit."""
    ensure_clone(SDK, SDK_URL)
    if tested:
        revision = versions[SDK]
        if git("-C", SDK, "cat-file", "-e", revision + "^{commit}", check=False).returncode:
            git("-C", SDK, "fetch", "--no-tags", "origin", revision)
    else:
        revision = resolve_main(SDK, "A1Z SDK")
    if git("-C", SDK, "rev-parse", "HEAD").stdout.strip() != revision:
        require_clean(SDK)
        git("-C", SDK, "checkout", "--detach", revision)
    mode = "tested commit" if tested else "origin/main"
    print(f"A1Z SDK ({mode}): {revision}")
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tested", action="store_true",
                        help="Use the tested commits from versions.json for a1z-data-collect and A1Z-T")
    parser.add_argument("--tested-a1z-t", action="store_true",
                        help="Use only the tested A1Z-T commit; keep a1z-data-collect on main")
    args = parser.parse_args()
    versions = json.loads((ROOT / "versions.json").read_text())
    git("submodule", "sync", "--", *PINNED)
    git("submodule", "update", "--init", "--checkout", "--", *PINNED)
    for name in PINNED:
        actual = git("-C", name, "rev-parse", "HEAD").stdout.strip()
        if actual != versions[name]:
            raise RuntimeError(f"Pinned version mismatch for {name}: {actual}; expected {versions[name]}")
    init_collector(versions, args.tested)
    init_leader(versions, args.tested or args.tested_a1z_t)
    init_sdk(versions, args.tested or args.tested_a1z_t)
    # Replace the upstream machine path only in the unchanged pinned config.
    memory_config = ROOT / "GalaxeaVLA/configs/model/g05_mem.yaml"
    if memory_config.is_file() and hashlib.sha256(memory_config.read_bytes()).hexdigest() == "b4514cb901c355bcbe33fb1fc9553e24b1ee82218e47b2bcd963e160cb35dc6f":
        lines = memory_config.read_text().splitlines(keepends=True)
        memory_config.write_text("".join(
            "pretrained_ckpt: ${oc.env:G05_MEM_CKPT,checkpoints/g05_pretrain_mem.pt}\n"
            if line.startswith("pretrained_ckpt:") else line for line in lines
        ))
    # Upgrade recognized generated configs; preserve user edits.
    config = ROOT / "GalaxeaVLA/configs/data/a1z.yaml"
    if config.is_file() and hashlib.sha256(config.read_bytes()).hexdigest() == "c1fd8c487266d01caabd6e9af64d53c9a661eeff38ce0c68eecfde2f3e3da390":
        config.write_text(config.read_text().replace("lerobot_ds_version: '3.0'", "lerobot_ds_version: '2.1'"))
    # Keep step-based training independent of epoch limits.
    task = ROOT / "GalaxeaVLA/configs/task/a1z.yaml"
    if task.is_file() and hashlib.sha256(task.read_bytes()).hexdigest() == "4e18ee65fa1f38f95294be226fc6ae78a76b812c6390420411707a7c488856a7":
        task.write_text(task.read_text().replace("  max_steps: 10000", "  max_epochs: null\n  max_steps: 10000"))
    # Add notices only to recognized, unedited integration-generated files.
    # Keep Hydra package directives first; upstream source notices are retained.
    integration_headers = {
        "configs/data/a1z.yaml": "21794e276541441bdeb8d9414bead341ccac37c30628862185221dd78e8b8b4e",
        "configs/data/a1z_single_left.yaml": "8a86a1c8fa0cb4df708ba90c28d5b5242214a8906161d2506fc369f9adbf4d1f",
        "configs/task/a1z.yaml": "e81a522c09563238715d47b610cd30f7b27a9582fb8d49cd270200c9d0cc2098",
        "configs/task/a1z_lora.yaml": "e3ed6c52f55069457aaab6527acad500ddfaed4258017dcea3914b1ce8aa0b4f",
        "configs/task/a1z_single_left_lora.yaml": "94eece1be06f65f543f137aa65734170507e2add24e3a7227d3940bc78d60d9c",
        "configs/task/a1z_single_left_full.yaml": "9495872c9c3c8d48f7cfc476c59badd1d128a88635ce858be640d33f97803fcc",
        "src/g05/models/g05/lora.py": "495b802b7f0cdeaf3748f2f31d8fb26997dce4e9b16a7ddb54e95f43ebada106"
    }
    header = (
        "# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.\n"
        "# See LICENSE.txt in the project root; upstream components retain their applicable licenses.\n\n"
    )
    for relative, expected in integration_headers.items():
        path = ROOT / "GalaxeaVLA" / relative
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            text = path.read_text()
            if text.startswith("# @package "):
                first, rest = text.split("\n", 1)
                text = first + "\n" + header + rest
            else:
                text = header + text
            path.write_text(text)
    pending = []
    series = json.loads((ROOT / "patches/series.json").read_text())
    if isinstance(series, dict):
        series = series["patches"]
    for name, filename in series:
        patch = str(ROOT / "patches" / filename)
        if git("-C", name, "apply", "--reverse", "--check", patch, check=False).returncode == 0:
            print(f"Already applied: {name}")
        elif git("-C", name, "apply", "--check", patch, check=False).returncode == 0:
            pending.append((name, patch))
        else:
            raise RuntimeError(f"Patch conflict in {name}; upstream may have changed. "
                               "Retry with --tested, or preserve your local edits and resolve manually")
    for name, patch in pending:
        git("-C", name, "apply", patch)
        print(f"Applied: {name}")
    print("Submodules ready. Run python3 scripts/verify_workspace.py next.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
