#!/usr/bin/env python3
# Copyright (c) 2026 The BTX developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://opensource.org/license/mit/.
"""Verify a HiveOS custom miner archive.

This checks the packaging contract used by HiveOS' built-in custom miner
scripts:

- custom-get derives the miner name from the archive URL by splitting the
  archive basename at the last hyphen.
- the version part must not contain a hyphen.
- the archive must contain a top-level directory matching the derived miner
  name.
- h-manifest.conf, h-config.sh, h-run.sh and h-stats.sh must be present there.

When bash is available, the verifier also runs a light simulation of HiveOS'
custom config/stats delegation without starting the miner.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import urlparse


REQUIRED_FILES = ("h-manifest.conf", "h-config.sh", "h-run.sh", "h-stats.sh")
DEFAULT_POOL = "stratum+tcp://stratum.minebtx.com:3333"
DEFAULT_WALLET = "btx1zckh4rkc7z94mhms8artz347y0apqlnytspat54a35csk5dtlyk5scj0cvs"
DEFAULT_WORKER = "hive-verify"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="HiveOS custom miner .tar.gz archive.")
    parser.add_argument(
        "--install-url",
        help="Optional install URL to use for HiveOS custom-get name parsing. "
        "Defaults to the archive filename.",
    )
    parser.add_argument(
        "--bash",
        default=None,
        help="Bash executable for script syntax/delegation checks. Defaults to bash on PATH.",
    )
    parser.add_argument(
        "--skip-bash",
        action="store_true",
        help="Skip bash syntax and HiveOS delegation simulations.",
    )
    parser.add_argument("--wallet", default=DEFAULT_WALLET)
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--worker", default=DEFAULT_WORKER)
    return parser.parse_args(argv)


def hive_archive_name(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme and parsed.path:
        return Path(parsed.path).name
    return Path(source).name


def derive_hive_names(archive_name: str) -> tuple[str, str, str]:
    if not archive_name.endswith(".tar.gz"):
        raise ValueError(f"HiveOS custom miner archive must end with .tar.gz: {archive_name}")
    basename = archive_name.removesuffix(".tar.gz")
    if "-" not in basename:
        raise ValueError(
            "HiveOS custom-get cannot derive a miner name without a version suffix: "
            f"{archive_name}"
        )
    miner, version = basename.rsplit("-", 1)
    if not miner:
        raise ValueError(f"HiveOS custom-get would derive an empty miner name: {archive_name}")
    if not version:
        raise ValueError(f"HiveOS custom-get would derive an empty version: {archive_name}")
    if "-" in version:
        raise ValueError(f"HiveOS custom miner version must not contain '-': {version}")
    return basename, miner, version


def safe_member_name(name: str) -> bool:
    path = Path(name)
    if path.is_absolute():
        return False
    return ".." not in path.parts


def top_level_directory(name: str) -> str | None:
    clean = name.strip("/")
    if not clean:
        return None
    return clean.split("/", 1)[0]


def read_archive_members(archive: Path) -> list[str]:
    if not archive.is_file():
        raise FileNotFoundError(f"Missing archive: {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        members = [member.name for member in handle.getmembers()]
    unsafe = [name for name in members if not safe_member_name(name)]
    if unsafe:
        raise ValueError(f"Archive contains unsafe paths: {unsafe[:5]}")
    return members


def verify_required_members(members: list[str], miner: str) -> None:
    top_levels = sorted({top for name in members if (top := top_level_directory(name))})
    if top_levels != [miner]:
        raise ValueError(
            f"Archive top-level directories must be exactly [{miner!r}], got {top_levels!r}"
        )
    member_set = {name.rstrip("/") for name in members}
    missing = [name for name in REQUIRED_FILES if f"{miner}/{name}" not in member_set]
    if missing:
        raise FileNotFoundError(f"Archive is missing required HiveOS files: {', '.join(missing)}")


def extract_manifest(archive: Path, miner: str) -> dict[str, str]:
    with tarfile.open(archive, "r:gz") as handle:
        manifest_file = handle.extractfile(f"{miner}/h-manifest.conf")
        if manifest_file is None:
            raise FileNotFoundError(f"{miner}/h-manifest.conf")
        manifest_text = manifest_file.read().decode("utf-8")
    manifest: dict[str, str] = {}
    for raw_line in manifest_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        manifest[key.strip()] = value
    custom_name = manifest.get("CUSTOM_NAME")
    if custom_name != miner:
        raise ValueError(f"CUSTOM_NAME must be {miner!r}, got {custom_name!r}")
    if not manifest.get("CUSTOM_CONFIG_FILENAME"):
        raise ValueError("h-manifest.conf must define CUSTOM_CONFIG_FILENAME")
    if not manifest.get("CUSTOM_LOG_BASENAME"):
        raise ValueError("h-manifest.conf must define CUSTOM_LOG_BASENAME")
    return manifest


def find_bash(requested: str | None) -> str | None:
    if requested:
        return requested
    return shutil.which("bash")


def windows_to_bash_path(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    if os.name == "nt" and re.match(r"^[A-Za-z]:\\", text):
        drive = text[0].lower()
        tail = text[3:].replace("\\", "/")
        return f"/{drive}/{tail}"
    return text


def run_bash(bash: str, script: str, cwd: Path) -> None:
    result = subprocess.run(
        [bash, "-lc", script],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout, result.stderr) if part.strip())
        raise RuntimeError(output.strip() or f"bash check failed with exit code {result.returncode}")


def run_bash_checks(
    archive: Path,
    miner: str,
    bash: str,
    wallet: str,
    pool: str,
    worker: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="btx-hiveos-verify-") as raw_tmp:
        temp_dir = Path(raw_tmp)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(temp_dir)
        miner_dir = temp_dir / miner
        bash_miner_dir = windows_to_bash_path(miner_dir)

        syntax_files = " ".join(f"./{name}" for name in REQUIRED_FILES)
        run_bash(bash, f"cd '{bash_miner_dir}' && bash -n {syntax_files} ./run.sh", temp_dir)

        template = f"{wallet}.{worker}"
        run_bash(
            bash,
            (
                f"cd '{bash_miner_dir}' && "
                f"CUSTOM_TEMPLATE='{template}' CUSTOM_URL='{pool}' WORKER_NAME='{worker}' "
                "CUSTOM_USER_CONFIG='BTX_BATCH_SIZE=512 BTX_WORKERS_PER_GPU=1 IGNORE_ME=bad' "
                "./h-config.sh >/tmp/btx-hiveos-verify-h-config.out && "
                f"grep -q '^BTX_WALLET={wallet}$' miner.env && "
                f"grep -q '^BTX_POOL={pool}$' miner.env && "
                f"grep -q '^BTX_WORKER_PREFIX={worker}$' miner.env && "
                "grep -q '^BTX_BATCH_SIZE=512$' miner.env && "
                "grep -q '^BTX_WORKERS_PER_GPU=1$' miner.env && "
                "! grep -q '^IGNORE_ME=' miner.env"
            ),
            temp_dir,
        )

        # Simulate HiveOS' built-in custom h-stats.sh delegating to this package.
        root_stats = temp_dir / "h-stats-root.sh"
        root_stats.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "if [[ -z $CUSTOM_MINER ]]; then",
                    "  echo '$CUSTOM_MINER is not defined' >&2",
                    "else",
                    "  source \"$MINER_DIR/$CUSTOM_MINER/h-stats.sh\"",
                    "fi",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        bash_temp_dir = windows_to_bash_path(temp_dir)
        run_bash(
            bash,
            (
                f"cd '{bash_temp_dir}' && "
                f"export CUSTOM_MINER='{miner}' MINER_DIR='{bash_temp_dir}' && "
                "source ./h-stats-root.sh && "
                "test -n \"${khs:-}\" && test -n \"${stats:-}\""
            ),
            temp_dir,
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    archive = args.archive.resolve()
    archive_name = hive_archive_name(args.install_url or str(archive))
    basename, miner, version = derive_hive_names(archive_name)
    members = read_archive_members(archive)
    verify_required_members(members, miner)
    manifest = extract_manifest(archive, miner)

    bash = None if args.skip_bash else find_bash(args.bash)
    if bash:
        run_bash_checks(archive, miner, bash, args.wallet, args.pool, args.worker)
    elif not args.skip_bash:
        print("warning: bash not found; skipped script/delegation checks", file=sys.stderr)

    print(f"OK HiveOS archive: {archive.name}")
    print(f"  basename: {basename}")
    print(f"  miner:    {miner}")
    print(f"  version:  {version}")
    print(f"  manifest: CUSTOM_NAME={manifest['CUSTOM_NAME']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
