param(
    [string]$Version = "0.32.12-opt1",
    [string]$RepoRoot = "",
    [string]$WrapperRoot = "C:\Source\_tmp\minebtx",
    [string]$LinuxSolver = "",
    [string]$WindowsSolver = "",
    [string]$CudaLabel = "cuda12",
    [string]$LinuxCudaLabel = "",
    [string]$WindowsCudaLabel = "",
    [string]$ArchLabel = "sm89",
    [string]$HiveArchLabel = "",
    [string]$ArchDisplay = "SM89/Ada",
    [string]$ReadyPool = "stratum+tcp://ninjaraider.com:44920",
    [string]$ReadyWallet = "btx1zwxtwvgt55h5smfxp7swxacp2qhavz9kpzt0fjvw8303w7kkl7pusgy9e73",
    [string]$StratumMinShareDifficulty = "",
    [switch]$AllowMissingWindows
)

$ErrorActionPreference = "Stop"

if (!$RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
if (!$LinuxCudaLabel) {
    $LinuxCudaLabel = $CudaLabel
}
if (!$WindowsCudaLabel) {
    $WindowsCudaLabel = $CudaLabel
}
if (!$HiveArchLabel) {
    $HiveArchLabel = $ArchLabel
}
if (!$StratumMinShareDifficulty) {
    if ($ReadyPool -match "bitminerpool\.xyz") {
        $StratumMinShareDifficulty = "0.05"
    } else {
        $StratumMinShareDifficulty = ""
    }
}
$DefaultStartStaggerSeconds = if ($ReadyPool -match "bitminerpool\.xyz") { "2" } else { "0" }
if (!$LinuxSolver) {
    $LinuxSolver = Join-Path $RepoRoot "build-docker-$LinuxCudaLabel-$ArchLabel-ubuntu22\bin\btx-gbt-solve"
    if (!(Test-Path -LiteralPath $LinuxSolver -PathType Leaf) -and $LinuxCudaLabel -eq "cuda12" -and $ArchLabel -eq "sm89") {
        $LinuxSolver = Join-Path $RepoRoot "build-docker-cuda12-sm89-ubuntu22\bin\btx-gbt-solve"
    }
}
if (!$WindowsSolver) {
    $candidates = @(
        (Join-Path $RepoRoot "build-win-$WindowsCudaLabel-$ArchLabel-miner-clangcl\bin\btx-gbt-solve.exe"),
        (Join-Path $RepoRoot "build-win-$WindowsCudaLabel-$ArchLabel-miner\bin\btx-gbt-solve.exe"),
        (Join-Path $RepoRoot "build-win-$WindowsCudaLabel-$ArchLabel-miner\src\btx-gbt-solve.exe"),
        (Join-Path $RepoRoot "build-win-cuda-$ArchLabel-miner-clangcl\bin\btx-gbt-solve.exe"),
        (Join-Path $RepoRoot "build-win-cuda-$ArchLabel-miner\bin\btx-gbt-solve.exe"),
        (Join-Path $RepoRoot "build-win-cuda-$ArchLabel-miner\src\btx-gbt-solve.exe")
    )
    if ($WindowsCudaLabel -eq "cuda12" -and $ArchLabel -eq "sm89") {
        $candidates += @(
            (Join-Path $RepoRoot "build-win-cuda-sm89-miner-clangcl\bin\btx-gbt-solve.exe"),
            (Join-Path $RepoRoot "build-win-cuda-sm89-miner\src\btx-gbt-solve.exe"),
            (Join-Path $RepoRoot "build-win-cuda-sm89-miner\bin\btx-gbt-solve.exe"),
            (Join-Path $RepoRoot "build-win-cuda-sm89-miner\btx-gbt-solve.exe")
        )
    }
    $WindowsSolver = ($candidates | Where-Object { Test-Path $_ } | Select-Object -First 1)
}

function Assert-File([string]$Path) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing file: $Path"
    }
}

function Assert-Dir([string]$Path) {
    if (!(Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Missing directory: $Path"
    }
}

function Write-Utf8NoBom([string]$Path, [string]$Content) {
    $parent = Split-Path -Parent $Path
    if ($parent -and !(Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Copy-Tree([string]$Source, [string]$Destination) {
    Assert-Dir $Source
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

function Copy-IfExists([string]$Source, [string]$Destination) {
    if (Test-Path -LiteralPath $Source -PathType Leaf) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Disable-PackagedPayoutSplits([string]$PythonDest) {
    $configPath = Join-Path $PythonDest "config.py"
    $stratumPath = Join-Path $PythonDest "stratum_client.py"
    $hardwarePath = Join-Path $PythonDest "hardware.py"

    $configText = Get-Content -LiteralPath $configPath -Raw
    $configText = [regex]::Replace(
        $configText,
        "(?ms)\r?\n    # Payout splits \(optional\).*?    payout_splits: list\[dict\[str, Any\]\] = dataclasses\.field\(default_factory=list\)\r?\n",
        "`r`n"
    )
    Write-Utf8NoBom $configPath $configText

    $stratumText = Get-Content -LiteralPath $stratumPath -Raw
    $stratumText = $stratumText.Replace("            payout_splits=self.cfg.payout_splits,`r`n", "")
    $stratumText = $stratumText.Replace("            payout_splits=self.cfg.payout_splits,`n", "")
    Write-Utf8NoBom $stratumPath $stratumText

    $hardwareText = Get-Content -LiteralPath $hardwarePath -Raw
    $hardwareText = [regex]::Replace(
        $hardwareText,
        "(?ms),\r?\n    payout_splits: list\[dict\[str, Any\]\] \| None = None",
        ""
    )
    $hardwareText = [regex]::Replace(
        $hardwareText,
        "(?ms)\r?\n        # v0\.4\.x .*?        `"payout_splits`": payout_splits or \[\],\r?\n",
        "`r`n"
    )
    Write-Utf8NoBom $hardwarePath $hardwareText
}

function Get-WslPath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $drive = $full.Substring(0,1).ToLowerInvariant()
    $tail = $full.Substring(3).Replace("\", "/")
    if ($drive -eq "c") {
        return "/mnt/c_full/$tail"
    }
    return "/mnt/$drive/$tail"
}

function Invoke-WslBash([string]$Command) {
    & wsl -e bash -lc "mountpoint -q /mnt/c_full || mount -t drvfs C: /mnt/c_full; $Command"
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed with exit code $LASTEXITCODE"
    }
}

function File-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

Assert-Dir $RepoRoot
Assert-Dir (Join-Path $WrapperRoot "src\dexbtx_miner")
$soloMinerSource = Join-Path $RepoRoot "scripts\mining\fast_solo_miner.py"
Assert-File $soloMinerSource
Assert-File $LinuxSolver
if (!$AllowMissingWindows) {
    if (!$WindowsSolver) { throw "Windows solver was not found." }
    Assert-File $WindowsSolver
}

$releaseName = "btx-miner-$Version"
$releaseRoot = Join-Path $RepoRoot "release-artifacts\$releaseName"
$stageRoot = Join-Path $releaseRoot "stage"
$distRoot = Join-Path $releaseRoot "dist"

$readyPoolTls = if ($ReadyPool -match "^stratum\+(ssl|tls)://") { "true" } else { "false" }
$readyPoolAuthority = [regex]::Replace($ReadyPool, "^[A-Za-z][A-Za-z0-9+.-]*://", "")
$readyPoolAuthority = ($readyPoolAuthority -split "/", 2)[0]
$readyPoolParts = $readyPoolAuthority -split ":", 2
$readyPoolHost = $readyPoolParts[0]
$readyPoolPort = if ($readyPoolParts.Count -gt 1 -and $readyPoolParts[1]) { [int]$readyPoolParts[1] } else { 3333 }

if (Test-Path -LiteralPath $stageRoot) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
if (Test-Path -LiteralPath $distRoot) {
    Remove-Item -LiteralPath $distRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageRoot,$distRoot | Out-Null

$linuxName = "$releaseName-linux-x86_64-$LinuxCudaLabel-$ArchLabel"
$hiveName = "$releaseName-hiveos-x86_64-$LinuxCudaLabel-$HiveArchLabel"
$windowsName = "$releaseName-windows-x86_64-$WindowsCudaLabel-$ArchLabel"
$hiveMinerName = "btx-miner"
$hiveVersionToken = $Version -replace "[^A-Za-z0-9._+~]", "_"
$hiveArchiveName = "$hiveMinerName-$hiveVersionToken"
$linuxDir = Join-Path $stageRoot $linuxName
$hiveDir = Join-Path $stageRoot $hiveName
$windowsDir = Join-Path $stageRoot $windowsName

New-Item -ItemType Directory -Force -Path $linuxDir,$hiveDir | Out-Null
if ($WindowsSolver) {
    New-Item -ItemType Directory -Force -Path $windowsDir | Out-Null
}

$pythonSource = Join-Path $WrapperRoot "src\dexbtx_miner"
$pyproject = Join-Path $WrapperRoot "pyproject.toml"
$license = Join-Path $WrapperRoot "LICENSE"
$linuxBinDir = Split-Path -Parent $LinuxSolver
$linuxBackendInfo = Join-Path $linuxBinDir "btx-matmul-backend-info"
$linuxBench = Join-Path $linuxBinDir "btx-matmul-solve-bench"

$linuxSha = File-Sha256 $LinuxSolver
$windowsSha = if ($WindowsSolver) { File-Sha256 $WindowsSolver } else { "" }
$sourceCommit = (& git -C $RepoRoot rev-parse --short=12 HEAD).Trim()
$sourceDirtyLines = & git -C $RepoRoot status --short -- . ":(exclude)release-artifacts"
$sourceDirty = ($sourceDirtyLines -join "`n").Trim()
$dirtyNote = if ($sourceDirty) { "yes" } else { "no" }

function Add-CommonFiles([string]$Dest, [string]$Platform, [string]$PackageCudaLabel) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Dest "bin"),(Join-Path $Dest "python") | Out-Null
    $pythonDest = Join-Path $Dest "python\dexbtx_miner"
    Copy-Tree $pythonSource $pythonDest
    Disable-PackagedPayoutSplits $pythonDest
    Copy-Item -LiteralPath $soloMinerSource -Destination (Join-Path $Dest "fast_solo_miner.py") -Force
    Get-ChildItem -LiteralPath (Join-Path $Dest "python\dexbtx_miner") -Directory -Filter "__pycache__" -Recurse -Force |
        Remove-Item -Recurse -Force
    Copy-IfExists $pyproject (Join-Path $Dest "pyproject.toml")
    Copy-IfExists $license (Join-Path $Dest "LICENSE.dxbtx-wrapper")

    Write-Utf8NoBom (Join-Path $Dest "requirements.txt") @'
pyyaml>=6.0
'@

    Write-Utf8NoBom (Join-Path $Dest "miner.env.example") @"
# Copy to miner.env, set BTX_WALLET, then run ./run.sh or run.bat.
BTX_MODE=pool
BTX_WALLET=
BTX_POOL=$ReadyPool
BTX_WORKER_PREFIX=my-rig
BTX_STRATUM_PASSWORD=
BTX_STRATUM_MIN_SHARE_DIFFICULTY=$StratumMinShareDifficulty

# Tuned defaults for the optimized SM89/Ada solver used on 4070 Ti SUPER.
BTX_SOLVER_THREADS=1
BTX_PREPARE_WORKERS=2
BTX_BATCH_SIZE=512
BTX_CUDA_POOL_SLOTS=
BTX_PREFETCH=8
BTX_GPU_INPUTS=1
BTX_WORKERS_PER_GPU=1
BTX_START_STAGGER_SECONDS=$DefaultStartStaggerSeconds
BTX_NONCES_PER_SLICE=100000000000
BTX_SOLVER_MAX_SECONDS_PER_SLICE=30
BTX_LOG_LEVEL=INFO
BTX_MATMUL_CUDA_DEVICE_PREPARED_INPUTS=1
BTX_MATMUL_CUDA_TILED_DIRECT_RHS=1
BTX_MATMUL_CUDA_WAIT_POLICY=blocking

# Solo mode uses the same solver but talks to your local or LAN BTX node RPC.
# Set BTX_MODE=solo and point these at a synced node.
BTX_SOLO_DATADIR=
BTX_SOLO_CONF=
BTX_SOLO_RPC_CONNECT=127.0.0.1
BTX_SOLO_RPC_PORT=19334
BTX_SOLO_RPC_USER=
BTX_SOLO_RPC_PASSWORD=
BTX_SOLO_RPC_COOKIE=
BTX_FASTSOLO_BATCH_SIZE=512
BTX_FASTSOLO_SOLVER_THREADS=1
BTX_FASTSOLO_POOL_SLOTS=0
BTX_FASTSOLO_SLICE_SECONDS=30
BTX_FASTSOLO_MAX_TRIES=100000000
BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS=30
BTX_FASTSOLO_STATS_FILE=

# Local RTX 5090 safe desktop profile measured on CUDA 13 / SM120:
#   BTX_SOLVER_THREADS=2
#   BTX_BATCH_SIZE=768
#   BTX_CUDA_POOL_SLOTS=4
#   BTX_WORKERS_PER_GPU=3
"@

    Write-Utf8NoBom (Join-Path $Dest "config.example.yaml") @"
pool_host: "$readyPoolHost"
pool_port: $readyPoolPort
pool_tls: $readyPoolTls

# Required. Set this to your payout address. The Windows START_MINING.bat
# convenience file defaults to the operator wallet requested for this build and
# can be edited before running.
payout_address: "btx1z...YOUR_BTX_ADDRESS_HERE..."
worker_name: "my-rig-gpu0"
stratum_password: ""
stratum_min_share_difficulty: "$StratumMinShareDifficulty"

gbt_solve_path: "./bin/btx-gbt-solve"
solver_backend: "cuda"
solver_threads: 1
solver_prepare_workers: 2
solver_batch_size: 512
solver_prefetch_depth: 8
gpu_inputs: 1
nonces_per_slice: 100000000000
log_level: "INFO"
"@

    Write-Utf8NoBom (Join-Path $Dest "NO_DEV_FEE.txt") @"
BTX Miner $Version

This package has no developer fee.

What that means:
- No developer-fee wallet is hard-coded in the launchers.
- No payout split is configured in config.example.yaml or miner.env.example.
- Windows START_MINING.bat defaults to the operator wallet requested for this
  build. Edit BTX_WALLET in that file before running to mine to a different
  wallet.
- The bundled launchers set DEXBTX_NO_SOLVER_AUTOUPDATE=1,
  DEXBTX_NO_WRAPPER_AUTOUPDATE=1, and DEXBTX_NO_SOLVER_RECHECK=1 so the
  local optimized solver is not replaced by an upstream updater.
- The only payout address used at runtime is the BTX_WALLET value supplied by
  miner.env, the environment, command-line flags, or START_MINING.bat.
"@

    Write-Utf8NoBom (Join-Path $Dest "README.md") @"
# BTX Miner $Version ($Platform)

No dev fee. Set your own BTX_WALLET and run the launcher. The Windows package
also includes START_MINING.bat with an editable operator default wallet.

Default pool:

$ReadyPool

## Pool and Solo Mode

BTX_MODE defaults to pool and connects to BTX_POOL. Set BTX_MODE=solo to mine
directly against a synced local or LAN BTX node using getblocktemplate and
submitblock. Solo mode uses the same optimized btx-gbt-solve binary; set
BTX_SOLO_DATADIR / BTX_SOLO_CONF or BTX_SOLO_RPC_* in miner.env before using it.

## Quick Start

Linux/HiveOS:

cp miner.env.example miner.env
nano miner.env
./run.sh

Windows:

copy miner.env.example miner.env
notepad miner.env
.\run.bat

Or use the ready-to-go Windows batch file:

notepad START_MINING.bat
.\START_MINING.bat

START_MINING.bat defaults to:

BTX_POOL=$ReadyPool
BTX_WALLET=$ReadyWallet

Edit those two lines before running if you want a different pool or wallet.

## Multi-GPU Behavior

The main launcher starts one miner process per visible NVIDIA GPU and sets
CUDA_VISIBLE_DEVICES per process. Worker names are:

<BTX_WORKER_PREFIX>-gpu0, <BTX_WORKER_PREFIX>-gpu1, ...

Set BTX_WORKERS_PER_GPU to run multiple miner processes against each visible
GPU. When BTX_WORKERS_PER_GPU is greater than 1, worker names are:

<BTX_WORKER_PREFIX>-gpu0w0, <BTX_WORKER_PREFIX>-gpu0w1, ...

Set BTX_SINGLE_GPU=1 to run only the current process/GPU.

## RTX 5090 Tuning

A local RTX 5090 desktop test reached about 341K N/s with:

BTX_SOLVER_THREADS=2
BTX_BATCH_SIZE=768
BTX_CUDA_POOL_SLOTS=4
BTX_WORKERS_PER_GPU=3

Four workers measured only about 0.6% higher and was not worth the extra
desktop load. Do not jump to high worker counts on an interactive PC.

## Binaries

Linux solver SHA256:

$linuxSha

Windows solver SHA256:

$windowsSha

BTX source commit: $sourceCommit
BTX source dirty at package time: $dirtyNote

CUDA target: $PackageCudaLabel / $ArchDisplay.

The original SM89 package was tuned on Ada cards such as RTX 4070 Ti SUPER.
Mixed-family packages include additional NVIDIA device code for the GPU
families named in the artifact. Validate tuning on each GPU generation before
assuming one set of batch/prefetch values is optimal everywhere.
"@
}

Add-CommonFiles $linuxDir "Linux x86_64 $LinuxCudaLabel $ArchLabel" $LinuxCudaLabel
Add-CommonFiles $hiveDir "HiveOS x86_64 $LinuxCudaLabel $HiveArchLabel" $LinuxCudaLabel
if ($WindowsSolver) {
    Add-CommonFiles $windowsDir "Windows x86_64 $WindowsCudaLabel $ArchLabel" $WindowsCudaLabel
}

Copy-Item -LiteralPath $LinuxSolver -Destination (Join-Path $linuxDir "bin\btx-gbt-solve") -Force
Copy-Item -LiteralPath $LinuxSolver -Destination (Join-Path $hiveDir "bin\btx-gbt-solve") -Force
Copy-IfExists $linuxBackendInfo (Join-Path $linuxDir "bin\btx-matmul-backend-info")
Copy-IfExists $linuxBackendInfo (Join-Path $hiveDir "bin\btx-matmul-backend-info")
Copy-IfExists $linuxBench (Join-Path $linuxDir "bin\btx-matmul-solve-bench")
Copy-IfExists $linuxBench (Join-Path $hiveDir "bin\btx-matmul-solve-bench")
if ($WindowsSolver) {
    Copy-Item -LiteralPath $WindowsSolver -Destination (Join-Path $windowsDir "bin\btx-gbt-solve.exe") -Force
}

$runOneSh = @'
#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DIR/miner.env" ]; then
  set -a
  . "$DIR/miner.env"
  set +a
fi

BTX_POOL="${BTX_POOL:-__BTX_READY_POOL__}"
BTX_WALLET="${BTX_WALLET:-}"
BTX_WORKER="${BTX_WORKER:-${BTX_WORKER_PREFIX:-$(hostname)}-gpu${CUDA_VISIBLE_DEVICES:-0}}"
BTX_SOLVER_THREADS="${BTX_SOLVER_THREADS:-1}"
BTX_PREPARE_WORKERS="${BTX_PREPARE_WORKERS:-2}"
BTX_BATCH_SIZE="${BTX_BATCH_SIZE:-512}"
BTX_CUDA_POOL_SLOTS="${BTX_CUDA_POOL_SLOTS:-}"
BTX_PREFETCH="${BTX_PREFETCH:-8}"
BTX_GPU_INPUTS="${BTX_GPU_INPUTS:-1}"
BTX_NONCES_PER_SLICE="${BTX_NONCES_PER_SLICE:-100000000000}"
BTX_SOLVER_MAX_SECONDS_PER_SLICE="${BTX_SOLVER_MAX_SECONDS_PER_SLICE:-30}"
BTX_LOG_LEVEL="${BTX_LOG_LEVEL:-INFO}"

if [ -z "$BTX_WALLET" ]; then
  echo "Set BTX_WALLET in miner.env or the environment." >&2
  exit 2
fi

export DEXBTX_NO_SOLVER_AUTOUPDATE=1
export DEXBTX_NO_WRAPPER_AUTOUPDATE=1
export DEXBTX_NO_SOLVER_RECHECK=1
export PYTHONPATH="$DIR/python${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "$BTX_CUDA_POOL_SLOTS" ]; then
  export BTX_MATMUL_CUDA_POOL_SLOTS="$BTX_CUDA_POOL_SLOTS"
fi
export BTX_MATMUL_CUDA_DEVICE_PREPARED_INPUTS="${BTX_MATMUL_CUDA_DEVICE_PREPARED_INPUTS:-1}"
export BTX_MATMUL_CUDA_TILED_DIRECT_RHS="${BTX_MATMUL_CUDA_TILED_DIRECT_RHS:-1}"
export BTX_MATMUL_CUDA_WAIT_POLICY="${BTX_MATMUL_CUDA_WAIT_POLICY:-blocking}"

exec python3 -m dexbtx_miner \
  --pool "$BTX_POOL" \
  --address "$BTX_WALLET" \
  --worker "$BTX_WORKER" \
  --gbt-solve "$DIR/bin/btx-gbt-solve" \
  --threads "$BTX_SOLVER_THREADS" \
  --prepare-workers "$BTX_PREPARE_WORKERS" \
  --batch-size "$BTX_BATCH_SIZE" \
  --prefetch "$BTX_PREFETCH" \
  --gpu-inputs "$BTX_GPU_INPUTS" \
  --nonces-per-slice "$BTX_NONCES_PER_SLICE" \
  --max-seconds-per-slice "$BTX_SOLVER_MAX_SECONDS_PER_SLICE" \
  --log-level "$BTX_LOG_LEVEL"
'@
$runOneSh = $runOneSh.Replace("__BTX_READY_POOL__", $ReadyPool)

$runAllSh = @'
#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DIR/miner.env" ]; then
  set -a
  . "$DIR/miner.env"
  set +a
fi

mkdir -p "$DIR/logs"
prefix="${BTX_WORKER_PREFIX:-$(hostname)}"
workers_per_gpu="${BTX_WORKERS_PER_GPU:-1}"
case "$workers_per_gpu" in
  ''|*[!0-9]*)
    echo "BTX_WORKERS_PER_GPU must be a positive integer." >&2
    exit 2
    ;;
esac
if [ "$workers_per_gpu" -lt 1 ]; then
  echo "BTX_WORKERS_PER_GPU must be at least 1." >&2
  exit 2
fi
start_stagger_seconds="${BTX_START_STAGGER_SECONDS:-${BTX_START_STAGGER_S:-0}}"

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
else
  gpu_count=0
fi

if [ "${gpu_count:-0}" -le 0 ]; then
  echo "No NVIDIA GPUs detected; starting one miner process without CUDA_VISIBLE_DEVICES." >&2
  exec "$DIR/run-one.sh"
fi

pids=""
cleanup() {
  for p in $pids; do
    kill "$p" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

i=0
started=0
while [ "$i" -lt "$gpu_count" ]; do
  w=0
  while [ "$w" -lt "$workers_per_gpu" ]; do
    if [ "$workers_per_gpu" -eq 1 ]; then
      worker="${prefix}-gpu${i}"
      log_name="gpu${i}"
    else
      worker="${prefix}-gpu${i}w${w}"
      log_name="gpu${i}w${w}"
    fi
    (
      export CUDA_VISIBLE_DEVICES="$i"
      export BTX_WORKER="$worker"
      exec "$DIR/run-one.sh"
    ) >> "$DIR/logs/${log_name}.log" 2>&1 &
    pids="$pids $!"
    started=$((started + 1))
    if [ "$start_stagger_seconds" != "0" ] && [ -n "$start_stagger_seconds" ]; then
      sleep "$start_stagger_seconds"
    fi
    w=$((w + 1))
  done
  i=$((i + 1))
done

echo "Started $started BTX miner process(es) across $gpu_count GPU(s). Logs are in $DIR/logs."
wait -n
status=$?
cleanup
exit "$status"
'@

$runSoloSh = @'
#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DIR/miner.env" ]; then
  set -a
  . "$DIR/miner.env"
  set +a
fi

BTX_WALLET="${BTX_WALLET:-}"
BTX_FASTSOLO_BATCH_SIZE="${BTX_FASTSOLO_BATCH_SIZE:-${BTX_BATCH_SIZE:-512}}"
BTX_FASTSOLO_SOLVER_THREADS="${BTX_FASTSOLO_SOLVER_THREADS:-${BTX_SOLVER_THREADS:-1}}"
BTX_FASTSOLO_POOL_SLOTS="${BTX_FASTSOLO_POOL_SLOTS:-0}"
BTX_FASTSOLO_SLICE_SECONDS="${BTX_FASTSOLO_SLICE_SECONDS:-30}"
BTX_FASTSOLO_MAX_TRIES="${BTX_FASTSOLO_MAX_TRIES:-100000000}"
BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS="${BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS:-30}"
BTX_FASTSOLO_TEMPLATE_POLL_SECONDS="${BTX_FASTSOLO_TEMPLATE_POLL_SECONDS:-0.5}"
BTX_FASTSOLO_STATS_FILE="${BTX_FASTSOLO_STATS_FILE:-$DIR/logs/solo-stats.json}"
BTX_FASTSOLO_GPUS="${BTX_FASTSOLO_GPUS:-${BTX_GPUS:-auto}}"

if [ -z "$BTX_WALLET" ]; then
  echo "Set BTX_WALLET in miner.env or the environment." >&2
  exit 2
fi

mkdir -p "$DIR/logs"
export BTX_FASTSOLO_STATS_FILE
export BTX_FASTSOLO_BATCH_SIZE
export BTX_FASTSOLO_SOLVER_THREADS
export BTX_FASTSOLO_POOL_SLOTS
export BTX_FASTSOLO_SLICE_SECONDS
export BTX_FASTSOLO_MAX_TRIES
export BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS
export BTX_FASTSOLO_TEMPLATE_POLL_SECONDS

cmd=(python3 "$DIR/fast_solo_miner.py"
  --address "$BTX_WALLET"
  --solver "$DIR/bin/btx-gbt-solve"
  --gpus "$BTX_FASTSOLO_GPUS"
  --stats-file "$BTX_FASTSOLO_STATS_FILE"
  --batch-size "$BTX_FASTSOLO_BATCH_SIZE"
  --solver-threads "$BTX_FASTSOLO_SOLVER_THREADS"
  --pool-slots "$BTX_FASTSOLO_POOL_SLOTS"
  --slice-seconds "$BTX_FASTSOLO_SLICE_SECONDS"
  --max-tries "$BTX_FASTSOLO_MAX_TRIES"
  --template-refresh-seconds "$BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS"
  --template-poll-seconds "$BTX_FASTSOLO_TEMPLATE_POLL_SECONDS")

[ -n "${BTX_SOLO_DATADIR:-}" ] && cmd+=(--datadir "$BTX_SOLO_DATADIR")
[ -n "${BTX_SOLO_CONF:-}" ] && cmd+=(--conf "$BTX_SOLO_CONF")
[ -n "${BTX_SOLO_RPC_CONNECT:-}" ] && cmd+=(--rpcconnect "$BTX_SOLO_RPC_CONNECT")
[ -n "${BTX_SOLO_RPC_PORT:-}" ] && cmd+=(--rpcport "$BTX_SOLO_RPC_PORT")
[ -n "${BTX_SOLO_RPC_USER:-}" ] && cmd+=(--rpcuser "$BTX_SOLO_RPC_USER")
[ -n "${BTX_SOLO_RPC_PASSWORD:-}" ] && cmd+=(--rpcpassword "$BTX_SOLO_RPC_PASSWORD")
[ -n "${BTX_SOLO_RPC_COOKIE:-}" ] && cmd+=(--rpccookiefile "$BTX_SOLO_RPC_COOKIE")

exec "${cmd[@]}"
'@

$runSh = @'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DIR/h-config.sh" ] && { [ -n "${CUSTOM_TEMPLATE:-}" ] || [ -n "${CUSTOM_WALLET:-}" ] || [ -n "${CUSTOM_URL:-}" ] || [ -n "${CUSTOM_USER_CONFIG:-}" ] || [ -n "${WORKER_NAME:-}" ]; }; then
  "$DIR/h-config.sh" >/dev/null
fi
if [ -f "$DIR/miner.env" ]; then
  set -a
  . "$DIR/miner.env"
  set +a
fi
if [ "${BTX_MODE:-pool}" = "solo" ]; then
  exec "$DIR/run-solo.sh" "$@"
fi
if [ "${BTX_SINGLE_GPU:-0}" = "1" ]; then
  exec "$DIR/run-one.sh" "$@"
fi
exec "$DIR/run-all-gpus.sh" "$@"
'@

foreach ($dir in @($linuxDir, $hiveDir)) {
    Write-Utf8NoBom (Join-Path $dir "run-one.sh") $runOneSh
    Write-Utf8NoBom (Join-Path $dir "run-all-gpus.sh") $runAllSh
    Write-Utf8NoBom (Join-Path $dir "run-solo.sh") $runSoloSh
    Write-Utf8NoBom (Join-Path $dir "run.sh") $runSh
}

Write-Utf8NoBom (Join-Path $hiveDir "h-manifest.conf") @"
CUSTOM_NAME=btx-miner
CUSTOM_VERSION=$Version
CUSTOM_ALGO=matmul
CUSTOM_MINERBIN=run.sh
CUSTOM_CONFIG_FILENAME=/hive/miners/custom/`$CUSTOM_NAME/miner.env
CUSTOM_LOG_BASENAME=/var/log/miner/`$CUSTOM_NAME/btx-miner
WEB_PORT=0
"@

$hConfigSh = @'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ ! -f ./h-manifest.conf ] && [ -d /hive/miners/custom/btx-miner ]; then
  cd /hive/miners/custom/btx-miner
fi
. ./h-manifest.conf
miner_dir="$(pwd)"
if [ -z "${CUSTOM_CONFIG_FILENAME:-}" ] || [ ! -d "$(dirname "$CUSTOM_CONFIG_FILENAME")" ]; then
  CUSTOM_CONFIG_FILENAME="$miner_dir/miner.env"
fi
export CUSTOM_CONFIG_FILENAME

template="${BTX_WALLET:-${CUSTOM_TEMPLATE:-${CUSTOM_WALLET:-}}}"
wallet="$template"
template_worker=""
if [[ "$template" == btx1*.* ]]; then
  wallet="${template%%.*}"
  template_worker="${template#*.}"
fi
if [[ "$wallet" != btx1* ]] && [ -n "${CUSTOM_WALLET:-}" ]; then
  wallet="$CUSTOM_WALLET"
fi
pool="${BTX_POOL:-${CUSTOM_URL:-__BTX_READY_POOL__}}"
worker="${BTX_WORKER_PREFIX:-${template_worker:-${WORKER_NAME:-$(hostname)}}}"
mode="${BTX_MODE:-pool}"

cat > "$CUSTOM_CONFIG_FILENAME" <<EOF
BTX_MODE=$mode
BTX_WALLET=$wallet
BTX_POOL=$pool
BTX_WORKER_PREFIX=$worker
BTX_STRATUM_PASSWORD=${BTX_STRATUM_PASSWORD:-}
BTX_STRATUM_MIN_SHARE_DIFFICULTY=${BTX_STRATUM_MIN_SHARE_DIFFICULTY:-__BTX_STRATUM_MIN_SHARE_DIFFICULTY__}
BTX_SOLVER_THREADS=${BTX_SOLVER_THREADS:-1}
BTX_PREPARE_WORKERS=${BTX_PREPARE_WORKERS:-2}
BTX_BATCH_SIZE=${BTX_BATCH_SIZE:-512}
BTX_CUDA_POOL_SLOTS=${BTX_CUDA_POOL_SLOTS:-}
BTX_PREFETCH=${BTX_PREFETCH:-8}
BTX_GPU_INPUTS=${BTX_GPU_INPUTS:-1}
BTX_WORKERS_PER_GPU=${BTX_WORKERS_PER_GPU:-1}
BTX_START_STAGGER_SECONDS=${BTX_START_STAGGER_SECONDS:-__BTX_START_STAGGER_SECONDS__}
BTX_NONCES_PER_SLICE=${BTX_NONCES_PER_SLICE:-100000000000}
BTX_SOLVER_MAX_SECONDS_PER_SLICE=${BTX_SOLVER_MAX_SECONDS_PER_SLICE:-30}
BTX_LOG_LEVEL=${BTX_LOG_LEVEL:-INFO}
BTX_MATMUL_CUDA_DEVICE_PREPARED_INPUTS=${BTX_MATMUL_CUDA_DEVICE_PREPARED_INPUTS:-1}
BTX_MATMUL_CUDA_TILED_DIRECT_RHS=${BTX_MATMUL_CUDA_TILED_DIRECT_RHS:-1}
BTX_MATMUL_CUDA_WAIT_POLICY=${BTX_MATMUL_CUDA_WAIT_POLICY:-blocking}
BTX_SOLO_DATADIR=${BTX_SOLO_DATADIR:-}
BTX_SOLO_CONF=${BTX_SOLO_CONF:-}
BTX_SOLO_RPC_CONNECT=${BTX_SOLO_RPC_CONNECT:-127.0.0.1}
BTX_SOLO_RPC_PORT=${BTX_SOLO_RPC_PORT:-19334}
BTX_SOLO_RPC_USER=${BTX_SOLO_RPC_USER:-}
BTX_SOLO_RPC_PASSWORD=${BTX_SOLO_RPC_PASSWORD:-}
BTX_SOLO_RPC_COOKIE=${BTX_SOLO_RPC_COOKIE:-}
BTX_FASTSOLO_BATCH_SIZE=${BTX_FASTSOLO_BATCH_SIZE:-${BTX_BATCH_SIZE:-512}}
BTX_FASTSOLO_SOLVER_THREADS=${BTX_FASTSOLO_SOLVER_THREADS:-${BTX_SOLVER_THREADS:-1}}
BTX_FASTSOLO_POOL_SLOTS=${BTX_FASTSOLO_POOL_SLOTS:-0}
BTX_FASTSOLO_SLICE_SECONDS=${BTX_FASTSOLO_SLICE_SECONDS:-30}
BTX_FASTSOLO_MAX_TRIES=${BTX_FASTSOLO_MAX_TRIES:-100000000}
BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS=${BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS:-30}
BTX_FASTSOLO_STATS_FILE=${BTX_FASTSOLO_STATS_FILE:-}
EOF

if [ -n "${CUSTOM_USER_CONFIG:-}" ]; then
  for token in $CUSTOM_USER_CONFIG; do
    if printf '%s' "$token" | grep -Eq '^(BTX|DEXBTX)_[A-Za-z0-9_]*=[A-Za-z0-9_./:+,=-]*$'; then
      printf '%s\n' "$token" >> "$CUSTOM_CONFIG_FILENAME"
    fi
  done
fi

echo "BTX miner config written to $CUSTOM_CONFIG_FILENAME"

# HiveOS miner-run sources h-config.sh and expects these functions to exist.
# Custom/local packages should not return a miner_ver, or Hive tries to install
# a nonexistent hive-miners-custom-$version package.
MINER_API_PORT=${MINER_API_PORT:-${WEB_PORT:-0}}
MINER_LOG_BASENAME=${MINER_LOG_BASENAME:-${CUSTOM_LOG_BASENAME:-/var/log/miner/btx-miner/btx-miner}}
miner_ver() { :; }
miner_config_gen() { :; }
'@
$hConfigSh = $hConfigSh.Replace("__BTX_READY_POOL__", $ReadyPool)
$hConfigSh = $hConfigSh.Replace("__BTX_STRATUM_MIN_SHARE_DIFFICULTY__", $StratumMinShareDifficulty)
$hConfigSh = $hConfigSh.Replace("__BTX_START_STAGGER_SECONDS__", $DefaultStartStaggerSeconds)
Write-Utf8NoBom (Join-Path $hiveDir "h-config.sh") $hConfigSh

Write-Utf8NoBom (Join-Path $hiveDir "h-run.sh") @'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ ! -f ./h-manifest.conf ] && [ -d /hive/miners/custom/btx-miner ]; then
  cd /hive/miners/custom/btx-miner
fi
. ./h-manifest.conf
miner_dir="$(pwd)"
if [ -z "${CUSTOM_CONFIG_FILENAME:-}" ] || [ ! -d "$(dirname "$CUSTOM_CONFIG_FILENAME")" ]; then
  CUSTOM_CONFIG_FILENAME="$miner_dir/miner.env"
fi
export CUSTOM_CONFIG_FILENAME

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
live_btx_wallet="${BTX_WALLET:-}"
live_btx_pool="${BTX_POOL:-}"
live_btx_worker_prefix="${BTX_WORKER_PREFIX:-}"
live_btx_stratum_password="${BTX_STRATUM_PASSWORD:-}"
live_btx_stratum_min_share_difficulty="${BTX_STRATUM_MIN_SHARE_DIFFICULTY:-}"
live_btx_start_stagger_seconds="${BTX_START_STAGGER_SECONDS:-}"

if [ -f "$CUSTOM_CONFIG_FILENAME" ]; then
  set -a
  . "$CUSTOM_CONFIG_FILENAME"
  set +a
fi

if [ -n "$live_btx_wallet" ]; then
  export BTX_WALLET="$live_btx_wallet"
elif [ -n "${CUSTOM_TEMPLATE:-}" ]; then
  export BTX_WALLET="$CUSTOM_TEMPLATE"
elif [ -n "${CUSTOM_WALLET:-}" ]; then
  export BTX_WALLET="$CUSTOM_WALLET"
fi

if [ -n "$live_btx_pool" ]; then
  export BTX_POOL="$live_btx_pool"
elif [ -n "${CUSTOM_URL:-}" ]; then
  export BTX_POOL="$CUSTOM_URL"
fi

if [ -n "$live_btx_worker_prefix" ]; then
  export BTX_WORKER_PREFIX="$live_btx_worker_prefix"
elif [ -n "${WORKER_NAME:-}" ]; then
  export BTX_WORKER_PREFIX="$WORKER_NAME"
fi

if [ -n "$live_btx_stratum_password" ]; then
  export BTX_STRATUM_PASSWORD="$live_btx_stratum_password"
fi
if [ -n "$live_btx_stratum_min_share_difficulty" ]; then
  export BTX_STRATUM_MIN_SHARE_DIFFICULTY="$live_btx_stratum_min_share_difficulty"
fi
if [ -n "$live_btx_start_stagger_seconds" ]; then
  export BTX_START_STAGGER_SECONDS="$live_btx_start_stagger_seconds"
fi

./h-config.sh

if [ -f "$CUSTOM_CONFIG_FILENAME" ]; then
  set -a
  . "$CUSTOM_CONFIG_FILENAME"
  set +a
fi

if [ -z "${BTX_WALLET:-}" ]; then
  echo "BTX_WALLET is required. Set the Hive wallet/template to your btx1z address." >&2
  exit 2
fi

exec ./run.sh 2>&1 | tee -a "$CUSTOM_LOG_BASENAME.log"
'@

Write-Utf8NoBom (Join-Path $hiveDir "h-stats.sh") @'
#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
miner_dir="${BTX_HIVE_MINER_DIR:-$script_dir}"
manifest="$miner_dir/h-manifest.conf"
[ -f "$manifest" ] && . "$manifest"
if [ -z "${CUSTOM_CONFIG_FILENAME:-}" ] || [ ! -d "$(dirname "$CUSTOM_CONFIG_FILENAME")" ]; then
  CUSTOM_CONFIG_FILENAME="$miner_dir/miner.env"
fi
[ -f "${CUSTOM_CONFIG_FILENAME:-}" ] && . "$CUSTOM_CONFIG_FILENAME"

log_dir="${BTX_HIVE_LOG_DIR:-$miner_dir/logs}"
version="${CUSTOM_VERSION:-unknown}"
mode="${BTX_MODE:-pool}"
solo_stats="${BTX_FASTSOLO_STATS_FILE:-$log_dir/solo-stats.json}"

json_number_array() {
  local out="[" sep="" v
  for v in "$@"; do
    case "$v" in
      ''|*[!0-9.]* ) v=0 ;;
    esac
    out="${out}${sep}${v}"
    sep=","
  done
  printf '%s]' "$out"
}

json_string_array() {
  local out="[" sep="" v
  for v in "$@"; do
    v="${v//\\/\\\\}"
    v="${v//\"/\\\"}"
    out="${out}${sep}\"${v}\""
    sep=","
  done
  printf '%s]' "$out"
}

collect_gpu_sensors() {
  bus_values=()
  temp_values=()
  fan_values=()
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  while IFS=',' read -r bus temp fan; do
    bus="$(printf '%s' "${bus:-}" | xargs)"
    temp="$(printf '%s' "${temp:-0}" | xargs)"
    fan="$(printf '%s' "${fan:-0}" | xargs)"
    temp="${temp%%.*}"
    fan="${fan%%.*}"
    case "$temp" in ''|*[!0-9]* ) temp=0 ;; esac
    case "$fan" in ''|*[!0-9]* ) fan=0 ;; esac
    bus_values+=("$bus")
    temp_values+=("$temp")
    fan_values+=("$fan")
  done < <(nvidia-smi --query-gpu=pci.bus_id,temperature.gpu,fan.speed --format=csv,noheader,nounits 2>/dev/null || true)
}

miner_uptime() {
  local pid
  pid="$(pgrep -fo 'dexbtx_miner|fast_solo_miner.py|btx-gbt-solve' 2>/dev/null || true)"
  if [ -n "$pid" ]; then
    ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ' || printf '0'
  else
    printf '0'
  fi
}

collect_gpu_sensors
bus_numeric_values=()
for bus in "${bus_values[@]}"; do
  b="${bus#*:}"
  b="${b%%:*}"
  b="${b#0x}"
  case "$b" in
    ''|*[!0-9A-Fa-f]* ) b=0 ;;
    * ) b=$((16#$b)) ;;
  esac
  bus_numeric_values+=("$b")
done
bus_json="$(json_number_array "${bus_numeric_values[@]}")"
temp_json="$(json_number_array "${temp_values[@]}")"
fan_json="$(json_number_array "${fan_values[@]}")"
uptime="$(miner_uptime)"
case "$uptime" in ''|*[!0-9]* ) uptime=0 ;; esac

hs_values=()
accepted_total=0
rejected_total=0

if [ -s "$solo_stats" ] && command -v jq >/dev/null 2>&1; then
  mode="solo"
  while IFS= read -r value; do
    hs_values+=("$value")
  done < <(jq -r '
    if (.miner.gpus // []) | length > 0 then
      .miner.gpus[] | ((.hashrate // .nonce_rate // .btx_work_rate // 0) / 1000)
    else
      ((.miner.hashrate // .miner.nonce_rate // .miner.btx_work_rate // 0) / 1000)
    end | @text
  ' "$solo_stats" 2>/dev/null)
  accepted_total="$(jq -r '[.miner.gpus[]?.accepted // 0] | add // 0' "$solo_stats" 2>/dev/null || echo 0)"
  rejected_total=0
elif [ -s "$solo_stats" ] && command -v python3 >/dev/null 2>&1; then
  mode="solo"
  solo_parsed="$(python3 - "$solo_stats" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
miner = payload.get("miner") or {}
gpus = miner.get("gpus") or []
if gpus:
    rates = [
        float(gpu.get("hashrate") or gpu.get("nonce_rate") or gpu.get("btx_work_rate") or 0.0) / 1000.0
        for gpu in gpus
    ]
    accepted = sum(int(gpu.get("accepted") or 0) for gpu in gpus)
else:
    rates = [float(miner.get("hashrate") or miner.get("nonce_rate") or miner.get("btx_work_rate") or 0.0) / 1000.0]
    accepted = int(miner.get("accepted") or 0)
print(" ".join(f"{rate:.3f}" for rate in rates))
print(accepted)
PY
)"
  solo_rates="$(printf '%s\n' "$solo_parsed" | sed -n '1p')"
  accepted_total="$(printf '%s\n' "$solo_parsed" | sed -n '2p')"
  read -r -a hs_values <<< "$solo_rates"
  rejected_total=0
elif [ -s "$solo_stats" ]; then
  mode="solo"
  solo_json="$(tr -d '\n\r\t ' < "$solo_stats")"
  if printf '%s' "$solo_json" | grep -q '"gpus":'; then
    solo_json="$(printf '%s' "$solo_json" | sed 's/.*"gpus":\[//; s/\].*//')"
  fi
  solo_parsed="$(printf '%s' "$solo_json" | tr '{}' '\n' | awk -F, '
    {
      rate="";
      acc=0;
      for (i=1;i<=NF;i++) {
        if ($i ~ /"hashrate":/) {
          split($i, a, ":");
          gsub(/[^0-9.]/, "", a[2]);
          rate=sprintf("%.3f", (a[2]+0)/1000.0);
        }
        if ($i ~ /"accepted":/) {
          split($i, a, ":");
          gsub(/[^0-9]/, "", a[2]);
          acc+=(a[2]+0);
        }
      }
      if (rate != "") {
        rates = rates sep rate;
        sep = " ";
      }
      accepted += acc;
    }
    END { print rates; print accepted+0 }
  ')"
  solo_rates="$(printf '%s\n' "$solo_parsed" | sed -n '1p')"
  accepted_total="$(printf '%s\n' "$solo_parsed" | sed -n '2p')"
  read -r -a hs_values <<< "$solo_rates"
  rejected_total=0
fi

if [ "${#hs_values[@]}" -eq 0 ]; then
  log_files=()
  while IFS= read -r file; do
    log_files+=("$file")
  done < <(find "$log_dir" -maxdepth 1 -type f -name 'gpu*.log' 2>/dev/null | sort)
  if [ "${#log_files[@]}" -eq 0 ] && [ -n "${CUSTOM_LOG_BASENAME:-}" ] && [ -f "$CUSTOM_LOG_BASENAME.log" ]; then
    log_files=("$CUSTOM_LOG_BASENAME.log")
  fi

  for file in "${log_files[@]}"; do
    parsed="$(awk '
      /solver: result/ {
        tries=0; work=0; elapsed=0;
        for (i=1;i<=NF;i++) {
          if ($i ~ /^tries_used=/) { sub("tries_used=","",$i); tries=$i+0 }
          if ($i ~ /^work=/) { sub("work=","",$i); work=$i+0 }
          if ($i ~ /^elapsed_s=/) { sub("elapsed_s=","",$i); elapsed=$i+0 }
        }
        if (tries <= 0) tries=work;
        if (tries > 0 && elapsed > 0) khs=tries / elapsed / 1000.0;
      }
      /share OK/ { accepted++ }
      /share REJECTED/ { rejected++ }
      END { printf "%.3f %d %d\n", khs+0, accepted+0, rejected+0 }
    ' "$file" 2>/dev/null)"
    read -r file_khs file_accepted file_rejected <<< "$parsed"
    hs_values+=("${file_khs:-0}")
    accepted_total=$((accepted_total + ${file_accepted:-0}))
    rejected_total=$((rejected_total + ${file_rejected:-0}))
  done
fi

[ "${#hs_values[@]}" -eq 0 ] && hs_values=(0)
khs="$(awk 'BEGIN { s=0; for (i=1; i<ARGC; i++) s+=ARGV[i]; printf "%.3f", s }' "${hs_values[@]}")"
hs_json="$(json_number_array "${hs_values[@]}")"

if command -v jq >/dev/null 2>&1; then
  stats="$(jq -nc \
    --argjson hs "$hs_json" \
    --argjson total_khs "$khs" \
    --argjson ar "[$accepted_total,$rejected_total]" \
    --argjson bus "$bus_json" \
    --argjson temp "$temp_json" \
    --argjson fan "$fan_json" \
    --argjson uptime "$uptime" \
    --arg ver "$version" \
    --arg mode "$mode" \
    '{total_khs:$total_khs, hs:$hs, hs_units:"khs", algo:"btx", ver:$ver, uptime:$uptime, ar:$ar, bus_numbers:$bus, temp:$temp, fan:$fan, miner_mode:$mode}')"
else
  stats="{\"total_khs\":$khs,\"hs\":$hs_json,\"hs_units\":\"khs\",\"algo\":\"btx\",\"ver\":\"$version\",\"uptime\":$uptime,\"ar\":[$accepted_total,$rejected_total],\"bus_numbers\":$bus_json,\"temp\":$temp_json,\"fan\":$fan_json,\"miner_mode\":\"$mode\"}"
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "$stats"
fi
'@

if ($WindowsSolver) {
    $runOnePs1 = @'
param(
    [string]$Wallet = $env:BTX_WALLET,
    [string]$Pool = $(if ($env:BTX_POOL) { $env:BTX_POOL } else { "__BTX_READY_POOL__" }),
    [string]$Worker = $env:BTX_WORKER,
    [string]$Gpu = $env:CUDA_VISIBLE_DEVICES,
    [int]$Threads = $(if ($env:BTX_SOLVER_THREADS) { [int]$env:BTX_SOLVER_THREADS } else { 1 }),
    [int]$PrepareWorkers = $(if ($env:BTX_PREPARE_WORKERS) { [int]$env:BTX_PREPARE_WORKERS } else { 2 }),
    [int]$BatchSize = $(if ($env:BTX_BATCH_SIZE) { [int]$env:BTX_BATCH_SIZE } else { 512 }),
    [string]$CudaPoolSlots = $env:BTX_CUDA_POOL_SLOTS,
    [int]$Prefetch = $(if ($env:BTX_PREFETCH) { [int]$env:BTX_PREFETCH } else { 8 }),
    [int]$GpuInputs = $(if ($env:BTX_GPU_INPUTS) { [int]$env:BTX_GPU_INPUTS } else { 1 }),
    [string]$NoncesPerSlice = $(if ($env:BTX_NONCES_PER_SLICE) { $env:BTX_NONCES_PER_SLICE } else { "100000000000" }),
    [double]$MaxSecondsPerSlice = $(if ($env:BTX_SOLVER_MAX_SECONDS_PER_SLICE) { [double]$env:BTX_SOLVER_MAX_SECONDS_PER_SLICE } else { 30.0 }),
    [string]$LogLevel = $(if ($env:BTX_LOG_LEVEL) { $env:BTX_LOG_LEVEL } else { "INFO" })
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $root "miner.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (!$line -or $line.StartsWith("#") -or !$line.Contains("=")) { return }
        $parts = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
    }
    if (!$Wallet) { $Wallet = $env:BTX_WALLET }
    if (!$Worker) { $Worker = $env:BTX_WORKER }
    if ($env:BTX_POOL) { $Pool = $env:BTX_POOL }
}

if (!$Wallet) { throw "Set BTX_WALLET in miner.env or pass -Wallet." }
if (!$Worker) {
    $prefix = if ($env:BTX_WORKER_PREFIX) { $env:BTX_WORKER_PREFIX } else { $env:COMPUTERNAME.ToLowerInvariant() }
    $suffix = if ($Gpu) { $Gpu } else { "0" }
    $Worker = "$prefix-gpu$suffix"
}
if ($Gpu) { $env:CUDA_VISIBLE_DEVICES = $Gpu }
if ($CudaPoolSlots) { $env:BTX_MATMUL_CUDA_POOL_SLOTS = $CudaPoolSlots }
if (!$env:BTX_MATMUL_CUDA_DEVICE_PREPARED_INPUTS) { $env:BTX_MATMUL_CUDA_DEVICE_PREPARED_INPUTS = "1" }
if (!$env:BTX_MATMUL_CUDA_TILED_DIRECT_RHS) { $env:BTX_MATMUL_CUDA_TILED_DIRECT_RHS = "1" }
if (!$env:BTX_MATMUL_CUDA_WAIT_POLICY) { $env:BTX_MATMUL_CUDA_WAIT_POLICY = "blocking" }

$env:DEXBTX_NO_SOLVER_AUTOUPDATE = "1"
$env:DEXBTX_NO_WRAPPER_AUTOUPDATE = "1"
$env:DEXBTX_NO_SOLVER_RECHECK = "1"
$env:PYTHONPATH = "$root\python;$env:PYTHONPATH"

$python = Get-Command python -ErrorAction SilentlyContinue
$argsPrefix = @()
if (!$python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if (!$python) { throw "Python 3.10+ is required on PATH." }
    $argsPrefix = @("-3")
}

$solver = Join-Path $root "bin\btx-gbt-solve.exe"
& $python.Source @argsPrefix -m dexbtx_miner `
    --pool $Pool `
    --address $Wallet `
    --worker $Worker `
    --gbt-solve $solver `
    --threads $Threads `
    --prepare-workers $PrepareWorkers `
    --batch-size $BatchSize `
    --prefetch $Prefetch `
    --gpu-inputs $GpuInputs `
    --nonces-per-slice $NoncesPerSlice `
    --max-seconds-per-slice $MaxSecondsPerSlice `
    --log-level $LogLevel
exit $LASTEXITCODE
'@
    $runOnePs1 = $runOnePs1.Replace("__BTX_READY_POOL__", $ReadyPool)
    Write-Utf8NoBom (Join-Path $windowsDir "run-one.ps1") $runOnePs1

    Write-Utf8NoBom (Join-Path $windowsDir "run-solo.ps1") @'
param(
    [string]$Wallet = $env:BTX_WALLET
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $root "miner.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (!$line -or $line.StartsWith("#") -or !$line.Contains("=")) { return }
        $parts = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
    }
    if (!$Wallet) { $Wallet = $env:BTX_WALLET }
}
if (!$Wallet) { throw "Set BTX_WALLET in miner.env or pass -Wallet." }

$python = Get-Command python -ErrorAction SilentlyContinue
$argsPrefix = @()
if (!$python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if (!$python) { throw "Python 3.10+ is required on PATH." }
    $argsPrefix = @("-3")
}

$solver = Join-Path $root "bin\btx-gbt-solve.exe"
$statsFile = if ($env:BTX_FASTSOLO_STATS_FILE) { $env:BTX_FASTSOLO_STATS_FILE } else { Join-Path $root "logs\solo-stats.json" }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $statsFile) | Out-Null

$soloArgs = @(
    (Join-Path $root "fast_solo_miner.py"),
    "--address", $Wallet,
    "--solver", $solver,
    "--gpus", $(if ($env:BTX_FASTSOLO_GPUS) { $env:BTX_FASTSOLO_GPUS } elseif ($env:BTX_GPUS) { $env:BTX_GPUS } else { "auto" }),
    "--stats-file", $statsFile,
    "--batch-size", $(if ($env:BTX_FASTSOLO_BATCH_SIZE) { $env:BTX_FASTSOLO_BATCH_SIZE } elseif ($env:BTX_BATCH_SIZE) { $env:BTX_BATCH_SIZE } else { "512" }),
    "--solver-threads", $(if ($env:BTX_FASTSOLO_SOLVER_THREADS) { $env:BTX_FASTSOLO_SOLVER_THREADS } elseif ($env:BTX_SOLVER_THREADS) { $env:BTX_SOLVER_THREADS } else { "1" }),
    "--pool-slots", $(if ($env:BTX_FASTSOLO_POOL_SLOTS) { $env:BTX_FASTSOLO_POOL_SLOTS } else { "0" }),
    "--slice-seconds", $(if ($env:BTX_FASTSOLO_SLICE_SECONDS) { $env:BTX_FASTSOLO_SLICE_SECONDS } else { "30" }),
    "--max-tries", $(if ($env:BTX_FASTSOLO_MAX_TRIES) { $env:BTX_FASTSOLO_MAX_TRIES } else { "100000000" }),
    "--template-refresh-seconds", $(if ($env:BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS) { $env:BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS } else { "30" }),
    "--template-poll-seconds", $(if ($env:BTX_FASTSOLO_TEMPLATE_POLL_SECONDS) { $env:BTX_FASTSOLO_TEMPLATE_POLL_SECONDS } else { "0.5" })
)

if ($env:BTX_SOLO_DATADIR) { $soloArgs += @("--datadir", $env:BTX_SOLO_DATADIR) }
if ($env:BTX_SOLO_CONF) { $soloArgs += @("--conf", $env:BTX_SOLO_CONF) }
if ($env:BTX_SOLO_RPC_CONNECT) { $soloArgs += @("--rpcconnect", $env:BTX_SOLO_RPC_CONNECT) }
if ($env:BTX_SOLO_RPC_PORT) { $soloArgs += @("--rpcport", $env:BTX_SOLO_RPC_PORT) }
if ($env:BTX_SOLO_RPC_USER) { $soloArgs += @("--rpcuser", $env:BTX_SOLO_RPC_USER) }
if ($env:BTX_SOLO_RPC_PASSWORD) { $soloArgs += @("--rpcpassword", $env:BTX_SOLO_RPC_PASSWORD) }
if ($env:BTX_SOLO_RPC_COOKIE) { $soloArgs += @("--rpccookiefile", $env:BTX_SOLO_RPC_COOKIE) }

& $python.Source @argsPrefix @soloArgs
exit $LASTEXITCODE
'@

    $runPs1 = @'
param(
    [string]$Wallet = $env:BTX_WALLET,
    [string]$Pool = $(if ($env:BTX_POOL) { $env:BTX_POOL } else { "__BTX_READY_POOL__" }),
    [string]$WorkerPrefix = $env:BTX_WORKER_PREFIX,
    [string]$Mode = $env:BTX_MODE,
    [int]$WorkersPerGpu = $(if ($env:BTX_WORKERS_PER_GPU) { [int]$env:BTX_WORKERS_PER_GPU } else { 1 }),
    [double]$StartStaggerSeconds = $(if ($env:BTX_START_STAGGER_SECONDS) { [double]$env:BTX_START_STAGGER_SECONDS } else { 0 }),
    [switch]$SingleGpu
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $root "miner.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (!$line -or $line.StartsWith("#") -or !$line.Contains("=")) { return }
        $parts = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
    }
    if (!$Wallet) { $Wallet = $env:BTX_WALLET }
    if (!$WorkerPrefix) { $WorkerPrefix = $env:BTX_WORKER_PREFIX }
    if ($env:BTX_POOL) { $Pool = $env:BTX_POOL }
    if ($env:BTX_MODE) { $Mode = $env:BTX_MODE }
    if ($env:BTX_WORKERS_PER_GPU) { $WorkersPerGpu = [int]$env:BTX_WORKERS_PER_GPU }
    if ($env:BTX_START_STAGGER_SECONDS) { $StartStaggerSeconds = [double]$env:BTX_START_STAGGER_SECONDS }
}
if (!$Wallet) { throw "Set BTX_WALLET in miner.env or pass -Wallet." }
if (!$WorkerPrefix) { $WorkerPrefix = $env:COMPUTERNAME.ToLowerInvariant() }
if (!$Mode) { $Mode = "pool" }
if ($WorkersPerGpu -lt 1) { throw "BTX_WORKERS_PER_GPU must be at least 1." }

if ($Mode.ToLowerInvariant() -eq "solo") {
    & (Join-Path $root "run-solo.ps1") -Wallet $Wallet
    exit $LASTEXITCODE
}

if ($SingleGpu -or $env:BTX_SINGLE_GPU -eq "1") {
    & (Join-Path $root "run-one.ps1") -Wallet $Wallet -Pool $Pool -Worker "$WorkerPrefix-gpu0"
    exit $LASTEXITCODE
}

$gpuLines = @()
try {
    $gpuLines = (& nvidia-smi -L 2>$null) | Where-Object { $_ -match '^GPU ' }
} catch {
    $gpuLines = @()
}
if ($gpuLines.Count -eq 0) {
    & (Join-Path $root "run-one.ps1") -Wallet $Wallet -Pool $Pool -Worker "$WorkerPrefix-gpu0"
    exit $LASTEXITCODE
}

$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$procs = @()
for ($i = 0; $i -lt $gpuLines.Count; $i++) {
    for ($w = 0; $w -lt $WorkersPerGpu; $w++) {
        if ($WorkersPerGpu -eq 1) {
            $worker = "$WorkerPrefix-gpu$i"
            $logName = "gpu$i"
        } else {
            $worker = "${WorkerPrefix}-gpu${i}w$w"
            $logName = "gpu${i}w$w"
        }
        $out = Join-Path $logs "$logName.log"
        $err = Join-Path $logs "$logName.err.log"
        $args = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", "`"$(Join-Path $root 'run-one.ps1')`"",
            "-Wallet", "`"$Wallet`"",
            "-Pool", "`"$Pool`"",
            "-Worker", "`"$worker`"",
            "-Gpu", "`"$i`""
        )
        $procs += Start-Process -FilePath "powershell.exe" -ArgumentList $args -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -WindowStyle Hidden
        if ($StartStaggerSeconds -gt 0) { Start-Sleep -Milliseconds ([int]($StartStaggerSeconds * 1000)) }
    }
}
Write-Host "Started $($procs.Count) BTX miner process(es) across $($gpuLines.Count) GPU(s). Logs are in $logs."
try {
    Wait-Process -Id ($procs.Id)
} finally {
    foreach ($p in $procs) {
        if (!$p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
'@
    $runPs1 = $runPs1.Replace("__BTX_READY_POOL__", $ReadyPool)
    Write-Utf8NoBom (Join-Path $windowsDir "run.ps1") $runPs1

    Write-Utf8NoBom (Join-Path $windowsDir "run.bat") @'
@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
'@

    Write-Utf8NoBom (Join-Path $windowsDir "START_MINING.bat") @"
@echo off
setlocal
cd /d "%~dp0"

REM Ready-to-go defaults. Edit these two lines before running if you want a
REM different pool or payout wallet.
set "BTX_MODE=pool"
set "BTX_POOL=$ReadyPool"
set "BTX_WALLET=$ReadyWallet"
set "BTX_STRATUM_MIN_SHARE_DIFFICULTY=$StratumMinShareDifficulty"
set "BTX_START_STAGGER_SECONDS=$DefaultStartStaggerSeconds"
set "BTX_SOLVER_MAX_SECONDS_PER_SLICE=30"

REM Worker names appear on the pool as <prefix>-gpu0, <prefix>-gpu1, ...
set "BTX_WORKER_PREFIX=%COMPUTERNAME%"

REM To solo mine instead, set BTX_MODE=solo and configure BTX_SOLO_DATADIR /
REM BTX_SOLO_CONF or BTX_SOLO_RPC_* in miner.env.

REM Optional RTX 5090 safe desktop profile measured on CUDA 13 / SM120.
REM Uncomment these four lines on a 5090 if you want the tested local profile.
REM set "BTX_SOLVER_THREADS=2"
REM set "BTX_BATCH_SIZE=768"
REM set "BTX_CUDA_POOL_SLOTS=4"
REM set "BTX_WORKERS_PER_GPU=3"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
"@

    Write-Utf8NoBom (Join-Path $windowsDir "START_MINEBTX_5090_FOREGROUND.bat") @"
@echo off
setlocal
cd /d "%~dp0"

REM Ready-to-go local RTX 5090 desktop profile. This window stays in the
REM foreground so you can watch the miner output.
set "BTX_MODE=pool"
set "BTX_POOL=stratum+tcp://stratum.minebtx.com:3333"
set "BTX_WALLET=$ReadyWallet"
set "BTX_WORKER_PREFIX=local-5090"

set "BTX_SOLVER_THREADS=2"
set "BTX_PREPARE_WORKERS=2"
set "BTX_BATCH_SIZE=768"
set "BTX_CUDA_POOL_SLOTS=4"
set "BTX_PREFETCH=8"
set "BTX_GPU_INPUTS=1"
set "BTX_WORKERS_PER_GPU=3"
set "BTX_START_STAGGER_SECONDS=1"
set "BTX_NONCES_PER_SLICE=100000000000"
set "BTX_SOLVER_MAX_SECONDS_PER_SLICE=30"
set "BTX_LOG_LEVEL=INFO"

title BTX MineBTX RTX 5090
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
"@
}

$hiveArchiveLayout = "_hive_archive"
$hiveArchiveRoot = Join-Path $stageRoot $hiveArchiveLayout
if (Test-Path -LiteralPath $hiveArchiveRoot) {
    Remove-Item -LiteralPath $hiveArchiveRoot -Recurse -Force
}
function Copy-HiveArchiveTree([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($child in Get-ChildItem -LiteralPath $Source -Force) {
        $target = Join-Path $Destination $child.Name
        if ($child.PSIsContainer) {
            Copy-HiveArchiveTree $child.FullName $target
        } else {
            Copy-Item -LiteralPath $child.FullName -Destination $target -Force
        }
    }
}
New-Item -ItemType Directory -Force -Path $hiveArchiveRoot | Out-Null
$hiveArchiveMinerRoot = Join-Path $hiveArchiveRoot $hiveMinerName
New-Item -ItemType Directory -Force -Path $hiveArchiveMinerRoot | Out-Null
Copy-HiveArchiveTree $hiveDir $hiveArchiveMinerRoot

$wslStage = Get-WslPath $stageRoot
$wslDist = Get-WslPath $distRoot
Invoke-WslBash "cd '$wslStage' && find '$linuxName' '$hiveName' '$hiveArchiveLayout' -type f \( -name '*.sh' -o -name 'btx-gbt-solve' -o -name 'btx-matmul-*' \) -exec chmod +x {} +"

$linuxArchive = Join-Path $distRoot "$linuxName.tar.gz"
$hiveArchive = Join-Path $distRoot "$hiveArchiveName.tar.gz"
$windowsArchive = Join-Path $distRoot "$windowsName.zip"

Invoke-WslBash "cd '$wslStage' && tar -czf '$wslDist/$linuxName.tar.gz' '$linuxName'"
Invoke-WslBash "cd '$wslStage/$hiveArchiveLayout' && tar -czf '$wslDist/$hiveArchiveName.tar.gz' '$hiveMinerName'"
if ($WindowsSolver) {
    Get-ChildItem -LiteralPath $windowsDir -Recurse -Force | ForEach-Object {
        if ($_.LastWriteTime.Year -lt 1980) { $_.LastWriteTime = Get-Date }
    }
    Compress-Archive -Path (Join-Path $windowsDir "*") -DestinationPath $windowsArchive -Force
}

$assets = Get-ChildItem -LiteralPath $distRoot -File | Sort-Object Name
$hashLines = foreach ($asset in $assets) {
    "$(File-Sha256 $asset.FullName)  $($asset.Name)"
}
Write-Utf8NoBom (Join-Path $distRoot "SHA256SUMS.txt") (($hashLines -join "`n") + "`n")

$releaseNotes = @"
# BTX Miner $Version

No dev fee.

This release packages the optimized BTX btx-gbt-solve pool miner for:

- Linux x86_64 $LinuxCudaLabel $ArchLabel
- HiveOS x86_64 $LinuxCudaLabel $HiveArchLabel
- Windows x86_64 $WindowsCudaLabel $ArchLabel

Pool endpoint: $ReadyPool

CUDA target: Linux/HiveOS $LinuxCudaLabel / Windows $WindowsCudaLabel / $ArchDisplay

HiveOS custom miner name: $hiveMinerName

HiveOS package root: $hiveMinerName

HiveOS asset: $hiveArchiveName.tar.gz

## Pool and Solo Support

- Default mode is BTX_MODE=pool against $ReadyPool.
- Classic Stratum pools that start at very low vardiff can use
  BTX_STRATUM_MIN_SHARE_DIFFICULTY. This build defaults it to
  "$StratumMinShareDifficulty" for the selected ready pool.
- BitMinerPool builds default BTX_START_STAGGER_SECONDS to
  "$DefaultStartStaggerSeconds" and wrapper reconnects use wider jitter so
  multi-rig restarts do not hammer the pool at once.
- Set BTX_MODE=solo to use the bundled fast async getblocktemplate solo miner
  against a synced BTX node RPC.
- Pool and solo mode use the same optimized btx-gbt-solve binary and the same
  BTX_WALLET payout setting.
- Pool launchers now default BTX_MATMUL_CUDA_WAIT_POLICY=blocking plus the
  device-prepared/tiled-direct CUDA fast paths. On the live 6x 4070 Ti SUPER
  MineBTX fleet this removed host CPU saturation and restored rig08 from about
  1.37 GN/s to about 1.91 GN/s.
- HiveOS stats support both modes: pool worker logs and solo JSON stats are
  reported as K N/s.
- HiveOS flight-sheet wallet/pool/worker changes are applied on miner start,
  even when a stale miner.env exists from a previous sheet.
- HiveOS packages use a Hive-compatible asset name and top-level btx-miner
  directory so custom-get installs h-manifest.conf, h-config.sh, h-run.sh, and
  h-stats.sh where Hive expects them.
- If HiveOS invokes run.sh directly, the launcher regenerates miner.env from
  the active flight sheet before starting.
- HiveOS extra config accepts BTX_...=... and DEXBTX_...=... assignments.

## Ready-to-Go Windows Batch File

The Windows package includes START_MINING.bat. It defaults to:

- BTX_POOL=$ReadyPool
- BTX_WALLET=$ReadyWallet

Edit those lines in START_MINING.bat before running if you want a different
pool or payout wallet.

## GPU Coverage

- NVIDIA 30-series Ampere support via SM86 device code.
- NVIDIA 40-series Ada support via SM89 device code, including the optimized
  4070 Ti SUPER work and default 40-series launcher profile.
- NVIDIA 50-series Blackwell support via SM120 device code, plus the documented
  RTX 5090 safe desktop profile.

The launchers start one miner process per visible NVIDIA GPU by default and
name workers as <BTX_WORKER_PREFIX>-gpuN. Set BTX_WORKERS_PER_GPU to run
multiple processes per GPU; those workers are named
<BTX_WORKER_PREFIX>-gpuNwM.

Local RTX 5090 safe desktop profile:

- BTX_SOLVER_THREADS=2
- BTX_BATCH_SIZE=768
- BTX_CUDA_POOL_SLOTS=4
- BTX_WORKERS_PER_GPU=3

## Validation

- Linux solver SHA256: $linuxSha
- Windows solver SHA256: $windowsSha
- BTX source commit: $sourceCommit
- BTX source dirty at package time: $dirtyNote

## Gains vs Previous Public/Reference Miner

- Measured on rig08 GPU5 against the MineBTX reference solver: 8.818M N/s vs
  3.739M N/s, about 2.36x reference throughput.
- The 40-series work from the prior BTX package remains included: SM89/Ada
  device code, the 4070 Ti SUPER launcher defaults, and the no-dev-fee
  packaging.
- The latest canaries after the previous package did not produce a deployable
  speed gain over the current known-good solver, so this release does not claim
  an additional hashrate uplift beyond the measured reference gain above.

## No Dev Fee

- No developer-fee wallet is included.
- No payout split is configured.
- START_MINING.bat includes the operator wallet requested for this build as a
  user-editable default.
- Auto-updaters are disabled in the launchers so the bundled optimized solver
  is not replaced during runtime.

## SHA256SUMS

$($hashLines -join "`n")
"@
Write-Utf8NoBom (Join-Path $releaseRoot "RELEASE_NOTES.md") $releaseNotes

Write-Host "Built $releaseName"
Write-Host "Artifacts: $distRoot"
