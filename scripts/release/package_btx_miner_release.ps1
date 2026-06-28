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
    [string]$ArchDisplay = "SM89/Ada",
    [string]$ReadyPool = "stratum.minebtx.com:3333",
    [string]$ReadyWallet = "btx1zwxtwvgt55h5smfxp7swxacp2qhavz9kpzt0fjvw8303w7kkl7pusgy9e73",
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
Assert-File $LinuxSolver
if (!$AllowMissingWindows) {
    if (!$WindowsSolver) { throw "Windows solver was not found." }
    Assert-File $WindowsSolver
}

$releaseName = "btx-miner-$Version"
$releaseRoot = Join-Path $RepoRoot "release-artifacts\$releaseName"
$stageRoot = Join-Path $releaseRoot "stage"
$distRoot = Join-Path $releaseRoot "dist"

if (Test-Path -LiteralPath $stageRoot) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
if (Test-Path -LiteralPath $distRoot) {
    Remove-Item -LiteralPath $distRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageRoot,$distRoot | Out-Null

$linuxName = "$releaseName-linux-x86_64-$LinuxCudaLabel-$ArchLabel"
$hiveName = "$releaseName-hiveos-x86_64-$LinuxCudaLabel-$ArchLabel"
$windowsName = "$releaseName-windows-x86_64-$WindowsCudaLabel-$ArchLabel"
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
    Get-ChildItem -LiteralPath (Join-Path $Dest "python\dexbtx_miner") -Directory -Filter "__pycache__" -Recurse -Force |
        Remove-Item -Recurse -Force
    Copy-IfExists $pyproject (Join-Path $Dest "pyproject.toml")
    Copy-IfExists $license (Join-Path $Dest "LICENSE.dxbtx-wrapper")

    Write-Utf8NoBom (Join-Path $Dest "requirements.txt") @'
pyyaml>=6.0
'@

    Write-Utf8NoBom (Join-Path $Dest "miner.env.example") @"
# Copy to miner.env, set BTX_WALLET, then run ./run.sh or run.bat.
BTX_WALLET=
BTX_POOL=stratum.minebtx.com:3333
BTX_WORKER_PREFIX=my-rig

# Tuned defaults for the optimized SM89/Ada solver used on 4070 Ti SUPER.
BTX_SOLVER_THREADS=1
BTX_PREPARE_WORKERS=2
BTX_BATCH_SIZE=512
BTX_CUDA_POOL_SLOTS=
BTX_PREFETCH=8
BTX_GPU_INPUTS=1
BTX_WORKERS_PER_GPU=1
BTX_NONCES_PER_SLICE=100000000000
BTX_LOG_LEVEL=INFO

# Local RTX 5090 safe desktop profile measured on CUDA 13 / SM120:
#   BTX_SOLVER_THREADS=2
#   BTX_BATCH_SIZE=768
#   BTX_CUDA_POOL_SLOTS=4
#   BTX_WORKERS_PER_GPU=3
"@

    Write-Utf8NoBom (Join-Path $Dest "config.example.yaml") @"
pool_host: "stratum.minebtx.com"
pool_port: 3333
pool_tls: false

# Required. Set this to your payout address. The Windows START_MINING.bat
# convenience file defaults to the operator wallet requested for this build and
# can be edited before running.
payout_address: "btx1z...YOUR_BTX_ADDRESS_HERE..."
worker_name: "my-rig-gpu0"

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

stratum+tcp://stratum.minebtx.com:3333

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
Add-CommonFiles $hiveDir "HiveOS x86_64 $LinuxCudaLabel $ArchLabel" $LinuxCudaLabel
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

BTX_POOL="${BTX_POOL:-stratum.minebtx.com:3333}"
BTX_WALLET="${BTX_WALLET:-}"
BTX_WORKER="${BTX_WORKER:-${BTX_WORKER_PREFIX:-$(hostname)}-gpu${CUDA_VISIBLE_DEVICES:-0}}"
BTX_SOLVER_THREADS="${BTX_SOLVER_THREADS:-1}"
BTX_PREPARE_WORKERS="${BTX_PREPARE_WORKERS:-2}"
BTX_BATCH_SIZE="${BTX_BATCH_SIZE:-512}"
BTX_CUDA_POOL_SLOTS="${BTX_CUDA_POOL_SLOTS:-}"
BTX_PREFETCH="${BTX_PREFETCH:-8}"
BTX_GPU_INPUTS="${BTX_GPU_INPUTS:-1}"
BTX_NONCES_PER_SLICE="${BTX_NONCES_PER_SLICE:-100000000000}"
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
  --log-level "$BTX_LOG_LEVEL"
'@

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

$runSh = @'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${BTX_SINGLE_GPU:-0}" = "1" ]; then
  exec "$DIR/run-one.sh" "$@"
fi
exec "$DIR/run-all-gpus.sh" "$@"
'@

foreach ($dir in @($linuxDir, $hiveDir)) {
    Write-Utf8NoBom (Join-Path $dir "run-one.sh") $runOneSh
    Write-Utf8NoBom (Join-Path $dir "run-all-gpus.sh") $runAllSh
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

Write-Utf8NoBom (Join-Path $hiveDir "h-config.sh") @'
#!/usr/bin/env bash
set -euo pipefail
cd /hive/miners/custom/btx-miner 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")"
. ./h-manifest.conf

wallet="${BTX_WALLET:-${CUSTOM_TEMPLATE:-${CUSTOM_WALLET:-}}}"
pool="${BTX_POOL:-${CUSTOM_URL:-stratum.minebtx.com:3333}}"
worker="${BTX_WORKER_PREFIX:-${WORKER_NAME:-$(hostname)}}"

cat > "$CUSTOM_CONFIG_FILENAME" <<EOF
BTX_WALLET=$wallet
BTX_POOL=$pool
BTX_WORKER_PREFIX=$worker
BTX_SOLVER_THREADS=${BTX_SOLVER_THREADS:-1}
BTX_PREPARE_WORKERS=${BTX_PREPARE_WORKERS:-2}
BTX_BATCH_SIZE=${BTX_BATCH_SIZE:-512}
BTX_CUDA_POOL_SLOTS=${BTX_CUDA_POOL_SLOTS:-}
BTX_PREFETCH=${BTX_PREFETCH:-8}
BTX_GPU_INPUTS=${BTX_GPU_INPUTS:-1}
BTX_WORKERS_PER_GPU=${BTX_WORKERS_PER_GPU:-1}
BTX_NONCES_PER_SLICE=${BTX_NONCES_PER_SLICE:-100000000000}
BTX_LOG_LEVEL=${BTX_LOG_LEVEL:-INFO}
EOF

echo "BTX miner config written to $CUSTOM_CONFIG_FILENAME"
'@

Write-Utf8NoBom (Join-Path $hiveDir "h-run.sh") @'
#!/usr/bin/env bash
set -euo pipefail
cd /hive/miners/custom/btx-miner 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")"
. ./h-manifest.conf

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
if [ ! -f "$CUSTOM_CONFIG_FILENAME" ]; then
  ./h-config.sh
fi

if [ -n "${CUSTOM_TEMPLATE:-}" ]; then
  export BTX_WALLET="${BTX_WALLET:-$CUSTOM_TEMPLATE}"
fi
if [ -n "${CUSTOM_URL:-}" ]; then
  export BTX_POOL="${BTX_POOL:-$CUSTOM_URL}"
fi
export BTX_WORKER_PREFIX="${BTX_WORKER_PREFIX:-${WORKER_NAME:-$(hostname)}}"

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
. /hive/miners/custom/btx-miner/h-manifest.conf 2>/dev/null || true
log_dir="/hive/miners/custom/btx-miner/logs"
nps="$(awk '
  /solver: result/ {
    work=0; elapsed=0;
    for (i=1;i<=NF;i++) {
      if ($i ~ /^work=/) { sub("work=","",$i); work=$i+0 }
      if ($i ~ /^elapsed_s=/) { sub("elapsed_s=","",$i); elapsed=$i+0 }
    }
    if (work > 0 && elapsed > 0) print work / elapsed;
  }
' "$log_dir"/gpu*.log 2>/dev/null | tail -n 1)"
[ -z "$nps" ] && nps=0
khs="$(awk -v n="$nps" 'BEGIN { printf "%.3f", n / 1000.0 }')"
stats="$(jq -nc --argjson khs "$khs" '{hs:[$khs], hs_units:"khs", algo:"btx-matmul", ar:[0,0], temp:[], fan:[], uptime:0}' 2>/dev/null || echo '{}')"
echo "$stats"
'@

if ($WindowsSolver) {
    Write-Utf8NoBom (Join-Path $windowsDir "run-one.ps1") @'
param(
    [string]$Wallet = $env:BTX_WALLET,
    [string]$Pool = $(if ($env:BTX_POOL) { $env:BTX_POOL } else { "stratum.minebtx.com:3333" }),
    [string]$Worker = $env:BTX_WORKER,
    [string]$Gpu = $env:CUDA_VISIBLE_DEVICES,
    [int]$Threads = $(if ($env:BTX_SOLVER_THREADS) { [int]$env:BTX_SOLVER_THREADS } else { 1 }),
    [int]$PrepareWorkers = $(if ($env:BTX_PREPARE_WORKERS) { [int]$env:BTX_PREPARE_WORKERS } else { 2 }),
    [int]$BatchSize = $(if ($env:BTX_BATCH_SIZE) { [int]$env:BTX_BATCH_SIZE } else { 512 }),
    [string]$CudaPoolSlots = $env:BTX_CUDA_POOL_SLOTS,
    [int]$Prefetch = $(if ($env:BTX_PREFETCH) { [int]$env:BTX_PREFETCH } else { 8 }),
    [int]$GpuInputs = $(if ($env:BTX_GPU_INPUTS) { [int]$env:BTX_GPU_INPUTS } else { 1 }),
    [string]$NoncesPerSlice = $(if ($env:BTX_NONCES_PER_SLICE) { $env:BTX_NONCES_PER_SLICE } else { "100000000000" }),
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
    --log-level $LogLevel
exit $LASTEXITCODE
'@

    Write-Utf8NoBom (Join-Path $windowsDir "run.ps1") @'
param(
    [string]$Wallet = $env:BTX_WALLET,
    [string]$Pool = $(if ($env:BTX_POOL) { $env:BTX_POOL } else { "stratum.minebtx.com:3333" }),
    [string]$WorkerPrefix = $env:BTX_WORKER_PREFIX,
    [int]$WorkersPerGpu = $(if ($env:BTX_WORKERS_PER_GPU) { [int]$env:BTX_WORKERS_PER_GPU } else { 1 }),
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
    if ($env:BTX_WORKERS_PER_GPU) { $WorkersPerGpu = [int]$env:BTX_WORKERS_PER_GPU }
}
if (!$Wallet) { throw "Set BTX_WALLET in miner.env or pass -Wallet." }
if (!$WorkerPrefix) { $WorkerPrefix = $env:COMPUTERNAME.ToLowerInvariant() }
if ($WorkersPerGpu -lt 1) { throw "BTX_WORKERS_PER_GPU must be at least 1." }

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
set "BTX_POOL=$ReadyPool"
set "BTX_WALLET=$ReadyWallet"

REM Worker names appear on the pool as <prefix>-gpu0, <prefix>-gpu1, ...
set "BTX_WORKER_PREFIX=%COMPUTERNAME%"

REM Optional RTX 5090 safe desktop profile measured on CUDA 13 / SM120.
REM Uncomment these four lines on a 5090 if you want the tested local profile.
REM set "BTX_SOLVER_THREADS=2"
REM set "BTX_BATCH_SIZE=768"
REM set "BTX_CUDA_POOL_SLOTS=4"
REM set "BTX_WORKERS_PER_GPU=3"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
"@
}

$wslStage = Get-WslPath $stageRoot
$wslDist = Get-WslPath $distRoot
Invoke-WslBash "cd '$wslStage' && find '$linuxName' '$hiveName' -type f \( -name '*.sh' -o -name 'btx-gbt-solve' -o -name 'btx-matmul-*' \) -exec chmod +x {} +"

$linuxArchive = Join-Path $distRoot "$linuxName.tar.gz"
$hiveArchive = Join-Path $distRoot "$hiveName.tar.gz"
$windowsArchive = Join-Path $distRoot "$windowsName.zip"

Invoke-WslBash "cd '$wslStage' && tar -czf '$wslDist/$linuxName.tar.gz' '$linuxName'"
Invoke-WslBash "cd '$wslStage' && tar -czf '$wslDist/$hiveName.tar.gz' '$hiveName'"
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
- HiveOS x86_64 $LinuxCudaLabel $ArchLabel
- Windows x86_64 $WindowsCudaLabel $ArchLabel

Pool endpoint: stratum+tcp://stratum.minebtx.com:3333

CUDA target: Linux/HiveOS $LinuxCudaLabel / Windows $WindowsCudaLabel / $ArchDisplay

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
- Measured on rig08 GPU5 against the MineBTX reference solver: 8.818M N/s vs
  3.739M N/s, about 2.36x reference throughput.

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
