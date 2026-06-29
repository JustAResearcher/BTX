#!/usr/bin/env python3
import argparse
import asyncio
import base64
import contextlib
import hashlib
import http.client
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path


BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BECH32M_CONST = 0x2BC830A3
PAYOUT_ADDRESS = "btx1zwxtwvgt55h5smfxp7swxacp2qhavz9kpzt0fjvw8303w7kkl7pusgy9e73"


def packaged_default_solver():
    exe = "btx-gbt-solve.exe" if os.name == "nt" else "btx-gbt-solve"
    script_dir = Path(__file__).resolve().parent
    for root in (script_dir, script_dir.parent, Path.cwd()):
        candidate = root / "bin" / exe
        if candidate.is_file():
            return str(candidate)
    return exe


def parse_bool(value):
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def sha256d(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def compact_size(n):
    if n < 0xFD:
        return bytes([n])
    if n <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", n)
    if n <= 0xFFFFFFFF:
        return b"\xfe" + struct.pack("<I", n)
    return b"\xff" + struct.pack("<Q", n)


def var_bytes(data):
    return compact_size(len(data)) + data


def script_num(n):
    if n == 0:
        return b""
    out = bytearray()
    value = n
    while value:
        out.append(value & 0xFF)
        value >>= 8
    if out[-1] & 0x80:
        out.append(0)
    return bytes(out)


def push_data(data):
    if len(data) < 0x4C:
        return bytes([len(data)]) + data
    if len(data) <= 0xFF:
        return b"\x4c" + bytes([len(data)]) + data
    if len(data) <= 0xFFFF:
        return b"\x4d" + struct.pack("<H", len(data)) + data
    return b"\x4e" + struct.pack("<I", len(data)) + data


def bech32_polymod(values):
    generators = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            if (top >> i) & 1:
                chk ^= generators[i]
    return chk


def bech32_hrp_expand(hrp):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def bech32_decode(addr):
    if addr.lower() != addr and addr.upper() != addr:
        raise ValueError("mixed-case bech32 address")
    addr = addr.lower()
    pos = addr.rfind("1")
    if pos < 1 or pos + 7 > len(addr):
        raise ValueError("invalid bech32 separator")
    hrp = addr[:pos]
    data = [BECH32_CHARSET.find(c) for c in addr[pos + 1 :]]
    if any(v < 0 for v in data):
        raise ValueError("invalid bech32 character")
    if bech32_polymod(bech32_hrp_expand(hrp) + data) != BECH32M_CONST:
        raise ValueError("invalid bech32m checksum")
    return hrp, data[:-6]


def convert_bits(data, from_bits, to_bits, pad):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << to_bits) - 1
    max_acc = (1 << (from_bits + to_bits - 1)) - 1
    for value in data:
        if value < 0 or value >> from_bits:
            raise ValueError("invalid base conversion value")
        acc = ((acc << from_bits) | value) & max_acc
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (to_bits - bits)) & maxv)
    elif bits >= from_bits or ((acc << (to_bits - bits)) & maxv):
        raise ValueError("invalid padding in base conversion")
    return bytes(ret)


def address_to_script_pubkey(addr):
    hrp, data = bech32_decode(addr)
    if hrp != "btx":
        raise ValueError(f"unexpected address hrp {hrp!r}")
    if not data:
        raise ValueError("empty witness address")
    version = data[0]
    program = convert_bits(data[1:], 5, 8, False)
    if version > 16 or len(program) not in (20, 32):
        raise ValueError("unsupported witness program")
    op = 0 if version == 0 else 0x50 + version
    return bytes([op, len(program)]) + program


def serialize_tx(version, vin, vout, locktime=0, witness=None):
    out = bytearray()
    out += struct.pack("<i", version)
    if witness is not None:
        out += b"\x00\x01"
    out += compact_size(len(vin))
    for txin in vin:
        out += txin["prev_hash"]
        out += struct.pack("<I", txin["prev_index"])
        out += var_bytes(txin["script_sig"])
        out += struct.pack("<I", txin["sequence"])
    out += compact_size(len(vout))
    for txout in vout:
        out += struct.pack("<q", txout["value"])
        out += var_bytes(txout["script_pubkey"])
    if witness is not None:
        for stack in witness:
            out += compact_size(len(stack))
            for item in stack:
                out += var_bytes(item)
    out += struct.pack("<I", locktime)
    return bytes(out)


def build_coinbase(height, value, payout_script, witness_commitment_hex):
    script_sig = push_data(script_num(height)) + b"\x00"
    vin = [
        {
            "prev_hash": b"\x00" * 32,
            "prev_index": 0xFFFFFFFF,
            "script_sig": script_sig,
            "sequence": 0xFFFFFFFF,
        }
    ]
    vout = [{"value": int(value), "script_pubkey": payout_script}]
    witness = None
    if witness_commitment_hex:
        vout.append({"value": 0, "script_pubkey": bytes.fromhex(witness_commitment_hex)})
        witness = [[b"\x00" * 32]]
    tx_no_witness = serialize_tx(2, vin, vout, witness=None)
    tx_full = serialize_tx(2, vin, vout, witness=witness)
    return tx_full, sha256d(tx_no_witness)


def merkle_root_internal(tx_hashes):
    if not tx_hashes:
        return b"\x00" * 32
    layer = list(tx_hashes)
    while len(layer) > 1:
        if len(layer) & 1:
            layer.append(layer[-1])
        layer = [sha256d(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


def u256_from_display(hex_value):
    return bytes.fromhex(hex_value)[::-1]


def u256_to_display(internal):
    return internal[::-1].hex()


def serialize_header(template, merkle_display, nonce64, digest_display, dim, seed_a_display, seed_b_display):
    return b"".join(
        [
            struct.pack("<i", int(template["version"])),
            u256_from_display(template["previousblockhash"]),
            u256_from_display(merkle_display),
            struct.pack("<I", int(template["curtime"])),
            struct.pack("<I", int(template["bits"], 16)),
            struct.pack("<Q", int(nonce64)),
            u256_from_display(digest_display),
            struct.pack("<H", int(dim)),
            u256_from_display(seed_a_display),
            u256_from_display(seed_b_display),
        ]
    )


def serialize_u32_vector_from_hex(hex_data):
    raw = bytes.fromhex(hex_data)
    if len(raw) % 4:
        raise ValueError("matrix_c_data_hex length is not a uint32 vector")
    return compact_size(len(raw) // 4) + raw


def build_block_hex(template, coinbase_tx, extra_txs, merkle_display, nonce64, digest_display, seed_a, seed_b, matrix_c_hex=None):
    dim = template.get("matmul_n") or template.get("matmul", {}).get("n") or 512
    header = serialize_header(template, merkle_display, nonce64, digest_display, dim, seed_a, seed_b)
    out = bytearray(header)
    out += compact_size(1 + len(extra_txs))
    out += coinbase_tx
    for tx in extra_txs:
        out += bytes.fromhex(tx["data"])
    out += b"\x00"  # matrix_a_data: seed-derived
    out += b"\x00"  # matrix_b_data: seed-derived
    if matrix_c_hex:
        out += serialize_u32_vector_from_hex(matrix_c_hex)
    return out.hex()


class RpcClient:
    def __init__(
        self,
        datadir,
        conf_path=None,
        rpcconnect=None,
        rpcport=None,
        rpcuser=None,
        rpcpassword=None,
        rpccookiefile=None,
    ):
        self.datadir = Path(datadir).expanduser() if datadir else None
        self.conf_path = (
            Path(conf_path).expanduser()
            if conf_path
            else (self.datadir / "btx.conf" if self.datadir else None)
        )
        self.host = rpcconnect or "127.0.0.1"
        self.port = int(rpcport) if rpcport else 19334
        self.user = rpcuser
        self.password = rpcpassword
        self.cookie_path = Path(rpccookiefile).expanduser() if rpccookiefile else None
        self._load_config()

    def _load_config(self):
        if self.conf_path and self.conf_path.exists():
            for line in self.conf_path.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key in ("rpcconnect", "rpcbind") and self.host == "127.0.0.1":
                    self.host = value
                elif key == "rpcport" and self.port == 19334:
                    self.port = int(value)
                elif key == "rpcuser" and not self.user:
                    self.user = value
                elif key == "rpcpassword" and not self.password:
                    self.password = value
        if not self.user or not self.password:
            cookie = self.cookie_path or (self.datadir / ".cookie" if self.datadir else None)
            if cookie is None or not cookie.exists():
                raise RuntimeError("missing BTX RPC credentials: set rpcuser/rpcpassword, rpccookiefile, or datadir with .cookie")
            userpass = cookie.read_text().strip()
            self.user, self.password = userpass.split(":", 1)

    def call(self, method, params=None, timeout=30):
        body = json.dumps({"jsonrpc": "1.0", "id": "fast-solo", "method": method, "params": params or []})
        token = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
            "Connection": "close",
        }
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request("POST", "/", body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
        finally:
            conn.close()
        if resp.status != 200:
            raise RuntimeError(f"RPC HTTP {resp.status}: {data[:300]!r}")
        payload = json.loads(data)
        if payload.get("error"):
            raise RuntimeError(f"RPC {method} error: {payload['error']}")
        return payload.get("result")


def build_template_work(rpc, address):
    template = rpc.call("getblocktemplate", [{"rules": ["segwit"], "mode": "template"}])
    prev_header = rpc.call("getblockheader", [template["previousblockhash"]])
    payout_script = address_to_script_pubkey(address)
    coinbase, coinbase_txid_internal = build_coinbase(
        int(template["height"]),
        int(template["coinbasevalue"]),
        payout_script,
        template.get("default_witness_commitment"),
    )
    tx_hashes = [coinbase_txid_internal]
    for tx in template.get("transactions", []):
        tx_hashes.append(u256_from_display(tx["txid"]))
    merkle_internal = merkle_root_internal(tx_hashes)
    merkle_display = u256_to_display(merkle_internal)
    return {
        "template": template,
        "parent_mtp": int(prev_header["mediantime"]),
        "coinbase": coinbase,
        "txs": template.get("transactions", []),
        "merkle": merkle_display,
    }


def work_prev_hash(work):
    return work["template"]["previousblockhash"]


class SolverWorker:
    def __init__(self, gpu, solver_path, args, lane_start, worker_index=0):
        self.gpu = int(gpu)
        self.solver_path = solver_path
        self.args = args
        self.lane_start = lane_start
        self.nonce_next = lane_start
        self.worker_index = int(worker_index)
        self.proc = None
        self.stderr_task = None
        self.preempt_capable = False
        self.inflight_parent = None
        self.preempt_count = 0
        self.stdin_lock = asyncio.Lock()

    @property
    def key(self):
        return f"{self.gpu}:{self.worker_index}"

    async def start(self):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        env.setdefault("BTX_MATMUL_BACKEND", "cuda")
        env.setdefault("BTX_MATMUL_GPU_INPUTS", "1")
        env.setdefault("BTX_MATMUL_PIPELINE_ASYNC", "1")
        env.setdefault("BTX_MATMUL_PREPARE_PREFETCH_DEPTH", "8")
        env.setdefault("BTX_MATMUL_PREPARE_WORKERS", "2")
        env.setdefault("BTX_MATMUL_SOLVER_THREADS", str(self.args.solver_threads))
        env.setdefault("BTX_MATMUL_SOLVE_BATCH_SIZE", str(self.args.batch_size))
        env.setdefault("BTX_MATMUL_NONCE_SEED_LOOKAHEAD", "1")
        env.setdefault("BTX_MATMUL_CUDA_DEVICE_PREPARED_INPUTS", "1")
        env.setdefault("BTX_MATMUL_CUDA_WAIT_POLICY", "blocking")
        env.setdefault("BTX_MINER_HEADER_TIME_REFRESH_ATTEMPTS", "4294967295")
        lib_dir = Path(self.solver_path).resolve().parent.parent / "lib"
        if os.name == "nt":
            env["PATH"] = f"{lib_dir};{env.get('PATH', '')}"
        else:
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        cmd = [
            self.solver_path,
            "--daemon",
            "--backend",
            "cuda",
            "--solver-threads",
            str(self.args.solver_threads),
            "--batch-size",
            str(self.args.batch_size),
            "--matmul-n",
            "512",
            "--matmul-b",
            "16",
            "--matmul-r",
            "8",
            "--epsilon-bits",
            str(self.args.epsilon_bits),
        ]
        if int(self.args.pool_slots) > 0:
            cmd += ["--pool-slots", str(int(self.args.pool_slots))]
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=16 * 1024 * 1024,
        )
        ready_text = await self._read_ready_line()
        try:
            ready_payload = json.loads(ready_text)
            self.preempt_capable = bool(ready_payload.get("preempt")) and bool(self.args.preempt) and hasattr(signal, "SIGUSR1")
        except (json.JSONDecodeError, TypeError):
            self.preempt_capable = False
        if self.proc.stderr is not None:
            self.stderr_task = asyncio.create_task(self._drain_stderr())

    async def _read_ready_line(self):
        if self.proc is None or self.proc.stderr is None:
            return "{}"
        deadline = time.time() + 30.0
        seen = []
        while time.time() < deadline:
            remaining = max(deadline - time.time(), 0.1)
            try:
                raw = await asyncio.wait_for(self.proc.stderr.readline(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise RuntimeError(f"solver gpu {self.gpu} did not signal daemon_ready; stderr={seen[-3:]}") from exc
            if not raw:
                raise RuntimeError(f"solver gpu {self.gpu} exited before daemon_ready; stderr={seen[-3:]}")
            text = raw.decode(errors="replace").strip()
            if "daemon_ready" in text:
                return text
            if text:
                seen.append(text)
        raise RuntimeError(f"solver gpu {self.gpu} did not signal daemon_ready; stderr={seen[-3:]}")

    async def _drain_stderr(self):
        if self.proc is None or self.proc.stderr is None:
            return
        while True:
            raw = await self.proc.stderr.readline()
            if not raw:
                return
            text = raw.decode(errors="replace").strip()
            if text:
                print(f"solver gpu {self.gpu} stderr: {text}", file=sys.stderr, flush=True)

    async def stop(self):
        if not self.proc:
            return
        if self.stderr_task:
            self.stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.stderr_task
            self.stderr_task = None
        if self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        self.proc = None
        self.preempt_capable = False

    def preempt(self, new_parent):
        if not self.preempt_capable or not self.proc or self.proc.returncode is not None:
            return False
        if self.inflight_parent is None or self.inflight_parent == new_parent:
            return False
        try:
            self.proc.send_signal(signal.SIGUSR1)
            self.preempt_count += 1
            return True
        except (ProcessLookupError, ValueError):
            return False

    async def solve_slice(self, work):
        if self.proc is None or self.proc.returncode is not None:
            await self.stop()
            await self.start()
        nonce_start = self.nonce_next
        job = {
            "version": int(work["template"]["version"]),
            "prev_hash": work["template"]["previousblockhash"],
            "merkle_root": work["merkle"],
            "time": int(work["template"]["curtime"]),
            "bits": work["template"]["bits"],
            "seed_a": work["template"]["seed_a"],
            "seed_b": work["template"]["seed_b"],
            "block_height": int(work["template"]["height"]),
            "parent_mtp": int(work["parent_mtp"]),
            "nonce_start": nonce_start,
            "matmul_n": 512,
            "matmul_b": 16,
            "matmul_r": 8,
            "epsilon_bits": int(self.args.epsilon_bits),
            "max_tries": int(self.args.max_tries),
            "max_seconds": float(self.args.slice_seconds),
            "gpu": self.gpu,
        }
        line = json.dumps(job, separators=(",", ":")) + "\n"
        started = time.time()
        self.inflight_parent = work_prev_hash(work)
        try:
            async with self.stdin_lock:
                self.proc.stdin.write(line.encode())
                await self.proc.stdin.drain()
                raw = await self.proc.stdout.readline()
        finally:
            self.inflight_parent = None
        elapsed = max(time.time() - started, 0.001)
        if not raw:
            raise RuntimeError(f"solver gpu {self.gpu} exited before returning a result")
        result = json.loads(raw)
        if "error" in result:
            raise RuntimeError(f"solver gpu {self.gpu} job error: {result['error']}")
        end = int(result.get("nonce64_end", nonce_start))
        self.nonce_next = end + 1
        tried = max(self.nonce_next - nonce_start, 0)
        result["local_gpu"] = self.gpu
        result["local_worker"] = self.worker_index
        result["local_nonce_start"] = nonce_start
        result["local_nonce_count"] = tried
        result["local_elapsed_s"] = elapsed
        result["local_nps"] = tried / elapsed
        return result


class WorkManager:
    def __init__(self, rpc, address, args):
        self.rpc = rpc
        self.address = address
        self.args = args
        self.work = None
        self.revision = 0
        self.ready = asyncio.Event()
        self.lock = asyncio.Lock()
        self.workers = []
        self.last_template_at = 0.0

    def set_workers(self, workers):
        self.workers = list(workers)

    async def get_work(self):
        await self.ready.wait()
        async with self.lock:
            return self.work, self.revision

    async def current_prev_hash(self):
        async with self.lock:
            if self.work is None:
                return None
            return work_prev_hash(self.work)

    async def refresh(self, reason):
        work = await asyncio.to_thread(build_template_work, self.rpc, self.address)
        new_parent = work_prev_hash(work)
        async with self.lock:
            old_parent = work_prev_hash(self.work) if self.work else None
            changed_parent = old_parent is not None and old_parent != new_parent
            self.work = work
            self.revision += 1
            revision = self.revision
            self.last_template_at = time.time()
            self.ready.set()
        if changed_parent:
            preempted = sum(1 for worker in self.workers if worker.preempt(new_parent))
            print(
                f"template tip change height={work['template']['height']} parent={new_parent[:16]} rev={revision} preempted={preempted}",
                flush=True,
            )
        elif reason != "poll":
            print(
                f"template {reason} height={work['template']['height']} parent={new_parent[:16]} rev={revision}",
                flush=True,
            )
        return changed_parent

    async def run(self):
        while True:
            try:
                if self.work is None:
                    await self.refresh("initial")
                    await asyncio.sleep(max(float(self.args.template_poll_seconds), 0.1))
                    continue

                try:
                    best_hash = await asyncio.to_thread(lambda: self.rpc.call("getbestblockhash", timeout=10))
                except Exception:
                    best_hash = None

                current_parent = await self.current_prev_hash()
                refresh_due = time.time() - self.last_template_at >= float(self.args.template_refresh_seconds)
                if best_hash and best_hash != current_parent:
                    await self.refresh("tip")
                elif refresh_due:
                    await self.refresh("refresh")
                await asyncio.sleep(max(float(self.args.template_poll_seconds), 0.1))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"template watcher error: {exc}", file=sys.stderr, flush=True)
                await asyncio.sleep(2.0)


def detect_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        gpus = [int(line.strip()) for line in out.splitlines() if line.strip()]
        return gpus
    except Exception:
        return []


def lane_for(hostname, gpu, worker_index=0):
    digest = hashlib.sha256(f"{hostname}:{gpu}:{worker_index}".encode()).digest()
    return (int.from_bytes(digest[:4], "big") % 1_000_000) * 1_000_000_000_000


def emit_final_seeds(args, work, nonce64):
    cmd = [
        args.solver,
        "--backend",
        "cpu",
        "--emit-seeds",
        "--version",
        str(int(work["template"]["version"])),
        "--prev-hash",
        work["template"]["previousblockhash"],
        "--merkle-root",
        work["merkle"],
        "--time",
        str(int(work["template"]["curtime"])),
        "--bits",
        "0x" + work["template"]["bits"].removeprefix("0x"),
        "--seed-a",
        work["template"]["seed_a"],
        "--seed-b",
        work["template"]["seed_b"],
        "--block-height",
        str(int(work["template"]["height"])),
        "--parent-mtp",
        str(int(work["parent_mtp"])),
        "--nonce-start",
        str(int(nonce64)),
    ]
    env = os.environ.copy()
    lib_dir = Path(args.solver).resolve().parent.parent / "lib"
    if os.name == "nt":
        env["PATH"] = f"{lib_dir};{env.get('PATH', '')}"
    else:
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    raw = subprocess.check_output(cmd, text=True, env=env, timeout=30)
    payload = json.loads(raw)
    return payload["seed_a"], payload["seed_b"]


def write_stats(args, worker_records, workers, work, status, revision):
    path = Path(args.stats_file)
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    by_gpu = {}
    total = 0.0
    raw_total = 0.0
    now = time.time()
    stale_after = float(args.stale_rate_seconds)
    for worker in workers:
        record = worker_records.get(worker.key)
        if record is None or now - float(record.get("ts", 0.0)) > stale_after:
            gpu = worker.gpu
            raw_nps = 0.0
            work_nps = 0.0
            share_count = 0
            solver_elapsed_s = 0.0
            tries_used = 0
            prehash = {}
            pipeline = {}
            backend = {}
        else:
            gpu = int(record.get("gpu", worker.gpu))
            raw_nps = float(record.get("nps", 0.0))
            share_count = int(record.get("share_count", 0))
            solver_elapsed_s = float(record.get("solver_elapsed_s", 0.0))
            tries_used = int(record.get("tries_used", 0))
            prehash = record.get("gpu_prehash_scan") or {}
            pipeline = record.get("solve_pipeline") or {}
            backend = record.get("backend_runtime") or {}
            work_attempts = int(pipeline.get("batched_nonce_attempts", 0) or 0)
            if work_attempts <= 0:
                work_attempts = int(prehash.get("total_selected_headers", 0) or 0)
            work_nps = (float(work_attempts) / solver_elapsed_s) if solver_elapsed_s > 0 else 0.0
        total += work_nps
        raw_total += raw_nps
        entry = by_gpu.setdefault(
            gpu,
            {
                "index": gpu,
                "hashrate": 0.0,
                "unit": "N/s",
                "nonce_rate": 0.0,
                "btx_work_rate": 0.0,
                "raw_nonce_rate": 0.0,
                "raw_nonce_rate_units": "N/s",
                "accepted": 0,
                "solver_elapsed_s": 0.0,
                "tries_used": 0,
                "prehash_attempts": 0,
                "prehash_scanned": 0,
                "prehash_elapsed_us": 0,
                "prehash_pass_flags": 0,
                "prehash_selected_headers": 0,
                "batched_digest_requests": 0,
                "batched_nonce_attempts": 0,
                "digest_requests": 0,
                "cuda_successes": 0,
                "cuda_variable_base_batches": 0,
                "cuda_variable_base_device_us": 0,
                "cuda_variable_base_host_digest_us": 0,
                "cuda_profile": {},
            },
        )
        entry["hashrate"] += work_nps
        entry["nonce_rate"] += work_nps
        entry["btx_work_rate"] += work_nps
        entry["raw_nonce_rate"] += raw_nps
        entry["accepted"] += share_count
        entry["solver_elapsed_s"] += solver_elapsed_s
        entry["tries_used"] += tries_used
        entry["prehash_attempts"] += int(prehash.get("attempts", 0) or 0)
        entry["prehash_scanned"] += int(prehash.get("total_scanned_count", 0) or 0)
        entry["prehash_elapsed_us"] += int(prehash.get("total_elapsed_us", 0) or 0)
        entry["prehash_pass_flags"] += int(prehash.get("total_pass_flags", 0) or 0)
        entry["prehash_selected_headers"] += int(prehash.get("total_selected_headers", 0) or 0)
        entry["batched_digest_requests"] += int(pipeline.get("batched_digest_requests", 0) or 0)
        entry["batched_nonce_attempts"] += int(pipeline.get("batched_nonce_attempts", 0) or 0)
        entry["digest_requests"] += int(backend.get("digest_requests", 0) or 0)
        entry["cuda_successes"] += int(backend.get("cuda_successes", 0) or 0)
        entry["cuda_variable_base_batches"] += int(backend.get("cuda_variable_base_batches", 0) or 0)
        entry["cuda_variable_base_device_us"] += int(backend.get("cuda_variable_base_last_device_us", 0) or 0)
        entry["cuda_variable_base_host_digest_us"] += int(backend.get("cuda_variable_base_last_host_digest_us", 0) or 0)
        profile = record.get("cuda_profile") if record else None
        if isinstance(profile, dict):
            entry["cuda_profile"] = profile
    gpu_stats = [by_gpu[index] for index in sorted(by_gpu)]
    payload = {
        "ts": now,
        "miner": {
            "name": "meowminer-btx",
            "mode": "solo-fast-async",
            "algorithm": "btx-matmul",
            "running": True,
            "wallet": args.address,
            "pool": "solo",
            "unit": "N/s",
            "hashrate": total,
            "nonce_rate": total,
            "btx_work_rate": total,
            "raw_nonce_rate": raw_total,
            "raw_nonce_rate_units": "N/s",
            "hashrate_source": "batched_nonce_attempts_per_solver_second",
            "gpus": gpu_stats,
            "backend": "cuda",
            "height": int(work["template"]["height"]),
            "previousblockhash": work["template"]["previousblockhash"],
            "status": status,
            "template_revision": revision,
            "preemptions": sum(worker.preempt_count for worker in workers),
        }
    }
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    tmp.replace(path)


async def submit_block(args, rpc, work, result, share):
    def do_submit():
        nonce64 = int(share["nonce64"])
        seed_a, seed_b = emit_final_seeds(args, work, nonce64)
        matrix_c_hex = share.get("matrix_c_data_hex")
        if not matrix_c_hex:
            raise RuntimeError("block share missing matrix_c_data_hex")
        block_hex = build_block_hex(
            work["template"],
            work["coinbase"],
            work["txs"],
            work["merkle"],
            nonce64,
            share["matmul_digest"],
            seed_a,
            seed_b,
            matrix_c_hex,
        )
        return rpc.call("submitblock", [block_hex], timeout=120)

    submit = await asyncio.to_thread(do_submit)
    print(
        f"FOUND block height={work['template']['height']} gpu={result.get('local_gpu')} submit={submit}",
        flush=True,
    )


async def worker_loop(worker, work_manager, result_queue):
    while True:
        work, revision = await work_manager.get_work()
        try:
            result = await worker.solve_slice(work)
            stale = (await work_manager.current_prev_hash()) != work_prev_hash(work)
            await result_queue.put(
                {
                    "worker": worker,
                    "work": work,
                    "revision": revision,
                    "result": result,
                    "stale": stale,
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"worker gpu={worker.gpu} idx={worker.worker_index} error: {exc}", file=sys.stderr, flush=True)
            await worker.stop()
            await asyncio.sleep(1.0)


async def result_collector(args, rpc, work_manager, workers, result_queue):
    worker_records = {}
    submit_tasks = set()
    submitted_blocks = set()
    last_stats = 0.0
    last_print = 0.0
    while True:
        timeout = max(float(args.stats_interval) - (time.time() - last_stats), 0.1)
        item = None
        try:
            item = await asyncio.wait_for(result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        if item is not None:
            worker = item["worker"]
            result = item["result"]
            stale = bool(item["stale"])
            worker_records[worker.key] = {
                "ts": time.time(),
                "gpu": result.get("local_gpu", worker.gpu),
                "worker": result.get("local_worker", worker.worker_index),
                "nps": float(result.get("local_nps", 0.0)),
                "tries_used": int(result.get("tries_used", 0)),
                "solver_elapsed_s": float(result.get("elapsed_s", 0.0)),
                "share_count": int(result.get("share_count", 0)),
                "preempted": bool(result.get("preempted", False)),
                "backend_runtime": result.get("backend_runtime") or {},
                "solve_pipeline": result.get("solve_pipeline") or {},
                "gpu_prehash_scan": result.get("gpu_prehash_scan") or {},
                "cuda_profile": result.get("cuda_profile") or {},
            }
            shares = result.get("shares") or ([result] if result.get("is_block") else [])
            if shares and stale:
                print(
                    f"dropping stale share(s) height={item['work']['template']['height']} gpu={worker.gpu}",
                    flush=True,
                )
            elif shares:
                for share in shares:
                    if not share.get("is_block"):
                        continue
                    block_key = (
                        item["work"]["template"]["previousblockhash"],
                        int(share.get("nonce64", 0)),
                        share.get("matmul_digest"),
                    )
                    if block_key in submitted_blocks:
                        continue
                    submitted_blocks.add(block_key)
                    task = asyncio.create_task(submit_block(args, rpc, item["work"], result, share))
                    submit_tasks.add(task)
                    task.add_done_callback(submit_tasks.discard)

        now = time.time()
        if now - last_stats >= float(args.stats_interval):
            work, revision = await work_manager.get_work()
            write_stats(args, worker_records, workers, work, "mining", revision)
            last_stats = now
            if now - last_print >= float(args.print_interval):
                active_records = [
                    record for record in worker_records.values()
                    if now - float(record.get("ts", 0.0)) <= float(args.stale_rate_seconds)
                ]
                total = sum(float(record.get("nps", 0.0)) for record in active_records)
                print(
                    f"height={work['template']['height']} gpus={len(set(w.gpu for w in workers))} workers={len(workers)} rate={total:,.0f} N/s mode=async preemptions={sum(w.preempt_count for w in workers)}",
                    flush=True,
                )
                last_print = now


async def run_miner(args):
    rpc = RpcClient(
        args.datadir,
        args.conf,
        args.rpcconnect,
        args.rpcport,
        args.rpcuser,
        args.rpcpassword,
        args.rpccookiefile,
    )
    gpus = args.gpus
    if args.gpus == ["auto"]:
        gpus = detect_gpus()
    gpus = [int(gpu) for gpu in gpus]
    if not gpus:
        raise RuntimeError("no GPUs detected or configured")
    hostname = socket.gethostname()
    workers = []
    for gpu in gpus:
        for worker_index in range(int(args.workers_per_gpu)):
            workers.append(
                SolverWorker(gpu, args.solver, args, lane_for(hostname, gpu, worker_index), worker_index)
            )
    await asyncio.gather(*(worker.start() for worker in workers))
    print(
        f"fast async solo mining on GPUs {','.join(map(str, gpus))} with {args.workers_per_gpu} worker(s)/GPU to {args.address}",
        flush=True,
    )
    work_manager = WorkManager(rpc, args.address, args)
    work_manager.set_workers(workers)
    result_queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(work_manager.run()),
        asyncio.create_task(result_collector(args, rpc, work_manager, workers, result_queue)),
    ]
    tasks.extend(asyncio.create_task(worker_loop(worker, work_manager, result_queue)) for worker in workers)
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for worker in workers:
            await worker.stop()


def proposal_test(args):
    rpc = RpcClient(
        args.datadir,
        args.conf,
        args.rpcconnect,
        args.rpcport,
        args.rpcuser,
        args.rpcpassword,
        args.rpccookiefile,
    )
    work = build_template_work(rpc, args.address)
    seed_a, seed_b = emit_final_seeds(args, work, 0)
    block_hex = build_block_hex(
        work["template"],
        work["coinbase"],
        work["txs"],
        work["merkle"],
        0,
        "00" * 32,
        seed_a,
        seed_b,
        None,
    )
    result = rpc.call("getblocktemplate", [{"mode": "proposal", "data": block_hex}], timeout=120)
    print(
        json.dumps(
            {
                "proposal": result,
                "height": work["template"]["height"],
                "txs": 1 + len(work["txs"]),
                "merkle": work["merkle"],
                "bytes": len(block_hex) // 2,
            },
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="BTX fast solo miner using getblocktemplate and btx-gbt-solve")
    parser.add_argument("--datadir", default=os.environ.get("BTX_SOLO_DATADIR") or os.environ.get("BTX_DATADIR") or str(Path.home() / ".btx"))
    parser.add_argument("--conf", default=os.environ.get("BTX_SOLO_CONF"))
    parser.add_argument("--rpcconnect", default=os.environ.get("BTX_SOLO_RPC_CONNECT"))
    parser.add_argument("--rpcport", default=os.environ.get("BTX_SOLO_RPC_PORT"))
    parser.add_argument("--rpcuser", default=os.environ.get("BTX_SOLO_RPC_USER"))
    parser.add_argument("--rpcpassword", default=os.environ.get("BTX_SOLO_RPC_PASSWORD"))
    parser.add_argument("--rpccookiefile", default=os.environ.get("BTX_SOLO_RPC_COOKIE"))
    parser.add_argument("--address", default=os.environ.get("BTX_WALLET", PAYOUT_ADDRESS))
    parser.add_argument("--solver", default=os.environ.get("BTX_FASTSOLO_SOLVER", packaged_default_solver()))
    parser.add_argument("--gpus", default=os.environ.get("BTX_FASTSOLO_GPUS") or os.environ.get("BTX_GPUS") or "auto")
    parser.add_argument("--slice-seconds", type=float, default=float(os.environ.get("BTX_FASTSOLO_SLICE_SECONDS", "30")))
    parser.add_argument("--loop-sleep", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BTX_FASTSOLO_BATCH_SIZE", "1024")))
    parser.add_argument("--max-tries", type=int, default=int(os.environ.get("BTX_FASTSOLO_MAX_TRIES", "100000000")))
    parser.add_argument("--workers-per-gpu", type=int, default=int(os.environ.get("BTX_FASTSOLO_WORKERS_PER_GPU", "1")))
    parser.add_argument("--solver-threads", type=int, default=int(os.environ.get("BTX_FASTSOLO_SOLVER_THREADS", "1")))
    parser.add_argument("--pool-slots", type=int, default=int(os.environ.get("BTX_FASTSOLO_POOL_SLOTS", "0")))
    parser.add_argument("--template-poll-seconds", type=float, default=float(os.environ.get("BTX_FASTSOLO_TEMPLATE_POLL_SECONDS", "0.5")))
    parser.add_argument("--template-refresh-seconds", type=float, default=float(os.environ.get("BTX_FASTSOLO_TEMPLATE_REFRESH_SECONDS", "30")))
    parser.add_argument("--stats-interval", type=float, default=float(os.environ.get("BTX_FASTSOLO_STATS_INTERVAL", "2")))
    parser.add_argument("--print-interval", type=float, default=float(os.environ.get("BTX_FASTSOLO_PRINT_INTERVAL", "10")))
    parser.add_argument("--stale-rate-seconds", type=float, default=float(os.environ.get("BTX_FASTSOLO_STALE_RATE_SECONDS", "90")))
    parser.add_argument("--preempt", type=parse_bool, default=parse_bool(os.environ.get("BTX_FASTSOLO_PREEMPT", "1")))
    parser.add_argument("--epsilon-bits", type=int, default=18)
    parser.add_argument("--stats-file", default=os.environ.get("BTX_FASTSOLO_STATS_FILE", "/var/run/mfarm/btx-fastsolo.json"))
    parser.add_argument("--proposal-test", action="store_true")
    args = parser.parse_args()
    if isinstance(args.gpus, str):
        args.gpus = ["auto"] if args.gpus == "auto" else [part for part in args.gpus.split(",") if part.strip()]
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be >= 1")
    if args.solver_threads < 1:
        raise ValueError("--solver-threads must be >= 1")
    if args.pool_slots < 0:
        raise ValueError("--pool-slots must be >= 0")
    return args


def main():
    args = parse_args()
    if args.proposal_test:
        proposal_test(args)
        return
    asyncio.run(run_miner(args))


if __name__ == "__main__":
    main()
