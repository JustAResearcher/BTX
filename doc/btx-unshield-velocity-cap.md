# Shielded unshield velocity cap (v0.31.1)

Defense-in-depth consensus rule that bounds the **rate** at which value can leave the shielded pool
(shielded→transparent / "unshield"), so a stolen spend key or a future inner-proof soundness
regression becomes a **slow, observable leak** rather than an instantaneous drain. Mirrors the
velocity-limit recommendation from the Zcash Orchard disclosure; complements (does not replace) the
turnstile and the C-002 per-tx bindings.

## Where it sits in the defense stack

| Layer | Guarantee | Mechanism |
|---|---|---|
| C-002 v3 bindings | per-tx soundness — forgery non-constructible | `ct_proof.cpp:3503-3509` (serial↔key), `:3847-3890` (balance) |
| Turnstile | net transparent supply inflation = 0; total loss ≤ pool | `ShieldedPoolBalance` (`turnstile.cpp`), enforced `validation.cpp:1082/6122` |
| **Velocity cap (this)** | egress **rate** bounded; drain is slow + detectable | `ShieldedUnshieldVelocity` leaky bucket |

The cap is *not* a substitute for the first two — it is the early-warning / blast-radius layer for
the residual no-one can rule out by code review alone (e.g. the open `f3` reduction failing
post-activation).

## Parameters (consensus)

`src/consensus/params.h`:
- `nShieldedUnshieldVelocityActivationHeight` — mainnet **130,000** (`chainparams.cpp`); INT_MAX (inert)
  on networks that don't set it. Fast-follow after the v0.31.0 C-002 fork (123,000) so the network has
  time to upgrade to v0.31.1; self-serve unshield does not exist before 123,000 anyway.
- `nShieldedUnshieldVelocityWindowBlocks` — **960** (~1 day at 90 s).
- `nShieldedUnshieldVelocityCapBps` — **1000** (10% of the pool may be unshielded per window).

%-of-pool means the cap auto-scales with the live pool and never needs retuning. 10%/day implies a
full drain takes ≥10 days even with a stolen key — ample time to detect and respond, while a normal
withdrawal stays far under.

## Mechanism (`src/shielded/unshield_velocity.{h,cpp}` — implemented + unit-tested)

Leaky bucket, all derived from the live pool balance:
- `capacity B = cap_bps/10000 · pool` (max burst)
- `refill R = B / window` per block (sustained rate)
- per block `h`: `bucket = min(B, bucket + R·(h−last)); if net_unshield(h) > 0: bucket −= net_unshield(h);`
  **block invalid iff `bucket < 0`** (`shielded-unshield-velocity-exceeded`).

`net_unshield(h)` = the block's net positive `value_balance` (Σ value_balance over the block; shields
in the same block offset unshields). Deterministic: pool, height, params only — no wall-clock, no I/O.
Reorg-safe: `Apply()` returns the pre-update `Snapshot`; the caller stashes it and hands it to
`Restore()` on disconnect (no lossy inverse). Serializable for persistence.

Unit tests (`src/test/shielded_unshield_velocity_tests.cpp`, green): capacity = %-of-pool + auto-scale;
burst-to-capacity then reject; full-window refill; net-ingress neither consumes nor overfills;
exact reorg restore; serialization round-trip.

## Remaining integration (consensus wiring — the final step)

The accumulator is done and tested; wiring it into block connection is the last piece:

1. **Member.** Add `ShieldedUnshieldVelocity m_shielded_unshield_velocity` to `ChainstateManager`,
   beside `m_shielded_pool_balance`. Persist its `Snapshot` with the shielded state (same record that
   carries the pool balance), and rebuild/restore it the same way on load and reorg.
2. **ConnectBlock** (`validation.cpp` ~6117, where per-bundle `value_balance` is summed into
   `next_pool_balance`): accumulate `block_value_balance += *value_balance`. After the bundle loop, if
   `consensus.IsShieldedUnshieldVelocityCapActive(pindex->nHeight)`, call
   `Apply(nHeight, block_value_balance, pool_balance_at_block_start, cap_bps, window, prev)`; on
   `false`, `state.Invalid(BLOCK_CONSENSUS, "shielded-unshield-velocity-exceeded")`. Stash `prev` for undo.
3. **DisconnectBlock** (`validation.cpp` ~5419, beside `UndoValueBalance`): `Restore(prev)` from the
   saved snapshot.
4. **Functional test:** regtest with the activation height lowered (add a `-con`-overridable opt
   mirroring the other shielded activation heights), mine past activation, drive unshields up to the
   cap (accept) and over it (reject `shielded-unshield-velocity-exceeded`), and reorg across a
   capped block to confirm exact restore.

Because the rule is consensus and pruning-independent, the velocity snapshot must be **persisted**
(not recomputed from blocks a pruned node may not have) so every node — pruned or full — evaluates it
identically. The single-scalar `Snapshot` makes that cheap.
