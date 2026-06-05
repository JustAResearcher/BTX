// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BTX_SHIELDED_UNSHIELD_VELOCITY_H
#define BTX_SHIELDED_UNSHIELD_VELOCITY_H

#include <consensus/amount.h>
#include <serialize.h>

#include <cstdint>

/**
 * Shielded-pool unshield (z->t egress) velocity cap — v0.31.1 defense-in-depth.
 *
 * Sits ALONGSIDE the turnstile (ShieldedPoolBalance, the net-supply firewall) and the C-002
 * per-tx value/serial bindings (soundness). Those make forgery non-constructible and bound total
 * loss; this bounds the RATE at which value can leave the pool, so a stolen spend key or a future
 * inner-proof regression becomes a slow, observable leak rather than an instant drain.
 *
 * Mechanism: a leaky bucket whose capacity and refill scale with the live shielded pool balance
 * (%-of-pool). Over any trailing window of W blocks at most cap_bps/10000 of the pool can be
 * unshielded; the bucket auto-scales as the pool grows, so it never needs retuning.
 *
 *   capacity B   = cap_bps/10000 * pool                 (max burst)
 *   refill   R   = B / W   per block                    (sustained rate; W full windows == one B/window)
 *   per block h: bucket = min(B, bucket + R*(h - last)); bucket -= net_unshield(h);
 *                reject the block if bucket < 0  ("shielded-unshield-velocity-exceeded").
 *
 * Reorg-safe by construction: Apply() returns the pre-update state so the caller can stash it in
 * block undo data and hand it back to Restore() on DisconnectBlock — no lossy inverse needed.
 * Consensus-deterministic: pool, height, and params are the only inputs; no wall-clock, no I/O.
 * Inert before nShieldedUnshieldVelocityActivationHeight (callers gate on it); self-serve unshield
 * does not exist before C-002 anyway.
 */
class ShieldedUnshieldVelocity
{
public:
    struct Snapshot {
        CAmount bucket{0};
        int32_t last_height{-1};
        bool initialized{false};
        SERIALIZE_METHODS(Snapshot, obj) { READWRITE(obj.bucket, obj.last_height, obj.initialized); }
    };

    /** Capacity (max burst) at the given pool balance: cap_bps/10000 of the pool. */
    static CAmount Capacity(CAmount pool_balance, uint32_t cap_bps);
    /** Per-block refill: capacity / window_blocks (>=1 sat so the bucket is never fully starved). */
    static CAmount RefillPerBlock(CAmount pool_balance, uint32_t cap_bps, uint32_t window_blocks);

    /**
     * Apply block `height` carrying `net_unshield` (sum of positive value_balance, i.e. value leaving
     * the pool) against `pool_balance` (the pool balance at the start of this block). On first call
     * the bucket initializes to full capacity. Returns the prior Snapshot (for undo) via `prev`.
     * Returns false iff the velocity cap is exceeded (=> block invalid).
     */
    [[nodiscard]] bool Apply(int32_t height,
                             CAmount net_unshield,
                             CAmount pool_balance,
                             uint32_t cap_bps,
                             uint32_t window_blocks,
                             Snapshot& prev);

    /** Restore the snapshot saved by a prior Apply() (DisconnectBlock / reorg). */
    void Restore(const Snapshot& prev) { m_state = prev; }

    /** Load/persist the running state. */
    void SetSnapshot(const Snapshot& s) { m_state = s; }
    [[nodiscard]] Snapshot GetSnapshot() const { return m_state; }

    SERIALIZE_METHODS(ShieldedUnshieldVelocity, obj) { READWRITE(obj.m_state); }

private:
    Snapshot m_state;
};

#endif // BTX_SHIELDED_UNSHIELD_VELOCITY_H
