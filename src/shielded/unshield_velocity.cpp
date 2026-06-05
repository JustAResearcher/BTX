// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <shielded/unshield_velocity.h>

#include <algorithm>

// All amounts here are money-range (|x| <= MAX_MONEY ~ 2^51), so sums/differences of a few of them
// stay well within int64; arithmetic below relies on that bound rather than checked helpers.

CAmount ShieldedUnshieldVelocity::Capacity(CAmount pool_balance, uint32_t cap_bps)
{
    if (pool_balance <= 0 || cap_bps == 0) return 0;
    // cap_bps/10000 * pool. pool <= MAX_MONEY (~2^51) and cap_bps < 2^16, so the naive product would
    // overflow int64; split into whole+fractional parts so every intermediate stays <= MAX_MONEY*1.
    const CAmount whole = (pool_balance / 10000) * static_cast<CAmount>(cap_bps);
    const CAmount frac = ((pool_balance % 10000) * static_cast<CAmount>(cap_bps)) / 10000;
    CAmount cap = whole + frac;
    if (cap < 0 || cap > MAX_MONEY) cap = MAX_MONEY;
    return cap;
}

CAmount ShieldedUnshieldVelocity::RefillPerBlock(CAmount pool_balance, uint32_t cap_bps, uint32_t window_blocks)
{
    const CAmount cap = Capacity(pool_balance, cap_bps);
    if (window_blocks == 0) return cap;
    const CAmount per_block = cap / static_cast<CAmount>(window_blocks);
    // Single-sat floor so a tiny-but-nonempty pool never wedges legitimate dust unshields.
    return (per_block <= 0 && cap > 0) ? 1 : per_block;
}

bool ShieldedUnshieldVelocity::Apply(int32_t height,
                                     CAmount net_unshield,
                                     CAmount pool_balance,
                                     uint32_t cap_bps,
                                     uint32_t window_blocks,
                                     Snapshot& prev)
{
    prev = m_state;

    const CAmount capacity = Capacity(pool_balance, cap_bps);

    // Initialize (or re-base after a gap/non-monotonic height) to a full bucket.
    if (!m_state.initialized || m_state.last_height < 0 || height <= m_state.last_height) {
        m_state.bucket = capacity;
    } else {
        const int64_t elapsed64 = static_cast<int64_t>(height) - static_cast<int64_t>(m_state.last_height);
        if (window_blocks == 0 || elapsed64 >= static_cast<int64_t>(window_blocks)) {
            // A full window (or more) elapsed -> the bucket refills completely.
            m_state.bucket = capacity;
        } else {
            // elapsed < window_blocks, so refill_per_block*elapsed <= capacity <= MAX_MONEY: no overflow.
            const CAmount refill_per_block = RefillPerBlock(pool_balance, cap_bps, window_blocks);
            const CAmount refilled = m_state.bucket + refill_per_block * static_cast<CAmount>(elapsed64);
            m_state.bucket = std::min(refilled, capacity);
        }
    }

    // Spend this block's net egress (value leaving the pool). Net ingress (net_unshield < 0) does not
    // over-fill the bucket. bucket and net_unshield are both money-range, so the difference is safe.
    if (net_unshield > 0) {
        m_state.bucket -= net_unshield;
    }

    m_state.last_height = height;
    m_state.initialized = true;

    if (m_state.bucket < 0) {
        // Velocity cap exceeded -> block invalid. Roll the running state back to the pre-Apply
        // snapshot so a re-evaluation starts clean.
        m_state = prev;
        return false;
    }
    return true;
}
