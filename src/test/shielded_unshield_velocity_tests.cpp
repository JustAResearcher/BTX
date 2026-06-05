// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <shielded/unshield_velocity.h>

#include <consensus/amount.h>
#include <streams.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(shielded_unshield_velocity_tests)

namespace {
constexpr uint32_t WIN = 960;     // ~1 day at 90s
constexpr uint32_t BPS = 1000;    // 10% of pool per window
constexpr CAmount POOL = 1000 * COIN;
} // namespace

BOOST_AUTO_TEST_CASE(capacity_is_pct_of_pool)
{
    // 10% of a 1000-coin pool == 100 coins; refill == capacity / window.
    BOOST_CHECK_EQUAL(ShieldedUnshieldVelocity::Capacity(POOL, BPS), 100 * COIN);
    BOOST_CHECK_EQUAL(ShieldedUnshieldVelocity::Capacity(POOL, 0), 0);
    BOOST_CHECK_EQUAL(ShieldedUnshieldVelocity::Capacity(0, BPS), 0);
    BOOST_CHECK_EQUAL(ShieldedUnshieldVelocity::RefillPerBlock(POOL, BPS, WIN), (100 * COIN) / WIN);
    // Auto-scales with the pool.
    BOOST_CHECK_EQUAL(ShieldedUnshieldVelocity::Capacity(POOL * 2, BPS), 200 * COIN);
}

BOOST_AUTO_TEST_CASE(burst_up_to_capacity_then_reject)
{
    ShieldedUnshieldVelocity v;
    ShieldedUnshieldVelocity::Snapshot prev;
    int32_t h = 100;
    // First block initializes a full bucket (== capacity == 100 coins). Spend 60 -> OK.
    BOOST_CHECK(v.Apply(h++, 60 * COIN, POOL, BPS, WIN, prev));
    // Next block (no meaningful refill yet) spend 40 -> exactly drains, OK.
    BOOST_CHECK(v.Apply(h++, 40 * COIN, POOL, BPS, WIN, prev));
    // Now the bucket is ~empty; a further 10-coin egress exceeds the cap -> reject.
    BOOST_CHECK(!v.Apply(h++, 10 * COIN, POOL, BPS, WIN, prev));
    // Rejection leaves running state unchanged (bucket not driven negative).
    BOOST_CHECK_GE(v.GetSnapshot().bucket, 0);
}

BOOST_AUTO_TEST_CASE(refill_over_a_full_window_restores_allowance)
{
    ShieldedUnshieldVelocity v;
    ShieldedUnshieldVelocity::Snapshot prev;
    BOOST_CHECK(v.Apply(100, 100 * COIN, POOL, BPS, WIN, prev)); // drain the whole bucket
    BOOST_CHECK(!v.Apply(101, 50 * COIN, POOL, BPS, WIN, prev)); // immediately too much
    // Skip a full window: the bucket refills completely, so a fresh burst is allowed again.
    BOOST_CHECK(v.Apply(100 + WIN + 1, 100 * COIN, POOL, BPS, WIN, prev));
}

BOOST_AUTO_TEST_CASE(net_ingress_does_not_consume_or_overfill)
{
    ShieldedUnshieldVelocity v;
    ShieldedUnshieldVelocity::Snapshot prev;
    BOOST_CHECK(v.Apply(100, 100 * COIN, POOL, BPS, WIN, prev)); // drain
    // A shield-in (negative value_balance) must not consume the bucket nor refill it past capacity.
    const CAmount before = v.GetSnapshot().bucket;
    BOOST_CHECK(v.Apply(101, -500 * COIN, POOL, BPS, WIN, prev));
    BOOST_CHECK_LE(v.GetSnapshot().bucket, ShieldedUnshieldVelocity::Capacity(POOL, BPS));
    BOOST_CHECK_GE(v.GetSnapshot().bucket, before); // only the tiny per-block refill, never negative
}

BOOST_AUTO_TEST_CASE(reorg_restore_is_exact)
{
    ShieldedUnshieldVelocity v;
    ShieldedUnshieldVelocity::Snapshot s100, s101;
    BOOST_CHECK(v.Apply(100, 30 * COIN, POOL, BPS, WIN, s100));
    const auto after100 = v.GetSnapshot();
    BOOST_CHECK(v.Apply(101, 30 * COIN, POOL, BPS, WIN, s101));
    // Disconnect block 101: restore the snapshot Apply() handed back -> exactly the post-100 state.
    v.Restore(s101);
    BOOST_CHECK_EQUAL(v.GetSnapshot().bucket, after100.bucket);
    BOOST_CHECK_EQUAL(v.GetSnapshot().last_height, after100.last_height);
}

BOOST_AUTO_TEST_CASE(serialization_round_trips)
{
    ShieldedUnshieldVelocity v;
    ShieldedUnshieldVelocity::Snapshot prev;
    BOOST_CHECK(v.Apply(123, 7 * COIN, POOL, BPS, WIN, prev));
    DataStream ss;
    ss << v;
    ShieldedUnshieldVelocity v2;
    ss >> v2;
    BOOST_CHECK_EQUAL(v2.GetSnapshot().bucket, v.GetSnapshot().bucket);
    BOOST_CHECK_EQUAL(v2.GetSnapshot().last_height, v.GetSnapshot().last_height);
}

BOOST_AUTO_TEST_SUITE_END()
