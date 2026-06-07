# BTX v0.32.2 C-002 and Shielded Sunset Fix Notes

Date: 2026-06-07

## Summary

The observed SLH-DSA failures were not random node noise. They were height-gated
C-002 compatibility bugs in wallet signing/finalization and mempool handling.
The shielded sunset also exposed a recovery-exit wallet accounting gap that
could leave exited notes visible as spendable.

This branch keeps production activation heights unchanged and uses regtest-only
activation overrides for deterministic boundary tests.

## Fixed Issues

- Mempool policy and the second mempool script check now use next-block C-002
  SLH-DSA/FIPS flags, matching wallet signing and block validation.
- Stale pre-C-002 legacy SLH transactions are revalidated and evicted before
  they can poison the C-002 activation block template.
- PSBT signing, finalization, extraction, analysis, wallet RPCs, and GUI paths
  now pass the same SLH-DSA/FIPS mode through local completeness checks.
- Recovery exits now bind the revealed claim to the actual shielded tree
  commitment, including SMILE compact-account commitments used by v2 notes.
- Recovery exits are wallet-accounted as spends in mempool, block scan, undo,
  conflict selection, and RPC output, so exited shielded balances do not remain
  selectable or appear frozen.
- Post-sunset `z_sendmany` tries exact single-note recovery exit when possible
  but falls back to strict zero-output V2_SEND transparent exit when needed.

## Defaults

- `-autoshieldcoinbase` remains default-off and is also blocked once shielded
  credits are disabled.
- Standard users should not need `-allowunpinnedshieldedsnapshot=1`; the current
  default allows unpinned snapshots. Flipping this default should wait until
  release-pinned shielded snapshot commitments are shipped.
- `-retainshieldedcommitmentindex=1` remains the sensible wallet default because
  post-sunset exits still need witnesses.

## Residual Limitation

External signer protocol negotiation still has no explicit C-002/FIPS mode
field. The wallet now finalizes/verifies returned PSBTs with the correct mode,
so wrong-mode signatures fail closed instead of broadcasting. A future signer
protocol revision should pass target validation height or SLH-DSA/FIPS mode to
the signer explicitly.

## Verification

Focused local checks:

- `cmake --build build-btx --target btxd btx-cli test_btx -j6`
- `build-btx/bin/test_btx --run_test=recovery_exit_tests,recovery_exit_wire_tests,recovery_exit_chain_tests,shielded_mempool_tests,validation_tests,psbt_wallet_tests`
- `test/functional/wallet_c002_slh_mempool_boundary.py --configfile=build-btx/test/config.ini --tmpdir=/tmp/btx-c002-slh-local`
- `test/functional/wallet_shielded_c002_sunset_lifecycle.py --configfile=build-btx/test/config.ini --tmpdir=/tmp/btx-shielded-sunset-local`

Docker/Linux checks:

- Ubuntu 24.04 container build of `build-linux` `btxd`/`btx-cli`
- `test/functional/wallet_c002_slh_mempool_boundary.py --configfile=build-linux/test/config.ini --tmpdir=/tmp/btx-c002-slh-docker`
- `test/functional/wallet_shielded_c002_sunset_lifecycle.py --configfile=build-linux/test/config.ini --tmpdir=/tmp/btx-shielded-sunset-docker`

The functional tests lower C-002/sunset heights only on regtest and cover:

- legacy SLH accepted before C-002 and evicted at the boundary,
- FIPS SLH direct wallet spend accepted and mined at activation,
- FIPS SLH PSBT spend accepted and mined after activation,
- pre-C-002 shielded notes spendable through the exact C-002 block,
- shielded ingress blocked at and after sunset,
- exact transparent exits accepted at and after sunset.
