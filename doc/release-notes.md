BTX version 0.31.1 is now available from:

  <https://github.com/btxchain/btx/releases/tag/v0.31.1>

This point release covers changes merged after BTX 0.31.0 was prepared on June
4, 2026. It includes shielded restart improvements, updated wallet coinbase
shielding defaults, shielded unshield velocity-cap enforcement, formal
verification artifacts, and release tooling hardening.

Please report bugs using the issue tracker at GitHub:

  <https://github.com/btxchain/btx/issues>

To receive security and update notifications, please subscribe to:

  <https://btx.dev/>

# How to Upgrade

If you are running an older version, shut it down. Wait until it has completely
shut down, then install the new binaries or replace the existing `btxd`,
`btx-cli`, and GUI binaries with the 0.31.1 release artifacts.

BTX 0.31.0 remains the network-upgrade release for block 123,000. BTX 0.31.1 is
a fast-follow operator update ahead of block 130,000; nodes, miners, pools,
exchanges, services, and explorers should upgrade before that height.

# Compatibility

BTX is supported on Linux, macOS 13+, and Windows 10+.

# Notable changes

- Zero-downtime shielded restart is now the default. When a node restarts with
  persisted shielded state that exactly matches the active chain tip (frontier
  root/size, commitment-index digest, anchor/registry windows, and nullifier-set
  accumulator all validated), it trusts that state and skips the restart audit
  path. If the local data is missing, out of date, on the wrong chain, or fails
  any check, the node falls back to a full rebuild from chain as before. Operator
  knobs are `-fastshieldedstartup` (default `1`) and `-shieldedstartupaudit`
  (default `1`).

- Coinbase auto-shielding is now opt-in. Mined rewards remain transparent by
  default unless the wallet is started with `-autoshieldcoinbase=1`. Even when
  enabled, automatic coinbase shielding waits until the C-002 shielded-pool
  activation height by default (block 123,000 on mainnet, 0 on regtest). Test
  networks can override this floor with `-autoshieldcoinbaseminheight=<n>`.

- The release enforces `ShieldedUnshieldVelocity`, a shielded-pool unshield
  velocity-cap consensus rule. The configured window is 960 blocks with a 1000
  bps cap, and the activation height is 130,000 on the configured networks. The
  running state is persisted with shielded state, restored exactly across reorgs,
  and covered by unit tests plus `wallet_shielded_velocity_cap.py`. See
  `doc/btx-unshield-velocity-cap.md` for the operator and integration details.

- A new `formal-verification/` suite documents the tiered shielded-pool proof
  plan and reproducible Z3 checks. Tier 1 covers accounting and velocity-cap
  invariants, Tier 2 covers verifier-relation algebra, and Tier 3 documents the
  Module-SIS reduction. Each tier includes a `PROOFS.md`, and
  `python3 formal-verification/run_all.py` runs the machine-checkable
  obligations.

- The auto-update installer has stricter shell cleanup guards, and release
  automation examples now point at the 0.31.1 artifacts.

# Credits

Thanks to everyone who contributed code, testing, operational validation, and
release engineering to this release.
