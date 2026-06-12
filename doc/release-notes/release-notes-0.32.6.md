BTX version 0.32.6 is now available from:

  <https://github.com/btxchain/btx/releases/tag/v0.32.6>

This v0.32.6 point release hardens the post-sunset shielded recovery-exit and
transparent-exit validation paths. It is intended for nodes, miners, pools,
exchanges, services, and explorers running the 0.32.x block-125,000 shielded
sunset series.

Please report bugs using the issue tracker at GitHub:

  <https://github.com/btxchain/btx/issues>

To receive release and update notifications, please subscribe to:

  <https://btx.dev/>

# How to Upgrade

If you are running an older version, shut it down. Wait until it has completely
shut down, then install the new binaries or replace the existing `btxd`,
`btx-cli`, and GUI binaries with the 0.32.6 release artifacts.

BTX 0.32.6 keeps the block-125,000 shielded sunset posture and adds the
block-128,000 cleanup boundary for proofless transparent-funded `V2_SEND`
public-flow shielding. Upgrade from earlier 0.32.x builds for the recovery-exit
validation hardening, mempool/template cleanup, and expanded regression
coverage.

# Compatibility

BTX is supported on Linux, macOS 13+, and Windows 10+.

# Notable changes

- Recovery exits now accept only notes whose frozen tree leaf is the
  deterministic SMILE compact account commitment. Legacy note commitments are
  rejected on this path because the normal spend identifier cannot be derived
  from public recovery fields. The practical lesson is encoded in validation:
  recovery must share the same consensus-retired identifier space as normal
  transparent unshielding, and validators must derive those identifiers
  themselves.

- Recovery-exit transactions must now be pure transparent exits with zero
  transparent inputs, zero shielded outputs, exact value/fee accounting, fixed
  ML-DSA-44 public-key and signature sizes, fully consumed membership-proof
  encoding, ownership binding to the transaction's transparent outputs, and
  atomic retirement of both the compact-account commitment and canonical normal
  nullifier.

- Mempool cleanup and block-template assembly now reject recovery exits whose
  derived nullifier or commitment has already been spent or retired on chain.
  Template validation also tracks same-block shielded nullifier conflicts and
  continues to run full `TestBlockValidity` for MatMul/KAWPOW templates.

- `V2_SEND` validation now binds every spend descriptor to the bundle spend
  anchor, checks user output coin commitments against their SMILE public coin,
  allows strict zero-output post-sunset transparent unshielding, and disables
  proofless transparent-funded public-flow shielding from block `128000`.

- Post-recovery-exit startup and snapshot restore are fail-closed. Rebuilt
  shielded state now includes retired recovery-exit commitments, legacy snapshots
  after recovery-exit activation require local blocks to rebuild that set, and
  fast startup runs the chain-derived audit when recovery-exit state is active
  instead of treating a locally self-consistent state pin as independent proof.

- Regression coverage was expanded across recovery-exit chain rebuilds,
  mempool eviction, mining-template exclusion, note anchor validation, SMILE
  commitment binding, shielded validation checks, and transaction validation
  edge cases.

- Release automation, fast-start, bootstrap, and mining examples now point at
  the 0.32.6 artifacts.

# Credits

Thanks to everyone who contributed code, testing, operational validation, and
release engineering to this release.
