# GUIDE_docs

## Part 1: Conceptual Explanation

The `docs/` folder is intentionally small. It does not duplicate the README or
folder guides. It stores ground-truth documents for contracts that are easy to
break silently and that multiple modules depend on.

Layout:

- `docs/user/`: user-facing policy and workflow docs.
- `docs/reference/`: reserved for stable developer reference material.

The primary user document is the offline raw-data policy. The repository is designed
to run from tracked Parquet files in `data/raw/` without database access in
normal workflows. Raw files are trusted only after validation against metadata
sidecars.

This folder should stay focused on durable contracts and reference material:

- data-access rules
- cache invariants
- architecture or specification documents requested explicitly
- other documentation that should remain stable even if implementation files move

It should not become a dumping ground for run logs, generated summaries, or duplicate explanations that already live in `README.md` or the `GUIDE_*.md` files.

## Part 2: Code Reference

- `user/offline_cache_policy.md`: defines the required `data/raw/` Parquet-plus-sidecar
  contract, validation checks, and ClickHouse fallback behavior.

Where to start:

1. Read `user/offline_cache_policy.md` before changing data-access logic.
2. Read `README.md` for offline-first run commands.
3. Read `GUIDE_ROOT.md` for project-level architecture.

## Part 3: Short Journal

- 2026-04-19: Updated docs focus to the `data/raw/` portability contract and
  optional ClickHouse refresh path.
- 2026-05-20: Moved user docs under `docs/user/` to match the standard layout.
