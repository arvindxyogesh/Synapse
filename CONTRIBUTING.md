# Contributing to Synapse

Thanks for considering a contribution. This is a young project, so the
process is intentionally lightweight.

## Getting set up

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
alembic upgrade head
pytest

# frontend
cd frontend
npm install
npm run lint && npm run build
```

Both directories have their own test suites and both run in CI on every
push (`.github/workflows/ci.yml`) — the same commands above are what CI
runs, so if they pass locally they'll pass there.

## Before opening a PR

- `cd backend && ruff check app tests && pytest`
- `cd frontend && npm run lint && npm run build`
- Add or update tests for behavior you change. This codebase leans on real
  tests over manual verification — e.g. `tests/test_openai_compat.py`
  exercises the actual `openai` SDK against the app, not just a schema
  comparison.
- Keep the diff scoped to what the PR is about. If you spot something
  unrelated worth fixing, open a separate PR or issue for it.

## Code style

- Backend: `ruff` (config in `backend/pyproject.toml`) is the source of
  truth — run it, don't hand-format to match nearby code.
- Comments explain *why*, not *what* — the code should already say what it
  does. Only add a comment for a non-obvious constraint, a workaround, or a
  reason a naive approach wouldn't work.
- Prefer the existing real-model-with-deterministic-fallback pattern
  (`app/embeddings.py`, `app/judge.py`) when adding anything that depends
  on an external model or network access: a real implementation when
  available, a clearly-labeled deterministic fallback otherwise, so the
  whole stack stays runnable and testable with zero external setup.

## Reporting bugs / requesting features

Open a GitHub issue. For a bug, include how to reproduce it (ideally a
failing test) and what you expected instead. For a feature, a short
description of the use case is more useful than a full design up front —
happy to discuss approach in the issue before a PR.

## Security

Please don't open a public issue for a security vulnerability — use
GitHub's private security advisory feature on this repo instead.
