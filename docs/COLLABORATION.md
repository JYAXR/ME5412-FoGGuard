# Lightweight GitHub collaboration

## Branch and review

- Protect `main` socially during the three-week sprint: no direct feature work on it.
- Branch names: `feat/<short-topic>`, `fix/<short-topic>`, or `docs/<short-topic>`.
- One Issue per deliverable and one primary owner. Pairing is encouraged; ownership stays clear.
- Pull requests should be reviewable in about 15 minutes and include evidence: test output, metric table, or UI screenshot.
- One approval is enough. The reviewer checks interfaces, safety fallback, and reproducibility.

## Daily rhythm

- 10-minute stand-up: yesterday, today, blocker.
- Update the task board after stand-up.
- Integrate to `main` at least every second day.
- Friday checkpoint: run the complete replay and record metrics, even if imperfect.

## Definition of done

A task is done only when code/docs are pushed, the acceptance checks in its Issue pass, and downstream owners can use it without private instructions.

## Suggested board columns

`Backlog` → `Ready` → `In progress` → `Review` → `Done`

Keep no more than one in-progress item per person. Use labels `week-1`, `week-2`, `week-3`, `integration`, and `design` for quick filtering.
