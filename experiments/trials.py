"""Trial definitions for Task 2. Each trial = a parent trial + a config delta.
Decisions (kept / discarded + reason) are written to experiments/decisions.json AFTER the
result has been seen, so the log never claims a result that was not measured."""
TRIALS = []   # filled in as experiments proceed: dict(id, parent, name, change, why, delta)
