# Fieldguide AI agent instructions

These instructions apply throughout this repository. This file is the entry point
for Codex and the source of truth for the validation policy shared with Cursor.

## Implementation and validation ownership

- Codex must not run post-modification formatting, linting, static type checking,
  or tests unless the user explicitly asks Codex to run them. This applies to
  initial implementation, repair turns, and work outside an orchestrator session.
- Do not run formatters in either write or check mode, lint autofixes, targeted
  tests, full test suites, or equivalent checks through scripts, task runners,
  pre-commit hooks, or subprocesses. Examples include `ruff format`, `ruff check`,
  `mypy`, `ty check`, `pyright`, `pytest`, and `unittest`.
- `codex-orchestrator` validates the starting checkout before phase implementation.
  Baseline failures stop the run before Codex starts and are handled as separate
  maintenance, without consuming the phase repair budget.
- After each Codex turn, the orchestrator runs `setup_commands` in order and stops
  if a prerequisite fails. Once setup succeeds, it applies formatting and runs all
  independent `commands`, collecting failures into one report for the same phase
  session. It reruns setup and all checks after each repair.
- Repair only regressions caused by the assigned phase, including affected
  contracts. Do not expand repairs into unrelated baseline maintenance or
  components explicitly excluded by the plan. If a failure requires such work,
  return `blocked` with the affected files and reason. A resumed legacy run may
  have no recorded baseline; do not assume its failures are phase regressions.
- Implement the requested behavior and add or update meaningful tests where
  required by the change or plan. Deferring execution does not remove test-writing
  responsibilities or allow weakening tests or checks to conceal failures.
- Review source, tests, configuration, documentation, and the diff by inspection.
  Fix issues reported by the orchestrator, then return control without rerunning
  the failed command or the validation sequence yourself.
- Do not launch or resume the orchestrator from an implementation or repair turn.
  Do not install tools or dependencies solely to perform deferred checks.
  Dependency and lockfile changes required by the assigned implementation remain
  in scope.
- State that checks were not run by Codex and are deferred to the orchestrator
  (or the user when working outside an orchestrated run). Do not claim they passed
  without actual results. Deferred validation alone is not an implementation
  blocker, and a completed implementation is not a claim of passing validation.

## Plan execution

When invoked by `codex-orchestrator`:

- Implement only the assigned phase and its acceptance criteria. Preserve
  unrelated edits and inspect partial changes when recovering an interrupted turn.
- Do not implement later phases, commit, change Git HEAD, push, publish, edit the
  source plan, or change `[tool.codex-orchestrator]` configuration.
- Return the structured result requested by the orchestrator. Use `completed`
  only when all phase implementation requirements are fulfilled; otherwise use
  `blocked` and describe the implementation blocker. Let the orchestrator decide
  whether validation passes and the next phase can begin.
- The orchestrator selects implementation and repair models from exported
  `CODEX_ORCHESTRATOR_MODEL` / `CODEX_ORCHESTRATOR_REASONING_EFFORT` and
  `CODEX_ORCHESTRATOR_REPAIR_MODEL` / `CODEX_ORCHESTRATOR_REPAIR_REASONING_EFFORT`.
  Repairs retain the phase session but may use a different model. Do not edit
  `.codex/config.toml` or orchestrator settings to change the assigned model.

## Python design

Before creating or modifying Python code, read and follow
[the Python design rules](.cursor/rules/python-oop-design-guidelines.mdc).
Apply them proportionally: preserve architectural boundaries, keep responsibilities
focused, encapsulate state and invariants, inject external dependencies, prefer
composition, and avoid speculative abstractions or unrelated rewrites. Design
tests around observable behavior. Review design by inspecting the final code;
this does not require running validation commands.

## Documentation and rule maintenance

- Read and follow [the README maintenance rules](.cursor/rules/readme-maintenance.mdc).
  Assess README impact for every task and update it in the same change when
  behavior, setup, commands, configuration, architecture, or limitations change.
  Avoid churn when there is no reader-visible impact.
- Verify documentation using source, configuration, test definitions, and existing
  output. Treat requests to verify the quickstart or examples as inspection during
  Codex work; leave execution to external validation unless the user asks for it.
- Read [the Cursor rule conventions](.cursor/rules/cursor-rules.mdc) before editing
  `.cursor/rules/`. The rule-location restriction applies to Cursor `.mdc` files;
  this root `AGENTS.md` is the Codex instruction file.
- Keep Cursor rules consistent with this file. General instructions in rules,
  plans, or the README to run checks before finishing do not override the validation
  ownership policy above. Explicit user requests to run checks take precedence.
