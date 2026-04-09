# 3DDL Optimization Docs

This `docs/` tree is the workspace-level source of truth for acceleration work across:

- `intelliscan/` for end-to-end pipeline execution
- `wp5-seg/` for segmentation training, pruning, export, and deployment tooling

Use this folder to track:

- what exists in code today
- what has been measured from saved outputs
- what is only reported in existing notes or prior discussions
- what should be tested next

## Evidence Rules

- Only treat a claim as validated when it can be traced to a file in this workspace or a measurement recomputed from those files.
- If a number only appears in an existing report, mark it as `Reported but not revalidated in this pass`.
- If a number came from the user but no local evidence was found, mark it as `User-reported result`.
- Every future experiment should include a comparison baseline on the same sample(s), environment, and code revision.

## Start Here

- `INDEX.md`: file map and evidence inventory
- `project_overview.md`: architecture and optimization layers
- `publication_strategy.md`: how to turn the current work into a report or paper
- `formal_report/3ddl_acceleration_report_submission.pdf`: formal LaTeX submission build with auto-generated TOC, list of figures, and list of tables
- `current_status.md`: 2-minute status snapshot
- `pipeline_demo_walkthrough.md`: one sample-based explanation of what goes into the full pipeline and what comes out
- `roadmap.md`: ranked next steps
- `experiment_registry.md`: registry of measured and reported artifacts
- `decision_log.md`: short reasoning log for current technical choices
- `optimization_history/`: per-optimization notes
- `templates/`: lightweight templates for future entries

## Scope Note

This docs tree sits at the workspace root because the acceleration work spans two separate Git repositories. Internal evidence paths are written relative to the workspace root, for example `intelliscan/output_formal/...` and `wp5-seg/pruning/...`.
