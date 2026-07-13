# Repository Audit, Cleanup, and Security Hardening Plan

## Goal

Instruct the agent to recursively inspect the entire project repository, determine which files and folders are actually needed, verify that required files are correctly linked into the project, remove or quarantine unwanted/corrupted files, reorganize the repository into a clean structure, and perform a deployment-focused security review.

This task is both a repository hygiene task and a secure code review task. Secure code review should focus on the full codebase, dependencies, data flow, configuration, and deployment exposure, not just the newest changes. [web:119][web:120]

---

## Scope

The agent must inspect:
- Every folder.
- Every subfolder.
- Every file.
- Every config, script, notebook, asset, dataset pointer, and test.
- Every deployment-facing component such as API, backend, frontend, environment config, and container files.

The agent must treat the repository as a living system and not delete anything unless it has confirmed that the item is unused, duplicated, obsolete, corrupted, or unsafe to keep. Manual review is important because automated tools alone miss context, trust boundaries, and business logic dependencies. [web:120]

---

## Discovery Phase

The agent must first build a complete inventory of the repository:
- Directory tree.
- File types.
- Import graph.
- Script entry points.
- Config references.
- Runtime references.
- Test references.
- Documentation references.
- Asset references.
- Generated artifacts.
- Possible duplicates or stale copies.

The agent should recursively trace file usage, because secure review and repository hygiene both require understanding the full application boundary and the dependencies inside it. [web:120]

---

## File Classification

For every file and folder, the agent must assign one of these states:
- `KEEP`.
- `UPDATE`.
- `MOVE`.
- `DEPRECATE`.
- `REMOVE`.
- `QUARANTINE`.

Classification rules:
- `KEEP`: file is needed, referenced, and healthy.
- `UPDATE`: file is needed but outdated, inconsistent, or partially broken.
- `MOVE`: file is needed but belongs in a different structure.
- `DEPRECATE`: file is legacy but still potentially useful for reference.
- `REMOVE`: file is confirmed unused, duplicated, or unnecessary.
- `QUARANTINE`: file appears corrupted, suspicious, malformed, or unsafe until reviewed.

The agent must always justify classification with evidence from imports, runtime wiring, config references, or release usage. Do not delete first and ask questions later.

---

## Linkage Verification

For each file that is marked `KEEP` or `UPDATE`, the agent must verify that it is correctly linked to the project by checking:
- Import statements.
- Module references.
- Routing references.
- Config references.
- Build pipeline references.
- Dataset references.
- Test references.
- Documentation references.
- Runtime call paths.

If a file exists but is not linked anywhere, the agent must determine whether it is:
- a future placeholder,
- a manual artifact,
- a legacy backup,
- or a true orphan.

If it is a true orphan, it should be moved to quarantine or removed after confirmation.

---

## Cleanup Rules

The agent may remove or quarantine files only when at least one of these is true:
- The file is confirmed unused.
- The file is a duplicate.
- The file is corrupted.
- The file is a stale build artifact.
- The file is a dead backup.
- The file is unsafe and not needed.
- The file violates project structure rules and has no valid runtime dependency.

The agent must never remove a file that is still needed by:
- code,
- tests,
- deployment,
- data loading,
- evaluation,
- model training,
- documentation,
- or reproducibility.

If a deletion is proposed, the agent must record:
- the reason,
- the evidence,
- the dependent files checked,
- and the rollback path.

---

## Repository Restructuring

After cleanup, the agent must reorganize the repository into a clear and maintainable structure.

Recommended structure:
- `backend/`
- `frontend/`
- `training/`
- `models/`
- `data/`
- `certified_data/`
- `configs/`
- `tests/`
- `docs/`
- `scripts/`
- `artifacts/`
- `reports/`

The agent should:
- group related files together,
- reduce nesting confusion,
- move temporary or experimental work into clearly labeled folders,
- separate source code from generated outputs,
- and ensure naming is consistent and descriptive.

The restructure must preserve functionality and not break imports or deployment paths.

---

## Corruption and Quality Checks

The agent must search for:
- corrupted files,
- malformed HTML, JSON, CSV, or config files,
- broken encodings,
- empty placeholders,
- duplicate versions of the same file,
- inconsistent filenames,
- impossible file extensions,
- and partially generated artifacts.

Any corrupted file should be quarantined first unless it is clearly a disposable artifact. The agent should prefer safe cleanup over aggressive deletion.

---

## Security Review

The agent must review the repository for security risks that could affect deployment or runtime safety. This must include:
- CORS misconfiguration,
- CSRF weaknesses,
- authentication/session issues,
- authorization bypasses,
- exposed secrets,
- hardcoded credentials,
- unsafe file uploads,
- path traversal,
- injection risks,
- unsafe deserialization,
- insecure direct object references,
- overly verbose error leakage,
- weak logging of sensitive data,
- vulnerable dependencies,
- unsafe environment variable handling,
- and unsafe debug or development settings.

The review should follow a secure code review mindset that traces data flow, trust boundaries, and deployment configuration. OWASP guidance emphasizes reviewing authentication, authorization, injection, error handling, logging, sensitive data exposure, and deployment configuration as core review areas. [web:119][web:120]

---

## Deployment Hardening

The agent must explicitly check deployment-facing surfaces such as:
- API route exposure.
- CORS allowlists.
- cookie settings.
- session cookie flags.
- security headers.
- environment variables.
- secret files.
- Docker or container configuration.
- reverse proxy settings.
- upload directories.
- public asset exposure.
- debug endpoints.
- health endpoints.
- log output.
- model artifact exposure.

The agent must ensure the project is safe for a deployable state and not leaving development shortcuts enabled.

---

## Dependency Review

The agent must inspect third-party dependencies and verify:
- they are required,
- they are pinned appropriately,
- they are not duplicated,
- they do not contain known vulnerabilities,
- and they are compatible with the project structure.

If dependency scanning tools or lockfiles exist, the agent should use them. Any dependency no longer needed should be removed carefully after confirming it is not imported or loaded dynamically.

---

## Validation Tests

After cleanup and restructuring, the agent must run checks to confirm:
- all imports still resolve,
- all tests still pass,
- training scripts still run,
- datasets still load correctly,
- deployment config still works,
- and no file links were broken.

The agent must also run a final repository tree check to confirm the structure is clean and coherent.

---

## Reporting Requirements

The agent must produce a final report containing:
- files kept,
- files moved,
- files deprecated,
- files removed,
- files quarantined,
- security issues found,
- fixes applied,
- unresolved risks,
- and validation status.

The report should be concise but evidence-based.

---

## Execution Order

1. Inventory all files and folders recursively.
2. Trace imports, references, and runtime usage.
3. Classify every file.
4. Quarantine suspicious or corrupted items.
5. Remove confirmed dead files.
6. Restructure folders and substructures.
7. Verify all links and imports.
8. Review security and deployment exposure.
9. Run validation tests.
10. Produce the final report.

---

## Agent Instruction Block

Use this exact instruction for the agent:

> Recursively inspect every folder, subfolder, and file in the repository. Build a full inventory, trace imports and runtime references, and classify every item as KEEP, UPDATE, MOVE, DEPRECATE, REMOVE, or QUARANTINE based on actual usage. Verify that needed files are properly linked to the project, quarantine corrupted or suspicious files, remove only confirmed dead or duplicate files, and reorganize the repository into a clean and maintainable folder structure without breaking functionality. Then perform a deployment-focused security review covering CORS, CSRF, secrets, authentication, authorization, logging, uploads, dependency risks, debug settings, and other vulnerabilities that could affect a production system. Validate the final structure with tests and produce a concise evidence-based report of all actions taken.

---

## Safety Rule

Do not perform destructive cleanup unless the file has been proven unnecessary or unsafe. Favor quarantine over deletion when uncertainty remains.