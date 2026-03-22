---
name: create-pr
description: Create a pull request from the current working branch to master for blog development. Use when the user wants to submit their draft or post changes via PR with squash merge.
argument-hint: "[optional PR title]"
allowed-tools: Bash(git *), Bash(gh *)
---

## Context

- Current branch: !`git branch --show-current`
- Git status: !`git status --short`
- Diff from master: !`git diff master --stat`

## Instructions

Create a pull request from the current branch to `master`.

### Steps

1. **Check current state**:
   - Verify you are NOT on `master`. If on `master`, create a new branch named `draft/<date>` (e.g., `draft/2026-03-22`) and switch to it.
   - If there are uncommitted changes, stage and commit them with a descriptive message.

2. **Push the branch**:
   - Push the current branch to origin with `-u` flag.

3. **Create the PR**:
   - Base branch: `master`
   - If the user provided `$ARGUMENTS`, use it as the PR title.
   - Otherwise, generate a concise title from the changes (e.g., "Add draft: DeepSeek V3 MLA" or "Update attention posts").
   - Body format:

```
gh pr create --base master --title "<title>" --body "$(cat <<'EOF'
## Changes
<bullet list summarizing what changed>

> Squash merge this PR to keep master history clean.
EOF
)"
```

4. **Output the PR URL** so the user can review and merge.

### Important

- Always use `--base master`.
- Remind the user to use **squash merge** when merging the PR to keep commit history clean.
