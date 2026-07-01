# Public Release Checklist

Use this checklist before switching the repository from private to public.

## Current Public Snapshot

- Public-ready branch: `public-release-clean`
- Public-ready commit style: one root commit without older local-path history
- Expected default branch after release: `main`

## Required Checks

Run these before making the repository public:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pytest -q
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

Run a current-tree and current-history scan:

```powershell
$pattern = '(secret|password|api[_-]?key|gho_|sk-[A-Za-z0-9]|\.env)'
git grep -n -i -E $pattern HEAD -- . ':(exclude).venv' ':(exclude)dist' ':(exclude)build'
foreach ($rev in git rev-list HEAD) {
  git grep -n -i -E $pattern $rev -- . ':(exclude).venv' ':(exclude)dist' ':(exclude)build'
}
```

Expected safe hits:

- `.gitignore` entries that ignore `.env`
- normal environment variable names such as `OLLAMA_HOST`, `OLLAMA_MODELS`, and `LOCALAPPDATA`

## Publish Procedure

Only run this after confirming `public-release-clean` is the branch you want to publish.

```powershell
git switch public-release-clean
git push origin public-release-clean:main --force-with-lease
gh repo edit --default-branch main
gh repo edit --visibility public
```

After publication:

```powershell
gh repo view --json nameWithOwner,defaultBranchRef,visibility,url
gh run list --branch main --limit 5
```

## Optional Cleanup

After the public `main` is verified, remove temporary release-prep branches:

```powershell
git push origin --delete codex/public-release-prep
git branch -D codex/public-release-prep
```

Keep `public-release-clean` locally until the public `main` state is verified.
