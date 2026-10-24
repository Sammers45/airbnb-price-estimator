# Committing to GitHub

Each phase gets its own branch. It's merged into `main` after you approve it, then tagged. CI runs lint and tests on every push.

## One time

```bash
# 1. Identity
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"

# 2. Create an empty repo named airbnb-price-estimator at https://github.com/new
#    (no README, no .gitignore - the project has them)

# 3. Inside the project folder
git init -b main
git add .gitignore README.md requirements.txt pyproject.toml config.yaml .github docs/GIT.md
git commit -m "chore: project setup"
git remote add origin https://github.com/<your-username>/airbnb-price-estimator.git
git push -u origin main
```

If you have the GitHub CLI (`gh auth login` once), step 2 and the remote can be replaced with:
`gh repo create airbnb-price-estimator --public --source=. --push`

## Every phase

```bash
# Start the phase
git checkout main && git pull
git checkout -b phase-1                  # phase-2, phase-3, ...

# Add the phase files, then check them
ruff format .
ruff check .
pytest

# Commit and push
git add src/ tests/ docs/PHASE_1.md
git commit -m "feat: phase 1 data pipeline (load, validate, clean, split)"
git push -u origin phase-1
```

After you run the pipeline on the real data, commit the generated report:

```bash
git add reports/
git commit -m "docs: phase 1 data audit"
git push
```

## Changes after feedback

```bash
git checkout phase-1
# edit files, then
pytest
git add -A
git commit -m "fix: <what changed>"
git push
```

## When the phase is approved

```bash
git checkout main
git merge --no-ff phase-1 -m "Merge phase 1"
git push
git tag phase-1 && git push origin phase-1
```

Or open a pull request on GitHub from `phase-1` into `main` and click **Merge** once CI is green.

## Rules

- Never commit `data/`, `mlruns/` or model files. `.gitignore` already excludes them.
- Commit reports only after a real run.
- Commit message prefixes: `feat:` (new code), `fix:`, `docs:`, `test:`, `chore:`.
