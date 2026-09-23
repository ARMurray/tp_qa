# correction/diagnostics

Output of `collect_diagnostics.py` lands here, and **this directory is
deliberately not gitignored** — unlike `logs/` and `data/`.

That is the whole point of it. The HPC and the machine doing the analysis are
different computers with no shared filesystem, and the login console is not
always somewhere things can be run interactively. Git already works in both
directions, so a diagnostic runs as a job, writes a text file here, and gets
committed and pushed. The other end pulls and reads it.

## Producing one

```bash
sbatch collect_diagnostics.slurm
```

Then, once it finishes:

```bash
cd /work/GRDVULN/tp_qa
git add correction/diagnostics/
git commit -m "diagnostics"
git push
```

## What lands here

`diag_<host>_<YYYYMMDD-HHMMSS>.txt` — one file per run, so they accumulate
rather than overwrite. Each covers:

1. environment, including the git commit the HPC is actually on
2. every configured path, and whether it exists
3. the deployed detector: age, size, and which classes it was trained on
4. parcel-store coverage against the training universe
5. `01a` NLCD output per state
6. detection output inventory per state, across all four roots
7. feature tables, with sizes and timestamps
8. the tail of the most recent job logs

All read-only. It writes nothing but its own report, so it is safe to run
while other jobs are going.

## Keep them or delete them?

They are small and textual, and having a record of what the cluster looked
like on a given day has repeatedly been worth more than it costs. Delete them
when they stop being interesting — but a handful is not a problem, and the
history of one is sometimes the fastest way to answer "when did this break?"

If they ever do become a problem, that is a signal to add
`correction/diagnostics/*.txt` to `.gitignore` and move to attaching them by
some other route — not a reason to stop producing them.
