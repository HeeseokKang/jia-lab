#!/usr/bin/env bash
# Stage 1 dry-run for QPM artifact migration.
#
# This script ONLY prints what Stage 2/3 would execute. It does not call
# mkdir/mv/rsync/ln on its own. Run it as many times as you want.
#
# Active dataset: 20260513_Hela_p15
# Decision: canonical artifact home is /data/Project_Data/QPM/<dataset>/analysis/
# ResearchRuntime is NOT the master location.

set -euo pipefail

REPO_ROOT="${HOME}/github/jia-lab/qpm-analysis"
DATASET="20260513_Hela_p15"
SRC="${REPO_ROOT}/analysis"
DST="/data/Project_Data/QPM/${DATASET}/analysis"

say() { printf '[dryrun] %s\n' "$*"; }
header() { printf '\n========== %s ==========\n' "$*"; }

# Sanity — confirm we are looking at the expected tree without modifying it.
header "Sanity checks (read-only)"
say "REPO_ROOT  = ${REPO_ROOT}"
say "DST root   = ${DST}"
[ -d "${SRC}" ] && say "SRC exists  OK" || say "SRC MISSING — abort before Stage 2"
[ -d "/data/Project_Data/QPM/${DATASET}" ] && \
    say "Dataset root exists  OK" || \
    say "Dataset root MISSING — abort before Stage 2"
[ -e "${DST}" ] && \
    say "WARN: ${DST} already exists; Stage 2 must reconcile" || \
    say "DST not yet created — Stage 2 will mkdir it"

# Stage 2a — create dataset-side skeleton.
header "Stage 2a: mkdir on dataset side (would run)"
for sub in calibration \
           dpc dpc/primary dpc/caveat dpc/empty \
           phase phase/primary phase/caveat \
           segmentation segmentation/primary segmentation/caveat \
           figures figures/previews \
           tables reports logs; do
    say "mkdir -p ${DST}/${sub}"
done

# Stage 2b — move heavy directories. Use rsync because /home and /data may be
# different filesystems; bare mv would silently fall back to cp+rm.
header "Stage 2b: move heavy directories (would run)"
for sub in calibration dpc phase segmentation figures tables logs; do
    if [ -d "${SRC}/${sub}" ] && [ ! -L "${SRC}/${sub}" ]; then
        n_files=$(find "${SRC}/${sub}" -type f | wc -l)
        size=$(du -sh "${SRC}/${sub}" | cut -f1)
        say "rsync -a --remove-source-files '${SRC}/${sub}/' '${DST}/${sub}/'   # ${n_files} files, ${size}"
        say "  then: find '${SRC}/${sub}' -type d -empty -delete"
    else
        say "skip ${SRC}/${sub}  (missing or already a symlink)"
    fi
done

# Stage 2c — move rendered HTML reports.
header "Stage 2c: move rendered HTML to dataset reports/ (would run)"
for f in "${SRC}"/0[1-5]_*.html; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    say "mv '${f}' '${DST}/reports/${base}'"
done

# Stage 3 — drop symlinks back into the repo so qmd renders keep working.
header "Stage 3: create repo-side symlinks (would run)"
for sub in calibration dpc phase segmentation figures tables logs; do
    say "ln -s '${DST}/${sub}' '${SRC}/${sub}'"
done

# Stage 3b — gitignore additions and index cleanup.
header "Stage 3b: git-level housekeeping (would run)"
say "echo 'analysis/*.html' >> ${REPO_ROOT}/.gitignore"
say "git -C ${REPO_ROOT} rm --cached analysis/20260513_Hela_p15/dpc_master_table.csv"

# Stage 4 — verification (would run, non-destructive but listed for completeness).
header "Stage 4: render verification (would run)"
for q in 01_dataset_inspection 02_illumination_calibration 03_dpc_frontend \
         04_wotf_reconstruction 05_phase_quantification; do
    say "(cd ${REPO_ROOT} && quarto render analysis/${q}.qmd)"
done

# Stage 5 — code/doc edits that follow successful render.
header "Stage 5: doc + script updates (manual review, listed only)"
say "edit CLAUDE.md L14-17 hard rule (new artifact home wording)"
say "edit CLAUDE.md run-command examples (5 sites: out_root → /data/.../analysis)"
say "edit src/dpc_analyze.py default out_root  (data_root / '_dpc_analysis' → 'analysis')"
say "edit configs/paths.py — register QPM dataset-side analysis path"

printf '\n[dryrun] complete — no filesystem changes were made.\n'
exit 0
