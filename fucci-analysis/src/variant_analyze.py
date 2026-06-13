"""Stage 4-6 -- 2D quadrant gating, drug-response, and cross-construct comparison.

PRIMARY endpoint (Heeseok 2026-06-10 eve, literature-driven reframe):
Our constructs are PIP-FUCCI-type. Green/488 = PIP = a DIRECT S-PHASE sensor
(HIGH in G1 *and* G2, LOW in S); red = Geminin (S/G2/M). So phase is read from a
**2D quadrant gate** on the green/red scatter, NOT the old 1D log2(red/green)
ratio. The 1D ratio mis-models G2 (double-positive -> mid ratio, not extreme),
which under-detected the RO->G2 arrest and produced metric artifacts (D900 RO
"weak", pBJ284 Palbo "wrong"). See library/cell_cycle/reporters/notes.md.

Quadrant map (per-channel on = SNR >= cfg.GATE_SNR):
  green-only -> G1 | red-only -> S | double-positive -> G2 | (M = diffuse, not gated)
Drug expectation: Palbo (CDK4/6i) -> up G1-frac ; RO (CDK1i) -> up G2-frac.

The 1D phase_score = log2(red/green) is kept as a DEMOTED reference column only.

Reads per_cell_measurements.csv (default erosion = cfg.EROSION_DEFAULT_PX) and
writes, dataset-side:
  figures/scatter_<construct>.png        per-cell green-vs-red, coloured by gated phase
  figures/phasefrac_<construct>.png      G1/S/G2 stacked fractions by drug (+ reps)
  figures/brightness_488_compare.png     within-488 brightness across constructs
  brightness_snr_summary.csv             per construct/channel SNR + bright-fraction
  separation_summary.csv                 anti-correlation per construct
  phase_fraction_summary.csv             per construct/drug G1/S/G2 fractions (+reps)
  drug_response_summary.csv              target-phase fraction shift + expected dir
  variant_comparison_summary.csv         cross-construct ranking vs D900
  RESULTS_SUMMARY.md                     headline numbers + caveats

Run:  conda activate fucci-analysis
      python fucci-analysis/src/variant_analyze.py [--erosion 7]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import variant_config as cfg  # noqa: E402

DRUG_ORDER = ["DMSO", "Palbo", "RO"]
DRUG_COLOR = {"DMSO": "0.4", "Palbo": "tab:blue", "RO": "tab:red"}
PHASE_ORDER = ["G1", "S", "G2"]
PHASE_COLOR = {"G1": "tab:green", "S": "tab:red", "G2": "tab:orange", "unassigned": "0.85"}

# Per-channel detection / gate threshold: SNR = (mean - bg_med)/bg_sigma >= this.
DETECT_SNR = cfg.GATE_SNR

# The target quadrant whose fraction should RISE under each drug.
DRUG_TARGET_PHASE = {"Palbo": "G1", "RO": "G2"}


def phase_score(df: pd.DataFrame) -> pd.Series:
    """DEMOTED reference metric only (mis-models G2). Kept for continuity."""
    g = df["green_bgsub"].clip(lower=1.0)
    r = df["red_bgsub"].clip(lower=1.0)
    return np.log2(r / g)


def gate_phase(df: pd.DataFrame) -> pd.Series:
    """2D quadrant gate (the PRIMARY phase call). green_on/red_on are SNR>=GATE_SNR.
      green-only -> G1 ; red-only -> S ; double-positive -> G2 ; neither -> unassigned
    """
    g, r = df["green_on"].to_numpy(), df["red_on"].to_numpy()
    out = np.full(len(df), "unassigned", dtype=object)
    out[g & ~r] = "G1"
    out[~g & r] = "S"
    out[g & r] = "G2"
    return pd.Series(out, index=df.index)


def load(erosion: int) -> pd.DataFrame:
    df = pd.read_csv(cfg.ANALYSIS_DIR / "per_cell_measurements.csv")
    df = df[df["erosion_px"] == erosion].copy()
    df["phase_score"] = phase_score(df)
    df["green_on"] = df["green_snr"] >= DETECT_SNR
    df["red_on"] = df["red_snr"] >= DETECT_SNR
    df["assignable"] = df["green_on"] | df["red_on"]   # >=1 reporter on (stateable)
    df["phase"] = gate_phase(df)
    return df


def red_gateable(df: pd.DataFrame, construct: str) -> bool:
    """A construct resolves S vs G2 only if its red reporter turns on in a real
    fraction of the asynchronous DMSO population (>= cfg.MIN_RED_GATE_FRAC) and
    labelling did not fail. Otherwise the 2-colour gate collapses to green-only."""
    if construct == "pBJ285" and cfg.PBJ285_LABELLING_FAILED:
        return False
    dmso = df[(df["construct"] == construct) & (df["drug"] == "DMSO")]
    return bool(len(dmso) and dmso["red_on"].mean() >= cfg.MIN_RED_GATE_FRAC)


# ---------- figures ---------------------------------------------------------
def fig_scatter(df: pd.DataFrame, construct: str, out: Path) -> None:
    """Per-cell green-vs-red scatter, faceted by drug, points coloured by gated phase."""
    sub = df[df["construct"] == construct]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, drug in zip(axes, DRUG_ORDER):
        d = sub[sub["drug"] == drug]
        for ph in ["unassigned", "S", "G1", "G2"]:   # unassigned drawn first (under)
            p = d[d["phase"] == ph]
            ax.scatter(p["green_bgsub"].clip(lower=1), p["red_bgsub"].clip(lower=1),
                       s=7, alpha=0.4, color=PHASE_COLOR[ph], edgecolors="none",
                       label=ph if ax is axes[0] else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{drug}  (n={len(d)})", fontsize=11)
        ax.set_xlabel(f"green {cfg.CONSTRUCTS[construct]['green']} bgsub  (PIP: hi=G1/G2)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(f"red {cfg.CONSTRUCTS[construct]['red']} bgsub  (Gem: hi=S/G2/M)")
    axes[0].legend(fontsize=8, markerscale=2, framealpha=0.9)
    gate = "red-gateable" if red_gateable(df, construct) else "RED NOT GATEABLE (green-only)"
    fig.suptitle(f"{construct}  2D quadrant gate  [{gate}]", fontsize=12)
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)


def fig_phasefrac(df: pd.DataFrame, construct: str, out: Path) -> None:
    """Stacked G1/S/G2 fraction (among stateable cells) per drug, with replicate dots."""
    sub = df[(df["construct"] == construct) & df["assignable"]]
    fig, ax = plt.subplots(figsize=(6, 4.6), constrained_layout=True)
    x = np.arange(len(DRUG_ORDER))
    bottoms = np.zeros(len(DRUG_ORDER))
    for ph in PHASE_ORDER:
        vals = []
        for drug in DRUG_ORDER:
            d = sub[sub["drug"] == drug]
            vals.append((d["phase"] == ph).mean() if len(d) else 0.0)
        ax.bar(x, vals, bottom=bottoms, color=PHASE_COLOR[ph], label=ph, width=0.6,
               edgecolor="white")
        bottoms += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(DRUG_ORDER)
    ax.set_ylabel("fraction of stateable cells")
    ax.set_ylim(0, 1)
    gate = "" if red_gateable(df, construct) else "  (RED NOT GATEABLE)"
    ax.set_title(f"{construct}  phase fractions{gate}", fontsize=12)
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight"); plt.close(fig)


def fig_brightness_488(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    order = list(cfg.CONSTRUCTS.keys())
    data = [df[df["construct"] == c]["green_snr"].dropna().values for c in order]
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("green 488 SNR (per cell)")
    ax.set_title("Within-488 brightness comparison (shared channel/exposure)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)


# ---------- tables ----------------------------------------------------------
def brightness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Brightness/usability = bright_fraction in DMSO (fraction of cells where the
    reporter is detectable, SNR>=GATE_SNR). In an unsynchronized DMSO population the
    cell-cycle distribution is comparable across lines, so this isolates reporter
    quality. p90 SNR is the secondary proxy; median is reference only (it undersells
    bimodal FUCCI reporters)."""
    rows = []
    for c in cfg.CONSTRUCTS:
        sub = df[df["construct"] == c]
        dmso = sub[sub["drug"] == "DMSO"]
        for ch, snr_col, on_col in [("green", "green_snr", "green_on"),
                                    ("red", "red_snr", "red_on")]:
            s = sub[snr_col].replace([np.inf, -np.inf], np.nan).dropna()
            unreliable = (c == "pBJ285" and ch == "red" and cfg.PBJ285_LABELLING_FAILED)
            rows.append({
                "construct": c, "channel_role": ch,
                "channel": cfg.CONSTRUCTS[c][ch],
                "bright_frac_dmso": round(float(dmso[on_col].mean()), 3) if len(dmso) else np.nan,
                "p90_snr": round(float(s.quantile(0.9)), 2) if len(s) else np.nan,
                "median_snr": round(float(s.median()), 2) if len(s) else np.nan,
                "n_cells": int(len(sub)),
                "reliable": not unreliable,
            })
    return pd.DataFrame(rows)


def separation_table(df: pd.DataFrame) -> pd.DataFrame:
    """DMSO green-vs-red anti-correlation (a true 2-colour FUCCI reporter should be
    anti-correlated as cells move G1<->S). Only meaningful where red is gateable."""
    rows = []
    for c in cfg.CONSTRUCTS:
        d = df[(df["construct"] == c) & (df["drug"] == "DMSO") & df["assignable"]]
        if len(d) > 5:
            rho, p = spearmanr(d["green_bgsub"], d["red_bgsub"])
        else:
            rho, p = float("nan"), float("nan")
        rows.append({"construct": c, "dmso_anticorr_rho": round(rho, 3),
                     "anticorr_p": f"{p:.1e}" if p == p else "nan",
                     "red_gateable": red_gateable(df, c),
                     "interpretation": "more negative rho = better anti-corr; valid only if red_gateable"})
    return pd.DataFrame(rows)


def phase_fraction_table(df: pd.DataFrame) -> pd.DataFrame:
    """PRIMARY readout. Per construct x drug, the G1/S/G2 fractions among STATEABLE
    cells (>=1 reporter on), plus the stateable fraction (gating efficiency) and the
    replicate-level spread of the drug TARGET phase."""
    st = df[df["assignable"]]
    rows = []
    for c in cfg.CONSTRUCTS:
        gateable = red_gateable(df, c)
        for drug in DRUG_ORDER:
            d = st[(st["construct"] == c) & (st["drug"] == drug)]
            n_all = int(((df["construct"] == c) & (df["drug"] == drug)).sum())
            n = len(d)
            fr = {ph: (float((d["phase"] == ph).mean()) if n else np.nan) for ph in PHASE_ORDER}
            # per-replicate target-phase fractions (for spread)
            rows.append({
                "construct": c, "drug": drug, "red_gateable": gateable,
                "n_stateable": n, "n_total": n_all,
                "stateable_frac": round(n / n_all, 3) if n_all else np.nan,
                "G1_frac": round(fr["G1"], 3), "S_frac": round(fr["S"], 3),
                "G2_frac": round(fr["G2"], 3),
            })
    return pd.DataFrame(rows)


def _rep_fraction(st: pd.DataFrame, construct: str, drug: str, phase: str) -> pd.Series:
    """Per-replicate fraction of `phase` among that well's stateable cells."""
    d = st[(st["construct"] == construct) & (st["drug"] == drug)]
    if not len(d):
        return pd.Series(dtype=float)
    return d.groupby("replicate")["phase"].apply(lambda s: (s == phase).mean())


def drug_response_table(df: pd.DataFrame) -> pd.DataFrame:
    """PRIMARY endpoint. For each drug, does the TARGET quadrant fraction rise vs DMSO?
      Palbo target = G1 (CDK4/6i -> G1 arrest) ; RO target = G2 (CDK1i -> G2 arrest).
    Scored on stateable cells; replicate-level fractions give the spread. Flagged
    not-reliable where the construct's red channel is not gateable (cannot call S/G2)."""
    st = df[df["assignable"]]
    rows = []
    for c in cfg.CONSTRUCTS:
        gateable = red_gateable(df, c)
        for drug in ["Palbo", "RO"]:
            target = DRUG_TARGET_PHASE[drug]
            base_r = _rep_fraction(st, c, "DMSO", target)
            treat_r = _rep_fraction(st, c, drug, target)
            base = float(base_r.median()) if len(base_r) else np.nan
            treat = float(treat_r.median()) if len(treat_r) else np.nan
            shift = treat - base if (treat == treat and base == base) else np.nan
            correct = bool(shift > 0) if shift == shift else False
            rows.append({
                "construct": c, "drug": drug,
                "target_phase": target, "expected": f"up {target}-frac",
                "dmso_target_frac": round(base, 3) if base == base else np.nan,
                "drug_target_frac": round(treat, 3) if treat == treat else np.nan,
                "target_frac_shift": round(shift, 3) if shift == shift else np.nan,
                "treat_rep_range": (f"{treat_r.min():.2f}-{treat_r.max():.2f}"
                                    if len(treat_r) else "nan"),
                "expected_direction_met": correct,
                # S vs G2 resolution needs a gateable red channel
                "reliable": gateable,
            })
    return pd.DataFrame(rows)


def comparison_summary(bright: pd.DataFrame, drug: pd.DataFrame,
                       phasefrac: pd.DataFrame) -> pd.DataFrame:
    """Cross-construct comparison on the PRIMARY metrics. A usable 2-colour FUCCI
    needs (a) a bright green S-sensor, (b) a red Geminin reporter that is GATEABLE,
    and (c) correct drug-arrest direction in BOTH drugs. Constructs whose red is not
    gateable cannot be ranked as 2-colour reporters (green-only)."""
    rows = []
    for c in cfg.CONSTRUCTS:
        g_bf = bright[(bright.construct == c) & (bright.channel_role == "green")]["bright_frac_dmso"].iloc[0]
        r_row = bright[(bright.construct == c) & (bright.channel_role == "red")].iloc[0]
        r_bf = r_row["bright_frac_dmso"]
        dr = drug[drug.construct == c]
        gateable = bool(dr["reliable"].all())
        n_correct = int(dr["expected_direction_met"].sum()) if gateable else np.nan
        mean_shift = float(dr["target_frac_shift"].mean()) if gateable else np.nan
        rows.append({
            "construct": c,
            "green488_bright_frac": g_bf,
            "red_bright_frac": r_bf,
            "red_gateable": gateable,
            "drug_dirs_correct_of_2": n_correct,
            "mean_target_frac_shift": round(mean_shift, 3) if mean_shift == mean_shift else np.nan,
        })
    out = pd.DataFrame(rows)
    # Rank only the constructs with a gateable red channel (true 2-colour reporters).
    rel = out[out["red_gateable"]].copy()
    if len(rel):
        rel["score"] = (rel["green488_bright_frac"].rank()
                        + rel["red_bright_frac"].rank()
                        + rel["drug_dirs_correct_of_2"].rank()
                        + rel["mean_target_frac_shift"].rank())
        rel = rel.sort_values("score", ascending=False).reset_index(drop=True)
        rel["rank"] = (rel.index + 1).astype(object)
    incomplete = out[~out["red_gateable"]].copy()
    incomplete["score"] = np.nan
    incomplete["rank"] = "not 2-colour gateable (green-only)"
    return pd.concat([rel, incomplete], ignore_index=True)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_summary_md(bright, sep, phasefrac, drug, comp, erosion: int) -> None:
    lines = [
        "# FUCCI Variant Characterization — Results Summary",
        f"\n_Dataset: 20260610_fucci_constructs_test · erosion={erosion}px · gate SNR≥"
        f"{cfg.GATE_SNR} · auto-generated by variant_analyze.py_\n",
        "> **Reframed 2026-06-10 (literature-driven).** Constructs are PIP-FUCCI-type: "
        "green/488 = PIP = a **direct S-phase sensor** (HIGH in G1 *and* G2, LOW in S); "
        "red = Geminin (S/G2/M). Phase is read from a **2D quadrant gate** "
        "(green-only=G1, red-only=S, double-positive=G2), NOT the old 1D log2(red/green) "
        "ratio — which mis-modelled G2 and produced artifacts (D900 RO 'weak', pBJ284 "
        "Palbo 'wrong'). See library/cell_cycle/reporters/notes.md.\n",
        "**Primary endpoint:** drug shifts the TARGET quadrant fraction up — "
        "Palbo (CDK4/6i) → ↑G1, RO (CDK1i) → ↑G2 (double-positive). Scored on stateable "
        "cells (≥1 reporter SNR≥%.0f), replicate-level fractions.\n" % cfg.GATE_SNR,
        "## Caveats (honor in interpretation)",
        f"- A construct can only call **S vs G2** if its red is **gateable** (DMSO red "
        f"bright-frac ≥ {cfg.MIN_RED_GATE_FRAC}). Where it is not, the gate collapses to "
        "green-only and S/G2 are not callable — flagged per row.",
        f"- Objective **NA 0.4** (sidecar yaml says {cfg.SIDECAR_NA}; stale config field).",
        "- **pBJ285 = LABELLING-FAILED** (SiRhP far-red ≈ background): red excluded, "
        "green-only.",
        "- Absolute per-cell SNR is dilution-limited (BF whole-cell mask ≫ nucleus at "
        f"10x); SNR rises monotonically with erosion. RELATIVE comparison is valid; "
        f"default erosion = {cfg.EROSION_DEFAULT_PX}px.",
        "- Illumination vignetting → per-cell LOCAL background (not flat-field).\n",
        "## Cross-construct comparison (ranked; only red-gateable constructs ranked)\n",
        _md_table(comp),
        "\n## Drug response — PRIMARY (Palbo→↑G1, RO→↑G2 double-positive)\n",
        _md_table(drug),
        "\n## Phase fractions (G1/S/G2 among stateable cells)\n",
        _md_table(phasefrac),
        "\n## Brightness / SNR\n",
        _md_table(bright),
        "\n## Channel separation (DMSO green-vs-red anti-corr)\n",
        _md_table(sep),
        "\n## Figures\n- figures/scatter_<construct>.png  (points coloured by gated phase)"
        "\n- figures/phasefrac_<construct>.png  (G1/S/G2 stacked by drug)"
        "\n- figures/brightness_488_compare.png\n",
    ]
    (cfg.ANALYSIS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--erosion", type=int, default=cfg.EROSION_DEFAULT_PX)
    args = ap.parse_args()

    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load(args.erosion)
    print(f"[ANALYZE] erosion={args.erosion}px, gate SNR>={cfg.GATE_SNR}, "
          f"cells={len(df)}, stateable={int(df['assignable'].sum())}")
    for c in cfg.CONSTRUCTS:
        print(f"  {c}: red_gateable={red_gateable(df, c)}")

    for c in cfg.CONSTRUCTS:
        fig_scatter(df, c, cfg.FIG_DIR / f"scatter_{c}.png")
        fig_phasefrac(df, c, cfg.FIG_DIR / f"phasefrac_{c}.png")
    fig_brightness_488(df, cfg.FIG_DIR / "brightness_488_compare.png")

    bright = brightness_table(df); bright.to_csv(cfg.ANALYSIS_DIR / "brightness_snr_summary.csv", index=False)
    sep = separation_table(df); sep.to_csv(cfg.ANALYSIS_DIR / "separation_summary.csv", index=False)
    phasefrac = phase_fraction_table(df); phasefrac.to_csv(cfg.ANALYSIS_DIR / "phase_fraction_summary.csv", index=False)
    drug = drug_response_table(df); drug.to_csv(cfg.ANALYSIS_DIR / "drug_response_summary.csv", index=False)
    comp = comparison_summary(bright, drug, phasefrac); comp.to_csv(cfg.ANALYSIS_DIR / "variant_comparison_summary.csv", index=False)
    write_summary_md(bright, sep, phasefrac, drug, comp, args.erosion)

    print("\n=== PHASE FRACTIONS ==="); print(phasefrac.to_string(index=False))
    print("\n=== DRUG RESPONSE (PRIMARY) ==="); print(drug.to_string(index=False))
    print("\n=== COMPARISON (ranked) ==="); print(comp.to_string(index=False))
    print(f"\n[SAVED] tables + figures under {cfg.ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
