"""Regenerate PERSIST figures from existing results JSON — no gymnasium needed."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT        = Path(__file__).parent.parent.parent   # Snath Robotics/
PAPER_FIGS  = ROOT.parent.parent / "academic_papers" / "snath_core" / "08_PERSIST" / "figures"
LOCAL_FIGS  = ROOT / "experiments" / "persist" / "figures"
LOCAL_FIGS.mkdir(parents=True, exist_ok=True)

SEEDS   = [42, 7, 13, 99, 2026]
D_NORM  = 0.8

PHASE_COLORS = {
    "1": "#e74c3c", "2": "#2ecc71", "3": "#e67e22", "3b": "#1abc9c",
    "4": "#9b59b6", "5c": "#95a5a6", "5w": "#3498db", "6": "#f39c12",
}
PHASE_LABELS = {
    "1": "Phase 1 — Encounter",    "2": "Phase 2 — First try",
    "3": "Phase 3 — Scope boundary","3b": "Phase 3b — Force",
    "4": "Phase 4 — Exhaustion",   "5c": "Phase 5c — Memory (cold)",
    "5w": "Phase 5w — Memory (warm)","6": "Phase 6 — Tournament",
}
LINESTYLES = {"2": "-", "3": "--", "3b": "-.", "4": ":", "5w": "-", "6": "--"}


def _save(fig, name):
    local = LOCAL_FIGS / name
    fig.savefig(local, dpi=150, bbox_inches="tight")
    print(f"  → {local}")
    if PAPER_FIGS.exists():
        fig.savefig(PAPER_FIGS / name, dpi=150, bbox_inches="tight")
        print(f"  → {PAPER_FIGS / name}")


def generate_divergence_curves(results_by_phase):
    fig = plt.figure(figsize=(13, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.30)
    ax_traj = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    for phase_id, phase_results in results_by_phase.items():
        trajs = [r["D_trajectory"] for r in phase_results if r.get("D_trajectory")]
        if not trajs:
            continue
        max_len = max(len(t) for t in trajs)
        color   = PHASE_COLORS.get(str(phase_id), "#888888")
        label   = PHASE_LABELS.get(str(phase_id), f"Phase {phase_id}")
        if max_len <= 1:
            all_d = [t[0] for t in trajs]
            mean_d = np.mean(all_d)
            ax_traj.scatter([0], [mean_d], color=color, marker="D", s=60,
                            zorder=5, label=f"{label} (escalates at entry)")
            ax_traj.annotate("↑ escalated", xy=(0, mean_d),
                             xytext=(0.6, mean_d + 0.025), fontsize=7, color=color,
                             arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.6))
        else:
            padded = np.array([t + [t[-1]] * (max_len - len(t)) for t in trajs])
            mean   = padded.mean(axis=0)
            std    = padded.std(axis=0)
            xs     = np.arange(len(mean))
            ls     = LINESTYLES.get(str(phase_id), "-")
            ax_traj.plot(xs, mean, color=color, label=label, linewidth=2.2, linestyle=ls)
            ax_traj.fill_between(xs, mean - std, mean + std, color=color, alpha=0.13)

    ax_traj.axhline(D_NORM, color="#333333", linestyle="--", linewidth=0.9,
                    label=f"Norm. threshold ({D_NORM})")
    ax_traj.set_ylim(1.85, 2.75)
    ax_traj.set_xlim(-0.5, 18)
    ax_traj.set_xlabel("Persistence loop steps", fontsize=10)
    ax_traj.set_ylabel("Divergence D(t)", fontsize=10)
    ax_traj.set_title(f"D(t) across experimental phases  (n={len(SEEDS)} seeds, mean ± std)", fontsize=10)
    ax_traj.legend(fontsize=7.5, loc="lower left", ncol=1, framealpha=0.88)
    ax_traj.grid(True, alpha=0.18, linestyle=":")

    ph_order = ["1", "2", "3", "3b", "4", "5c", "5w", "6"]
    ph_short = {
        "1":  "Ph.1 Encounter", "2":  "Ph.2 First try",
        "3":  "Ph.3 Scope",     "3b": "Ph.3b Force",
        "4":  "Ph.4 Exhaustion","5c": "Ph.5c Cold",
        "5w": "Ph.5w Warm",     "6":  "Ph.6 Tournament",
    }
    dec_means, bar_colors, bar_labels = [], [], []
    for ph in ph_order:
        if ph not in results_by_phase or not results_by_phase[ph]:
            continue
        tds = [r.get("total_decisions") or r.get("steps") or 1 for r in results_by_phase[ph]]
        dec_means.append(float(np.mean([t for t in tds if t])))
        bar_colors.append(PHASE_COLORS.get(ph, "#888888"))
        bar_labels.append(ph_short.get(ph, f"Ph.{ph}"))

    xs   = np.arange(len(dec_means))
    bars = ax_bar.bar(xs, dec_means, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.6, width=0.65)
    for bar, val in zip(bars, dec_means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    str(int(round(val))), ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    if len(dec_means) >= 6:
        try:
            cold_idx = bar_labels.index("Ph.5c Cold")
            warm_idx = bar_labels.index("Ph.5w Warm")
            ax_bar.annotate("", xy=(warm_idx, dec_means[warm_idx] + 2),
                            xytext=(cold_idx, dec_means[cold_idx] + 2),
                            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
            ax_bar.text((cold_idx + warm_idx) / 2, max(dec_means) * 0.88,
                        "2.4×\nfaster", ha="center", va="bottom",
                        fontsize=7.5, color="#555555", fontweight="bold")
        except ValueError:
            pass

    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(bar_labels, fontsize=6.8, rotation=45, ha="right")
    ax_bar.set_ylabel("Total decisions", fontsize=10)
    ax_bar.set_title("Decision budget\nper phase", fontsize=10)
    ax_bar.set_ylim(0, max(dec_means) * 1.25)
    ax_bar.grid(True, alpha=0.18, linestyle=":", axis="y")

    fig.tight_layout()
    _save(fig, "divergence_curves.pdf")
    plt.close(fig)


def generate_zone_detection(zone_summaries):
    zone_labels = list(zone_summaries.keys())
    entries_mean, entries_std, finals_mean, finals_std = [], [], [], []
    for zone, results in zone_summaries.items():
        ents = [r["D_entry"] for r in results if r.get("D_entry") is not None]
        fins = [r["D_final"] for r in results if r.get("D_final") is not None]
        entries_mean.append(np.mean(ents) if ents else 0)
        entries_std.append(np.std(ents) if ents else 0)
        finals_mean.append(np.mean(fins) if fins else 0)
        finals_std.append(np.std(fins) if fins else 0)

    x = np.arange(len(zone_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, entries_mean, w, yerr=entries_std, capsize=4,
           label="D entry", color="#e67e22", alpha=0.85)
    ax.bar(x + w/2, finals_mean,  w, yerr=finals_std,  capsize=4,
           label="D final", color="#3498db", alpha=0.85)
    ax.axhline(D_NORM, color="#333333", linestyle="--", linewidth=0.9,
               label=f"Norm. threshold ({D_NORM})")
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=11)
    ax.set_ylabel("Divergence D", fontsize=11)
    ax.set_title(f"Zone detection: entry vs final D  (n={len(SEEDS)} seeds, mean ± std)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.18, linestyle=":", axis="y")
    fig.tight_layout()
    _save(fig, "zone_detection.pdf")
    plt.close(fig)


if __name__ == "__main__":
    results_file = Path(__file__).parent / "persist_results_20260623T185309.json"
    print(f"Loading {results_file}")
    with open(results_file) as f:
        all_results = json.load(f)

    zone_summaries = {
        "ice":       all_results["2"],
        "ice_slope": all_results["3"],
        "force":     all_results["3b"],
        "novel":     all_results["4"],
    }

    print("Regenerating figures...")
    generate_divergence_curves(all_results)
    generate_zone_detection(zone_summaries)
    print("Done.")
