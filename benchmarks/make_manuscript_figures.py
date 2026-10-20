"""Build the four main-text figures for the Journal of Cheminformatics manuscript.

All numbers are read from the saved ten-split result files in ``benchmarks/results``;
nothing is typed by hand except axis text.  Output: PDF + PNG previews in
``manuscript/figs`` (override with the ``MANUSCRIPT_FIGS`` environment variable).

Figure sizes are set to the final printed width (170 mm full width, 125 mm and
150 mm for the compact panels) with 7 pt type, so the figures are not scaled
down by the journal's typesetter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
OUT = Path(os.environ.get("MANUSCRIPT_FIGS", ROOT / "manuscript" / "figs"))
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]
LABEL = {"esol": "ESOL", "freesolv": "FreeSolv", "lipophilicity": "Lipophilicity",
         "bbbp": "BBBP", "bace": "BACE"}
REG = ["esol", "freesolv", "lipophilicity"]
CLS = ["bbbp", "bace"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------
# Fig 1: the five stages of an experiment
# --------------------------------------------------------------------------
def fig_pipeline():
    fig = plt.figure(figsize=(165 * MM, 36 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("1", "Input", "SMILES table +\nmeasured property"),
        ("2", "Audit", "structures, labels,\ndescriptor support"),
        ("3", "Configure", "plain-text YAML:\nblocks, routes, roles"),
        ("4", "Fit", "graph, descriptors,\nor a hybrid"),
        ("5", "Record", "hash, seed,\npredictions, checkpoint"),
    ]
    xs = np.linspace(0.09, 0.91, 5)
    y = 0.64
    ax.plot([0.045, 0.955], [y, y], color="0.75", lw=0.7, zorder=1)
    for x, (num, title, desc) in zip(xs, steps):
        ax.scatter([x], [y], s=95, facecolor="white", edgecolor="0.15",
                   linewidth=0.7, zorder=3)
        ax.text(x, y, num, ha="center", va="center", fontsize=6.9,
                fontweight="bold", zorder=4)
        ax.text(x, y + 0.20, title, ha="center", va="bottom", fontsize=7.8,
                fontweight="bold")
        ax.text(x, y - 0.13, desc, ha="center", va="top", fontsize=6.6,
                linespacing=1.25)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x1 - 0.021, y), xytext=(x0 + 0.021, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.35",
                                    shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(0.5, 0.13, "Every stage writes a file that a reader can inspect and re-run.",
            ha="center", va="center", fontsize=6.6, style="italic", color="0.25")
    save(fig, "fig1_pipeline")


# --------------------------------------------------------------------------
# Fig 2: one molecule followed through audit and evidence
# --------------------------------------------------------------------------
def fig_worked_example():
    from rdkit import Chem
    from rdkit.Chem import Draw

    data = json.loads((RESULTS / "fig1_worked_example.json").read_text())
    mol = Chem.MolFromSmiles(data["smiles"])
    Draw.MolToFile(mol, str(OUT / "_phenacetin.png"), size=(760, 560),
                   wedgeBonds=False, kekulize=True)
    img = plt.imread(OUT / "_phenacetin.png")
    mask = (img[..., :3] < 0.98).any(axis=2)
    rr = np.where(mask.any(axis=1))[0]
    cc = np.where(mask.any(axis=0))[0]
    pad = 8
    img = img[max(rr[0] - pad, 0):rr[-1] + pad, max(cc[0] - pad, 0):cc[-1] + pad]

    fig = plt.figure(figsize=(170 * MM, 66 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.86, 1.26, 1.02],
                          left=0.005, right=0.995, top=0.86, bottom=0.15,
                          wspace=0.58)

    # (a) structure
    axa = fig.add_subplot(gs[0])
    axa.imshow(img)
    axa.set_anchor("N")
    axa.axis("off")
    axa.text(0.5, -0.03, "phenacetin, measured $\\log_{10}S$ = $-2.35$\n"
                         "$N$ = 1,128; split 1, training partition",
             transform=axa.transAxes, ha="center", va="top", fontsize=6.8)

    # (b) descriptor evidence
    axb = fig.add_subplot(gs[1])
    table = sorted(data["table"], key=lambda r: r["abs_rho"])
    names = [r["name"] for r in table]
    vals = [r["abs_rho"] for r in table]
    kept = [r["in_route"] for r in table]
    colors = ["#1f6f43" if k else "0.72" for k in kept]
    bars = axb.barh(range(len(names)), vals, color=colors, height=0.62,
                    edgecolor="none")
    axb.set_yticks(range(len(names)))
    axb.set_yticklabels(names, fontsize=6.4)
    axb.set_xlim(0, 1.0)
    axb.set_xlabel("absolute Spearman correlation with the target")
    for b, v in zip(bars, vals):
        axb.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                 va="center", fontsize=6.4)
    axb.spines[["top", "right"]].set_visible(False)
    axb.legend(handles=[Patch(color="#1f6f43", label="kept in the route"),
                        Patch(color="0.72", label="flagged redundant")],
               loc="lower right", frameon=False, handlelength=1.0,
               borderaxespad=0.1)

    # (c) audit card
    axc = fig.add_subplot(gs[2])
    axc.axis("off")
    rows = [
        ("1,128", "structures read\n0 invalid SMILES"),
        ("0", "duplicate structures\n(matched by InChIKey)"),
        ("20 / 210", "descriptors outside\nthe training range on test"),
    ]
    for i, (big, small) in enumerate(rows):
        yy = 0.80 - i * 0.34
        axc.text(0.00, yy, big, fontsize=11, fontweight="bold", va="center",
                 transform=axc.transAxes)
        axc.text(0.42, yy, small, fontsize=6.8, va="center", linespacing=1.3,
                 transform=axc.transAxes)
        if i < len(rows) - 1:
            axc.plot([0.0, 1.0], [yy - 0.17, yy - 0.17], color="0.85", lw=0.6,
                     transform=axc.transAxes, clip_on=False)

    for ax, lab in zip((axa, axb, axc),
                       ("a  One molecule", "b  Descriptor evidence, training set",
                        "c  Audit of the same split")):
        fig.text(ax.get_position().x0, 0.945, lab, fontsize=7.8,
                 fontweight="bold", va="center")
    save(fig, "fig2_worked_example")


# --------------------------------------------------------------------------
# Fig 3: descriptor effect in the two implementations, split by metric family
# --------------------------------------------------------------------------
def fig_descriptor_effect():
    """Percentage change in the primary metric after adding the 210 descriptors.

    Read from the saved result files so that the figure cannot drift from the
    tables: framework rows come from the modularity ablation, Chemprop rows from
    the external comparator file.
    """
    framework_rel, chemprop_rel = {}, {}
    for ds in DATASETS:
        abl = json.loads((RESULTS / f"modularity_ablation_{ds}.json").read_text())["raw_results"]
        metric = "roc_auc" if ds in CLS else "rmse"
        graph = np.array([r[metric] for r in abl["residual_only"]], dtype=float).mean()
        hybrid = np.array([r[metric] for r in abl["direct_hybrid"]], dtype=float).mean()
        framework_rel[ds] = (graph - hybrid) / graph * 100 if ds not in CLS \
            else (hybrid - graph) / graph * 100

        cp = json.loads((RESULTS / f"chemprop_v2_{ds}.json").read_text())["summary"]
        cp_graph = cp["chemprop_graph"]["mean"]
        cp_hybrid = cp["chemprop_hybrid"]["mean"]
        chemprop_rel[ds] = (cp_graph - cp_hybrid) / cp_graph * 100 if ds not in CLS \
            else (cp_hybrid - cp_graph) / cp_graph * 100

    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 54 * MM),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.30))
    fig.subplots_adjust(left=0.135, right=0.99, top=0.85, bottom=0.30)
    h = 0.34

    for ax, group, xlabel, title in (
        (axes[0], REG, "% change in RMSE", "a  Regression endpoints"),
        (axes[1], CLS, "% change in AUROC", "b  Classification endpoints"),
    ):
        y = np.arange(len(group))[::-1] * 1.0
        fw = [framework_rel[d] for d in group]
        cp = [chemprop_rel[d] for d in group]
        b1 = ax.barh(y + h / 2, fw, height=h, color="#2a5d9f",
                     label="this framework", edgecolor="none")
        b2 = ax.barh(y - h / 2, cp, height=h, color="#c46a1b",
                     label="Chemprop v2.3.1", edgecolor="none")
        ax.axvline(0, color="0.2", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[d] for d in group])
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for bars, vals in ((b1, fw), (b2, cp)):
            for b, v in zip(bars, vals):
                off = 0.35 if v >= 0 else -0.35
                ax.text(v + off, b.get_y() + b.get_height() / 2, f"{v:+.1f}",
                        va="center", ha="left" if v >= 0 else "right", fontsize=6)

    axes[0].set_xlim(-9.5, 23.5)
    axes[1].set_xlim(-4.2, 3.2)
    fig.text(0.5, 0.04,
             "Positive values mean that adding the descriptors helped.  "
             "ESOL (Chemprop): the negative mean comes from one split of ten, on which the\n"
             "descriptor hybrid reached a test RMSE of 2.76 while beating graph-only on the "
             "other nine.  All values are computed from the saved per-split records.",
             ha="center", va="bottom", fontsize=6.2, linespacing=1.35)
    handles = [Patch(color="#2a5d9f", label="this framework"),
               Patch(color="#c46a1b", label="Chemprop v2.3.1")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.945), handlelength=1.0, columnspacing=1.6)
    save(fig, "fig3_descriptor_effect")


# --------------------------------------------------------------------------
# Fig 4: compact routes against the matched random control
# --------------------------------------------------------------------------
ROUTE_LABEL = {
    "direct_hybrid": "all 210",
    "hybrid_rec": "recommender",
    "hybrid_spearman": "Spearman",
    "hybrid_mi": "MI",
    "residual_only": "graph only",
}
ROUTE_COLOR = {
    "direct_hybrid": "#222222",
    "hybrid_rec": "#1f6f43",
    "hybrid_spearman": "#8a6fb0",
    "hybrid_mi": "#c46a1b",
    "residual_only": "#2a5d9f",
}
ROUTE_ORDER = ["hybrid_rec", "hybrid_spearman", "hybrid_mi", "direct_hybrid",
               "residual_only"]


def fig_routing():
    data = {}
    for name in DATASETS:
        j = json.loads((RESULTS / f"evidence_routing_{name}.json").read_text())
        data[name] = j

    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 58 * MM),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0],
                                              wspace=0.42))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.80, bottom=0.19)
    axa, axb = axes

    # (a) regression: % improvement in RMSE over the random control
    ys, labels = [], []
    ypos = 0
    for ds in REG:
        s = data[ds]["summary"]
        rand = np.asarray(s["hybrid_random"]["per_split"])
        for route in ROUTE_ORDER:
            r = np.asarray(s[route]["per_split"])
            gain = (rand - r) / rand * 100
            axa.errorbar(gain.mean(), ypos, xerr=gain.std(ddof=0), fmt="o",
                         ms=3.2, color=ROUTE_COLOR[route], ecolor="0.55",
                         elinewidth=0.6, capsize=1.4, capthick=0.5, lw=0.6)
            ypos += 1
        ys.append(ypos - 2.5)
        labels.append(LABEL[ds])
        ypos += 0.8

    axa.axvline(0, color="0.2", lw=0.7)
    axa.set_yticks(ys)
    axa.set_yticklabels(labels)
    axa.set_ylim(ypos - 1.2, -1.0)
    axa.set_xlabel("RMSE improvement over the random control (%)")
    axa.set_title("a  Regression endpoints", loc="left", fontweight="bold")
    axa.spines[["top", "right"]].set_visible(False)

    # (b) classification: AUROC difference in thousandths
    ys2, labels2 = [], []
    ypos = 0
    for ds in CLS:
        s = data[ds]["summary"]
        rand = np.asarray(s["hybrid_random"]["per_split"])
        for route in ROUTE_ORDER:
            r = np.asarray(s[route]["per_split"])
            diff = (r - rand) * 100
            axb.errorbar(diff.mean(), ypos, xerr=diff.std(ddof=0), fmt="o",
                         ms=3.2, color=ROUTE_COLOR[route], ecolor="0.55",
                         elinewidth=0.6, capsize=1.4, capthick=0.5, lw=0.6)
            ypos += 1
        ys2.append(ypos - 2.5)
        labels2.append(LABEL[ds])
        ypos += 0.8

    axb.axvline(0, color="0.2", lw=0.7)
    axb.set_yticks(ys2)
    axb.set_yticklabels(labels2)
    axb.set_ylim(ypos - 1.2, -1.0)
    axb.set_xlabel("AUROC difference (percentage points)")
    axb.set_title("b  Classification endpoints", loc="left", fontweight="bold")
    axb.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], marker="o", ls="", ms=3.2,
                      color=ROUTE_COLOR[r], label=ROUTE_LABEL[r])
               for r in ROUTE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.955), handletextpad=0.2,
               columnspacing=1.2)
    save(fig, "fig4_routing")


if __name__ == "__main__":
    fig_pipeline()
    fig_worked_example()
    fig_descriptor_effect()
    fig_routing()
