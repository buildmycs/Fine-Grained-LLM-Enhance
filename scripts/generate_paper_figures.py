"""Generate publication-ready figures for the dual-text ALMT paper.

Outputs are written to ``figures/paper`` in SVG, PDF, and 600-DPI PNG.
The SVG/PDF files remain editable and should be preferred for typesetting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures" / "paper"

COLORS = {
    "ink": "#172033",
    "muted": "#59657A",
    "line": "#8793A6",
    "paper": "#FFFFFF",
    "panel": "#F7F9FC",
    "almt": "#DDEBFA",
    "almt_edge": "#3974A8",
    "proposed": "#F2E7FA",
    "proposed_edge": "#7A4EAB",
    "llm": "#FFF0D8",
    "llm_edge": "#C97819",
    "audio": "#DDF3ED",
    "audio_edge": "#27806E",
    "visual": "#E6F1E2",
    "visual_edge": "#54813F",
    "bad": "#FCE1E3",
    "bad_edge": "#B83A45",
    "good": "#DCF3E5",
    "good_edge": "#237A4B",
    "neutral": "#EEF1F5",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.linewidth": 0.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def canvas(figsize=(7.2, 4.8), xlim=(0, 16), ylim=(0, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def rounded_box(
    ax,
    xy,
    width,
    height,
    text="",
    *,
    facecolor=None,
    edgecolor=None,
    fontsize=7.0,
    weight="normal",
    color=None,
    radius=0.13,
    linewidth=1.0,
    linestyle="-",
    zorder=2,
    pad=0.08,
    ha="center",
    va="center",
):
    facecolor = facecolor or COLORS["paper"]
    edgecolor = edgecolor or COLORS["line"]
    color = color or COLORS["ink"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        if ha == "left":
            text_x = xy[0] + 0.18
        elif ha == "right":
            text_x = xy[0] + width - 0.18
        else:
            text_x = xy[0] + width / 2
        ax.text(
            text_x,
            xy[1] + height / 2,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            weight=weight,
            color=color,
            zorder=zorder + 1,
            linespacing=1.22,
        )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=None,
    linewidth=1.05,
    linestyle="-",
    mutation_scale=8,
    connectionstyle="arc3",
    zorder=1,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color or COLORS["line"],
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def small_label(ax, x, y, text, *, color=None, fontsize=5.9, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color or COLORS["muted"], zorder=6)


def section_label(ax, x, y, text, color):
    ax.text(x, y, text.upper(), ha="left", va="center", fontsize=5.4,
            weight="bold", color=color)


def save_all(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def _draw_framework_detailed() -> None:
    fig, ax = canvas(figsize=(7.2, 4.65), xlim=(0, 16), ylim=(0, 10.2))

    # Background groups communicate provenance without dominating the pipeline.
    rounded_box(
        ax, (0.12, 6.35), 8.0, 3.45,
        facecolor="#FCF9FE", edgecolor=COLORS["proposed_edge"],
        linewidth=0.8, linestyle="--", radius=0.16, zorder=0,
    )
    section_label(ax, 0.35, 9.55, "Proposed dual-text semantic enhancement", COLORS["proposed_edge"])

    rounded_box(
        ax, (3.45, 0.68), 9.68, 5.20,
        facecolor="#F8FBFE", edgecolor=COLORS["almt_edge"],
        linewidth=0.8, linestyle="--", radius=0.16, zorder=0,
    )
    section_label(ax, 3.70, 5.62, "ALMT backbone", COLORS["almt_edge"])

    rounded_box(
        ax, (13.25, 0.68), 2.62, 8.15,
        facecolor="#FCF9FE", edgecolor=COLORS["proposed_edge"],
        linewidth=0.8, linestyle="--", radius=0.16, zorder=0,
    )
    section_label(ax, 13.45, 8.55, "Proposed objectives", COLORS["proposed_edge"])

    # Inputs and offline LLM enhancement.
    rounded_box(ax, (0.35, 7.55), 1.55, 0.85, "Raw text\n$x^r$",
                facecolor=COLORS["neutral"], fontsize=6.0, weight="bold")
    rounded_box(ax, (0.35, 8.65), 1.55, 0.64, "Local context\nC4",
                facecolor=COLORS["neutral"], fontsize=6.0)
    rounded_box(ax, (2.25, 8.10), 1.65, 1.05, "Local Qwen2.5\nC4-Explicit",
                facecolor=COLORS["llm"], edgecolor=COLORS["llm_edge"],
                fontsize=6.6, weight="bold")
    rounded_box(ax, (4.23, 8.35), 1.55, 0.72, "Enhanced text\n$x^e$",
                facecolor=COLORS["llm"], edgecolor=COLORS["llm_edge"], fontsize=6.5)
    arrow(ax, (1.90, 8.98), (2.24, 8.85), color=COLORS["llm_edge"])
    arrow(ax, (1.90, 7.98), (2.24, 8.40), color=COLORS["llm_edge"])
    arrow(ax, (3.90, 8.62), (4.22, 8.69), color=COLORS["llm_edge"])

    rounded_box(ax, (4.25, 6.73), 1.52, 1.25, "Shared BERT\n+ shared $W_l$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.8, weight="bold")
    arrow(ax, (1.90, 7.82), (4.24, 7.20), color=COLORS["proposed_edge"],
          connectionstyle="arc3,rad=0.06")
    arrow(ax, (5.00, 8.35), (5.00, 7.99), color=COLORS["proposed_edge"])

    rounded_box(ax, (6.18, 6.83), 1.65, 1.08, "Gated cross-\nattention",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=6.0, weight="bold")
    small_label(ax, 7.00, 6.56, r"$Q=H_o,\;K=V=H_e$", color=COLORS["proposed_edge"])
    arrow(ax, (5.78, 7.35), (6.17, 7.35), color=COLORS["proposed_edge"])

    rounded_box(ax, (8.33, 6.70), 1.75, 1.34,
                "Multi-scale\nlanguage states\n$H_l^1,H_l^2,H_l^3$",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=5.35, weight="bold")
    arrow(ax, (7.84, 7.35), (8.32, 7.35), color=COLORS["almt_edge"])

    # Audio / visual embedding streams.
    rounded_box(ax, (0.35, 4.15), 1.55, 0.72, "Audio  $x^a$",
                facecolor=COLORS["audio"], edgecolor=COLORS["audio_edge"],
                fontsize=6.7, weight="bold")
    rounded_box(ax, (0.35, 2.70), 1.55, 0.72, "Visual  $x^v$",
                facecolor=COLORS["visual"], edgecolor=COLORS["visual_edge"],
                fontsize=6.7, weight="bold")
    rounded_box(ax, (3.82, 3.97), 1.82, 1.08, "Audio embedding\nTransformer",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=5.2, weight="bold")
    rounded_box(ax, (3.82, 2.52), 1.82, 1.08, "Visual embedding\nTransformer",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=5.2, weight="bold")
    arrow(ax, (1.90, 4.51), (3.81, 4.51), color=COLORS["audio_edge"])
    arrow(ax, (1.90, 3.06), (3.81, 3.06), color=COLORS["visual_edge"])

    rounded_box(ax, (6.30, 2.78), 2.18, 1.72,
                "Adaptive Hyper-modality\nLearning (AHL)\n3 guided stages",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=5.0, weight="bold")
    arrow(ax, (5.65, 4.50), (6.29, 4.05), color=COLORS["audio_edge"])
    arrow(ax, (5.65, 3.06), (6.29, 3.30), color=COLORS["visual_edge"])
    arrow(ax, (9.05, 6.70), (7.78, 4.51), color=COLORS["almt_edge"],
          connectionstyle="arc3,rad=-0.12")
    small_label(ax, 8.38, 5.25, "language guidance", color=COLORS["almt_edge"])

    rounded_box(ax, (9.03, 3.08), 1.63, 1.15, "Hyper-modality\n$H_h$",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=5.7, weight="bold")
    arrow(ax, (8.49, 3.64), (9.02, 3.64), color=COLORS["almt_edge"])

    rounded_box(ax, (10.91, 5.10), 1.83, 1.42,
                "Cross-modality\nFusion Transformer",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=5.4, weight="bold")
    small_label(ax, 11.82, 4.84, r"$Q=H_l^3,\;K=V=H_h$", color=COLORS["almt_edge"])
    arrow(ax, (10.09, 7.20), (10.90, 6.05), color=COLORS["almt_edge"])
    arrow(ax, (10.66, 3.64), (11.50, 5.09), color=COLORS["almt_edge"])
    rounded_box(ax, (11.16, 7.05), 1.34, 0.67, "Fused $z$",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=6.7, weight="bold")
    arrow(ax, (11.82, 6.53), (11.82, 7.04), color=COLORS["almt_edge"])

    # Prediction heads and training objectives.
    rounded_box(ax, (13.48, 6.85), 2.15, 0.85, "Regression head\n" + r"$\hat y_{reg}$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.6, weight="bold")
    rounded_box(ax, (13.48, 5.60), 2.15, 0.90, "Ordinal head\n6 ordered thresholds",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.0, weight="bold")
    rounded_box(ax, (13.48, 4.25), 2.15, 0.95, "Intensity projection\n$h_i$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.0, weight="bold")
    arrow(ax, (12.50, 7.38), (13.47, 7.28), color=COLORS["proposed_edge"])
    arrow(ax, (12.50, 7.32), (13.47, 6.05), color=COLORS["proposed_edge"])
    arrow(ax, (12.50, 7.24), (13.47, 4.74), color=COLORS["proposed_edge"])

    rounded_box(ax, (13.48, 2.82), 2.15, 0.90,
                "Inference calibration\n" + r"$\hat y=(1-\rho)\hat y_{reg}+\rho\hat y_{ord}$",
                facecolor=COLORS["good"], edgecolor=COLORS["good_edge"],
                fontsize=4.8, weight="bold")
    arrow(ax, (14.60, 6.85), (14.60, 3.73), color=COLORS["proposed_edge"])
    arrow(ax, (14.60, 5.60), (14.60, 3.73), color=COLORS["proposed_edge"])

    rounded_box(ax, (13.48, 1.05), 2.15, 1.25,
                r"$\mathcal{L}=\lambda_r\mathcal{L}_{reg}$" + "\n"
                + r"$+\,w(t)\lambda_o\mathcal{L}_{ord}$" + "\n"
                + r"$+\,w(t)\lambda_c\mathcal{L}_{con}$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.5, weight="bold")
    arrow(ax, (14.60, 4.25), (14.60, 2.31), color=COLORS["proposed_edge"])
    arrow(ax, (14.23, 5.60), (14.14, 2.31), color=COLORS["proposed_edge"],
          connectionstyle="arc3,rad=0.10")
    arrow(ax, (14.05, 6.85), (13.92, 2.31), color=COLORS["proposed_edge"],
          connectionstyle="arc3,rad=0.13")
    small_label(ax, 14.56, 0.83, "warm-up + class balance", fontsize=4.8)

    # Compact legend and equation for the gated residual.
    ax.text(0.35, 1.65, "Gated residual text fusion", fontsize=6.2,
            weight="bold", color=COLORS["proposed_edge"], ha="left")
    ax.text(0.35, 1.22,
            r"$C=\mathrm{MHA}(H_o,H_e,H_e),\quad G=\sigma(W[H_o;C;|H_o-C|;H_o\odot C])$",
            fontsize=4.9, color=COLORS["ink"], ha="left")
    ax.text(0.35, 0.83,
            r"$H_{dual}=\mathrm{LN}(H_o+\sigma(\alpha)\,G\odot C)$",
            fontsize=5.7, color=COLORS["ink"], ha="left")

    ax.add_patch(Rectangle((7.95, 1.20), 0.25, 0.25, facecolor=COLORS["almt"],
                           edgecolor=COLORS["almt_edge"], linewidth=0.8))
    ax.text(8.32, 1.32, "Original ALMT", fontsize=4.8, va="center", color=COLORS["muted"])
    ax.add_patch(Rectangle((9.65, 1.20), 0.25, 0.25, facecolor=COLORS["proposed"],
                           edgecolor=COLORS["proposed_edge"], linewidth=0.8))
    ax.text(10.02, 1.32, "Proposed", fontsize=4.8, va="center", color=COLORS["muted"])
    ax.add_patch(Rectangle((11.02, 1.20), 0.25, 0.25, facecolor=COLORS["llm"],
                           edgecolor=COLORS["llm_edge"], linewidth=0.8))
    ax.text(11.39, 1.32, "Offline LLM", fontsize=4.8, va="center", color=COLORS["muted"])

    save_all(fig, "fig1_proposed_framework_detailed")


def draw_framework() -> None:
    """Draw the compact paper-facing architecture diagram."""
    fig, ax = canvas(figsize=(7.2, 3.85), xlim=(0, 16), ylim=(0.25, 8.35))

    # Stage labels.
    ax.text(0.35, 8.03, "OFFLINE SEMANTIC ENHANCEMENT", fontsize=6.0,
            weight="bold", color=COLORS["llm_edge"], ha="left")
    ax.plot([0.35, 5.95], [7.82, 7.82], color=COLORS["llm_edge"],
            linewidth=0.8, alpha=0.5)
    ax.text(6.55, 8.03, "MULTIMODAL SENTIMENT MODEL", fontsize=6.0,
            weight="bold", color=COLORS["proposed_edge"], ha="left")
    ax.plot([6.55, 15.65], [7.82, 7.82], color=COLORS["proposed_edge"],
            linewidth=0.8, alpha=0.5)

    # Offline enhancement and dual-text pathway.
    rounded_box(ax, (0.35, 6.50), 1.45, 0.65, "Context  C4",
                facecolor=COLORS["neutral"], fontsize=6.0)
    rounded_box(ax, (0.35, 5.20), 1.45, 0.75, "Raw text",
                facecolor=COLORS["neutral"], fontsize=6.4, weight="bold")
    rounded_box(ax, (2.15, 6.35), 1.85, 0.95, "Qwen2.5\nC4-Explicit",
                facecolor=COLORS["llm"], edgecolor=COLORS["llm_edge"],
                fontsize=6.2, weight="bold")
    rounded_box(ax, (4.35, 6.48), 1.55, 0.70, "Enhanced text",
                facecolor=COLORS["llm"], edgecolor=COLORS["llm_edge"],
                fontsize=5.8, weight="bold")
    arrow(ax, (1.80, 6.82), (2.14, 6.91), color=COLORS["llm_edge"])
    arrow(ax, (1.80, 5.72), (2.14, 6.57), color=COLORS["llm_edge"],
          connectionstyle="arc3,rad=-0.12")
    arrow(ax, (4.00, 6.83), (4.34, 6.83), color=COLORS["llm_edge"])

    rounded_box(ax, (4.15, 5.05), 1.95, 0.85, "Shared text\nencoder",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.8, weight="bold")
    arrow(ax, (1.80, 5.57), (4.14, 5.48), color=COLORS["proposed_edge"])
    arrow(ax, (5.12, 6.48), (5.12, 5.91), color=COLORS["proposed_edge"])

    rounded_box(ax, (6.65, 5.05), 1.90, 0.85, "Gated dual-text\nfusion",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.8, weight="bold")
    arrow(ax, (6.11, 5.48), (6.64, 5.48), color=COLORS["proposed_edge"])

    rounded_box(ax, (9.00, 5.05), 1.85, 0.85, "Multi-scale\nlanguage encoder",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=5.7, weight="bold")
    arrow(ax, (8.56, 5.48), (8.99, 5.48), color=COLORS["almt_edge"])

    # Audio and visual streams.
    rounded_box(ax, (0.35, 3.18), 1.45, 0.68, "Audio",
                facecolor=COLORS["audio"], edgecolor=COLORS["audio_edge"],
                fontsize=6.4, weight="bold")
    rounded_box(ax, (0.35, 1.75), 1.45, 0.68, "Visual",
                facecolor=COLORS["visual"], edgecolor=COLORS["visual_edge"],
                fontsize=6.4, weight="bold")
    rounded_box(ax, (2.20, 3.05), 1.90, 0.90, "Audio encoder",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=5.8, weight="bold")
    rounded_box(ax, (2.20, 1.62), 1.90, 0.90, "Visual encoder",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=5.8, weight="bold")
    arrow(ax, (1.80, 3.52), (2.19, 3.52), color=COLORS["audio_edge"])
    arrow(ax, (1.80, 2.09), (2.19, 2.09), color=COLORS["visual_edge"])

    rounded_box(ax, (6.55, 2.25), 1.85, 1.05, "AHL",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=9.0, weight="bold")
    small_label(ax, 7.47, 1.92, "language-guided", color=COLORS["almt_edge"], fontsize=5.4)
    arrow(ax, (4.10, 3.50), (6.54, 2.98), color=COLORS["audio_edge"],
          connectionstyle="arc3,rad=-0.05")
    arrow(ax, (4.10, 2.08), (6.54, 2.56), color=COLORS["visual_edge"],
          connectionstyle="arc3,rad=0.05")
    arrow(ax, (9.75, 5.04), (8.02, 3.31), color=COLORS["almt_edge"],
          connectionstyle="arc3,rad=-0.10")

    rounded_box(ax, (10.15, 3.15), 1.95, 1.00, "Cross-modal\nfusion",
                facecolor=COLORS["almt"], edgecolor=COLORS["almt_edge"],
                fontsize=6.0, weight="bold")
    arrow(ax, (8.41, 2.78), (10.14, 3.50), color=COLORS["almt_edge"])
    arrow(ax, (10.85, 5.44), (11.24, 4.16), color=COLORS["almt_edge"],
          connectionstyle="arc3,rad=0.08")

    rounded_box(ax, (12.70, 3.15), 1.75, 1.00, "Regression &\nordinal heads",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.7, weight="bold")
    arrow(ax, (12.11, 3.65), (12.69, 3.65), color=COLORS["proposed_edge"])
    rounded_box(ax, (14.65, 3.25), 1.10, 0.80, "Sentiment\nscore",
                facecolor=COLORS["good"], edgecolor=COLORS["good_edge"],
                fontsize=5.1, weight="bold")
    arrow(ax, (14.46, 3.65), (14.64, 3.65), color=COLORS["good_edge"])

    rounded_box(ax, (10.15, 1.15), 4.30, 0.75,
                "Multi-objective training\nregression  ·  ordinal  ·  contrastive",
                facecolor="#FCF9FE", edgecolor=COLORS["proposed_edge"],
                fontsize=5.35, weight="bold", linestyle="--")
    arrow(ax, (13.57, 3.14), (13.00, 1.91), color=COLORS["proposed_edge"],
          linestyle="--", connectionstyle="arc3,rad=0.08")

    save_all(fig, "fig1_proposed_framework")


def draw_framework_vertical() -> None:
    """Draw a bottom-up architecture aligned with the source paper's Figure 2."""
    fig, ax = canvas(figsize=(7.2, 7.1), xlim=(0, 16), ylim=(0.15, 15.65))

    language_fill, language_edge = "#FFF0B8", "#C58A12"
    audio_fill, audio_edge = "#D9EAF7", "#3974A8"
    visual_fill, visual_edge = "#E3F0D9", "#54813F"
    ahl_fill, ahl_edge = "#FBE2D3", "#C56E38"
    fusion_fill, fusion_edge = "#E7E4FA", "#6252A5"
    innovation_fill, innovation_edge = "#F1E4FA", "#7A4EAB"
    panel_fill, panel_edge = "#FBFCFE", "#AEB8C7"

    def panel(x, y, w, h):
        return rounded_box(
            ax, (x, y), w, h, facecolor=panel_fill, edgecolor=panel_edge,
            linewidth=0.75, radius=0.18, zorder=0, pad=0.04,
        )

    def right_stage(y, label, color=COLORS["muted"]):
        ax.text(14.62, y, label, ha="center", va="center", fontsize=6.0,
                weight="bold", color=color, linespacing=1.25)

    def new_badge(x, y):
        ax.text(
            x, y, "NEW", ha="center", va="center", fontsize=4.5,
            weight="bold", color="white", zorder=8,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=innovation_edge,
                      edgecolor=innovation_edge, linewidth=0.0),
        )

    # Layer containers follow the bottom-to-top hierarchy of the source figure.
    panel(0.35, 0.42, 13.35, 1.48)
    panel(0.35, 2.12, 13.35, 2.28)
    panel(0.35, 4.62, 13.35, 4.36)
    panel(0.35, 9.20, 13.35, 1.90)
    panel(0.35, 11.32, 13.35, 4.03)
    right_stage(1.16, "MULTIMODAL\nINPUT")
    right_stage(3.25, "MODALITY\nEMBEDDING")
    right_stage(6.80, "ADAPTIVE\nHYPER-MODALITY\nLEARNING", ahl_edge)
    right_stage(10.15, "CROSS-MODAL\nFUSION", fusion_edge)
    right_stage(13.34, "PREDICTION", innovation_edge)

    # Bottom row: original text, enhanced text, audio, and video.
    rounded_box(ax, (0.72, 0.78), 2.15, 0.72, "Original text  $x^r$",
                facecolor=language_fill, edgecolor=language_edge,
                fontsize=5.9, weight="bold")
    rounded_box(ax, (3.18, 0.72), 2.15, 0.84,
                "Enhanced text  $x^e$\nC4-Explicit (offline)",
                facecolor=COLORS["llm"], edgecolor=COLORS["llm_edge"],
                fontsize=5.1, weight="bold")
    rounded_box(ax, (7.05, 0.78), 2.05, 0.72, "Audio  $x^a$",
                facecolor=audio_fill, edgecolor=audio_edge,
                fontsize=6.0, weight="bold")
    rounded_box(ax, (10.18, 0.78), 2.05, 0.72, "Video  $x^v$",
                facecolor=visual_fill, edgecolor=visual_edge,
                fontsize=6.0, weight="bold")
    # Parameter-shared dual-language encoding and modality embedding.
    rounded_box(ax, (0.88, 2.45), 4.35, 0.83,
                "Shared BERT + shared\nlanguage projection",
                facecolor=innovation_fill, edgecolor=innovation_edge,
                fontsize=5.35, weight="bold")
    new_badge(1.18, 3.20)
    arrow(ax, (1.79, 1.50), (1.95, 2.44), color=language_edge)
    arrow(ax, (4.26, 1.50), (4.15, 2.44), color=COLORS["llm_edge"])
    small_label(ax, 1.78, 3.38, "$H_o$", color=innovation_edge, fontsize=5.3)
    small_label(ax, 4.32, 3.38, "$H_e$", color=innovation_edge, fontsize=5.3)

    rounded_box(ax, (1.20, 3.48), 3.72, 0.72,
                "Gated cross-attention\n" + r"$Q=H_o,\;K=V=H_e$  +  original residual",
                facecolor=innovation_fill, edgecolor=innovation_edge,
                fontsize=4.85, weight="bold")
    arrow(ax, (1.78, 3.29), (2.12, 3.47), color=innovation_edge)
    arrow(ax, (4.32, 3.29), (4.02, 3.47), color=innovation_edge)

    rounded_box(ax, (6.98, 2.62), 2.22, 0.92, "Audio embedding\nTransformer",
                facecolor=audio_fill, edgecolor=audio_edge,
                fontsize=5.5, weight="bold")
    rounded_box(ax, (10.10, 2.62), 2.22, 0.92, "Visual embedding\nTransformer",
                facecolor=visual_fill, edgecolor=visual_edge,
                fontsize=5.5, weight="bold")
    arrow(ax, (8.08, 1.50), (8.08, 2.61), color=audio_edge)
    arrow(ax, (11.20, 1.50), (11.20, 2.61), color=visual_edge)
    small_label(ax, 8.08, 3.82, "$H_a^1$", color=audio_edge, fontsize=5.2)
    small_label(ax, 11.20, 3.82, "$H_v^1$", color=visual_edge, fontsize=5.2)

    # Three language scales guide three stacked AHL stages.
    rounded_box(ax, (1.32, 5.68), 3.15, 0.68, "Language Transformer",
                facecolor=language_fill, edgecolor=language_edge,
                fontsize=5.7, weight="bold")
    rounded_box(ax, (1.32, 7.04), 3.15, 0.68, "Language Transformer",
                facecolor=language_fill, edgecolor=language_edge,
                fontsize=5.7, weight="bold")
    arrow(ax, (3.06, 4.20), (2.90, 5.67), color=language_edge)
    arrow(ax, (2.90, 6.37), (2.90, 7.03), color=language_edge)
    arrow(ax, (2.90, 7.73), (2.90, 8.66), color=language_edge)

    rounded_box(ax, (7.55, 4.70), 3.95, 0.38,
                "Initial hyper-modality  $H_{hyper}^{0}$",
                facecolor=COLORS["neutral"], edgecolor=panel_edge,
                fontsize=4.85, weight="bold", pad=0.04)
    arrow(ax, (8.08, 3.55), (8.55, 4.69), color=audio_edge,
          connectionstyle="arc3,rad=-0.06")
    arrow(ax, (11.20, 3.55), (10.50, 4.69), color=visual_edge,
          connectionstyle="arc3,rad=0.06")

    ahl_boxes = [
        (5.20, "AHL stage 1", "$H_{hyper}^{1}$"),
        (6.56, "AHL stage 2", "$H_{hyper}^{2}$"),
        (7.92, "AHL stage 3", "$H_{hyper}^{3}$"),
    ]
    for y, title, output in ahl_boxes:
        rounded_box(ax, (7.03, y), 5.02, 0.68,
                    f"{title}\nlanguage-guided update",
                    facecolor=ahl_fill, edgecolor=ahl_edge,
                    fontsize=5.15, weight="bold")
        small_label(ax, 12.42, y + 0.34, output, color=ahl_edge,
                    fontsize=4.8, ha="left")

    arrow(ax, (9.53, 5.09), (9.53, 5.19), color=ahl_edge)
    arrow(ax, (9.53, 5.89), (9.53, 6.55), color=ahl_edge)
    arrow(ax, (9.53, 7.25), (9.53, 7.91), color=ahl_edge)

    scale_y = [5.54, 6.90, 8.26]
    scale_labels = ["$H_l^1$", "$H_l^2$", "$H_l^3$"]
    for y, label in zip(scale_y, scale_labels):
        arrow(ax, (4.48, y), (7.02, y), color=language_edge)
        small_label(ax, 4.78, y + 0.23, label, color=language_edge,
                    fontsize=5.2)

    # Cross-modal fusion retains the query / key-value roles.
    rounded_box(ax, (4.55, 9.72), 6.25, 0.78,
                "Cross-modality Fusion Transformer",
                facecolor=fusion_fill, edgecolor=fusion_edge,
                fontsize=6.1, weight="bold")
    arrow(ax, (2.90, 8.67), (5.85, 9.71), color=language_edge,
          connectionstyle="arc3,rad=-0.10")
    arrow(ax, (9.53, 8.61), (9.15, 9.71), color=ahl_edge,
          connectionstyle="arc3,rad=0.06")
    small_label(ax, 5.18, 9.38, "$H_l^3$: Query",
                color=language_edge, fontsize=5.0)
    small_label(ax, 9.74, 9.38, "$H_{hyper}^3$: Key & Value",
                color=ahl_edge, fontsize=5.0)

    rounded_box(ax, (6.30, 10.62), 2.75, 0.38, "Final feature  $z$",
                facecolor=fusion_fill, edgecolor=fusion_edge,
                fontsize=5.3, weight="bold", pad=0.04)
    arrow(ax, (7.67, 10.51), (7.67, 10.61), color=fusion_edge)

    # Final feature branches into regression, ordinal, and training-only heads.
    rounded_box(ax, (2.15, 12.10), 2.60, 0.72, "Regression head",
                facecolor=fusion_fill, edgecolor=fusion_edge,
                fontsize=5.8, weight="bold")
    rounded_box(ax, (5.65, 12.10), 2.75, 0.72,
                "Ordinal head\n6 ordered thresholds",
                facecolor=innovation_fill, edgecolor=innovation_edge,
                fontsize=5.2, weight="bold")
    new_badge(5.92, 12.75)
    rounded_box(ax, (9.28, 12.10), 2.75, 0.72,
                "Intensity projection\ncontrastive learning",
                facecolor=innovation_fill, edgecolor=innovation_edge,
                fontsize=5.0, weight="bold", linestyle="--")
    small_label(ax, 10.65, 11.78, "training only",
                color=innovation_edge, fontsize=4.7)

    arrow(ax, (7.67, 11.01), (3.45, 12.09), color=fusion_edge,
          connectionstyle="arc3,rad=0.08")
    arrow(ax, (7.67, 11.01), (7.02, 12.09), color=innovation_edge)
    arrow(ax, (7.67, 11.01), (10.65, 12.09), color=innovation_edge,
          linestyle="--", connectionstyle="arc3,rad=-0.08")

    rounded_box(ax, (4.65, 13.28), 4.55, 0.62,
                "Regression-ordinal prediction calibration",
                facecolor=innovation_fill, edgecolor=innovation_edge,
                fontsize=5.45, weight="bold")
    arrow(ax, (3.45, 12.83), (5.55, 13.27), color=fusion_edge,
          connectionstyle="arc3,rad=-0.08")
    arrow(ax, (7.02, 12.83), (7.02, 13.27), color=innovation_edge)

    rounded_box(ax, (5.18, 14.38), 3.48, 0.64,
                "Sentiment prediction\nnegative  ↔  neutral  ↔  positive",
                facecolor=COLORS["good"], edgecolor=COLORS["good_edge"],
                fontsize=5.25, weight="bold")
    arrow(ax, (6.92, 13.91), (6.92, 14.37), color=COLORS["good_edge"])

    save_all(fig, "fig1_proposed_framework")


def conversation_card(ax, *, enhanced=False):
    rounded_box(ax, (0.45, 1.05), 7.05, 6.33, facecolor=COLORS["panel"],
                edgecolor="#C8D0DC", radius=0.18, linewidth=0.9, zorder=0)
    ax.text(0.78, 6.93, "LOCAL DISCOURSE CONTEXT", fontsize=6.3, weight="bold",
            color=COLORS["muted"], ha="left")
    rounded_box(ax, (0.78, 5.78), 5.65, 0.72,
                'Turn -2   “The trailer looked exciting.”',
                facecolor=COLORS["paper"], edgecolor="#D6DCE5", fontsize=6.45,
                ha="left")
    rounded_box(ax, (1.25, 4.65), 5.82, 0.82,
                'Turn -1   “But the plot went nowhere.\nI kept checking the time.”',
                facecolor=COLORS["paper"], edgecolor="#D6DCE5", fontsize=6.1,
                ha="left")
    target_face = COLORS["llm"] if enhanced else COLORS["bad"]
    target_edge = COLORS["llm_edge"] if enhanced else COLORS["bad_edge"]
    rounded_box(ax, (0.78, 3.28), 6.28, 0.92,
                'Target      “Yeah, that was just great.”',
                facecolor=target_face, edgecolor=target_edge, fontsize=6.7,
                weight="bold", ha="left", linewidth=1.15)
    ax.text(0.80, 2.75, "Audio", fontsize=6.1, weight="bold", color=COLORS["audio_edge"], ha="left")
    ax.plot([1.50, 1.74, 1.96, 2.18, 2.40, 2.62, 2.84, 3.06, 3.28, 3.50],
            [2.73, 2.61, 2.82, 2.66, 2.78, 2.64, 2.76, 2.68, 2.74, 2.69],
            color=COLORS["audio_edge"], linewidth=1.0)
    ax.text(3.72, 2.73, "subtle / ambiguous prosody", fontsize=6.0,
            color=COLORS["muted"], ha="left", va="center")
    ax.text(0.80, 2.04, "Visual", fontsize=6.1, weight="bold", color=COLORS["visual_edge"], ha="left")
    ax.plot([1.55, 1.95, 2.35, 2.75, 3.15, 3.50],
            [2.01, 2.04, 2.00, 2.03, 2.01, 2.02],
            color=COLORS["visual_edge"], linewidth=1.0, marker="o", markersize=1.8)
    ax.text(3.72, 2.04, "weak facial evidence", fontsize=6.0,
            color=COLORS["muted"], ha="left", va="center")
    ax.text(0.78, 1.42, "Illustrative MOSI-style example (not a verbatim dataset sample)",
            fontsize=5.4, color=COLORS["muted"], style="italic", ha="left")


def result_card(ax, *, correct: bool):
    edge = COLORS["good_edge"] if correct else COLORS["bad_edge"]
    face = COLORS["good"] if correct else COLORS["bad"]
    symbol = "✓" if correct else "×"
    label = "CORRECT" if correct else "POLARITY ERROR"
    predicted = "Negative  (-2)" if correct else "Positive  (+1)"
    rounded_box(ax, (12.45, 2.05), 3.05, 4.38, facecolor=face, edgecolor=edge,
                radius=0.20, linewidth=1.2, zorder=1)
    ax.text(13.98, 5.90, symbol, fontsize=25, weight="bold", color=edge,
            ha="center", va="center")
    ax.text(13.98, 5.16, label, fontsize=7.2, weight="bold", color=edge,
            ha="center")
    ax.text(12.82, 4.36, "Prediction", fontsize=6.0, color=COLORS["muted"], ha="left")
    ax.text(12.82, 3.92, predicted, fontsize=8.0, weight="bold", color=COLORS["ink"], ha="left")
    ax.plot([12.82, 15.13], [3.56, 3.56], color=edge, alpha=0.35, linewidth=0.8)
    ax.text(12.82, 3.14, "Reference", fontsize=6.0, color=COLORS["muted"], ha="left")
    ax.text(12.82, 2.70, "Negative  (-2)", fontsize=8.0, weight="bold",
            color=COLORS["ink"], ha="left")


def intro_header(ax, panel, title, subtitle, color):
    ax.text(0.45, 7.98, panel, fontsize=7.0, weight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor=color))
    ax.text(1.04, 8.02, title, fontsize=10.0, weight="bold", color=COLORS["ink"],
            ha="left", va="center")
    ax.text(1.04, 7.64, subtitle, fontsize=6.2, color=COLORS["muted"],
            ha="left", va="center")


def draw_raw_failure() -> None:
    fig, ax = canvas(figsize=(7.2, 3.75), xlim=(0, 16), ylim=(0.6, 8.55))
    intro_header(
        ax, "A", "Surface text can reverse the predicted polarity",
        "Unresolved reference and implicit sarcasm make a negative utterance appear positive.",
        COLORS["bad_edge"],
    )
    conversation_card(ax, enhanced=False)

    arrow(ax, (7.50, 4.28), (8.05, 4.28), color=COLORS["bad_edge"], linewidth=1.25)
    rounded_box(ax, (8.08, 5.18), 3.55, 1.10,
                "Surface lexical cue\n“great”  →  positive",
                facecolor=COLORS["bad"], edgecolor=COLORS["bad_edge"],
                fontsize=7.1, weight="bold")
    rounded_box(ax, (8.08, 3.55), 3.55, 1.10,
                "Unresolved reference\nWhat does “that” denote?",
                facecolor=COLORS["neutral"], edgecolor=COLORS["line"],
                fontsize=6.8)
    rounded_box(ax, (8.08, 1.92), 3.55, 1.10,
                "Implicit meaning\nSarcastic intent\nremains hidden",
                facecolor=COLORS["neutral"], edgecolor=COLORS["line"],
                fontsize=6.2)
    arrow(ax, (11.64, 5.73), (12.44, 4.95), color=COLORS["bad_edge"])
    arrow(ax, (11.64, 4.10), (12.44, 4.32), color=COLORS["line"])
    arrow(ax, (11.64, 2.47), (12.44, 3.72), color=COLORS["line"])
    result_card(ax, correct=False)

    ax.text(9.85, 1.28, "Text ambiguity dominates when nonverbal evidence is weak.",
            fontsize=5.8, color=COLORS["bad_edge"], weight="bold", ha="center")
    save_all(fig, "fig2_raw_text_failure")


def draw_llm_correction() -> None:
    fig, ax = canvas(figsize=(7.2, 3.75), xlim=(0, 16), ylim=(0.6, 8.55))
    intro_header(
        ax, "B", "C4-Explicit enhancement restores the intended polarity",
        "Context-grounded rewriting resolves reference, disambiguates pragmatics, and states the implicit sentiment.",
        COLORS["good_edge"],
    )
    conversation_card(ax, enhanced=True)

    arrow(ax, (7.50, 4.28), (7.93, 4.28), color=COLORS["llm_edge"], linewidth=1.25)
    rounded_box(ax, (7.90, 4.86), 4.32, 1.75,
                "C4-Explicit rewrite\n“‘That’ refers to the movie.\n‘Great’ is sarcastic; the speaker\nactually found the movie\nvery disappointing.”",
                facecolor=COLORS["llm"], edgecolor=COLORS["llm_edge"],
                fontsize=4.75, weight="bold", linewidth=1.15)

    chips = [
        ("Coreference", 7.96, 4.12, 1.18),
        ("Disambiguation", 9.30, 4.12, 1.42),
        ("Implicit → explicit", 10.88, 4.12, 1.48),
    ]
    for text, x, y, w in chips:
        rounded_box(ax, (x, y), w, 0.43, text, facecolor=COLORS["paper"],
                    edgecolor=COLORS["llm_edge"], fontsize=4.45, weight="bold",
                    radius=0.10, pad=0.04)

    rounded_box(ax, (8.10, 2.30), 1.65, 0.72, "Raw branch\n$H_o$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.5, weight="bold")
    rounded_box(ax, (10.08, 2.30), 1.78, 0.72, "Enhanced branch\n$H_e$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.25, weight="bold")
    rounded_box(ax, (9.12, 1.18), 1.68, 0.66, "Gated fusion\n$H_{dual}$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=6.1, weight="bold")
    arrow(ax, (8.92, 2.29), (9.50, 1.85), color=COLORS["proposed_edge"])
    arrow(ax, (10.97, 2.29), (10.42, 1.85), color=COLORS["proposed_edge"])
    arrow(ax, (11.95, 5.45), (12.44, 4.92), color=COLORS["llm_edge"])
    arrow(ax, (10.81, 1.51), (12.44, 3.40), color=COLORS["proposed_edge"],
          connectionstyle="arc3,rad=-0.15")
    result_card(ax, correct=True)

    save_all(fig, "fig3_llm_text_correction")


def intensity_conversation_card(ax, *, enhanced=False):
    rounded_box(ax, (0.45, 1.05), 7.05, 6.33, facecolor=COLORS["panel"],
                edgecolor="#C8D0DC", radius=0.18, linewidth=0.9, zorder=0)
    ax.text(0.78, 6.93, "LOCAL DISCOURSE CONTEXT", fontsize=6.3, weight="bold",
            color=COLORS["muted"], ha="left")
    rounded_box(ax, (0.78, 5.78), 5.65, 0.72,
                'Turn -2   “I had waited months for this sequel.”',
                facecolor=COLORS["paper"], edgecolor="#D6DCE5", fontsize=6.1,
                ha="left")
    rounded_box(ax, (1.25, 4.55), 5.82, 0.92,
                'Turn -1   “The story made no sense. The acting was painful.\n'
                'The ending ruined the entire series.”',
                facecolor=COLORS["paper"], edgecolor="#D6DCE5", fontsize=5.35,
                ha="left")
    target_face = COLORS["llm"] if enhanced else "#FFF0D8"
    target_edge = COLORS["llm_edge"]
    rounded_box(ax, (0.78, 3.18), 6.28, 0.92,
                'Target      “I did not really enjoy it.”',
                facecolor=target_face, edgecolor=target_edge, fontsize=6.55,
                weight="bold", ha="left", linewidth=1.15)

    ax.text(0.80, 2.67, "Audio", fontsize=6.1, weight="bold",
            color=COLORS["audio_edge"], ha="left")
    ax.plot([1.50, 1.74, 1.96, 2.18, 2.40, 2.62, 2.84, 3.06, 3.28, 3.50],
            [2.65, 2.57, 2.70, 2.59, 2.67, 2.58, 2.66, 2.60, 2.65, 2.61],
            color=COLORS["audio_edge"], linewidth=1.0)
    ax.text(3.72, 2.65, "restrained prosody", fontsize=6.0,
            color=COLORS["muted"], ha="left", va="center")
    ax.text(0.80, 1.98, "Visual", fontsize=6.1, weight="bold",
            color=COLORS["visual_edge"], ha="left")
    ax.plot([1.55, 1.95, 2.35, 2.75, 3.15, 3.50],
            [1.96, 1.98, 1.95, 1.97, 1.95, 1.96],
            color=COLORS["visual_edge"], linewidth=1.0,
            marker="o", markersize=1.8)
    ax.text(3.72, 1.98, "subtle facial evidence", fontsize=6.0,
            color=COLORS["muted"], ha="left", va="center")
    ax.text(0.78, 1.42,
            "Illustrative MOSI-style example (not a verbatim dataset sample)",
            fontsize=5.4, color=COLORS["muted"], style="italic", ha="left")


def intensity_result_card(ax, *, correct):
    edge = COLORS["good_edge"] if correct else COLORS["llm_edge"]
    face = COLORS["good"] if correct else COLORS["llm"]
    symbol = "✓" if correct else "!"
    label = "CORRECT LEVEL" if correct else "INTENSITY ERROR"
    predicted = "Strong negative\n(-3)" if correct else "Mild negative\n(-1)"
    rounded_box(ax, (12.45, 2.05), 3.05, 4.38, facecolor=face, edgecolor=edge,
                radius=0.20, linewidth=1.2, zorder=1)
    ax.text(13.98, 5.90, symbol, fontsize=23, weight="bold", color=edge,
            ha="center", va="center")
    ax.text(13.98, 5.16, label, fontsize=7.0, weight="bold", color=edge,
            ha="center")
    ax.text(13.98, 4.46, "Prediction", fontsize=6.0,
            color=COLORS["muted"], ha="center")
    ax.text(13.98, 3.93, predicted, fontsize=6.25, weight="bold",
            color=COLORS["ink"], ha="center", va="center", linespacing=1.05)
    ax.plot([12.82, 15.13], [3.40, 3.40], color=edge, alpha=0.35, linewidth=0.8)
    ax.text(13.98, 3.02, "Reference", fontsize=6.0,
            color=COLORS["muted"], ha="center")
    ax.text(13.98, 2.52, "Strong negative\n(-3)", fontsize=6.25, weight="bold",
            color=COLORS["ink"], ha="center", va="center", linespacing=1.05)


def draw_raw_intensity_error() -> None:
    fig, ax = canvas(figsize=(7.2, 3.75), xlim=(0, 16), ylim=(0.6, 8.55))
    intro_header(
        ax, "C", "Correct polarity, wrong sentiment intensity",
        "Under-specified target wording compresses a strongly negative opinion into a mild negative class.",
        COLORS["llm_edge"],
    )
    intensity_conversation_card(ax, enhanced=False)

    arrow(ax, (7.50, 4.20), (8.05, 4.20), color=COLORS["llm_edge"], linewidth=1.25)
    rounded_box(ax, (8.08, 5.18), 3.55, 1.10,
                "Surface wording\n“did not really enjoy”\n→ mild negative",
                facecolor=COLORS["llm"], edgecolor=COLORS["llm_edge"],
                fontsize=6.1, weight="bold")
    rounded_box(ax, (8.08, 3.55), 3.55, 1.10,
                "Contextual intensity omitted\nmonths of anticipation\n+ multiple severe failures",
                facecolor=COLORS["neutral"], edgecolor=COLORS["line"],
                fontsize=5.7)
    rounded_box(ax, (8.08, 1.92), 3.55, 1.10,
                "Polarity is negative,\nbut its severity\nremains implicit",
                facecolor=COLORS["neutral"], edgecolor=COLORS["line"],
                fontsize=6.0)
    arrow(ax, (11.64, 5.73), (12.44, 4.95), color=COLORS["llm_edge"])
    arrow(ax, (11.64, 4.10), (12.44, 4.32), color=COLORS["line"])
    arrow(ax, (11.64, 2.47), (12.44, 3.72), color=COLORS["line"])
    intensity_result_card(ax, correct=False)

    ax.text(9.85, 0.78, "Polarity is preserved; only the ordinal level is underestimated.",
            fontsize=5.65, color=COLORS["llm_edge"], weight="bold", ha="center")
    save_all(fig, "fig4_raw_text_intensity_error")


def draw_llm_intensity_correction() -> None:
    fig, ax = canvas(figsize=(7.2, 3.75), xlim=(0, 16), ylim=(0.6, 8.55))
    intro_header(
        ax, "D", "C4-Explicit restores the correct sentiment level",
        "Context-grounded enhancement makes severity explicit while preserving the original negative polarity.",
        COLORS["good_edge"],
    )
    intensity_conversation_card(ax, enhanced=True)

    arrow(ax, (7.50, 4.20), (7.92, 4.20), color=COLORS["llm_edge"], linewidth=1.25)
    rounded_box(ax, (7.90, 4.82), 4.32, 1.80,
                "C4-Explicit rewrite\n“The sequel was deeply disappointing.\n"
                "Its incoherent story, poor acting,\n"
                "and ruined ending made the speaker\n"
                "strongly dislike it.”",
                facecolor=COLORS["llm"], edgecolor=COLORS["llm_edge"],
                fontsize=5.1, weight="bold", linewidth=1.15)

    chips = [
        ("Context cues", 7.90, 4.08, 1.25),
        ("Explicit intensity", 9.34, 4.08, 1.38),
        ("Polarity kept", 10.91, 4.08, 1.25),
    ]
    for text, x, y, w in chips:
        rounded_box(ax, (x, y), w, 0.43, text,
                    facecolor=COLORS["paper"], edgecolor=COLORS["llm_edge"],
                    fontsize=3.95, weight="bold", radius=0.10, pad=0.04)

    rounded_box(ax, (8.10, 2.30), 1.65, 0.72, "Raw branch\n$H_o$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.5, weight="bold")
    rounded_box(ax, (10.08, 2.30), 1.78, 0.72, "Enhanced branch\n$H_e$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=5.25, weight="bold")
    rounded_box(ax, (9.12, 1.18), 1.68, 0.66, "Gated fusion\n$H_{dual}$",
                facecolor=COLORS["proposed"], edgecolor=COLORS["proposed_edge"],
                fontsize=6.1, weight="bold")
    arrow(ax, (8.92, 2.29), (9.50, 1.85), color=COLORS["proposed_edge"])
    arrow(ax, (10.97, 2.29), (10.42, 1.85), color=COLORS["proposed_edge"])
    arrow(ax, (12.23, 5.45), (12.44, 4.92), color=COLORS["llm_edge"])
    arrow(ax, (10.81, 1.51), (12.44, 3.40), color=COLORS["proposed_edge"],
          connectionstyle="arc3,rad=-0.15")
    intensity_result_card(ax, correct=True)

    save_all(fig, "fig5_llm_intensity_correction")


def main() -> None:
    configure_style()
    draw_framework_vertical()
    draw_raw_failure()
    draw_llm_correction()
    draw_raw_intensity_error()
    draw_llm_intensity_correction()
    print(f"Generated paper figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
