"""Draw code-faithful, editable ordinal/regression prediction diagrams.

No experiment results are encoded in these architecture figures. Both heads
consume the final fused feature. MSE supervises the blended final prediction.
The ordinal BCE is balanced per cumulative threshold using training labels.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures" / "paper"
PDFS = ROOT / "output" / "pdf"
INK = "#233044"
MUTED = "#627087"
PURPLE = "#7951A5"
PURPLE_BG = "#F1E9F8"
BLUE = "#42759F"
BLUE_BG = "#EAF2FA"
GREEN = "#35765A"
GREEN_BG = "#E7F2EB"
TRAIN = "#A66425"
TRAIN_BG = "#FCF4E8"
EDGE = "#A7B2C2"
TEXT_CHECKS = []


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def canvas(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def label(ax, x, y, value, size=15, color=INK, bold=False, align="center"):
    return ax.text(x, y, value, fontsize=size, color=color,
                   weight="bold" if bold else "normal", ha=align,
                   va="center", linespacing=1.35, zorder=8)


def frame(ax, x, y, w, h, fill="white", edge=EDGE, dashed=False, radius=12,
          lw=1.5):
    item = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill,
        linestyle=(0, (5, 4)) if dashed else "-", zorder=1,
    )
    ax.add_patch(item)
    return item


def node(ax, x, y, w, h, title, formula=None, color=INK, fill="white",
         size=15, formula_size=16):
    rect = frame(ax, x, y, w, h, fill, color)
    texts = []
    if formula is None:
        texts.append(label(ax, x + w / 2, y + h / 2, title, size, bold=True))
    else:
        texts.append(label(ax, x + w / 2, y + h * .29, title, size, bold=True))
        texts.append(label(ax, x + w / 2, y + h * .69, formula, formula_size))
    TEXT_CHECKS.extend((ax, rect, text) for text in texts)
    return rect


def arrow(ax, points, color=INK, train=False, width=1.65, head=14):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    a = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=head,
                        linewidth=width, edgecolor=color, facecolor=color,
                        linestyle=(0, (5, 3.5)) if train else "-", zorder=4)
    ax.add_patch(a)


def branch(ax, x, y, color=INK):
    ax.add_patch(Circle((x, y), 3.5, facecolor=color, edgecolor="none", zorder=5))


def export(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    PDFS.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, rect, text in TEXT_CHECKS:
        if ax.figure is not fig:
            continue
        rb = rect.get_window_extent(renderer)
        tb = text.get_window_extent(renderer)
        if not (rb.x0 + 3 <= tb.x0 and tb.x1 <= rb.x1 - 3
                and rb.y0 + 2 <= tb.y0 and tb.y1 <= rb.y1 - 2):
            raise RuntimeError(f"Label does not fit: {text.get_text()!r}")
    for suffix in ("svg", "png"):
        path = FIGURES / f"{name}.{suffix}"
        fig.savefig(path, dpi=300)
        print(path)
    path = PDFS / f"{name}.pdf"
    fig.savefig(path, metadata={"Title": name.replace("_", " "),
                                "Author": "", "Subject": "Model architecture"})
    print(path)
    plt.close(fig)


def detailed():
    fig, ax = canvas(1820, 1130)
    label(ax, 40, 37, "Ordinal-guided regression", 23, bold=True, align="left")
    arrow(ax, [(1130, 38), (1190, 38)], color=INK)
    label(ax, 1205, 38, "Prediction flow", 12, align="left")
    arrow(ax, [(1430, 38), (1490, 38)], color=TRAIN, train=True)
    label(ax, 1505, 38, "Training only", 12, align="left")

    label(ax, 40, 96, "(a) Joint prediction", 16, bold=True, align="left")
    node(ax, 30, 315, 180, 120, "Fused feature", r"$\mathbf{z}\in\mathbb{R}^{128}$",
         color=EDGE, fill="#F5F7FA", size=15, formula_size=19)
    label(ax, 120, 470, "After cross-modal\nfusion", 12, MUTED)
    arrow(ax, [(210, 375), (250, 375), (250, 204), (310, 204)], color=BLUE)
    arrow(ax, [(250, 375), (250, 494), (315, 494)], color=PURPLE)
    branch(ax, 250, 375)

    node(ax, 310, 148, 410, 112, "Regression head", r"$\hat y_{\mathrm{reg}}=\mathbf{w}_r^\top\mathbf{z}+b_r$",
         color=BLUE, fill=BLUE_BG, size=18, formula_size=20)
    arrow(ax, [(720, 204), (1595, 204), (1595, 310)], color=BLUE)
    label(ax, 1055, 177, "Continuous sentiment estimate", 14, BLUE)
    label(ax, 1652, 258, r"$1-\rho$", 18, BLUE)

    frame(ax, 288, 298, 1117, 370, fill="#FCFAFE", edge=PURPLE, radius=16)
    label(ax, 315, 326, "Monotonic ordinal head", 18, PURPLE, True, "left")
    node(ax, 315, 440, 200, 108, "Shared score", r"$s=\mathbf{w}_o^\top\mathbf{z}+b_o$",
         color=PURPLE, fill=PURPLE_BG, size=14, formula_size=16)
    label(ax, 414, 584, "One score shared across\nall six threshold tasks", 12, MUTED)

    frame(ax, 564, 355, 374, 118, fill=PURPLE_BG, edge=PURPLE)
    label(ax, 751, 378, "Learnable ordered thresholds", 14, bold=True)
    label(ax, 751, 413, r"$\theta_0 < \theta_1 < \cdots < \theta_5$", 19, PURPLE)
    label(ax, 751, 449, r"$\theta_k=\theta_{k-1}+\mathrm{softplus}(\delta_k)+\epsilon$", 13)
    node(ax, 564, 510, 374, 106, "Cumulative decisions", r"$p_k=\sigma(s-\theta_k),\quad k=0,\ldots,5$",
         color=PURPLE, fill="white", size=15, formula_size=17)
    arrow(ax, [(515, 494), (537, 494), (537, 563), (564, 563)], color=PURPLE)
    arrow(ax, [(751, 473), (751, 510)], color=PURPLE)

    node(ax, 981, 501, 150, 116, "Ordered probs.", r"$\mathbf{p}=[p_0,\ldots,p_5]$",
         color=PURPLE, fill=PURPLE_BG, size=12, formula_size=13)
    arrow(ax, [(938, 563), (981, 563)], color=PURPLE)
    label(ax, 1056, 644, r"$p_0\geq p_1\geq\cdots\geq p_5$", 12, PURPLE)

    node(ax, 1170, 502, 205, 116, "Expected score", r"$\hat y_{\mathrm{ord}}=\sum_{k=0}^{5}p_k-3$",
         color=PURPLE, fill=PURPLE_BG, size=14, formula_size=16)
    arrow(ax, [(1131, 559), (1170, 559)], color=PURPLE)
    arrow(ax, [(1375, 559), (1425, 559), (1425, 415), (1450, 415)], color=PURPLE)
    label(ax, 1435, 477, r"$\rho$", 19, PURPLE, align="left")

    node(ax, 1450, 310, 330, 143, "Prediction fusion",
         r"$\hat y=(1-\rho)\hat y_{\mathrm{reg}}+\rho\hat y_{\mathrm{ord}}$",
         color=GREEN, fill=GREEN_BG, size=18, formula_size=17)
    arrow(ax, [(1615, 453), (1615, 510)], color=GREEN)
    node(ax, 1465, 510, 300, 108, "Final sentiment", r"$\hat y$",
         color=GREEN, fill=GREEN_BG, size=17, formula_size=25)

    ax.plot([30, 1790], [711, 711], color="#D1D8E2", linewidth=1.0, zorder=0)
    label(ax, 40, 740, "(b) Balanced ordinal supervision and joint optimization", 16, bold=True, align="left")

    node(ax, 30, 870, 175, 103, "Ground truth", r"$y\in[-3,3]$",
         color=TRAIN, fill=TRAIN_BG, size=14, formula_size=17)
    node(ax, 277, 845, 410, 137, "Cumulative target encoding",
         r"$c=\mathrm{round}(\mathrm{clip}(y,-3,3))+3$" + "\n" + r"$q_k=\mathbb{1}[c>k]$",
         color=TRAIN, fill=TRAIN_BG, size=15, formula_size=15)
    arrow(ax, [(205, 922), (277, 922)], color=TRAIN, train=True)
    node(ax, 814, 774, 350, 88, "Train-set threshold counts",
         r"$w_k^+,\ w_k^-\quad\mathrm{(capped)}$",
         color=TRAIN, fill=TRAIN_BG, size=14, formula_size=15)
    node(ax, 814, 910, 350, 93, "Balanced ordinal BCE", r"$\mathcal{L}_{\mathrm{ord}}(\mathbf{p},\mathbf{q};\mathbf{w})$",
         color=TRAIN, fill=TRAIN_BG, size=16, formula_size=19)
    arrow(ax, [(687, 951), (814, 951)], color=TRAIN, train=True)
    arrow(ax, [(1100, 862), (1100, 910)], color=TRAIN, train=True)
    arrow(ax, [(1131, 596), (1149, 596), (1149, 689), (759, 689), (759, 885), (873, 885), (873, 910)],
          color=TRAIN, train=True)

    label(ax, 1757, 775, r"$y$", 20, TRAIN)
    node(ax, 1465, 831, 300, 101, "Final-output MSE", r"$\mathcal{L}_{\mathrm{mse}}=(\hat y-y)^2$",
         color=TRAIN, fill=TRAIN_BG, size=16, formula_size=18)
    arrow(ax, [(1550, 618), (1550, 831)], color=TRAIN, train=True)
    arrow(ax, [(1735, 787), (1735, 831)], color=TRAIN, train=True)
    node(ax, 1250, 1018, 515, 85, "Joint objective",
         r"$\mathcal{L}=\mathcal{L}_{\mathrm{mse}}+\lambda_{\mathrm{ord}}\,r(e)\,\mathcal{L}_{\mathrm{ord}}$",
         color=TRAIN, fill=TRAIN_BG, size=15, formula_size=19)
    arrow(ax, [(990, 1003), (990, 1062), (1250, 1062)], color=TRAIN, train=True)
    arrow(ax, [(1615, 932), (1615, 1018)], color=TRAIN, train=True)
    label(ax, 35, 1067, r"$r(e)$: ordinal-loss warm-up; threshold weights use training labels only.",
          12, MUTED, align="left")
    export(fig, "ordinal_regression_heads_detail")


def compact():
    fig, ax = canvas(1100, 1030)
    label(ax, 550, 36, "Ordinal-guided regression", 22, bold=True)
    node(ax, 335, 78, 430, 91, "Final sentiment prediction", r"$\hat y$",
         color=GREEN, fill=GREEN_BG, size=16, formula_size=21)
    node(ax, 259, 231, 582, 128, "Regression-ordinal prediction fusion",
         r"$\hat y=(1-\rho)\hat y_{\mathrm{reg}}+\rho\hat y_{\mathrm{ord}}$",
         color=GREEN, fill=GREEN_BG, size=18, formula_size=21)
    arrow(ax, [(550, 231), (550, 169)], color=GREEN)
    node(ax, 60, 507, 365, 137, "Regression head",
         r"$\hat y_{\mathrm{reg}}=\mathbf{w}_r^\top\mathbf{z}+b_r$",
         color=BLUE, fill=BLUE_BG, size=20, formula_size=23)
    frame(ax, 560, 414, 487, 370, fill="#FCFAFE", edge=PURPLE, radius=15, lw=1.7)
    label(ax, 804, 446, "Monotonic ordinal head", 20, PURPLE, True)
    label(ax, 804, 739, r"$s=\mathbf{w}_o^\top\mathbf{z}+b_o$", 21)
    frame(ax, 589, 580, 429, 111, fill=PURPLE_BG, edge=PURPLE)
    label(ax, 804, 608, "Six learnable ordered thresholds", 16, bold=True)
    label(ax, 804, 651, r"$p_k=\sigma(s-\theta_k),\quad \theta_0<\cdots<\theta_5$", 21)
    arrow(ax, [(804, 718), (804, 691)], color=PURPLE, head=12)
    arrow(ax, [(804, 580), (804, 558)], color=PURPLE, head=12)
    label(ax, 804, 511, r"$\hat y_{\mathrm{ord}}=\sum_{k=0}^{5}p_k-3$", 23)
    arrow(ax, [(242, 507), (242, 392), (390, 392), (390, 359)], color=BLUE)
    arrow(ax, [(804, 414), (804, 390), (710, 390), (710, 359)], color=PURPLE)
    label(ax, 188, 407, r"$1-\rho$", 21, BLUE)
    label(ax, 873, 390, r"$\rho$", 22, PURPLE)
    node(ax, 343, 860, 414, 101, "Shared fused representation", r"$\mathbf{z}\in\mathbb{R}^{128}$",
         color=EDGE, fill="#F5F7FA", size=17, formula_size=21)
    arrow(ax, [(550, 860), (550, 825), (242, 825), (242, 644)], color=BLUE)
    arrow(ax, [(550, 825), (804, 825), (804, 784)], color=PURPLE)
    branch(ax, 550, 825)
    label(ax, 550, 1000, "Input from the cross-modal fusion module; inference path shown.", 13, MUTED)
    export(fig, "ordinal_regression_heads_compact")


if __name__ == "__main__":
    setup()
    detailed()
    compact()
