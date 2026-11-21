"""Draw the README figure of session alignment, in light and dark themes.

    uv run python docs/media/make_figures.py

Reads the cross-subject accuracies from ``research/results/alignment.md`` (three
classes, montage C) and writes ``alignment-{light,dark}.png`` next to this script.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
RESULTS = OUT.parents[1] / "research" / "results" / "alignment.md"
MODELS = ["csp-lda", "mdm", "ts-lr", "eegnet"]
CONDITIONS = {
    "none": "no alignment",
    "euclid": "Euclidean alignment",
    "riemann": "Riemannian re-centering",
}
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e4e3df",
        "none": "#a3a29c",
        "euclid": "#2a78d6",
        "riemann": "#1baf7a",
    },
    "dark": {
        "bg": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "grid": "#34332f",
        "none": "#6f6e69",
        "euclid": "#3987e5",
        "riemann": "#199e70",
    },
}
CHANCE = 100 / 3


def read_results() -> dict[str, list[float]]:
    """Cross-subject mean accuracy per condition, in the order of MODELS."""
    text = RESULTS.read_text(encoding="utf-8")
    result = {}
    for condition in CONDITIONS:
        section = text.split(f"## bnci2014_001-3class-c-{condition}\n", 1)[1].split("\n## ", 1)[0]
        scores = {}
        for model, cross_subject in re.findall(
            r"^\| `([\w-]+)` \| .*? \| ([\d.]+) ±", section, re.M
        ):
            scores[model] = float(cross_subject)
        result[condition] = [scores[model] for model in MODELS]
    return result


def draw(theme_name: str, theme: dict[str, str], data: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=110, facecolor=theme["bg"])
    ax.set_facecolor(theme["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=theme["ink2"], length=0, labelsize=11)
    x = np.arange(len(MODELS))
    width = 0.26
    for k, (condition, label) in enumerate(CONDITIONS.items()):
        positions = x + (k - 1) * width
        values = data[condition]
        ax.bar(positions, values, width=width - 0.03, color=theme[condition], label=label, zorder=2)
        for position, value in zip(positions, values, strict=True):
            ax.text(
                position,
                value + 0.8,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=theme["ink2"],
            )
    ax.axhline(CHANCE, color=theme["ink2"], lw=1, ls=(0, (4, 3)), zorder=3)
    ax.set_xlim(-0.55, 4.1)
    ax.text(
        4.08, CHANCE + 0.8, "chance 33 %", ha="right", va="bottom", fontsize=9, color=theme["ink2"]
    )
    ax.set_xticks(x, MODELS)
    ax.set_ylim(0, 75)
    ax.set_yticks([0, 20, 40, 60])
    ax.yaxis.grid(True, color=theme["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("accuracy (%)", color=theme["ink2"])
    ax.set_title(
        "Decoding a new user without calibration (cross-subject, 3 classes, mean of 9 subjects)",
        color=theme["ink"],
        loc="left",
        fontsize=12,
    )
    ax.legend(frameon=False, labelcolor=theme["ink"], fontsize=10, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"alignment-{theme_name}.png", facecolor=theme["bg"])
    plt.close(fig)


def main() -> None:
    data = read_results()
    for name, theme in THEMES.items():
        draw(name, theme, data)


if __name__ == "__main__":
    main()
