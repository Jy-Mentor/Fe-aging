"""Project-wide color palettes for publication-quality figures.

Compatible with matplotlib, seaborn, and plotly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PALETTE_PATH = _PROJECT_ROOT / "paper" / "palettes.json"


def load_palettes(path: str | Path | None = None) -> dict[str, Any]:
    """Load all palettes from palettes.json.

    Args:
        path: Path to palettes.json. Defaults to ``paper/palettes.json``.

    Returns:
        Nested dictionary of palette definitions.
    """
    if path is None:
        path = _DEFAULT_PALETTE_PATH
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_palette(name: str, n: int | None = None, path: str | Path | None = None) -> list[str]:
    """Return a palette as a list of hex colors.

    Args:
        name: Palette name, e.g. ``"nature_cancer"``.
        n: Number of colors to return. Colors are recycled if needed.
        path: Path to palettes.json.

    Returns:
        List of hex color strings.
    """
    pals = load_palettes(path)
    if name not in pals:
        available = ", ".join(sorted(pals.keys()))
        raise ValueError(f"Unknown palette: {name}. Available: {available}")
    colors = list(pals[name]["colors"])
    if n is not None:
        colors = [colors[i % len(colors)] for i in range(n)]
    return colors


# Convenience accessors
nature_cancer = get_palette("nature_cancer")
nature_npg = get_palette("nature_npg")
science_aaas = get_palette("science_aaas")
cell_press = get_palette("cell_press")
lancet = get_palette("lancet")
nejm = get_palette("nejm")
jama = get_palette("jama")
okabe_ito = get_palette("okabe_ito")
iron_aging = get_palette("iron_aging")


def register_matplotlib(name: str | None = None) -> None:
    """Register project palettes as named matplotlib ListedColormaps.

    After registration you can use ``plt.get_cmap("nature_cancer")``.
    """
    from matplotlib.colors import ListedColormap

    pals = load_palettes()
    for pal_name, definition in pals.items():
        if pal_name.startswith("_"):
            continue
        colors = definition.get("colors", [])
        if not colors:
            continue
        cmap_name = name if name else pal_name
        ListedColormap(colors, name=cmap_name)


def to_seaborn_palette(name: str, n: int | None = None) -> list[str]:
    """Return a palette suitable for seaborn's ``palette`` argument."""
    return get_palette(name, n=n)


if __name__ == "__main__":
    # Quick sanity check when run directly
    print("Nature Cancer:", nature_cancer)
    print("Okabe-Ito:", okabe_ito)
    print("Available palettes:", ", ".join(sorted(load_palettes().keys())))
