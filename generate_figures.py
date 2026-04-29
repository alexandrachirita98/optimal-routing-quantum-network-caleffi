"""Run all Figures methods and save outputs into ./images/."""

from __future__ import annotations

import os
import inspect
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.handlers.figures import Figures


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(out_dir, exist_ok=True)

    figs = Figures()

    methods = [
        name for name, _ in inspect.getmembers(figs, predicate=inspect.ismethod)
        if not name.startswith("_")
    ]

    for name in methods:
        method = getattr(figs, name)
        save_path = os.path.join(out_dir, f"{name}.png")
        print(f"[+] Running {name} → {save_path}")
        try:
            method(show=False, save_path=save_path)
        except TypeError:
            method()
            plt.savefig(save_path, dpi=150)
        plt.close("all")

    print(f"\nDone. Figures saved in: {out_dir}")


if __name__ == "__main__":
    main()
