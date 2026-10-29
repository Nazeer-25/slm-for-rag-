"""Render the report's equations as 600-dpi PNGs (STIX = Times-like math).

The same LaTeX strings are printed in the report's appendix so the paper can
typeset them natively.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures"
plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})

EQUATIONS = {
    "eq1_score": r"$s(c)\;=\;\max_{w\,\in\,\mathcal{W}(c)}\;p_{\theta}(\mathrm{inj}\mid w)$",
    "eq2_decision": r"$\hat{y}(c)\;=\;1\ \ \mathrm{if}\ \ s(c)\geq\tau,\ \ \mathrm{otherwise}\ \ 0$",
    "eq3_threshold": r"$\tau^{*}\;=\;\arg\max_{\tau}\ \mathrm{TPR}_{\mathrm{val}}(\tau)\quad\mathrm{subject\ to}\quad\mathrm{FPR}_{\mathrm{val}}(\tau)\leq\alpha$",
    "eq4_wilson": r"$\dfrac{\hat{p}+\dfrac{z^{2}}{2n}\ \pm\ z\sqrt{\dfrac{\hat{p}(1-\hat{p})}{n}+\dfrac{z^{2}}{4n^{2}}}}{1+\dfrac{z^{2}}{n}},\qquad z=1.96$",
}

for name, tex in EQUATIONS.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=13)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03, transparent=False,
                facecolor="white")
    plt.close(fig)
print("wrote", ", ".join(EQUATIONS))
