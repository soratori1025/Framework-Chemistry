import io

p = "benchmarks/make_manuscript_figures.py"
s = io.open(p, encoding="utf-8").read()

pairs = [
    ("fontsize=6.5,\n                fontweight=\"bold\", zorder=4",
     "fontsize=6.9,\n                fontweight=\"bold\", zorder=4"),
    ("ax.text(x, y + 0.20, title, ha=\"center\", va=\"bottom\", fontsize=7.5,",
     "ax.text(x, y + 0.20, title, ha=\"center\", va=\"bottom\", fontsize=7.8,"),
    ("ax.text(x, y - 0.13, desc, ha=\"center\", va=\"top\", fontsize=6.2,",
     "ax.text(x, y - 0.13, desc, ha=\"center\", va=\"top\", fontsize=6.6,"),
    ("ha=\"center\", va=\"center\", fontsize=6.4, style=\"italic\", color=\"0.25\")",
     "ha=\"center\", va=\"center\", fontsize=6.6, style=\"italic\", color=\"0.25\")"),
    ("transform=axa.transAxes, ha=\"center\", va=\"top\", fontsize=6.2)",
     "transform=axa.transAxes, ha=\"center\", va=\"top\", fontsize=6.8)"),
    ("axb.set_yticklabels(names, fontsize=5.8)",
     "axb.set_yticklabels(names, fontsize=6.4)"),
    ("                 va=\"center\", fontsize=6)",
     "                 va=\"center\", fontsize=6.4)"),
    ("axc.text(0.42, yy, small, fontsize=6.2, va=\"center\", linespacing=1.3,",
     "axc.text(0.42, yy, small, fontsize=6.8, va=\"center\", linespacing=1.3,"),
    ("fig.text(ax.get_position().x0, 0.945, lab, fontsize=7.5,",
     "fig.text(ax.get_position().x0, 0.945, lab, fontsize=7.8,"),
]
missing = []
for old, new in pairs:
    if old not in s:
        missing.append(old[:60])
        continue
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched; missing:", missing)
