"""Export the preregistered R28 epoch curves from inspectable query-macro tables."""
from __future__ import annotations

import csv
import os
import sys

from prepare_stage1_r28 import OUT

sys.path.insert(0,str(OUT / "plot_dependencies"))
os.environ.setdefault("MPLCONFIGDIR",str(OUT / "plot_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def run() -> None:
    with (OUT / "statistics/main_table.csv").open() as handle:
        table = list(csv.DictReader(handle))
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    colors = ("#4477AA","#EE6677","#228833")
    for section,arms,panels in (
        ("student",("S-EDGE-LONG","S-PATH-LONG","S-COV-LONG"),
         (("U","RawRecall","Own U raw recall"),("U_OFFLINE_T0","R@10","Own Full-U + frozen T0 Recall@10"))),
        ("teacher",("T-EDGE-CONT","T-PATH-SPLIT-LSE","T-PATH-SPLIT-COV"),
         (("D","R@10","Teacher QT-only Full-U Recall@10"),("E-LSE","EO_STRICT_hits@10","E-LSE strict EO Top10 retention"),
          ("E-COV","EO_STRICT_hits@10","E-COV strict EO Top10 retention"))),
    ):
        fig,axes = plt.subplots(len(panels),1,figsize=(9,3.6*len(panels)),squeeze=False)
        for axis,(view,metric,title) in zip(axes[:,0],panels):
            for color,arm in zip(colors,arms):
                for seed,style in (("13","-"),("29","--")):
                    points = [r for r in table if r["section"] == section and r["arm"] == arm and r["seed"] == seed
                              and r["view"] == view and r["metric"] == metric and r["kind"] == "overall" and r["condition"] == "Real"
                              and (section != "teacher" or r["budget"] == "Full-U")]
                    if section == "teacher":
                        points += [r for r in table if r["arm"] == "T0" and r["view"] == view and r["metric"] == metric
                                   and r["kind"] == "overall" and r["condition"] == "Real" and r["budget"] == "Full-U"]
                    points.sort(key=lambda r:float(r["epoch"]))
                    assert [float(r["epoch"]) for r in points] == [0,.5,1,2,3,5]
                    values = [float(r["total"])/207 if "EO_STRICT" in metric else float(r["value"]) for r in points]
                    axis.plot([float(r["epoch"]) for r in points],values,style,color=color,marker="o",markersize=4,label=f"{arm} seed{seed}")
            axis.set(title=title,xlabel="Query-graph epoch",xticks=[0,.5,1,2,3,5])
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, ncol=3, loc="upper center")
        fig.tight_layout(rect=(0, 0, 1, .91))
        for extension in ("png","pdf"):
            fig.savefig(directory / f"{section}_trajectory.{extension}",dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    run()
