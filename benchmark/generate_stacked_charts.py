import json

import matplotlib.pyplot as plt
import numpy as np

with open("benchmark_detailed_v0.3.3.json") as f:
    v3 = json.load(f)
with open("benchmark_detailed_v0.4.0.dev.json") as f:
    v4 = json.load(f)

resolutions = ["2.5x2.5", "0.25x0.25"]
backends = [
    ("serial", "1"),
    ("dask", "2"),
    ("dask", "4"),
    ("dask", "8"),
    ("mpi", "2"),
    ("mpi", "4"),
    ("mpi", "8"),
]

md_lines = [
    "# PyStormTracker Benchmark Report",
    "",
    "This document provides a detailed breakdown of execution time (in seconds) "
    "comparing the legacy nested-object architecture (`v0.3.3`) and the modern, "
    "Numba JIT-compiled array architecture (`v0.4.0.dev`).",
    "",
    "## Methodology",
    "- **Hardware**: AMD Ryzen 7 5800X (16 Threads), 48GB WSL Memory Limit.",
    "- **Datasets**: ERA5 Mean Sea Level Pressure (MSL).",
    "  - `2.5x2.5`: 144x73 grid, 360 time steps.",
    "  - `0.25x0.25`: 1440x721 grid, 60 time steps.",
    "- **Execution**: Component timings (Detection, Linking, Export, IO/Overhead) "
    "were extracted from the CLI.",
    "",
]

for res in resolutions:
    md_lines.append(f"## Resolution: {res}")
    md_lines.append(
        f"![{res} Breakdown](benchmark_{res.replace('.', '_')}_breakdown.png)"
    )
    md_lines.append("")
    md_lines.append(
        "| Version | Backend | Workers | Det (s) | Link (s) | Exp (s) | IO (s) | "
        "Total (s) |"
    )
    md_lines.append("|---|---|---|---|---|---|---|---|")

    labels = []
    v3_det = []
    v3_link = []
    v3_exp = []
    v3_io = []
    v4_det = []
    v4_link = []
    v4_exp = []
    v4_io = []

    for b, w in backends:
        label = f"{b.upper()}-{w}w"
        labels.append(label)

        t3 = v3[res][b].get(
            w,
            {
                "detection": 0,
                "linking": 0,
                "export": 0,
                "io_overhead": 0,
                "wall": 0,
            },
        )
        t4 = v4[res][b].get(
            w,
            {
                "detection": 0,
                "linking": 0,
                "export": 0,
                "io_overhead": 0,
                "wall": 0,
            },
        )

        md_lines.append(
            f"| v0.3.3 | {b.upper()} | {w} | {t3['detection']:.2f} | "
            f"{t3['linking']:.2f} | {t3['export']:.2f} | {t3['io_overhead']:.2f} | "
            f"**{t3['wall']:.2f}** |"
        )
        md_lines.append(
            f"| v0.4.0.dev | {b.upper()} | {w} | {t4['detection']:.2f} | "
            f"{t4['linking']:.2f} | {t4['export']:.2f} | {t4['io_overhead']:.2f} | "
            f"**{t4['wall']:.2f}** |"
        )

        v3_det.append(t3["detection"])
        v3_link.append(t3["linking"])
        v3_exp.append(t3["export"])
        v3_io.append(t3["io_overhead"])

        v4_det.append(t4["detection"])
        v4_link.append(t4["linking"])
        v4_exp.append(t4["export"])
        v4_io.append(t4["io_overhead"])

    md_lines.append("")

    # Plotting
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    # v0.3.3 stacked bars
    ax.bar(
        x - width / 2,
        v3_det,
        width,
        label="v0.3 Detection",
        color="#ff9999",
        edgecolor="white",
    )
    ax.bar(
        x - width / 2,
        v3_link,
        width,
        bottom=v3_det,
        label="v0.3 Linking",
        color="#66b3ff",
        edgecolor="white",
    )
    ax.bar(
        x - width / 2,
        v3_exp,
        width,
        bottom=np.array(v3_det) + np.array(v3_link),
        label="v0.3 Export",
        color="#99ff99",
        edgecolor="white",
    )
    ax.bar(
        x - width / 2,
        v3_io,
        width,
        bottom=np.array(v3_det) + np.array(v3_link) + np.array(v3_exp),
        label="v0.3 IO/Overhead",
        color="#ffcc99",
        edgecolor="white",
    )

    # v0.4.0.dev stacked bars
    ax.bar(
        x + width / 2,
        v4_det,
        width,
        label="v0.4 Detection",
        color="#cc0000",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        v4_link,
        width,
        bottom=v4_det,
        label="v0.4 Linking",
        color="#0055ff",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        v4_exp,
        width,
        bottom=np.array(v4_det) + np.array(v4_link),
        label="v0.4 Export",
        color="#00cc00",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        v4_io,
        width,
        bottom=np.array(v4_det) + np.array(v4_link) + np.array(v4_exp),
        label="v0.4 IO/Overhead",
        color="#cc6600",
        edgecolor="black",
    )

    ax.set_ylabel("Time (Seconds)")
    ax.set_title(f"Component Breakdown: Legacy vs Modern ({res})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Add a custom legend to show the matching colors clearly
    handles, legends = ax.get_legend_handles_labels()
    # Group by component
    new_handles = [
        handles[0],
        handles[4],
        handles[1],
        handles[5],
        handles[2],
        handles[6],
        handles[3],
        handles[7],
    ]
    new_labels = [
        legends[0],
        legends[4],
        legends[1],
        legends[5],
        legends[2],
        legends[6],
        legends[3],
        legends[7],
    ]
    ax.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add total time annotations
    for i in range(len(labels)):
        tot3 = v3_det[i] + v3_link[i] + v3_exp[i] + v3_io[i]
        tot4 = v4_det[i] + v4_link[i] + v4_exp[i] + v4_io[i]

        ax.text(
            x[i] - width / 2,
            tot3 + (tot3 * 0.01),
            f"{tot3:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            x[i] + width / 2,
            tot4 + (tot4 * 0.01),
            f"{tot4:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(f"benchmark_{res.replace('.', '_')}_breakdown.png", dpi=150)
    plt.close()

with open("BENCHMARK.md", "w") as f:
    f.write("\n".join(md_lines))

print("Stacked breakdown report generated successfully.")
