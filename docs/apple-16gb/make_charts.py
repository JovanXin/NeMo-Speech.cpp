"""Charts for docs/apple-16gb/README.md: one light and one dark SVG per chart.

    uvx --with matplotlib python docs/apple-16gb/make_charts.py

All numbers are wall-clock seconds measured on a 16 GB M4 MacBook (macOS 27.2) with
Firefox, Spotify and Claude open (about 7 GB of other apps), on a 114 s two-speaker
clip unless noted. Runs vary with memory pressure, so bars show the best run and the
whisker spans every run of that configuration.
"""

from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "img"
REALTIME = 114.3  # clip length, seconds

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", before="#c3c2b7",
                  s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", before="#5a5955",
                 s=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}


def frame(ax, t, xlabel=None, ylabel=None):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["text2"], labelsize=9, length=0)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=t["text2"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["text2"], fontsize=9)


def fig_for(t, w, h):
    fig = plt.figure(figsize=(w, h), facecolor=t["surface"])
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    return fig


def title(fig, t, main, sub):
    fig.text(0.012, 0.965, main, color=t["text"], fontsize=12, fontweight="bold", va="top")
    fig.text(0.012, 0.895, sub, color=t["text2"], fontsize=9, va="top")


def save(fig, name, mode):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


# 1. End-to-end time on the 2-minute clip, step by step ------------------------------------
JOURNEY = [  # label, every run (s)
    ("CPU only (mmap)", [152.5, 355.1]),
    ("Metal, experts on CPU (cpu_moe)", [217.4, 221.7, 245.5, 250.9, 260.9]),
    ("Expert streaming, 32 slots", [88.0]),
    ("Expert streaming, 48 slots", [78.8, 83.4]),
    ("Expert streaming, 64 slots", [66.2, 68.3]),
    ("80 slots + n_batch 2048", [50.9, 54.9, 57.3, 58.7, 59.2, 59.6]),
    ("80 slots + F_NOCACHE reads", [52.4, 52.8]),
]


def chart_journey(mode):
    t = THEMES[mode]
    fig = fig_for(t, 8.4, 4.2)
    title(fig, t, "Transcribing a 114 s clip on a 16 GB M4",
          "Seconds, end to end (lower is better). Bar = best run, whisker = range of runs.")
    ax = fig.add_axes([0.33, 0.13, 0.52, 0.68])
    frame(ax, t, xlabel="seconds")
    labels = [j[0] for j in JOURNEY][::-1]
    best = [min(j[1]) for j in JOURNEY][::-1]
    lo = [min(j[1]) for j in JOURNEY][::-1]
    hi = [max(j[1]) for j in JOURNEY][::-1]
    colors = [t["s"][0] if i < 5 else t["before"] for i in range(len(labels))]  # streaming configs in blue
    y = range(len(labels))
    ax.barh(y, best, height=0.62, color=colors, edgecolor=t["surface"], linewidth=2)
    for yi, l, h in zip(y, lo, hi):
        if h > l:
            ax.plot([l, h], [yi, yi], color=t["text2"], linewidth=1.2, solid_capstyle="round")
    for yi, b, h in zip(y, best, hi):
        extra = f"  (up to {h:.0f})" if h - b > 15 else ""
        ax.text(h + 5, yi, f"{b:.0f} s{extra}", color=t["text"], fontsize=9, va="center")
    ax.axvline(REALTIME, color=t["muted"], linewidth=1, linestyle=(0, (3, 3)))
    ax.text(REALTIME + 3, len(labels) - 0.45, "real time (114 s)", color=t["text2"], fontsize=8.5)
    ax.set_yticks(list(y), labels, color=t["text"])
    ax.set_xlim(0, 400)
    save(fig, "journey", mode)


# 2. Where the time goes ---------------------------------------------------------------------
PHASES = [  # same session, same memory pressure (NEMO_SALM_PROFILE)
    ("CPU only", dict(enc=4.64, pre=62.75, dec=258.71, total=355.05)),
    ("Metal + experts on CPU", dict(enc=0.82, pre=54.82, dec=133.98, total=217.37)),
    ("Expert streaming, 80 slots", dict(enc=0.98, pre=15.00, dec=29.54, total=50.85)),
]


def chart_phases(mode):
    t = THEMES[mode]
    fig = fig_for(t, 8.4, 3.6)
    title(fig, t, "Where the time goes",
          "Seconds per phase on the 114 s clip. Decode (one token at a time) dominated until experts were cached on the GPU.")
    ax = fig.add_axes([0.25, 0.16, 0.71, 0.54])
    frame(ax, t, xlabel="seconds")
    names = ["Encoder", "Prefill (audio)", "Decode (text)", "Load + other"]
    rows = PHASES[::-1]
    for yi, (_, p) in enumerate(rows):
        segs = [p["enc"], p["pre"], p["dec"], p["total"] - p["enc"] - p["pre"] - p["dec"]]
        left = 0
        for k, v in enumerate(segs):
            ax.barh(yi, v, left=left, height=0.6, color=t["s"][k], edgecolor=t["surface"], linewidth=2,
                    label=names[k] if yi == 0 else None)
            if v >= 14:
                ax.text(left + v / 2, yi, f"{v:.0f}", color=t["surface"] if mode == "light" else t["text"],
                        fontsize=8.5, ha="center", va="center", fontweight="bold")
            left += v
        ax.text(left + 4, yi, f"{p['total']:.0f} s", color=t["text"], fontsize=9, va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=t["text"])
    ax.set_xlim(0, 390)
    leg = fig.legend(loc="upper left", bbox_to_anchor=(0.005, 0.86), ncol=4, frameon=False, fontsize=8.5,
                     handlelength=1.2)
    for txt in leg.get_texts():
        txt.set_color(t["text2"])
    save(fig, "phases", mode)


# 3. Cache size: time, hit rate, memory (n_batch 512 sweep) ----------------------------------
SLOTS = [32, 48, 64, 80, 96]
SLOT_TIME = [88.0, 83.4, 67.2, 66.0, 190.9]
HIT = {64: 84.0, 80: 88.8, 96: 92.7, 104: 94.2}
PEAK = {32: 6.0, 48: 6.7, 64: 8.1, 80: 9.3, 96: 10.6, 104: 11.3}


def chart_slots(mode):
    t = THEMES[mode]
    fig = fig_for(t, 8.4, 3.2)
    title(fig, t, "Expert cache size: more slots help until the Mac starts swapping",
          "Slots per layer (80 slots = about 7 GB of experts in GPU memory). Three measures, three axes.")
    panels = [
        ("seconds (114 s clip)", SLOTS, SLOT_TIME, "{:.0f}"),
        ("cache hit rate, %", list(HIT), list(HIT.values()), "{:.0f}%"),
        ("peak memory of the process, GB", list(PEAK), list(PEAK.values()), "{:.1f}"),
    ]
    for i, (lab, xs, ys, fmt) in enumerate(panels):
        ax = fig.add_axes([0.06 + i * 0.32, 0.14, 0.24, 0.6])
        frame(ax, t)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.plot(xs, ys, color=t["s"][0], linewidth=2, marker="o", markersize=5,
                markeredgecolor=t["surface"], markeredgewidth=1.5)
        best = ys.index(min(ys)) if i == 0 else len(ys) - 1
        ax.annotate(fmt.format(ys[best]), (xs[best], ys[best]), xytext=(-8, -12 if i == 0 else 0),
                    textcoords="offset points", ha="right", va="center", color=t["text"], fontsize=8.5)
        ax.set_title(lab, color=t["text2"], fontsize=9, loc="left")
        ax.set_xticks([32, 64, 96])
        ax.set_xlim(26, 110)
        ax.margins(y=0.15)
        if i == 0:
            ax.annotate("swapping", (96, 190.9), xytext=(-8, 0), textcoords="offset points",
                        ha="right", va="center", color=t["text2"], fontsize=8.5)
    save(fig, "slots", mode)


# 4. 12 s clip end to end (includes model load) ----------------------------------------------
SHORT = [
    ("CPU only", 55.9),
    ("Metal + experts on CPU", 66.7),
    ("Expert streaming, 32 slots", 18.9),
    ("80 slots + MTP draft 7", 16.6),
    ("80 slots + cross-layer prefetch", 13.1),
]


def chart_short(mode):
    t = THEMES[mode]
    fig = fig_for(t, 8.4, 3.0)
    title(fig, t, "A 12 s voice memo, end to end",
          "Seconds including model load (lower is better). One run each.")
    ax = fig.add_axes([0.31, 0.15, 0.64, 0.63])
    frame(ax, t, xlabel="seconds")
    rows = SHORT[::-1]
    colors = [t["s"][0] if i == 0 else t["before"] for i in range(len(rows))]
    ax.barh(range(len(rows)), [r[1] for r in rows], height=0.62, color=colors, edgecolor=t["surface"], linewidth=2)
    for yi, (_, v) in enumerate(rows):
        ax.text(v + 1, yi, f"{v:.1f} s", color=t["text"], fontsize=9, va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=t["text"])
    ax.set_xlim(0, 75)
    save(fig, "short", mode)


# 5. Things that did not help: MTP drafting --------------------------------------------------
MTP = [("off", 57.3), ("3 tokens", 120.7), ("5 tokens", 104.5), ("7 tokens", 173.2)]


def chart_mtp(mode):
    t = THEMES[mode]
    fig = fig_for(t, 8.4, 2.8)
    title(fig, t, "MTP speculative decoding: correct, but slower when experts stream",
          "Seconds on the 114 s clip, 80 slots. Each verify pass needs the experts of several tokens at once.")
    ax = fig.add_axes([0.16, 0.17, 0.79, 0.58])
    frame(ax, t, xlabel="seconds")
    rows = MTP[::-1]
    colors = [t["s"][0] if r[0] == "off" else t["before"] for r in rows]
    ax.barh(range(len(rows)), [r[1] for r in rows], height=0.6, color=colors, edgecolor=t["surface"], linewidth=2)
    for yi, (_, v) in enumerate(rows):
        ax.text(v + 2, yi, f"{v:.0f} s", color=t["text"], fontsize=9, va="center")
    ax.set_yticks(range(len(rows)), [f"draft {r[0]}" if r[0] != "off" else "MTP off" for r in rows], color=t["text"])
    ax.set_xlim(0, 195)
    save(fig, "mtp", mode)


# 6. Disk reads: pread vs page faults --------------------------------------------------------
READS = [("1 thread", 1994, 668), ("6 threads", 3701, 1307)]


def chart_reads(mode):
    t = THEMES[mode]
    fig = fig_for(t, 8.4, 3.1)
    title(fig, t, "Reading 1.4 MB expert slices from the SSD",
          "MB/s, random offsets in the model file, page cache bypassed. Page faults (mmap) are what the CPU path relied on.")
    ax = fig.add_axes([0.13, 0.16, 0.82, 0.52])
    frame(ax, t, xlabel="MB/s")
    for i, (lab, pread, fault) in enumerate(READS[::-1]):
        ax.barh(i + 0.17, pread, height=0.32, color=t["s"][0], edgecolor=t["surface"], linewidth=2,
                label="pread (expert streaming)" if i == 0 else None)
        ax.barh(i - 0.17, fault, height=0.32, color=t["s"][1], edgecolor=t["surface"], linewidth=2,
                label="mmap page faults" if i == 0 else None)
        ax.text(pread + 40, i + 0.17, f"{pread:,}", color=t["text"], fontsize=9, va="center")
        ax.text(fault + 40, i - 0.17, f"{fault:,}", color=t["text"], fontsize=9, va="center")
    ax.set_yticks(range(len(READS)), [r[0] for r in READS[::-1]], color=t["text"])
    ax.set_xlim(0, 4300)
    leg = fig.legend(loc="upper left", bbox_to_anchor=(0.005, 0.84), ncol=2, frameon=False, fontsize=8.5,
                     handlelength=1.2)
    for txt in leg.get_texts():
        txt.set_color(t["text2"])
    save(fig, "reads", mode)


if __name__ == "__main__":
    for m in THEMES:
        chart_journey(m)
        chart_phases(m)
        chart_slots(m)
        chart_short(m)
        chart_mtp(m)
        chart_reads(m)
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
