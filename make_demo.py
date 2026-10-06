"""Build a short silent slideshow video (out/demo.mp4) from the outputs of run.py.

Needs ffmpeg on PATH. Run after `python run.py --replay 2025-05-03 --replay 2024-04-18`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
SLIDES = OUT / "demo_slides"
W, H = 8.5, 11  # same page size as the card


def text_slide(path: Path, title: str, body: str, mono: bool = False) -> None:
    fig = plt.figure(figsize=(W, H))
    fig.text(0.07, 0.92, title, fontsize=24, weight="bold", va="top")
    fig.text(0.07, 0.84, body, fontsize=10.5 if mono else 14, va="top",
             family="DejaVu Sans Mono" if mono else "DejaVu Sans", linespacing=1.5)
    fig.text(0.07, 0.04, "Lakefront Window · Built with PriorLabs-TabPFN · built by Alfred (AI agent)",
             fontsize=9, color="#555555")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def offset_slide(path: Path, offsets: dict) -> None:
    fig = plt.figure(figsize=(W, H))
    fig.text(0.07, 0.92, "Why: 'cooler by the lake'", fontsize=24, weight="bold", va="top")
    fig.text(0.07, 0.86, "Oak Street lakefront sensor minus Midway airport temperature,\n"
             "average by month, 7am-7pm, 2015-2023 (training years)", fontsize=13, va="top")
    ax = fig.add_axes([0.12, 0.3, 0.8, 0.45])
    months = [int(m) for m in offsets]
    vals = [offsets[str(m)] if str(m) in offsets else offsets[m] for m in months]
    ax.bar(months, vals, color=["#1565c0" if v < 0 else "#ef6c00" for v in vals])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(months, "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())
    ax.set_ylabel("°C (lakefront − airport)")
    fig.text(0.07, 0.2, textwrap.fill("In May and June the lakefront averages ~3 °C colder than the airport, "
             "and onshore winds make individual days far colder: in 2015-2026 there were 77 days with hours where "
             "Midway read 60 °F or warmer while the lakefront sensor read under 50 °F.", 70), fontsize=12, va="top")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")
    res = json.loads((OUT / "results.json").read_text())
    SLIDES.mkdir(parents=True, exist_ok=True)
    slides = []

    p = SLIDES / "01_title.png"
    text_slide(p, "Lakefront Window", textwrap.fill(
        "An offline, open-weight AI tool that picks tomorrow's best 2 hours to be outside on Chicago's "
        "lakefront, then prints it on a one-page card so the screen is the shortest part of your day.", 55)
        + "\n\nTabPFN-v2 on a laptop CPU · Chicago beach sensors · free NWS forecast")
    slides.append(p)

    p = SLIDES / "02_why.png"
    offset_slide(p, res["lake_minus_midway_temp_by_month_c"])
    slides.append(p)

    p = SLIDES / "03_results.png"
    ho = res["holdout"]
    keys = [k for k in ho]
    lines = [f"Holdout: {res['split']['test'][2]:,} hours, 2024-01-01 .. 2026-10-05", "",
             f"{'method':34} {'acc':>6} {'brier':>6} {'window':>7}"]
    for k in keys:
        m = ho[k]
        lines.append(f"{k:34} {m['accuracy']:6.3f} {m['brier']:6.3f} {m['window_hit_rate']:7.3f}")
    lines += ["", "window = best 2h pick was actually good", "(ceiling: " +
              f"{ho[keys[0]]['window_oracle_any_good_window']:.3f} of days had one)", "",
              "Inputs = real airport observations, not", "archived forecasts: forecast error is",
              "NOT included. Gradient boosting on all", "29k training hours ties TabPFN; TabPFN",
              "used a 3,000-hour sample."]
    text_slide(p, "Honest holdout check", "\n".join(lines), mono=True)
    slides.append(p)

    for name in ["card.png", "replay_2025-05-03.png", "replay_2024-04-18.png"]:
        if (OUT / name).exists():
            slides.append(OUT / name)

    listfile = SLIDES / "list.txt"
    with listfile.open("w") as f:
        for s in slides:
            f.write(f"file '{s.resolve()}'\nduration 5\n")
        f.write(f"file '{slides[-1].resolve()}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listfile),
                    "-vf", "scale=936:1210:force_original_aspect_ratio=decrease,pad=936:1210:(ow-iw)/2:(oh-ih)/2:white,format=yuv420p",
                    "-r", "25", str(OUT / "demo.mp4")], check=True)
    print(f"wrote {OUT / 'demo.mp4'} ({len(slides)} slides x 5 s)")


if __name__ == "__main__":
    main()
