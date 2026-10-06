"""Lakefront Window: pick tomorrow's best 2-hour window to be outside on Chicago's lakefront.

Usage:
    python fetch_data.py      # once, needs internet (data + TabPFN-v2 weights)
    python run.py             # offline from then on
    python run.py --fetch     # refresh data first (e.g. a new NWS forecast)
    python run.py --quick     # smaller TabPFN context (1000 rows), faster

Outputs land in ./out: results.json / results.md (holdout check),
card.pdf + card.png + card.txt (the one-page printable card).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(ROOT / "models"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lakefront as L  # noqa: E402

OUT = ROOT / "out"
SEED = 0
TEST_START = "2024-01-01"   # holdout = every hour from 2024 on; training = 2015-2023
OAK_ST = (41.9028, -87.6225)


# --------------------------------------------------------------------------- models
def tabpfn_model(context: int):
    from tabpfn import TabPFNClassifier
    ckpts = sorted((ROOT / "models").glob("tabpfn-v2-classifier*.ckpt"))
    if not ckpts:
        raise SystemExit("TabPFN-v2 weights not found in ./models. Run `python fetch_data.py` first.")
    return TabPFNClassifier(model_path=str(ckpts[0]), device="cpu", random_state=SEED,
                            ignore_pretraining_limits=context > 1000)


def fit_tabpfn(train: pd.DataFrame, context: int):
    ctx = train.sample(min(context, len(train)), random_state=SEED)
    clf = tabpfn_model(context)
    clf.fit(ctx[L.FEATURES].to_numpy(), ctx.label.to_numpy())
    return clf


def proba(clf, X: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(X[L.FEATURES].to_numpy())[:, 1]


def monthly_temp_offset(train: pd.DataFrame) -> pd.Series:
    """Mean (lakefront - Midway) temperature by month, learned on training years only."""
    return (train.lake_temp_c - train.temp_c).groupby(train.index.month).mean()


def naive(df: pd.DataFrame, offset: pd.Series | None = None, wet=None) -> np.ndarray:
    temp = df.temp_c.to_numpy()
    if offset is not None:
        temp = temp + offset.reindex(df.index.month).fillna(0).to_numpy()
    wet = (df.precip > 0) if wet is None else wet
    return L.good_hour(temp, df.wind_ms, wet)


# --------------------------------------------------------------------------- metrics
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
    yhat = (p >= 0.5).astype(int)
    out = {
        "accuracy": float((yhat == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "f1_good": float(f1_score(y, yhat)),
        "brier": float(brier_score_loss(y, p)),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(p)) > 2 else None
    return out


def window_hits(df: pd.DataFrame, score: np.ndarray) -> dict:
    """Each test day, pick the 2 consecutive hours with the highest score.
    A 'hit' means both picked hours were actually good at the lakefront."""
    d = df[["label"]].copy()
    d["score"] = score
    d["date"] = d.index.date
    hits, oracle, rand, days = 0, 0, 0.0, 0
    for _, g in d.groupby("date"):
        g = g.sort_index()
        if len(g) != len(L.HOURS) or (g.index.hour.to_list() != L.HOURS):
            continue  # need every hour 7..19 that day
        s, y = g.score.to_numpy(), g.label.to_numpy()
        pair_s = s[:-1] + s[1:]
        pair_y = (y[:-1] == 1) & (y[1:] == 1)
        best = int(np.argmax(pair_s))  # ties -> earliest window
        hits += int(pair_y[best])
        oracle += int(pair_y.any())
        rand += float(pair_y.mean())
        days += 1
    return {"days": days, "hit_rate": hits / days, "oracle_any_good_window": oracle / days,
            "random_window": rand / days, "hit_rate_on_days_with_a_good_window": hits / oracle}


# --------------------------------------------------------------------------- evaluation
def evaluate(df: pd.DataFrame, context: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, test = df[df.index < TEST_START], df[df.index >= TEST_START]
    y = test.label.to_numpy()
    off = monthly_temp_offset(train)
    clim = train.groupby([train.index.month, train.index.hour]).label.mean()
    clim_p = np.array([clim.get((m, h), train.label.mean())
                       for m, h in zip(test.index.month, test.index.hour)])

    print(f"train {len(train):,} hours ({train.index.min():%Y-%m-%d}..{train.index.max():%Y-%m-%d}), "
          f"test {len(test):,} hours ({test.index.min():%Y-%m-%d}..{test.index.max():%Y-%m-%d})")
    preds = {
        "naive_forecast_as_is": naive(test).astype(float),
        "naive_plus_lake_temp_offset": naive(test, off).astype(float),
        "climatology_month_hour": clim_p,
    }
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(train[L.FEATURES], train.label)
    preds["logistic_regression_all_train"] = lr.predict_proba(test[L.FEATURES])[:, 1]
    gbm = HistGradientBoostingClassifier(random_state=SEED).fit(train[L.FEATURES], train.label)
    preds["gradient_boosting_all_train"] = gbm.predict_proba(test[L.FEATURES])[:, 1]

    t0 = time.time()
    tab = fit_tabpfn(train, context)
    preds[f"tabpfn_v2_{context}_rows"] = proba(tab, test)
    tab_secs = time.time() - t0
    print(f"TabPFN-v2 ({context}-row context) scored {len(test):,} hours in {tab_secs:.0f}s on CPU")

    res = {}
    for name, p in preds.items():
        res[name] = metrics(y, p)
        res[name].update({f"window_{k}": v for k, v in window_hits(test, p).items()})

    # Synthetic stress test: perturb the regional inputs the way a day-ahead forecast
    # error might (NOT real forecast error; we have no forecast archive).
    rng = np.random.default_rng(SEED)
    noisy = test.copy()
    noisy["temp_c"] += rng.normal(0, 2.0, len(noisy))
    noisy["wind_ms"] = (noisy.wind_ms + rng.normal(0, 1.5, len(noisy))).clip(lower=0)
    noisy["rh"] = (noisy.rh + rng.normal(0, 8, len(noisy))).clip(0, 100)
    stress = {
        "naive_forecast_as_is": metrics(y, naive(noisy).astype(float)),
        "naive_plus_lake_temp_offset": metrics(y, naive(noisy, off).astype(float)),
        "gradient_boosting_all_train": metrics(y, gbm.predict_proba(noisy[L.FEATURES])[:, 1]),
        f"tabpfn_v2_{context}_rows": metrics(y, proba(tab, noisy)),
    }
    tab_p = preds[f"tabpfn_v2_{context}_rows"]
    bins = pd.cut(tab_p, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
    calib = pd.DataFrame({"p": tab_p, "y": y}).groupby(bins, observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    hold = test[["temp_c", "wind_ms", "precip", "lake_temp_c", "lake_gust_ms", "lake_wet", "label"]].copy()
    for k, p in preds.items():
        hold[k] = np.round(p, 4)
    hold.to_csv(OUT / "holdout_predictions.csv")
    err = pd.DataFrame({"tab": (tab_p >= 0.5) != y, "naive": preds["naive_forecast_as_is"].astype(int) != y,
                        "date": test.index.date})
    per_day = err.groupby("date").sum()
    day_tally = {"tabpfn_fewer_wrong_hours": int((per_day.tab < per_day.naive).sum()),
                 "naive_fewer_wrong_hours": int((per_day.tab > per_day.naive).sum()),
                 "tie": int((per_day.tab == per_day.naive).sum())}
    results = {
        "label": {
            "definition": (f"Oak Street lakefront sensor hour with air temp {L.TEMP_MIN_C:g}-"
                           f"{L.TEMP_MAX_C:g} C, hourly max gust < {L.WIND_MAX_MS:g} m/s, and no "
                           "rain/snow detected; hours 07:00-19:00 local only"),
            "base_rate_train": float(train.label.mean()),
            "base_rate_test": float(test.label.mean()),
        },
        "split": {"train": [str(train.index.min()), str(train.index.max()), len(train)],
                  "test": [str(test.index.min()), str(test.index.max()), len(test)]},
        "tabpfn_context_rows": context,
        "tabpfn_cpu_seconds": round(tab_secs, 1),
        "holdout": res,
        "stress_test_noisy_inputs": stress,
        "lake_minus_midway_temp_by_month_c": {int(k): round(float(v), 2) for k, v in off.items()},
        "days_tabpfn_vs_naive": day_tally,
        "tabpfn_calibration": [{"bin": str(i), "n": int(r.n), "mean_pred": round(float(r.mean_pred), 3),
                                "observed": round(float(r.observed), 3)} for i, r in calib.iterrows()],
    }
    return results, tab, off


# --------------------------------------------------------------------------- card
def sun_times(date, lat, lon, tz="America/Chicago"):
    """Approximate sunrise/sunset (NOAA sunrise equation, +-2 min)."""
    n = pd.Timestamp(date).dayofyear
    g = 2 * math.pi / 365 * (n - 1)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.degrees(math.acos(math.cos(math.radians(90.833)) / (math.cos(math.radians(lat)) * math.cos(decl))
                                - math.tan(math.radians(lat)) * math.tan(decl)))
    base = pd.Timestamp(date).tz_localize("UTC")
    rise = base + pd.Timedelta(minutes=720 - 4 * (lon + ha) - eqt)
    sets = base + pd.Timedelta(minutes=720 - 4 * (lon - ha) - eqt)
    return rise.tz_convert(tz).tz_localize(None), sets.tz_convert(tz).tz_localize(None)


def hour_label(h: int) -> str:
    return f"{(h - 1) % 12 + 1}{'am' if h < 12 else 'pm'}"


def daylight_mask(index: pd.DatetimeIndex, rise, sset) -> np.ndarray:
    """True for hours whose full 60 minutes fall between sunrise and sunset."""
    starts = index
    ends = index + pd.Timedelta(hours=1)
    return np.asarray((starts >= rise.floor("min")) & (ends <= sset), bool)


def pick_window(day: pd.DataFrame) -> tuple[int, float] | None:
    hrs = day.index.hour.to_numpy()
    best = None
    for i in range(len(day) - 1):
        if hrs[i + 1] == hrs[i] + 1 and day.daylight.iloc[i] and day.daylight.iloc[i + 1]:
            # Round to 2 decimals so CPU run-to-run jitter doesn't flip near-ties; ties -> earliest.
            s = round((day.p_good.iloc[i] + day.p_good.iloc[i + 1]) / 2, 2)
            if best is None or s > best[1]:
                best = (int(hrs[i]), float(s))
    return best


def make_card(day: pd.DataFrame, target, *, title_note: str, source_note: str, actual=None) -> dict:
    rise, sset = sun_times(target, *OAK_ST)
    day = day.copy()
    day["daylight"] = daylight_mask(day.index, rise, sset)
    best = pick_window(day)
    if best is None:
        verdict, h0, s = "No daylight window in range", None, 0.0
    else:
        h0, s = best
        verdict = (f"Best 2 hours: {hour_label(h0)}-{hour_label(h0 + 2)}" if s >= 0.5
                   else f"No great window. Least-bad: {hour_label(h0)}-{hour_label(h0 + 2)}")
    hours = []
    for t, r in day.iterrows():
        h = {"hour": int(t.hour), "temp_f": round(r.temp_c * 9 / 5 + 32), "wind_mph": round(r.wind_ms / 0.44704),
             "pop_pct": round(r["pop"] * 100), "p_good": round(float(r.p_good), 3), "naive_ok": bool(r.naive),
             "sky": r.get("short", ""), "daylight": bool(r.daylight)}
        if actual is not None:
            h["actual_good"] = int(actual.loc[t])
        hours.append(h)
    return {"date": pd.Timestamp(target).strftime("%a %b %-d, %Y"), "verdict": verdict, "window_start_hour": h0,
            "window_mean_p_good": round(float(s), 3), "sunrise": rise.strftime("%-I:%M %p"),
            "sunset": sset.strftime("%-I:%M %p"), "title_note": title_note, "source_note": source_note,
            "hours": hours}


def build_card(df: pd.DataFrame, results: dict, context: int) -> dict:
    fc, meta = L.load_forecast()
    issued = pd.Timestamp(meta["generated_at"]).tz_convert("America/Chicago").tz_localize(None)
    target = (issued + pd.Timedelta(days=1)).normalize()
    day = fc[(fc.index.normalize() == target) & fc.index.hour.isin(L.HOURS)].copy()
    if day.empty:
        raise SystemExit("Cached forecast does not cover tomorrow 7am-7pm; run `python run.py --fetch`.")
    model = fit_tabpfn(df, context)  # final model: context sampled from all years
    dry, wet = day.copy(), day.copy()
    dry["precip"], wet["precip"] = 0.0, 1.0
    day["p_good"] = (1 - day["pop"]) * proba(model, dry) + day["pop"] * proba(model, wet)  # average over rain chance
    day["naive"] = naive(day, wet=day["pop"].to_numpy() >= 0.5)
    issued_s = issued.strftime("%a %b %-d %-I:%M %p CT")
    card = make_card(day, target, title_note="Forecast card",
                     source_note=f"NWS hourly forecast issued {issued_s} (gridpoint {meta['gridpoint']}, Midway area).")
    render_card(card, results, context, OUT / "card")
    return card


def build_replay(df: pd.DataFrame, date: str, model, offset, results: dict, context: int) -> dict:
    """Re-run a past (holdout) day using the airport observations as a perfect 'forecast',
    with the train-only model, and show what actually happened at the lakefront."""
    target = pd.Timestamp(date).normalize()
    if target < pd.Timestamp(TEST_START):
        raise SystemExit(f"--replay must be a holdout day (on/after {TEST_START}).")
    day = df[df.index.normalize() == target].copy()
    if day.empty:
        raise SystemExit(f"No joined data for {date}.")
    day["pop"] = day.precip
    day["short"] = ""
    day["p_good"] = proba(model, day)
    day["naive"] = naive(day)
    card = make_card(day, target, title_note="Replay of a holdout day",
                     source_note="Inputs: actual Midway airport observations (a perfect 'forecast'); "
                                 "model trained on 2015-2023 only.", actual=day.label)
    render_card(card, results, context, OUT / f"replay_{target:%Y-%m-%d}")
    return card


def render_card(card: dict, results: dict, context: int, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab = results["holdout"][f"tabpfn_v2_{context}_rows"]
    nai = results["holdout"]["naive_forecast_as_is"]
    hours = card["hours"]
    replay = "actual_good" in hours[0]
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.955, "LAKEFRONT WINDOW", fontsize=26, weight="bold")
    fig.text(0.07, 0.93, f"Oak Street Beach, Chicago  ·  {card['date']}  ·  {card['title_note']}", fontsize=12)
    fig.text(0.93, 0.958, "Built with PriorLabs-TabPFN", fontsize=9, ha="right", color="#555555")
    fig.text(0.07, 0.882, card["verdict"], fontsize=22, weight="bold", color="#1b5e20")
    fig.text(0.07, 0.858, f"Chance the lakefront is a 'good outside hour' in that window: "
             f"{card['window_mean_p_good']:.0%}   ·   sunrise {card['sunrise']}, sunset {card['sunset']}", fontsize=10)

    ax = fig.add_axes([0.08, 0.57, 0.86, 0.25])
    xs = [h["hour"] for h in hours]
    ps = [h["p_good"] for h in hours]
    colors = [("#2e7d32" if p >= 0.5 else "#e0e0e0") if h["daylight"] else "#9e9e9e" for p, h in zip(ps, hours)]
    bars = ax.bar(xs, ps, color=colors, width=0.8)
    for b, h in zip(bars, hours):
        if not h["daylight"]:
            b.set_hatch("//")
    ax.axhline(0.5, color="black", lw=0.6, ls="--")
    if card["window_start_hour"] is not None:
        h0 = card["window_start_hour"]
        ax.axvspan(h0 - 0.5, h0 + 1.5, color="#fff59d", zorder=0)
    ax.set_xticks(xs, [hour_label(x) for x in xs], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("P(good outside hour)")
    ax.set_title("Hour by hour: TabPFN-v2 lakefront estimate (hatched = not full daylight)", fontsize=10, loc="left")
    for x, h in zip(xs, hours):
        ax.text(x, 1.02, "app ✓" if h["naive_ok"] else "app ✗", ha="center", fontsize=6.5, color="#0d47a1")

    cols = ["Hour", "Temp", "Wind", "Rain", "Lakefront P(good)", "App says OK?"]
    rows = [[hour_label(h["hour"]) + ("" if h["daylight"] else " (dark)"), f"{h['temp_f']}°F", f"{h['wind_mph']} mph",
             f"{h['pop_pct']}%", f"{h['p_good']:.0%}", "yes" if h["naive_ok"] else "no"] for h in hours]
    if replay:
        cols += ["Actual at lakefront"]
        for r, h in zip(rows, hours):
            r.append("good" if h["actual_good"] else "not good")
    else:
        cols += ["Sky"]
        for r, h in zip(rows, hours):
            r.append(h["sky"][:22])
    tax = fig.add_axes([0.07, 0.25, 0.88, 0.29])
    tax.axis("off")
    t = tax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.25)

    foot = (
        f"What 'good' means: lakefront sensor reads {L.TEMP_MIN_C:g}-{L.TEMP_MAX_C:g} °C (50-81 °F), hourly gust under "
        f"{L.WIND_MAX_MS:g} m/s (~20 mph), no rain or snow.\n"
        "How: TabPFN-v2 (open-weight tabular model, run locally on CPU) learned how the Oak Street lakefront sensor\n"
        "differs from Midway airport conditions (spring/summer lake cooling, onshore wind) and applies that to the inputs.\n"
        f"Holdout 2024-2026 ({results['split']['test'][2]:,} daylight-range hours, airport obs as input): TabPFN accuracy "
        f"{tab['accuracy']:.1%} vs {nai['accuracy']:.1%}\n"
        f"for 'trust the weather app as-is'; best-window hit rate {tab['window_hit_rate']:.0%} vs "
        f"{nai['window_hit_rate']:.0%}. Real forecast error is NOT included in these numbers.\n"
        f"{card['source_note']}\n"
        "Data: Chicago Data Portal k7hf-8y75 · IEM ASOS archive (KMDW) · api.weather.gov. Not a safety tool: check\n"
        "beach and marine advisories. Built by Alfred, an AI agent, for Brooks Moore. MIT licensed.\n"
        "Now fold this card, leave the phone, and go."
    )
    fig.text(0.07, 0.045, foot, fontsize=7.6, va="bottom")
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=110)
    plt.close(fig)

    lines = [f"LAKEFRONT WINDOW, {card['date']} (Oak Street Beach) - {card['title_note']}", card["verdict"],
             f"sunrise {card['sunrise']} / sunset {card['sunset']}", ""]
    for h in hours:
        extra = (f"actual={'good' if h['actual_good'] else 'not good'}" if replay else h["sky"])
        lines.append(f"{hour_label(h['hour']):>4}{'' if h['daylight'] else '*'}  {h['temp_f']:>3}F  {h['wind_mph']:>2}mph  "
                     f"rain {h['pop_pct']:>3}%  P(good) {h['p_good']:>4.0%}  app-ok={'Y' if h['naive_ok'] else 'N'}  {extra}")
    lines += ["", "* = not full daylight", card["source_note"], "Not a safety tool. Built with PriorLabs-TabPFN."]
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n")


def results_md(r: dict) -> str:
    out = ["# Holdout results", "", f"Label: {r['label']['definition']}.", "",
           f"Train {r['split']['train'][2]:,} hours ({r['split']['train'][0][:10]} to {r['split']['train'][1][:10]}); "
           f"test {r['split']['test'][2]:,} hours ({r['split']['test'][0][:10]} to {r['split']['test'][1][:10]}). "
           f"Good-hour base rate: train {r['label']['base_rate_train']:.1%}, test {r['label']['base_rate_test']:.1%}.",
           "", "Inputs are actual Midway airport observations standing in for a forecast; "
           "real forecast error is not included.", "",
           "| Method | Accuracy | Balanced acc. | F1 (good) | Brier | AUC | Best-window hit rate |",
           "|---|---|---|---|---|---|---|"]
    for k, m in r["holdout"].items():
        auc = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
        out.append(f"| {k} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['f1_good']:.3f} | "
                   f"{m['brier']:.3f} | {auc} | {m['window_hit_rate']:.3f} |")
    any_m = next(iter(r["holdout"].values()))
    out += ["", f"Window check over {any_m['window_days']} test days with all 13 hours: a day had at least one fully "
            f"good 2-hour window {any_m['window_oracle_any_good_window']:.1%} of the time (ceiling); a random "
            f"window was good {any_m['window_random_window']:.1%} of the time. On days that had a good window, "
            "the picked window was good: " + ", ".join(
                f"{k} {m['window_hit_rate_on_days_with_a_good_window']:.1%}" for k, m in r["holdout"].items()) + ".",
            "", f"Per test day, TabPFN had fewer wrong hours than the as-is app reading on "
            f"{r['days_tabpfn_vs_naive']['tabpfn_fewer_wrong_hours']} days, more on "
            f"{r['days_tabpfn_vs_naive']['naive_fewer_wrong_hours']} days, tied on {r['days_tabpfn_vs_naive']['tie']}.",
            "", "## TabPFN calibration on the holdout", "", "| Predicted bin | Hours | Mean predicted | Observed good |",
            "|---|---|---|---|"] + [f"| {c['bin']} | {c['n']} | {c['mean_pred']:.2f} | {c['observed']:.2f} |"
                                    for c in r["tabpfn_calibration"]] + ["",
            "## Synthetic stress test (noise on inputs: temp sd 2 C, wind sd 1.5 m/s, RH sd 8)", "",
            "| Method | Accuracy | Brier |", "|---|---|---|"]
    for k, m in r["stress_test_noisy_inputs"].items():
        out.append(f"| {k} | {m['accuracy']:.3f} | {m['brier']:.3f} |")
    out += ["", f"TabPFN-v2 context: {r['tabpfn_context_rows']} randomly sampled training hours; "
            f"CPU time to score the holdout: {r['tabpfn_cpu_seconds']} s.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download fresh data first (needs internet)")
    ap.add_argument("--quick", action="store_true", help="1000-row TabPFN context instead of 3000")
    ap.add_argument("--context", type=int, default=3000)
    ap.add_argument("--replay", metavar="YYYY-MM-DD", action="append", default=[],
                    help="also render a card for a past holdout day, with what actually happened")
    args = ap.parse_args()
    if args.fetch:
        import fetch_data
        fetch_data.main()
    os.environ["HF_HUB_OFFLINE"] = "1"  # never reach the network from here on
    warnings.filterwarnings("ignore", message=".*Running on CPU.*")
    context = 1000 if args.quick else args.context
    for f in ("beach_oak_street.csv", "mdw_asos.csv", "nws_hourly_mdw.json"):
        if not (L.DATA / f).exists():
            raise SystemExit(f"data/{f} missing. Run `python fetch_data.py` once (internet needed).")
    OUT.mkdir(exist_ok=True)
    df = L.build_dataset()
    results, tab_eval, offset = evaluate(df, context)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    (OUT / "results.md").write_text(results_md(results))
    print(results_md(results))
    card = build_card(df, results, context)
    (OUT / "card.json").write_text(json.dumps(card, indent=1))
    print(f"\n{card['date']}: {card['verdict']} (mean P(good) {card['window_mean_p_good']:.0%})")
    print(f"Card written to {OUT / 'card.pdf'} and card.png / card.txt")
    for d in args.replay:
        rc = build_replay(df, d, tab_eval, offset, results, context)
        (OUT / f"replay_{d}.json").write_text(json.dumps(rc, indent=1))
        print(f"Replay {rc['date']}: {rc['verdict']} -> out/replay_{d}.pdf")


if __name__ == "__main__":
    main()
