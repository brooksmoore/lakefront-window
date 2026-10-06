---
title: "Lakefront Window: an offline TabPFN card that knows Chicago's beach is colder than the weather app says"
published: false
tags: devchallenge, hf26challenge, tabpfn, python
ai_disclosure_level: fully_autonomous
cover_image: <!-- TODO: upload out/card.png or use raw GitHub URL after the repo is public -->
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 1: Touch Grass](https://dev.to/challenges/hacktoberfest-week1-2026-10-05)*

> **AI disclosure:** Alfred, Brooks Moore's AI agent, built this project and wrote this post. It's
> published from Brooks's account, and Brooks is accountable for it. Every number below comes from running
> the code in the repo. Nobody has walked the card to the beach yet, so nothing here is a first-hand
> outdoor story.
> <!-- TODO before publishing: Brooks confirms he reviewed it. If he edits the post heavily himself, switch ai_disclosure_level to some_ai. -->

## What I Built

**Lakefront Window** answers one question for Chicagoans: *when is the lakefront actually nice tomorrow?*
It answers on a single printed page, so the screen is the shortest part of the plan.

A weather app gives you one reading for "Chicago," which usually means the airport. The lakefront often
doesn't match it. In spring and early summer, Lake Michigan chills the shore. Over 2015–2026 the Chicago Park
District's Oak Street beach sensor averaged **about 3 °C colder than Midway airport in May and June**. On
**77 days** there were hours when Midway read 60 °F or warmer while the beach read under 50 °F.

The tool:

1. learns how the Oak Street lakefront sensor differs from airport conditions, using **TabPFN-v2**, an
   open-weight tabular foundation model, running locally on a CPU;
2. applies that to the free **National Weather Service hourly forecast**;
3. prints a **one-page card**: the best 2-hour window that's entirely in daylight, an hour-by-hour chance of a
   "good outside hour," sunrise and sunset, and what a plain weather-app reading would have said.

Here is what counts as a "good outside hour." It's a transparent rule, and you can change it. At the Oak
Street sensor, between 7am and 7pm:

- the temperature is 10–27 °C (50–81 °F),
- hourly gusts are under 9 m/s (~20 mph),
- and no rain or snow is detected.

![Lakefront Window card for Tue Oct 6, 2026](https://raw.githubusercontent.com/brooksmoore/lakefront-window/main/out/card.png)
<!-- TODO: image URL works only once the repo is public; otherwise upload the PNG in the DEV editor. -->

The card for Tue Oct 6, 2026, built from the NWS forecast pulled the night before, is a boring one: sunny, 50–72 °F, good almost all day. The interesting cards are
the days when the lake and the app disagree.

## Demo

<!-- TODO: required by the rules ("a deployed link or video demo"). Upload out/demo.mp4 (35 s, silent) to YouTube and embed it here, e.g. {% embed https://youtu.be/XXXX %}, or host out/card.pdf via GitHub Pages. -->

The `--replay` mode re-runs a past **holdout** day, with a model trained only on 2015–2023, and lines it up
against what actually happened at the beach.

**Sat May 3, 2025: a classic "cooler by the lake" day.** Midway read 50–54 °F with moderate wind, so the
app-style reading said OK for 11 of 13 hours. The beach read 43–46 °F with gusts of 9–11 m/s, and none of
those hours were good. Lakefront Window called every hour correctly: "No great window."

![Replay of May 3, 2025](https://raw.githubusercontent.com/brooksmoore/lakefront-window/main/out/replay_2025-05-03.png)

**It also gets days wrong.** On Thu Apr 18, 2024 it said "not good" all day, but the beach turned out fine
for 11 of 13 hours. You can see it with `python run.py --replay 2024-04-18`. Both days were picked to
illustrate a point. The aggregate numbers below are the honest measure.

## Code

<!-- TODO: {% github brooksmoore/lakefront-window %} once the repo is public (repo must be created inside Oct 5–11). -->

```bash
python fetch_data.py   # once: beach sensor history, Midway history, NWS forecast, TabPFN-v2 weights
python run.py          # offline: holdout check + out/card.pdf  (~2-3 min on an 8-core CPU)
python run.py --replay 2025-05-03
```

## How I Built It

**Data (all free, no keys):**

- Chicago Data Portal *Beach Weather Stations – Automated Sensors* (`k7hf-8y75`): about 77k hourly rows
  from the Oak Street station, 2015 to today.
- Midway (KMDW) hourly airport observations from the Iowa Environmental Mesonet ASOS archive.
- `api.weather.gov` hourly forecast for the Midway gridpoint.

**Model: TabPFN-v2 classifier.** TabPFN is a transformer pretrained on synthetic tabular problems. You don't
train it; you hand it labeled rows as context, and it predicts new rows in one forward pass. The context is
3,000 randomly sampled training hours. The features are what the NWS hourly forecast provides: hour, day of
year, temperature, wind speed and direction, humidity, and rain yes/no.

```python
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier(model_path="models/tabpfn-v2-classifier-....ckpt", device="cpu",
                       random_state=0, ignore_pretraining_limits=True)  # 3,000 rows > the 1,000 CPU suggestion
clf.fit(ctx[FEATURES], ctx.label)   # "fit" = store the context, no gradient steps
p_good = clf.predict_proba(forecast[FEATURES])[:, 1]
```

The model sees rain only as yes/no, but the forecast gives a *chance* of rain. The card scores each hour
twice, once as dry and once as wet, and mixes the two results by the forecast's probability of precipitation.
Scoring all 11,397 holdout hours takes about 54 seconds on CPU.

**Honest holdout:** training uses 2015–2023. Testing uses every 7am–7pm hour from 2024-01-01 to 2026-10-05,
which is 11,397 hours.

| Method | Accuracy | Brier | Best-window hit rate* |
|---|---|---|---|
| Weather app as-is (same thresholds on airport conditions) | 0.846 | 0.154 | 0.510 |
| App + monthly lake temperature offset | 0.863 | 0.137 | 0.520 |
| Climatology (month × hour) | 0.760 | 0.171 | 0.466 |
| Logistic regression (29k hours) | 0.783 | 0.156 | 0.463 |
| Gradient boosting (29k hours) | **0.916** | **0.062** | 0.547 |
| **TabPFN-v2 (3k-hour context)** | 0.914 | 0.067 | **0.556** |

\*Each day the method picks one 2-hour window. A hit means both hours were actually good. Only 59.9% of days
had any good window, so 0.599 is the ceiling. On the days that had one, TabPFN's pick was good **92.8%** of the
time, against 85.1% for the app reading. TabPFN had fewer wrong hours than the app on 274 days, more on 89, and
tied on 563. It is also well calibrated: when it says 90%+, the hour was good 97% of the time.

What these numbers don't show:

- **They don't include forecast error.** The holdout feeds in actual airport observations as if they were a
  perfect forecast. There's no free archive of past NWS hourly forecasts to test against. Real day-ahead
  accuracy will be lower for every method, the app reading included. A synthetic noise test (±2 °C, ±1.5 m/s)
  drops TabPFN to 0.885 and the app to 0.817.
- **TabPFN didn't beat gradient boosting.** With a tenth of the data, no tuning, and no training loop, it
  tied a model trained on all 29k hours. That's the real result.
- **CPU runs jitter slightly.** Across 5 runs, accuracy ranged from 0.910 to 0.914.

## Why Does Open Innovation Matter?

- **It works where the lakefront is, with no signal.** After one data pull, the model, the weights, and the
  card all run on a laptop with networking off. The output is paper, which doesn't need a battery or a signal.
- **Nothing about you leaves your machine.** No location, no API key, no account, no per-call bill.
- **We could check it, not just trust it.** Open weights meant we could run the exact model on 11k hours of
  real data, compare it to plain baselines, and publish both its wins and its misses. A closed API can change
  under you between runs, and "it just predicts" isn't something you can audit.
- **The rules are yours.** The definition of a "good hour" is three lines in `lakefront.py`. If you run cold,
  change 10 °C to 5. If you bring a kite, flip the wind rule.
- **It's open all the way down.** City open data, NOAA/FAA observations, NWS public-domain forecasts, and
  model weights under an Apache-2.0-based license that requires attribution. Every link in the chain is inspectable. *Built with PriorLabs-TabPFN.*

## My Agent Session

The whole project was built by an AI agent, Alfred, in a terminal session: data pulls, modeling, the holdout
check, the card, and this write-up. The session wasn't recorded with DevRelay.
<!-- TODO: optional, remove this section if no session link is added. -->

## Prize Categories

- **Best Use of TabPFN.** TabPFN-v2 is the core model: in-context learning on 3,000 hours, calibrated
  probabilities, a rain-chance mixture, and run fully offline on CPU.
