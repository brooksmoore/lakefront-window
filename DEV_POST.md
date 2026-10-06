---
title: "Lakefront Window: a one-page TabPFN card for when Chicago's beach is colder than your weather app"
published: false
tags: devchallenge, hf26challenge, tabpfn, python
ai_disclosure_level: fully_autonomous
cover_image: ""
---

<!-- TODO cover_image: upload out/card.png in the DEV editor, or use the raw GitHub URL once the repo is public -->
*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 1: Touch Grass](https://dev.to/challenges/hacktoberfest-week1-2026-10-05)*

> **AI disclosure:** Alfred, Brooks Moore's AI agent, built this project and wrote this post. Claude, a second
> AI agent on the same team, fact-checked the numbers against the code's output and edited the text. It's
> published from Brooks's account, and Brooks is accountable for it. Nobody has carried the card to the beach yet,
> so nothing here is a first-hand outdoor story. Every scene below comes from sensor logs.
> <!-- TODO before publishing: Brooks confirms he reviewed it. If he rewrites it himself, switch ai_disclosure_level to some_ai. -->

## The 10-mile problem

On Saturday, May 3, 2025, Chicago Midway Airport logged a mild afternoon: 53 °F, no rain. Ten miles
northeast, the Park District's weather sensor at Oak Street Beach logged 43–46 °F all day, with gusts up to
11 m/s (about 25 mph). By an ordinary weather-app reading, that was a fine day for a walk on the lakefront.
At the lakefront, it wasn't.

This is Chicago's "cooler by the lake" effect. Lake Michigan stays cold well into summer, and onshore wind
carries that chill onto the shore. Over 2015–2023, during 7am–7pm, the Oak Street sensor averaged **3.0 °C
colder than Midway in May** and **2.8 °C colder in June**. Across the joined 7am–7pm record
(2015–2026), there were **77 days** with hours when Midway read 60 °F or warmer while the beach
sensor read under 50 °F (see `midway_warm_lakefront_cold_hours` in `out/results.json`). Your weather
app gives you one number for "Chicago," and that number usually comes from the airport.

## What I Built

**Lakefront Window** answers one question: *when is the lakefront actually nice tomorrow?* It's for anyone
who walks, runs, or bikes along Chicago's 18-mile Lakefront Trail and has dressed for the airport and then
met the lake instead.

You run it the night before. It prints one page and then gets out of the way. You fold the card, leave the
phone, and go. The screen is the shortest part of the plan.

How it works:

1. **TabPFN-v2**, an open-weight tabular foundation model running on a laptop CPU, learns how the Oak Street
   sensor differs from airport conditions.
2. It applies that to tomorrow's free **National Weather Service hourly forecast**.
3. It prints a **one-page card** with the best 2-hour daylight window, an hour-by-hour chance of a "good
   outside hour," sunrise and sunset, and what a plain weather-app reading would have said.

A "good outside hour" is a simple, visible rule, not a black box. At the Oak Street sensor, between 7am and
7pm:

- the temperature is 10–27 °C (50–81 °F),
- hourly gusts are under 9 m/s (~20 mph),
- and no rain or snow is detected.

If you run cold or fly kites, change it. It's three lines in `lakefront.py`.

![Lakefront Window forecast card](https://brooksmoore.github.io/lakefront-window/card.png)
<!-- TODO: regenerate the card from a fresh forecast on publish day and update this caption; the image URL works only once the repo is public (otherwise upload the PNG in the DEV editor). -->

*A forecast card for Tue Oct 6, 2026, built from the NWS forecast issued the night before. It's a boring
day: sunny, 50–72 °F, good almost all day. The interesting cards are the days when the lake and the app
disagree.*

## Demo

**Live demo:** [brooksmoore.github.io/lakefront-window](https://brooksmoore.github.io/lakefront-window/) — today's card (PDF + image) and two holdout replays.

There's no archive of old NWS forecasts, so the on-page replays use `--replay` instead. It re-runs a past day the model
never saw. The model is trained only on 2015–2023, and the replay feeds in **what Midway actually observed**
as a stand-in for a perfect forecast. Then it lines the predictions up against what the beach sensor recorded.

**Sat May 3, 2025: the day from the top.** Midway read 46–54 °F, so an app-style reading said "OK" for 11 of 13
hours. Lakefront Window gave every hour a 13% chance or less and printed "No great window." The beach sensor
agreed on all 13 hours. To be fair, the simplest baseline, "airport temperature plus the average monthly lake
offset," also got this day right. The model earns its keep on the averages below, not on one dramatic day.

![Replay of May 3, 2025](https://brooksmoore.github.io/lakefront-window/replay_2025-05-03.png)

**And a day it got wrong: Thu Apr 18, 2024.** It said "not good" all day, but the beach turned out fine for
11 of 13 hours. The plain app reading did better that day. You can check it with
`python run.py --replay 2024-04-18`. Both replay days were chosen to illustrate a point, one good and one bad.
The holdout numbers are the honest measure.

## Code

<!-- TODO: {% github brooksmoore/lakefront-window %} once the repo is public (the repo must be created Oct 5–11). -->

```bash
python fetch_data.py               # once, online: beach + Midway history, NWS forecast, TabPFN-v2 weights
python run.py                      # offline: holdout check + out/card.pdf
python run.py --replay 2025-05-03  # re-run a past day next to what really happened
```

## How I Built It

**Data (free, no API keys):**

- Chicago Data Portal, *Beach Weather Stations – Automated Sensors* (`k7hf-8y75`): hourly readings from the Oak
  Street station since 2015.
- Midway (KMDW) hourly airport observations from the Iowa Environmental Mesonet ASOS archive.
- The `api.weather.gov` hourly forecast for Midway's gridpoint.

**Model: the TabPFN-v2 classifier.** TabPFN is a transformer pretrained on millions of synthetic tabular
problems. You don't train it. You hand it labeled rows as context, and it predicts new rows in a single
forward pass. Here the context is 3,000 randomly sampled training hours. The features are only things an NWS
hourly forecast provides: hour, day of year, temperature, wind speed and direction, humidity, and rain yes/no.

```python
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier(model_path="models/tabpfn-v2-classifier-....ckpt", device="cpu",
                       random_state=0, ignore_pretraining_limits=True)  # 3,000 rows > the 1,000 CPU suggestion
clf.fit(ctx[FEATURES], ctx.label)   # "fit" = store the context, no gradient steps
p_good = clf.predict_proba(forecast[FEATURES])[:, 1]
```

The model sees rain only as yes/no, but forecasts give a *chance* of rain. So the card scores each hour twice,
once dry and once wet, and blends the two by the forecast's precipitation probability. Fitting the model and
scoring all 11,397 holdout hours takes about 54 seconds on CPU.

**The holdout test.** Training data is 2015–2023. Testing uses every 7am–7pm hour from 2024-01-01 to
2026-10-05 where both stations reported: 11,397 hours the model never saw.
**Read this before the table:** the inputs are actual airport observations standing in for a perfect forecast,
so real forecast error isn't included (more on that below).

| Method | Accuracy | Brier ↓ | Best-window hit rate* |
|---|---|---|---|
| Weather app as-is (same rule on airport conditions) | 0.846 | 0.154 | 0.510 |
| App + monthly lake temperature offset | 0.863 | 0.137 | 0.520 |
| Climatology (month × hour) | 0.760 | 0.171 | 0.466 |
| Logistic regression (29k hours) | 0.783 | 0.156 | 0.463 |
| Gradient boosting (29k hours) | **0.916** | **0.062** | 0.547 |
| **TabPFN-v2 (3k-hour context)** | 0.914 | 0.067 | **0.556** |

\*Each day, the method picks one 2-hour window, and a hit means both hours really were good. Only 59.9% of test
days had any good window, so 0.599 is the ceiling.

What the table means in plain terms:

- **The pick is usually right.** On days that had a good window, TabPFN's pick was good **92.8%** of the time.
  The app reading's pick was good 85.1% of the time.
- **It's wrong less often, day by day.** TabPFN had fewer wrong hours than the app on 274 days and more on 89.
  On 563 days they tied.
- **Its probabilities are conservative.** When it says 70–90%, the hour was good 93% of the time. When it says
  90%+, the hour was good 99% of the time. Below 50%, its numbers match reality closely. Above 50%, the real
  odds are better than it says, so a green bar on the card, if anything, undersells the hour.

What the numbers don't show:

- **Forecast error.** No free archive of past NWS hourly forecasts exists, so every method above got a perfect
  "forecast." Real day-ahead accuracy will be lower for all of them. A synthetic noise test (σ 2 °C, 1.5 m/s)
  drops TabPFN to 0.885, gradient boosting to 0.888, and the app to 0.817.
- **A TabPFN win over gradient boosting.** There isn't one. With a tenth of the data, no tuning, and no
  training loop, TabPFN *tied* a boosted model trained on all 29k hours. That's the honest result, and I think
  it's the interesting one.
- **Bit-for-bit repeatability.** TabPFN on CPU jitters slightly between runs; window picks round to 2 decimals
  so near-ties do not flip. (A multi-run accuracy band was observed on this box but is not checked into `out/`.)
- **Other beaches.** This is one sensor at one beach.

## Why Does Open Innovation Matter?

- **It works at the lake, with no signal.** After one data pull, the model, the weights, and the card all run on
  a laptop with networking off. The output is paper, which needs no battery or bars.
- **Nothing about you leaves your machine.** No location, no API key, no account, no per-call bill.
- **We could check it, not just trust it.** Open weights let us run the exact model on 11k hours of real data,
  compare it to plain baselines, and publish its misses next to its wins. A closed API can change between runs,
  and "trust us, it predicts" isn't something you can audit.
- **The rules are yours.** The idea of a "good hour" is a personal judgment, so it lives in plain code you can
  edit, not in a vendor's settings page.
- **It's open all the way down.** City open data, NOAA/FAA observations, public-domain NWS forecasts, and model
  weights under an Apache-2.0-based license with attribution. You can inspect every link in the chain.
  *Built with PriorLabs-TabPFN.*

## My Agent Session

An AI agent, Alfred, built the whole project in a terminal session: the data pulls, modeling, holdout check,
card, and first draft of this post. Claude, another AI agent, audited every number against the run output
before publishing. The session wasn't recorded with DevRelay.
<!-- TODO: optional; delete this section if no session link is added. -->

## Prize Categories

- **Best Use of TabPFN.** TabPFN-v2 is the core model. It does in-context learning from a 3,000-hour sample,
  tied a fully trained gradient-boosting model with no tuning, gives conservative probabilities that a rain-chance
  blend can build on, and runs fully offline on CPU.
