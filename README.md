# Fantasy Cricket AI Challenge

Traditional-ML solution for **pre-toss fantasy-point prediction and legal best-XI selection** using Cricsheet T20 data.

Given a match date, two full squads and declared roles, the system predicts every candidate's fantasy points and returns an exactly valid XI in descending order of predicted points. No toss, official-XI, scorecard or target-match information is ever used.

- **Model:** Histogram Gradient Boosting (MAE loss) on 22 leakage-safe rolling features
- **Selector:** exact, deterministic role-constrained XI optimizer
- **Data cutoff:** fixed at `2025-12-31`. No 2026 dataset or result is included.

---

## Headline results

Temporal holdout: all matches from 2024-01-01 to 2025-12-31 (3,577 matches, 116,521 candidate rows), with training on earlier matches only.

| Model | MAE | RMSE | Spearman | Top-11 capture | Top-11 overlap |
|---|---:|---:|---:|---:|---:|
| Last-5 mean baseline | 28.992 | 37.145 | 0.251 | 0.5716 | 0.4810 |
| Extra Trees | 22.190 | **30.785** | 0.550 | 0.6654 | 0.5611 |
| **Histogram Gradient Boosting (final)** | **19.474** | 32.277 | **0.565** | **0.6667** | **0.5648** |

- MAE improves by **32.8%** over the baseline, and top-11 capture improves by **9.51 percentage points**.
- Extra Trees has the lowest RMSE but weaker MAE and selection metrics, so the booster is the final model.
- The model artifact records an appearance ROC-AUC of **0.8860265**.

**Capture** is the points earned by the predicted top 11 divided by the hindsight-oracle top 11, before role constraints.

---

## Required files

- `model_artifacts/model_2025-12-31.pkl`: ready model with appearance ROC-AUC **0.8860265**
- `src/`: download, training, prediction, scoring, feature and XI-selection code
- `requirements.txt`: Python dependencies
- `run.sh` / `run.ps1`: complete pipeline (Linux/macOS and Windows)
- `reports/`: technical report and literature review (`reports/literature_review.md`)
- `outputs/figures/`: feature importance, target distribution and validation residual plots
- `results/hidden_predictions.csv`: generated after the organizer supplies the hidden fixture

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/download_data.py --raw-dir data/raw --manifest data/raw/download_manifest.json
```

Run the unit tests (scoring, optimizer constraints and cutoff behaviour):

```bash
python -m pytest -q
```

## Required training command

```bash
python src/train.py --train-start 2000-01-01 --train-end 2025-12-31
```

`train.py` refuses any `--train-end` after `2025-12-31`.

## Required evaluation command

Place the organizer-provided fixture at `data/raw/test_fixtures.json`, then run:

```bash
python src/evaluate.py --model-path model_artifacts/model_2025-12-31.pkl --test-file data/raw/test_fixtures.json --output results/hidden_predictions.csv
```

## Full pipeline

```bash
chmod +x run.sh
./run.sh          # Windows PowerShell: ./run.ps1
```

The official hidden CSV is not fabricated or committed before `test_fixtures.json` is supplied.

---

## Problem definition

The task combines zero-inflated regression, within-match ranking and constrained optimization.

**Available before the toss:** match date, teams, venue or city when supplied, team type, gender, full squads, player names or IDs, and declared fantasy roles.

**Not available (never used):** toss, official XI, batting order, innings result, and any event from the target match.

### Valid prediction

- Exactly **11 players**, sorted by predicted points (descending)
- **WK 1-4, BAT 3-6, AR 1-4, BOWL 3-6**
- **At least one player from each side**
- No captain multipliers

### Output format

`src/evaluate.py` writes one wide CSV row per fixture (36 columns). It contains `Match Date`, `Team 1` and `Team 2`, followed by 11 repeated groups of player name, role and predicted points, in descending order of predicted points.

---

## Data

The only match source is the official [Cricsheet JSON archive](https://cricsheet.org/downloads/).

| Item | Value |
|---|---|
| Matches | 13,085 T20 / IT20 matches |
| Date range | 2005-02-17 to 2025-12-31 |
| Unique players (registry IDs) | 11,458 |
| Candidate rows | 424,084 |
| Mean / median target | 24.90 / 9.0 points |
| Played rate | 64.4% |

- Ball-level events are converted to the supplied fantasy rules. Super Overs are ignored.
- Cricsheet does not publish full historical squads, so each team's **five most recent XIs** approximate its pre-toss candidate pool. This adds realistic zero labels for rotation risk without pretending every observed player was certain to appear.
- The target is zero-inflated (absent candidates score 0) and right-skewed (centuries and wicket hauls). Rows within a match are highly dependent.

### Data quality

- Missing target values: 0
- Exact duplicates on `(match_id, team, player_id)`: 0
- Stable registry IDs are used when available; names are kept in outputs for readability.
- Historical transfer candidates that appear for the opponent are removed from the stale prior-team pool.
- Team-history state includes gender, so men's and women's national teams never share state.

![Target distribution](outputs/figures/target_distribution.png)

---

## Method

### Features (22)

Features encode:

- recent team selection frequency (`team_selection_rate5`) and days since last appearance (`days_since_last`)
- rolling form at three horizons (last 3, 5 and 10)
- career batting, bowling and fielding components
- workload proxies
- team and opponent form
- venue scoring level
- team type and gender

Every rolling statistic is built by walking matches in date order. State is updated only after the feature rows for a match are emitted.

### Model

Histogram Gradient Boosting with:

| Setting | Value |
|---|---|
| Loss | MAE |
| Iterations | 250 |
| Learning rate | 0.06 |
| Leaves | 31 |
| L2 regularization | 2.0 |

This handles nonlinear interactions between availability, recent form and career role without large categorical encodings.

### XI selection

The downstream exact optimizer enumerates feasible role counts and uses sorted role lists, with a minimum-loss swap if both teams are not initially represented. The selection is deterministic and always legal. Fixture-provided roles override historical role inference.

### Final model artifact

The artifact contains the fitted booster, feature list, cutoff metadata, registry-name map and all chronological state needed for future fixtures. New-name cold starts use conservative zero history.

---

## Validation strategy

All matches from **2024-01-01 to 2025-12-31** form a blind-like temporal holdout. A random row split would place a player's future matches in training for an earlier validation match and overstate performance.

- **MAE** is the primary calibration metric: interpretable in fantasy points and less dominated by rare explosions than RMSE.
- **RMSE** highlights costly misses on explosive performances.
- **Spearman** and **top-11 capture** measure the actual ranking decision.
- **Top-11 overlap** measures agreement between the predicted and oracle top 11.

---

## Ablation

| Features | MAE | Spearman | Capture |
|---|---:|---:|---:|
| Full 22-feature model | **19.474** | **0.565** | **0.6667** |
| Remove recent selection and days since last match | 21.938 | 0.334 | 0.5972 |
| Form-only seven features | 22.251 | 0.303 | 0.5902 |

Availability features provide the largest measurable gain. Permuting `team_selection_rate5` raises MAE by 2.26 and permuting `days_since_last` raises it by 2.17; recent form is the next most important family. Appearance modeling contributes more than adding additional form horizons alone.

![Feature importance](outputs/figures/feature_importance.png)

---

## Error analysis

- Played-player MAE is **27.14**, against **5.53** for absent candidates. The model predicts absences accurately.
- It compresses the upside of players who appear. For the 15+ match group, the mean actual score is 41.54 points and the mean prediction is 23.01.
- Established players are hardest, because they combine high expectation with high match-to-match variance.
- MAE training stabilizes median predictions but compresses high upside.

![Validation residuals](outputs/figures/validation_residuals.png)

---

## Leakage audit

- **Hard date guard:** the raw archive may contain matches after the cutoff, but `iter_matches` filters by event date and `train.py` refuses any `--train-end` after `2025-12-31`. The latest processed match date is exactly `2025-12-31`.
- **Prior-state features only:** target-match updates happen after the feature row is emitted.
- **No forbidden signals:** toss, official-XI flags, scorecard results and target-match events are never used as features.
- **Temporal holdout:** no random row splits.
- **Venue and opponent aggregates** are computed chronologically, not over the full dataset.
- **Manual audit:** found and fixed a gender collision in national-team histories and stale cross-team transfer candidates.
- **Integrity check:** zero duplicates on `(match_id, team, player_id)`.

---

## Sample fixture check

The checked sample fixture produced a 36-column CSV row with 11 players, normalized roles, descending predictions, legal role counts, and representation from both teams (India and Pakistan). Unit tests cover scoring, optimizer constraints and cutoff date behaviour.

---

## Limitations

- Cricsheet does not publish full historical squads, so candidate pools are approximations.
- Direct-hit run-outs cannot be distinguished.
- Inferred historical roles are approximate (fixture-provided roles override them at inference).
- Cold-start players have little signal.
- The pre-toss setting creates irreducible uncertainty around selection and role.
- The model compresses the upside of high-scoring outcomes.

## Future work

A two-stage appearance model plus conditional-points model, conditional quantile prediction, or a ranking loss are the most promising extensions.

---

## Literature and references

- [Cricsheet JSON format](https://cricsheet.org/format/json/) and [download catalogue](https://cricsheet.org/downloads/): the data foundation
- *PlayerRank: Leveraging Learning-to-Rank AI for Player Positioning in Cricket*: motivates Spearman and top-11 capture alongside MAE
- *FanCric: Multi-Agentic Framework for Crafting Fantasy 11 Cricket Teams*: prediction plus operations-research framing; this project uses a deterministic exact optimizer instead of agentic inference
- [Predicting Outcome of Indian Premier League Matches Using Machine Learning](https://arxiv.org/abs/1809.09813): relevance of player form and context
- No commercial APIs, no post-toss information and no LLM in the scoring path

The full review is in `reports/literature_review.md`.

---

## Reproducibility

1. Install `requirements.txt`
2. `python -m pytest -q`
3. Run the training and evaluation commands above, or `run.sh` / `run.ps1`

The archive hash, experiment tables, internal predictions, plots and final model are stored in the repository.
