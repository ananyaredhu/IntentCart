# IntentCart — Conversational Shopping Copilot

A submission for TikTok TechJam 2026's "Shopping Copilot: AI Conversational
Search and Recommendations" challenge: an agent that finds a customer's
hidden target product, out of 50,000 catalog items, in as few conversation
turns as possible — combining hybrid retrieval, adaptive slot-filling
dialogue, dynamic re-ranking, and transparent explanations, entirely offline.

**Verified result on the 200-session public set**: Hit Rate@10 `0.89`, MRR
`0.452`, MTTC `3.65`, **TechnicalScore `0.728`** — up from the provided
starter baseline's `0.107` (Hit Rate@10 `0.125`, MRR `0.068`).

## 1. Project overview

The organizer's starter agent is a stateless BM25 search that only looks at
the current turn's message. This submission (`starter/agent.py`,
`starter/retrieval.py`, `starter/state.py`) replaces it with a full pipeline:

- **Hybrid multi-route retrieval** — keyword search (SQLite FTS5/BM25),
  category-taxonomy overlap, and dense vector similarity (a small local
  sentence-transformer), fused with Reciprocal Rank Fusion and continuously
  reweighted by a live Buying-vs-Browsing signal recomputed every turn.
- **A category-first hard filter** — category is the one attribute the
  simulator discloses for free and reliably every time, unlike
  material/color/budget which are probabilistically parsed from free text —
  so it's used to narrow the 50,000-product search universe down to the
  genuinely relevant category before anything else has to discriminate.
- **A structured dialogue state machine** — slot-filling across 7
  simulator-answerable attributes, override detection (both phrase cues and
  structural conflict, so it doesn't depend on exact wording), abstention
  memory, and fallback capture for disclosures that don't match a known
  vocabulary.
- **Adaptive, information-theoretic clarification** — decides whether the
  candidate pool is still too broad (Shannon entropy of category spread,
  price variance) and, if so, asks whichever unasked attribute would split
  the current candidates most usefully — restricted to the 7 attribute types
  the grading simulator actually answers.
- **IDF-weighted structured re-ranking** — a Bayesian-shrunk rating prior,
  profile-tag overlap, and embedding similarity to a plain-text session
  summary, plus attribute-match boosts scaled by how rare each matched term
  actually is across the catalog (a boilerplate word like "imported," on
  15,300 of 50,000 products, barely moves the ranking; a genuinely rare,
  distinguishing word moves it much more).
- **Failure detection and strategy switching** — a hard filter that would
  empty the candidate pool gets dropped instead of returning nothing;
  stalled narrowing (no entropy improvement for 2 turns) triggers a broader
  search; the last few turns of an unconfident session diversify the top-10
  across categories instead of staying clustered around one guess.
- **Transparent, state-grounded explanations** — the customer-facing message
  is built from the same structured state and re-ranking signals that
  actually drove that turn's ranking (e.g. *"Showing leather options under
  $80, prioritizing well-rated items"*), not a generic template.

Every one of the design decisions above was validated by running the full
200-session local evaluator before and after, and kept only when it
measurably helped — several plausible-sounding ideas (a consecutive-
abstention cutoff, a cross-encoder re-ranker, scoping extraction to only the
asked attribute) were tried, measured to make things worse, and reverted.
See Section 5 for what didn't make the cut and why.

## 2. Setup and installation

Python 3.10+ required (developed and tested on 3.12).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the frozen catalog from the organizer's participant-kit release
(not bundled in this repo, since it's a large binary asset):
[TechJam2026/techjam-conversational-search releases/participant-kit](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit)

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the download against the release's published `SHA256SUMS` file before use.

## 3. Steps to reproduce the results

```bash
python3 -m evaluator.local_evaluator
```

This prints the aggregate metrics (and a per-scenario breakdown for
Buying/Browsing/Intent Override/Boundary) and writes the full per-session
results to `results.json`. No arguments or configuration are needed — the
agent is fully self-contained.

**First run only**: this downloads a small local sentence-transformer model
(`all-MiniLM-L6-v2`, ~80MB) from Hugging Face and computes embeddings for
all 50,000 catalog products once, caching the result to
`data/catalog.embeddings.npy` (~77MB, regenerated automatically if
missing). **This is the only network access the agent ever needs**, and
only on this first run — every subsequent run, including official grading,
works entirely offline from that cache. If network access is unavailable
and no cache exists yet, dense retrieval degrades gracefully (skipped, not
crashed) rather than failing the whole agent.

Expect the full 200-session run to take roughly 60-90 seconds after the
first (embedding-building) run.

## 4. Model, cost, and network disclosure

No LLM API is used anywhere in the shipped critical path. The only model
involved is the local, offline sentence-transformer above — no API key, no
credentials, and no per-request cost. Token usage is reported as `0` for
every turn since no metered model is called. Latency is dominated by the
one-time embedding build; per-turn inference is well under a second.

We did prototype an optional Gemini API reranking layer, built with a
strict fallback (available only if `GEMINI_API_KEY` is set, with a hard
timeout and full response validation, silently no-op on any failure) so the
core system would remain unaffected either way. Direct measurement showed
real per-call latency of 40+ seconds against the account/tier we tested —
impractical within a 10-turn conversation budget — so it was removed from
the final submission rather than shipped as dead weight. This is a
deliberate, evidence-based choice: fully offline and reliably reproducible
was judged more valuable than a capability that would be pure no-op risk
under judging conditions where network access may be restricted.

## 5. Limitations and what we'd improve with more time

- **Ranking precision (MRR) has more headroom than coverage (Hit Rate).**
  Pool recall — whether the true target ever appears anywhere in the raw
  retrieval pool — measures ~99% on the public set, meaning the retrieval
  side is close to its ceiling; the remaining gap is concentrated in
  ranking precision (getting an already-findable candidate into the top
  few positions, not just the top 10). The most promising next step we
  identified but didn't build is a small learned ranking model (logistic
  regression or gradient-boosted trees over hand-engineered features:
  BM25/category/embedding rank, rating, price match, slot overlap),
  trained on synthetic conversations generated from the evaluator's own
  deterministic `intent_card()`/`behavior_for()` functions applied to many
  more catalog products than the 200 we were given labels for. This stays
  within the "no full-model training" rule (it's a lightweight classifier
  on top of features, not fine-tuning the embedding/LLM foundation models)
  but needs real time to build and validate without overfitting to the
  synthetic generation process.
- **No held-out validation split.** Every tuning decision this session was
  measured and kept/reverted against the same 200 public sessions — there
  is no internal train/validation split, so the reported score carries some
  unmeasured optimism about generalization to the private 800 sessions.
  A stratified k-fold check on the public set (which we did run once,
  finding a ±0.03 spread around the mean) is a reasonable proxy but isn't a
  substitute for genuinely unseen data.
- **Hand-vocabulary limits.** Material/color/style/use_case/feature
  recognition relies on vocabulary lists mined from real catalog word
  frequency (not guessed), but any fixed list has coverage gaps. The
  fallback-capture path (storing the raw reply text when nothing matches)
  covers this partially but not with the same precision as a recognized
  term.
- **A genuinely more capable semantic signal (e.g. a well-integrated LLM
  reranker) remains an open, likely-valuable direction** if a low-latency
  provider or a much faster local model were available — our own
  measurement of one specific free-tier option showed it wasn't practical
  here, but that's a property of the specific model/tier tested, not
  evidence against the general idea.

## 6. Team

Solo submission — all design, implementation, testing, and this writeup by
Ananya Redhu.

## 7. Agent interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## 8. Technical metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario (Buying/Browsing/Intent Override/Boundary).

## 9. Repository layout

```text
starter/agent.py       Agent entry point: orchestrates retrieval, dialogue state, and re-ranking
starter/retrieval.py   BM25/category/dense routes, RRF fusion, IDF weighting, vocabularies
starter/state.py       Slot-filling state machine, clarification policy, explanations
data/public_set.jsonl  200 labeled development sessions
evaluator/local_evaluator.py  public-set simulator and scorer (unmodified, organizer-provided)
docs/                  competition specification, agent API contract, submission rules
```

## 10. Data source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data. Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
