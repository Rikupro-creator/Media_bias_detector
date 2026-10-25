import os
import re
import sys
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
from sklearn.metrics import cohen_kappa_score, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "political_annotated.parquet"

# Checkpoint files (safe to stop and re-run; finished rows are skipped)
RESULTS_FILE = "gemini_judge_results.jsonl"
ERRORS_FILE = "gemini_judge_errors.jsonl"

# Final outputs
OUTPUT_FILE = "political_verified.parquet"
DISAGREEMENTS_CSV = "judge_disagreements.csv"
METRICS_FILE = "judge_agreement_metrics.json"

MODEL = "gemini-3.1-flash-lite"          # verify this model ID is valid for your key
THINKING_LEVEL = "high"             # low | medium | high

ARTICLE_COLUMN = "article_text"
LUNA_LABEL_COLUMN = "bias_classification"
LUNA_REASON_COLUMN = "bias_reason"

# Concurrency / retry settings
MAX_WORKERS = 8         # lower this if you hit 429s
MAX_RETRIES = 5
BASE_BACKOFF = 2.0      # seconds, doubles each retry (with jitter)

# Evaluation settings
CLASSES = ["biased", "not_biased"]
N_BOOTSTRAP = 2000
SEED = 42


# ============================================================
# LOAD + VALIDATE DATA
# ============================================================

df = pd.read_parquet(INPUT_FILE)
print(f"Dataset loaded: {len(df):,} rows")

missing = [c for c in (ARTICLE_COLUMN, LUNA_LABEL_COLUMN, LUNA_REASON_COLUMN) if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

valid_luna_labels = {"biased", "not_biased"}
observed = set(df[LUNA_LABEL_COLUMN].dropna().astype(str).str.strip().str.lower().unique())
invalid = sorted(observed - valid_luna_labels)
if invalid:
    raise ValueError(f"Unexpected Luna labels: {invalid}. Expected {sorted(valid_luna_labels)}")


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are an expert evaluator of political news media framing
and an independent verifier of LLM-generated annotations.

You are evaluating annotations of Kenyan political news
articles.

The ARTICLE TEXT is the primary source of evidence.

Your task is NOT to determine whether the political claims
made in the article are factually true.

Your task is to determine whether the ARTICLE'S FRAMING
contains evidence of media bias.

CORE PRINCIPLES

1. Treat the article as the primary evidence.
2. Do NOT assume the initial LLM annotation is correct.
3. Perform an independent analysis of the article BEFORE
   evaluating the initial annotation.
4. The initial LLM annotation is a hypothesis to evaluate.
   It is NOT evidence.
5. Do not infer media bias merely from:
   - the political affiliation of a person
   - criticism or praise of a politician
   - criticism or support of government or opposition
   - controversial political content
6. Distinguish political content from biased framing.
7. Evidence of framing bias must be supported by the article.
8. Do not invent quotations, facts, sources, political
   positions, omitted information, or motives.
9. If the article does not provide sufficient evidence,
   classify it as "not_biased".
10. Be conservative when the evidence is ambiguous.

POSSIBLE FRAMING CUES

- loaded or evaluative language
- selective framing
- asymmetric treatment of political actors
- disproportionate positive or negative characterization
- selective omission that materially affects framing
- repeated emphasis creating a particular interpretation
- unequal attribution or sourcing
- systematic presentation of one side as credible and
  another as unreliable
- other identifiable framing mechanisms

IMPORTANT DISTINCTION

Political disagreement is NOT automatically media bias.
An article reporting that an opposition politician criticized
the government is not automatically biased. An article can
contain strongly negative political claims while still being
ordinary reporting if those claims are clearly attributed and
relevant opposing views are represented appropriately.

EVIDENCE REQUIREMENT

If you classify an article as "biased", identify concrete
textual evidence. The evidence quote MUST appear verbatim in
the article, be short, directly support the reasoning, and
not be paraphrased or fabricated. If there is no adequate
textual evidence, do not classify the article as biased.

INDEPENDENCE REQUIREMENT

Analyze the article and decide your independent label and
evidence BEFORE examining the Luna annotation. The Luna
annotation must never be treated as evidence.
"""


def build_prompt(article, luna_label, luna_reason):
    return f"""
============================================================
ARTICLE
============================================================

{article}

============================================================
INITIAL LLM ANNOTATION
============================================================

Luna label:
{luna_label}

Luna reason:
{luna_reason}

============================================================
EVALUATION TASK
============================================================

STAGE 1 - INDEPENDENT ANALYSIS

Ignore the Luna annotation initially. Determine independently:
1. Does the article contain evidence of media framing bias?
2. What specific textual evidence supports your conclusion?
3. What framing mechanism, if any, is present?
4. Is the evidence strong enough to justify a biased label?

STAGE 2 - VERIFY LUNA

After completing your independent assessment, determine:
1. Is Luna's label supported by the article?
2. Is Luna's stated reason supported by the article?
3. Does Luna's reasoning logically follow from the article?
4. If Luna is incorrect, explain why.

Do not change your independent judgment simply because Luna
gave a different label.

Return JSON matching the required schema.
"""


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "independent_label": {"type": "STRING", "enum": ["biased", "not_biased"]},
        "initial_label_supported": {"type": "BOOLEAN"},
        "initial_reason_supported": {"type": "BOOLEAN"},
        "evidence_present": {"type": "BOOLEAN"},
        "evidence_type": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": [
                    "loaded_language", "selective_framing", "asymmetric_treatment",
                    "source_imbalance", "omission", "positive_characterization",
                    "negative_characterization", "other",
                ],
            },
        },
        "evidence_quote": {"type": "STRING"},
        "verifier_reason": {"type": "STRING"},
        "disagreement_reason": {"type": "STRING"},
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "independent_label", "initial_label_supported", "initial_reason_supported",
        "evidence_present", "evidence_type", "evidence_quote",
        "verifier_reason", "disagreement_reason", "confidence",
    ],
    # Reasoning fields first, then the label
    "property_ordering": [
        "evidence_present", "evidence_type", "evidence_quote", "verifier_reason",
        "independent_label", "initial_label_supported", "initial_reason_supported",
        "disagreement_reason", "confidence",
    ],
}


# ============================================================
# CLIENT + GENERATION CONFIG
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)

# temperature intentionally left at default (recommended for Gemini 3 models)
GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    response_mime_type="application/json",
    response_schema=RESPONSE_SCHEMA,
    thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
)


# ============================================================
# PHASE 1: JUDGING (with checkpointing + progress bar)
# ============================================================

def load_done_keys():
    keys = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        keys.add(str(json.loads(line)["key"]))
                    except Exception:
                        pass
    return keys


write_lock = threading.Lock()


def judge_row(idx, row):
    article = str(row[ARTICLE_COLUMN]).strip()
    luna_label = str(row[LUNA_LABEL_COLUMN]).strip().lower()
    luna_reason = str(row[LUNA_REASON_COLUMN]).strip()
    prompt = build_prompt(article, luna_label, luna_reason)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt, config=GEN_CONFIG
            )
            text = resp.text
            if not text:
                raise ValueError("empty response text")
            json.loads(text)  # validate before saving
            return str(idx), text, None
        except Exception as e:
            last_err = e
            time.sleep(BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1))

    return str(idx), None, repr(last_err)


def save_success(key, text):
    record = {"key": key, "response": {"candidates": [{"content": {"parts": [{"text": text}]}}]}}
    with write_lock, open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_error(key, err):
    with write_lock, open(ERRORS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "error": err}, ensure_ascii=False) + "\n")


done_keys = load_done_keys()
if done_keys:
    print(f"Resuming: {len(done_keys):,} rows already judged")

todo = []
skipped_empty = 0
for idx, row in df.iterrows():
    if str(idx) in done_keys:
        continue
    if not str(row[ARTICLE_COLUMN]).strip():
        skipped_empty += 1
        continue
    todo.append((idx, row))

if skipped_empty:
    print(f"Skipped {skipped_empty:,} rows with empty articles")

# Clear old errors: failed rows are retried on this run
if os.path.exists(ERRORS_FILE):
    os.remove(ERRORS_FILE)

ok, failed = 0, 0

if todo:
    print(f"To judge: {len(todo):,} (workers={MAX_WORKERS}, thinking={THINKING_LEVEL})")
    print("Press Ctrl+C to stop safely; re-run to resume.\n")

    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = [pool.submit(judge_row, idx, row) for idx, row in todo]

    try:
        with tqdm(total=len(df) - skipped_empty, initial=len(done_keys),
                  desc="Judging", unit="article") as bar:
            for fut in as_completed(futures):
                key, text, err = fut.result()
                if err is None:
                    save_success(key, text)
                    ok += 1
                else:
                    save_error(key, err)
                    failed += 1
                bar.update(1)
                bar.set_postfix(ok=ok, failed=failed)
        pool.shutdown(wait=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Cancelling pending requests...")
        pool.shutdown(wait=False, cancel_futures=True)
        print(f"Progress saved in {RESULTS_FILE}. Re-run the script to resume.")
        sys.exit(0)

    print(f"\nJudging complete. Succeeded: {ok:,} | Failed: {failed:,}"
          + (f" (see {ERRORS_FILE}; re-run to retry)" if failed else ""))
else:
    print("Nothing left to judge.")


# ============================================================
# PHASE 2: PARSE RESULTS
# ============================================================

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


records, parse_failures = [], 0

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        try:
            text = rec["response"]["candidates"][0]["content"]["parts"][0]["text"]
            parsed = extract_json(text)
            parsed["_key"] = str(rec["key"])
            records.append(parsed)
        except Exception:
            parse_failures += 1

print(f"\nParsed results: {len(records):,} | Unparseable: {parse_failures:,}")

if not records:
    sys.exit("No judge results to evaluate.")

judge = (pd.DataFrame(records)
         .drop_duplicates(subset="_key", keep="last")
         .add_prefix("judge_")
         .rename(columns={"judge__key": "_key"}))

df["_key"] = df.index.astype(str)
valid = df.merge(judge, on="_key", how="inner")
valid["luna_label_norm"] = valid[LUNA_LABEL_COLUMN].astype(str).str.strip().str.lower()
valid = valid.dropna(subset=["judge_independent_label"]).copy()
valid["agree"] = valid["luna_label_norm"] == valid["judge_independent_label"]

n = len(valid)
print(f"Rows used for agreement: {n:,} of {len(df):,}")


# ============================================================
# PHASE 3: AGREEMENT METRICS
# ============================================================

po = valid["agree"].mean()

rng = np.random.default_rng(SEED)
agree_arr = valid["agree"].to_numpy()
boot = [rng.choice(agree_arr, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)]
ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

kappa = cohen_kappa_score(valid["luna_label_norm"], valid["judge_independent_label"], labels=CLASSES)
pabak = 2 * po - 1   # prevalence-adjusted; useful when classes are imbalanced

print("\n" + "=" * 60)
print("AGREEMENT: original (Luna) label vs judge independent label")
print("=" * 60)
print(f"N:                     {n:,}")
print(f"Raw agreement:         {po:.3f}  (95% bootstrap CI {ci_low:.3f} to {ci_high:.3f})")
print(f"Cohen's kappa:         {kappa:.3f}")
print(f"PABAK:                 {pabak:.3f}")

print("\nLabel distribution")
dist = pd.DataFrame({
    "luna": valid["luna_label_norm"].value_counts(),
    "judge": valid["judge_independent_label"].value_counts(),
}).fillna(0).astype(int)
print(dist)

cm = confusion_matrix(valid["luna_label_norm"], valid["judge_independent_label"], labels=CLASSES)
cm_df = pd.DataFrame(cm, index=[f"luna_{c}" for c in CLASSES], columns=[f"judge_{c}" for c in CLASSES])
print("\nConfusion matrix (rows = Luna, cols = judge)")
print(cm_df)

print("\nPer-class view (judge treated as reference)")
per_class = {}
for c in CLASSES:
    luna_c = valid["luna_label_norm"] == c
    judge_c = valid["judge_independent_label"] == c
    confirmed = int((luna_c & judge_c).sum())
    prec = confirmed / luna_c.sum() if luna_c.sum() else float("nan")
    rec = confirmed / judge_c.sum() if judge_c.sum() else float("nan")
    per_class[c] = {"luna_count": int(luna_c.sum()), "judge_confirms": prec, "luna_recovers_judge": rec}
    print(f"  {c:<11} Luna-labelled: {luna_c.sum():>5} | judge confirms: {prec:.3f} | Luna recovers judge's: {rec:.3f}")

print("\nAgreement by judge confidence")
by_conf = valid.groupby("judge_confidence")["agree"].agg(["count", "mean"]).round(3)
print(by_conf)


# ============================================================
# PHASE 4: JUDGE QUALITY CHECKS
# ============================================================

print("\n" + "=" * 60)
print("JUDGE SELF-CONSISTENCY CHECKS")
print("=" * 60)

quality = {}

incons = (((valid["judge_independent_label"] == "biased") & (~valid["judge_evidence_present"])) |
          ((valid["judge_independent_label"] == "not_biased") & (valid["judge_evidence_present"])))
quality["evidence_present_inconsistent"] = float(incons.mean())
print(f"evidence_present inconsistent with label:   {incons.mean():.1%}")

# Judge's own claim of whether Luna's label is supported vs computed agreement
claimed = valid["judge_initial_label_supported"].astype(bool)
mismatch = (claimed != valid["agree"]).mean()
quality["label_supported_vs_computed_mismatch"] = float(mismatch)
print(f"'initial_label_supported' != computed agree: {mismatch:.1%}")


def norm_text(s):
    s = str(s).lower()
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()


def quote_in_article(row):
    q = norm_text(row.get("judge_evidence_quote", ""))
    if not q:
        return np.nan
    return q in norm_text(row[ARTICLE_COLUMN])


valid["quote_grounded"] = valid.apply(quote_in_article, axis=1)
checked = valid["quote_grounded"].dropna()
if len(checked):
    quality["quote_grounded_rate"] = float(checked.astype(float).mean())
    print(f"Evidence quotes found verbatim in article:  {checked.astype(float).mean():.1%} ({len(checked):,} checked)")

biased_rows = valid["judge_independent_label"] == "biased"
if biased_rows.any():
    has_quote = valid.loc[biased_rows, "judge_evidence_quote"].astype(str).str.strip().ne("").mean()
    quality["biased_verdicts_with_quote"] = float(has_quote)
    print(f"'biased' verdicts that include a quote:     {has_quote:.1%}")


# ============================================================
# PHASE 5: DISAGREEMENT ANALYSIS
# ============================================================

dis = valid[~valid["agree"]].copy()
print(f"\nDisagreements: {len(dis):,} of {n:,}")

if len(dis):
    print("\nDirection of disagreement")
    print(dis.groupby(["luna_label_norm", "judge_independent_label"]).size().rename("count"))

    types_count = dis.explode("judge_evidence_type")["judge_evidence_type"].value_counts()
    print("\nEvidence types cited in disagreements")
    print(types_count)

    cols = [
        "_key", ARTICLE_COLUMN, LUNA_LABEL_COLUMN, LUNA_REASON_COLUMN,
        "judge_independent_label", "judge_confidence", "judge_evidence_quote",
        "judge_verifier_reason", "judge_disagreement_reason", "quote_grounded",
    ]
    dis[[c for c in cols if c in dis.columns]].to_csv(DISAGREEMENTS_CSV, index=False)
    print(f"\nSaved disagreements for manual review: {DISAGREEMENTS_CSV}")


# ============================================================
# PHASE 6: SAVE OUTPUTS
# ============================================================

valid.drop(columns=["_key"]).to_parquet(OUTPUT_FILE, index=False)

metrics = {
    "n_evaluated": int(n),
    "n_total": int(len(df)),
    "raw_agreement": float(po),
    "agreement_ci95": [float(ci_low), float(ci_high)],
    "cohens_kappa": float(kappa),
    "pabak": float(pabak),
    "confusion_matrix": cm_df.to_dict(),
    "per_class": per_class,
    "agreement_by_confidence": by_conf.to_dict(),
    "judge_quality": quality,
    "model": MODEL,
    "thinking_level": THINKING_LEVEL,
}
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, default=str)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Verified dataset:  {OUTPUT_FILE}")
print(f"Metrics summary:   {METRICS_FILE}")
print(f"Disagreements:     {DISAGREEMENTS_CSV}")