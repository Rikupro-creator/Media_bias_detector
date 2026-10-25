import os
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI


# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = "political.parquet"
OUTPUT_FILE = "political_annotated.parquet"

MODEL = "gpt-5.6-luna"

MAX_WORKERS = 8          # concurrent API calls; raise/lower based on your rate limits
CHECKPOINT_EVERY = 100
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2     # seconds; doubles each attempt (+ jitter)


# =========================================================
# OPENAI CLIENT (thread-safe, shared across workers)
# =========================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Set your API key as an environment variable first."
    )

# max_retries=0 so our own backoff loop below is the only retry layer
client = OpenAI(api_key=api_key, max_retries=0)


# =========================================================
# PROMPT
# =========================================================

PROMPT_TEMPLATE = """
You are an expert annotator of political news media and media framing, with deep knowledge of Kenya's political, cultural, social, and journalistic context and how political language and narratives are understood by Kenyan audiences.

Your task is to determine whether the following Kenyan political news article contains evidence of MEDIA BIAS in its framing.

## CRITICAL EVIDENCE REQUIREMENT

YOUR CLASSIFICATION MUST BE BASED ON EVIDENCE PRESENT IN THE ARTICLE TEXT.

You MUST identify and use specific evidence from the provided article to justify your classification.

Do NOT classify an article as "biased" based on:

* your general knowledge about the politician, party, government, opposition, or institution;
* the political reputation of the people mentioned;
* your knowledge of Kenyan politics;
* assumptions about the journalist or news organization;
* the political topic being discussed;
* information that is not contained in the article.

If you classify an article as "biased", the `bias_reason` MUST identify concrete evidence from the article text that demonstrates the bias.

The evidence MUST be directly traceable to the article. For example, identify:

* specific words or phrases that use loaded or emotive language;
* specific descriptions or labels applied to a political actor;
* an identifiable imbalance in how competing political sides are presented;
* specific statements or sections that favor or disadvantage one side;
* relevant perspectives that are visibly omitted from the article;
* differences in the treatment of political actors that can be observed in the text.

DO NOT provide a generic justification such as:

* "The article is biased because it favors one side."
* "The article uses negative language."
* "The article is one-sided."

Those statements alone are NOT sufficient.

Instead, explain WHAT in the article demonstrates the bias.

For example:

Weak justification:
"The article is biased against the government because it is critical."

Acceptable justification:
"The article repeatedly describes the government's actions using strongly negative terms such as '...' and '...', while presenting the opposition's allegations without comparable scrutiny or an alternative government explanation."

The justification must therefore connect the classification to observable evidence in the article.

## EVIDENCE FOR NOT_BIASED

Evidence is also required when classifying an article as `not_biased`.

The justification should explain what is present in the article that supports a predominantly balanced or factual presentation.

For example, the article may:

* attribute allegations clearly to their sources;
* present relevant responses from opposing political actors;
* distinguish facts from political claims;
* use predominantly descriptive rather than loaded language;
* provide multiple relevant perspectives;
* avoid systematically favorable or unfavorable descriptions.

Do not simply say:
"The article is neutral."

Explain briefly what in the text supports that assessment.

## IMPORTANT DISTINCTION

Political criticism is NOT automatically media bias.

An article may report:

* criticism of the government;
* accusations against a politician;
* opposition claims;
* allegations of corruption;
* protests;
* controversial statements;
* government failures;

without being biased.

The question is whether the ARTICLE'S OWN PRESENTATION OR FRAMING provides identifiable evidence of favoring or disadvantaging a political actor, political side, or political perspective.

When a statement is clearly attributed to a source, distinguish the source's claim from the journalist's own framing.

For example:

"Opposition leader X accused the government of corruption."

This alone does not demonstrate that the article is biased against the government because the accusation is explicitly attributed.

However, if the article repeatedly presents accusations as established facts, uses loaded descriptions, omits a relevant response, or otherwise frames the issue in a consistently one-sided manner, those textual features may constitute evidence of bias.

## WHAT TO CONSIDER

Consider evidence such as:

1. LOADED OR EMOTIVE LANGUAGE

Language that portrays a political actor, group, or institution in a strongly favorable or unfavorable way beyond straightforward factual reporting.

2. ONE-SIDED FRAMING

The article substantially presents one political side's position, interpretation, or narrative while providing little or no meaningful representation of relevant opposing perspectives.

3. UNEQUAL TREATMENT

Similar actions or claims by different political actors are treated differently through descriptions, scrutiny, context, or presentation.

4. SELECTIVE EMPHASIS

Particular facts, events, allegations, or viewpoints are emphasized in a way that creates a discernible favorable or unfavorable portrayal.

5. SOURCE OR PERSPECTIVE OMISSION

A relevant perspective is absent in a way that materially affects the framing of the political issue, where the absence is reasonably identifiable from the article itself.

6. FRAMING THROUGH DESCRIPTION

The choice of descriptions, labels, quotations, headlines, or contextual information creates a discernible favorable or unfavorable portrayal.

## DO NOT CLASSIFY AS BIASED MERELY BECAUSE:

* the article reports criticism of a politician;
* the article reports an allegation;
* the article quotes someone making a political accusation;
* the article contains facts unfavorable to one side;
* one political actor receives more coverage because they are the main subject;
* opposing views are genuinely unavailable or irrelevant to the story;
* the article reports a controversial political event;
* the article discusses a politically sensitive issue.

## REQUIRED DECISION

Answer the following question:

"Does the article's presentation or framing contain reasonably identifiable evidence of favoring or disadvantaging a political actor, political side, or political perspective?"

## REQUIRED OUTPUT

Return exactly two fields:

`bias_classification`

Must be exactly one of:

* `biased`
* `not_biased`

`bias_reason`

A concise, evidence-based justification.

### If `biased`:

You MUST identify concrete evidence from the article text and explain how that evidence demonstrates biased framing.

Where possible, quote or closely identify the relevant wording from the article.

### If `not_biased`:

You MUST explain what features of the article support a predominantly factual or balanced presentation.

Do not invent evidence.

If the article does not contain sufficient textual evidence to establish bias, classify it as `not_biased` and explain that no sufficiently identifiable framing evidence was found.


ARTICLE:

{article}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "bias_classification": {
            "type": "string",
            "enum": ["biased", "not_biased"],
        },
        "bias_reason": {"type": "string"},
    },
    "required": ["bias_classification", "bias_reason"],
    "additionalProperties": False,
}


# =========================================================
# BIAS ANNOTATION FUNCTION (runs inside worker threads)
# =========================================================

def annotate_bias(article):
    # .replace instead of .format so any braces in the article text are safe
    prompt = PROMPT_TEMPLATE.replace("{article}", article)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "bias_annotation",
                        "strict": True,
                        "schema": SCHEMA,
                    }
                },
            )

            output = response.output_text.strip()
            if not output:
                raise ValueError("Model returned an empty response.")

            result = json.loads(output)
            classification = result["bias_classification"]
            reason = result["bias_reason"]

            if classification not in ("biased", "not_biased"):
                raise ValueError(f"Invalid classification: {classification}")

            return classification, reason

        except Exception as e:
            if attempt == MAX_RETRIES:
                raise

            # exponential backoff with jitter, so workers don't retry in lockstep
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(
                f"   API attempt {attempt}/{MAX_RETRIES} failed: {e} "
                f"-> retrying in {delay:.1f}s"
            )
            time.sleep(delay)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# Resume from the previous output if it exists, otherwise start from input
source_file = OUTPUT_FILE if os.path.exists(OUTPUT_FILE) else INPUT_FILE
print(f"Reading: {source_file}")

data = pd.read_parquet(source_file)

print("Columns:")
print(data.columns)
print("\nShape:")
print(data.shape)

if "article_text" not in data.columns:
    raise ValueError("The dataset does not contain the required column: article_text")

if "bias_classification" not in data.columns:
    data["bias_classification"] = None

if "bias_reason" not in data.columns:
    data["bias_reason"] = None


# =========================================================
# FIND UNANNOTATED ROWS
# =========================================================

remaining = data[
    data["bias_classification"].isna() | data["bias_reason"].isna()
].index

total_articles = len(data)
already_annotated = total_articles - len(remaining)

print("\n" + "=" * 60)
print("ANNOTATION STATUS")
print("=" * 60)
print(f"Total articles:     {total_articles}")
print(f"Already annotated:  {already_annotated}")
print(f"Remaining:          {len(remaining)}")
print(f"Workers:            {MAX_WORKERS}")


def save():
    data.to_parquet(OUTPUT_FILE, index=False)


# =========================================================
# HANDLE EMPTY ARTICLES UP FRONT (no API call needed)
# =========================================================

to_annotate = []
processed = 0
failed = 0

for i in remaining:
    article = data.at[i, "article_text"]
    if pd.isna(article) or not str(article).strip():
        data.at[i, "bias_classification"] = "not_biased"
        data.at[i, "bias_reason"] = "No article text available for annotation."
        processed += 1
    else:
        to_annotate.append(i)

if processed:
    print(f"\nMarked {processed} empty articles.")


# =========================================================
# PARALLEL ANNOTATION LOOP
# All DataFrame writes happen in the main thread only.
# =========================================================

total_to_do = len(remaining)
api_done = 0
start = time.time()

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
futures = {}

try:
    for i in to_annotate:
        fut = executor.submit(annotate_bias, str(data.at[i, "article_text"]))
        futures[fut] = i

    for fut in as_completed(futures):
        i = futures[fut]

        try:
            classification, reason = fut.result()
        except Exception as e:
            failed += 1
            print(f"\nERROR on row {i}: {e}")
            print("Row will remain unannotated and be retried on the next run.\n")
            continue

        data.at[i, "bias_classification"] = classification
        data.at[i, "bias_reason"] = reason
        processed += 1
        api_done += 1

        elapsed = time.time() - start
        rate = api_done / elapsed if elapsed > 0 else 0
        print(
            f"[{processed}/{total_to_do}] Row {i}: {classification} "
            f"({rate:.2f} articles/s)"
        )

        if api_done % CHECKPOINT_EVERY == 0:
            save()
            print("\n" + "-" * 60)
            print(f"CHECKPOINT SAVED: {processed} articles -> {OUTPUT_FILE}")
            print("-" * 60 + "\n")

except KeyboardInterrupt:
    print("\nInterrupted. Cancelling pending requests and saving progress...")
    executor.shutdown(wait=False, cancel_futures=True)

finally:
    executor.shutdown(wait=True, cancel_futures=True)
    save()


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("ANNOTATION COMPLETE")
print("=" * 60)

print(f"Total articles:       {total_articles}")
print(f"Processed this run:   {processed}")
print(f"Failed this run:      {failed}")
print(f"Output file:          {OUTPUT_FILE}")
print(f"Elapsed:              {time.time() - start:.1f}s")

print("\nBias distribution:")
print(data["bias_classification"].value_counts(dropna=False))

remaining_after = data[
    data["bias_classification"].isna() | data["bias_reason"].isna()
]
print(f"\nRemaining unannotated: {len(remaining_after)}")
print("\nDone.")