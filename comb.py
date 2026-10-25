import os
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# LOAD & COMBINE DATA
# ---------------------------------------------------------

df = pd.read_parquet('/home/riku/Documents/diss1/diss/kenyamoja_articles1.parquet')
df1 = pd.read_parquet('/home/riku/Documents/diss/kenyamoja_articles1.parquet')
df2=pd.read_parquet('/home/riku/Downloads/final1.parquet')
combined_vertical = pd.concat([df, df1, df2], axis=0).reset_index(drop=True)
combined_vertical.to_parquet('final.parquet')

print(df1.shape)
print(df.shape)
print(combined_vertical.shape)

# Use the combined dataframe for classification
df = combined_vertical

# ---------------------------------------------------------
# CLIENT
# ---------------------------------------------------------

# NOTE: pull the key from an environment variable rather than hardcoding it
# export OPENAI_API_KEY="sk-..." before running this script
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL_NAME = "gpt-5.6-luna"

# How many requests to run at once. 8-16 is a reasonable range for most
# API rate limits; raise/lower depending on what your account allows.
MAX_WORKERS = 10

# ---------------------------------------------------------
# PROMPT / CLASSIFY FUNCTION
# ---------------------------------------------------------

def classify_political(article):

    prompt = f"""
You are an expert annotator of Kenyan news media.

Your task is to determine whether the following news article is
PRIMARILY about Kenyan politics.

Return exactly ONE label:

Political
Not Political

DEFINITION OF POLITICAL:

Classify an article as Political when its MAIN SUBJECT concerns
Kenyan political actors, political institutions, political
processes, political decision-making, political parties,
elections, political opposition, government accountability,
legislation, governance, or political conflict.

This includes articles primarily concerning:

- The President or Deputy President in their political/government role
- Members of Parliament, Senators, Governors, MCAs, Cabinet Secretaries,
  or other political office holders
- Political parties and political campaigns
- Elections and electoral politics
- Government versus opposition relations or disputes
- Parliamentary or Senate political proceedings
- Political appointments, resignations, impeachment, or succession
- Government accountability and political scandals
- Corruption involving political actors or government institutions
- Political protests and demonstrations
- Political legislation and major government political decisions
- Conflicts between political actors
- Kenyan government–opposition relations

IMPORTANT:

Do NOT classify an article as Political merely because it mentions
a politician, political party, government institution, President,
Governor, MP, or another political keyword.

The MAIN SUBJECT of the article must be political.

For example, an article should be classified as Not Political if:

- A politician opens a hospital but the article mainly discusses
  healthcare or the hospital.
- A politician attends a sporting event but the article mainly
  discusses the sport.
- A crime article merely mentions a politician.
- A business article briefly mentions government.
- A health, education, technology, entertainment, or sports article
  merely mentions a political figure.
- A government agency is mentioned but political decision-making is
  not the main subject.
- Kenya is mentioned in an article whose main subject is foreign
  politics.

KENYA FOCUS:

The political subject should primarily concern Kenya.

DECISION RULE:

Ask:

"If I had to summarize this article in one sentence, would the
summary primarily describe a political issue, political actor,
political institution, political decision, election, government
accountability issue, or political conflict in Kenya?"

If YES → Political

If NO → Not Political

Do not provide an explanation.
Do not provide additional text.

Return exactly:

Political

or

Not Political

ARTICLE:
{article}
"""

    # Simple retry with backoff for transient errors / rate limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.choices[0].message.content.strip()

            if result.lower() == "political":
                return "Political"
            if result.lower() == "not political":
                return "Not Political"
            return "Uncertain"

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
                continue
            raise e


def classify_row(index_and_article):
    """Wrapper so we can track which row a result belongs to."""
    idx, article = index_and_article

    if pd.isna(article) or not str(article).strip():
        return idx, "Empty"

    try:
        return idx, classify_political(str(article))
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        return idx, "Error"


# ---------------------------------------------------------
# CLASSIFY DATASET IN PARALLEL
# ---------------------------------------------------------

df["political_classification"] = None

# Build the list of (index, article) pairs to process
work_items = list(df["article_text"].items())

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(classify_row, item): item[0] for item in work_items
    }

    for future in tqdm(as_completed(futures), total=len(futures)):
        idx, label = future.result()
        df.at[idx, "political_classification"] = label

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv("kenyan_news_political_classified.csv", index=False)
df.to_parquet("kenyan_news_political_classified.parquet", index=False)

print("Done.")
print(df["political_classification"].value_counts())