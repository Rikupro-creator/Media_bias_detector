# Kenyan News Media Bias Pipeline & Dataset Construction

This repository contains the full end-to-end data engineering, machine learning annotation, judge verification, and document compilation pipeline for constructing and evaluating the **Kenyan Political News Media Bias Dataset**.

---

## 🏗️ Architecture & Pipeline Overview

The project is structured into **6 key phases**:

```
[1. Scraping] ──> [2. Text Extraction] ──> [3. Political Classification]
                                                        │
[6. Technical Reports] <── [5. Gemini Judge Verification] <── [4. Bias Framing Annotation]
```

1. **Scraping & Aggregation:** Harvesting article links across leading Kenyan news outlets via KenyaMoja.
2. **Text Extraction & Recovery:** Robust multi-threaded scraping with headless Edge/Chrome Selenium fallbacks for dynamic JS rendering.
3. **Political Relevance Filtering:** Zero-shot classification using `gpt-5.6-luna` to isolate political articles.
4. **Media Bias Framing Annotation:** Detailed evidence-based media bias framing analysis via `gpt-5.6-luna`.
5. **Independent Judge Verification:** Secondary evaluation using `gemini-3.1-flash-lite` with high reasoning level to calculate inter-model agreement metrics (Cohen's Kappa, PABAK, Bootstrap CIs).
6. **Model Fine-Tuning Experiments & Technical Reports:** Jupyter Notebooks for training transformer baselines (DeBERTa/BERT) and automated Word (`.docx`) technical analysis generation.

---

## 📜 File-by-File Inventory

### 1. Web Scraping & Text Extraction

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `scraper.py` | Python Script | Primary web scraper harvesting article links, headlines, dates, and source outlets from the KenyaMoja news aggregator. | KenyaMoja web pages | `kenyamoja_news.csv` |
| `scraper2.py` | Python Script | Requests & Selenium article text scraper handling 9 key news sites (Citizen, TV47, Star, Tuko, Standard, Nation, Capital FM, K24, KBC). | `kenyamoja_news.csv` | Extracted article texts in memory / pandas |
| `scraper3.py` | Python Script | High-performance multi-threaded Edge/Selenium scraper targeting JavaScript-heavy sites (`Nation`, `K24`). | URLs list | Extracted DOM paragraph elements |
| `paragraphs.py` | Python Script | Main robust text extraction engine using ThreadPoolExecutor and site-specific BeautifulSoup parsers with fallback handling. | `kenyamoja_news.csv` | `kenyamoja_articles.csv`, `failed_links.csv`, `kenyamoja_skipped.csv` |
| `failed_urls.py` | Python Script | Retries failed extractions logged in `failed_links.csv` using headless Selenium. Merges recovered content back into main dataset. | `failed_links.csv`, `kenyamoja_news_with_articles.csv` | `still_failed_after_selenium.csv`, updated `kenyamoja_news_with_articles.csv` |
| `paragraphs2.py` | Python Script | Prototype parsing script for testing BeautifulSoup extraction of Citizen Digital article containers. | Hardcoded URL | Test stdout output |

---

### 2. Data Cleaning, Merging & Utilities

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `conver.py` | Python Script | Utility script converting raw CSV article dataset to Parquet format for faster I/O. | `kenyamoja_articles.csv` | `kenyamoja_articles1.parquet` |
| `cleaner.py` | Python Script | Diagnostic script comparing column schemas between `kenyamoja_subset.parquet` and `kenyamoja_articles.csv`. | `kenyamoja_subset.parquet`, `kenyamoja_articles.csv` | Diagnostic stdout |
| `comb.py` | Python Script | Combines multiple scraped Parquet datasets vertically into a single consolidated dataset. | `kenyamoja_articles1.parquet`, `final1.parquet` | `final.parquet` |
| `comb2.py` | Python Script | Standardizes column names, concatenates classification runs, and filters for political news. | `kenyan_news_political_classified.parquet`, `kenyan_news_political_classified1.parquet` | `data.parquet`, `political.parquet` |
| `change.py` | Python Script | Utility exporting verified Parquet dataset into CSV format for external inspection. | `political_verified.parquet` | `political_verified.csv` |
| `analysis.py` | Python Script | Quick data quality check inspecting missing values across dataset columns. | `kenyamoja_articles1.parquet` | Diagnostic stdout |
| `data_analyzer.py` | Python Script | Summary statistics reporter printing article counts by outlet and label distribution. | `political_verified.parquet` | Summary statistics stdout |
| `final_dataset.py` | Python Script | Placeholder script reserved for final dataset assembly. | - | - |
| `verifier.py` | Python Script | Placeholder script reserved for manual sanity checks. | - | - |

---

### 3. Classification & Annotation (LLM Pipelines)

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `classifier.py` | Python Script | Concurrent LLM classifier using `gpt-5.6-luna` to perform zero-shot political relevance classification (`Political` vs `Not Political`). | `kenyamoja_subset.parquet` / `final.parquet` | `kenyan_news_political_classified.parquet` |
| `annotator.py` | Python Script | Primary media bias framing annotator using `gpt-5.6-luna`. Requires text evidence before outputting JSON with `bias_classification` and `bias_reason`. | `political.parquet` | `political_annotated.parquet` |
| `judge.py` | Python Script | Independent verification script using `gemini-3.1-flash-lite` (with reasoning) to audit Luna's bias labels. Calculates Cohen's Kappa, PABAK, and bootstrap CIs. | `political_annotated.parquet` | `gemini_judge_results.jsonl`, `judge_agreement_metrics.json`, `judge_disagreements.csv`, `political_verified.parquet` |

---

### 4. Machine Learning & Experimentation Notebooks

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `media-bias-continued (1).ipynb` | Jupyter Notebook | Kaggle notebook for fine-tuning transformer models (DeBERTa/BERT) on political bias classification. | Dataset parquet | Model weights & evaluation plots |
| `media-bias-continued (2).ipynb` | Jupyter Notebook | Extended training notebook covering hyperparameter optimization, error analysis, and validation metrics. | Dataset parquet | Model checkpoints & loss curves |
| `mediabias1 (5).ipynb` | Jupyter Notebook | Exploratory analysis notebook evaluating token distributions, framing frequencies, and baseline classifiers. | Dataset parquet | Exploratory plots & tables |
| `testings.ipynb` | Jupyter Notebook | Quick sandbox notebook for testing LLM prompt templates and API calls. | Sample text | Test prompt outputs |

---

### 5. Document & Technical Report Builders

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `create_docx.py` | Python Script | Master generator for `Kenyan_News_Pipeline_and_Scripts_Analysis.docx`, generating a professional technical report covering methodology, prompts, and verification metrics. | Codebase analysis data | `Kenyan_News_Pipeline_and_Scripts_Analysis.docx` |

---

## 📊 Dataset & Artifact Files Reference

| File | Format | Description |
|---|---|---|
| `kenyamoja_news.csv` | CSV | Raw scraped news metadata (URLs, headlines, sources, publish dates). |
| `kenyamoja_articles.csv` | CSV | Scraped news metadata joined with extracted full article text. |
| `kenyamoja_articles1.parquet` | Parquet | Parquet conversion of `kenyamoja_articles.csv`. |
| `kenyamoja_news_with_articles.csv` | CSV | Master article text compilation post-Selenium retry. |
| `failed_links.csv` | CSV | URLs that failed HTTP GET requests during initial scraping pass. |
| `still_failed_after_selenium.csv` | CSV | Hard-failed URLs that could not be rendered even with headless Selenium. |
| `kenyamoja_skipped.csv` | CSV | Links skipped due to unsupported domains or video/audio content. |
| `kenyamoja_subset.parquet` | Parquet | Cleaned working subset of scraped article records. |
| `final.parquet` | Parquet | Merged dataset containing all collected articles across scraping runs. |
| `kenyan_news_political_classified.parquet` | Parquet | Full dataset with `gpt-5.6-luna` political classification labels. |
| `kenyan_news_political_classified1.parquet` | Parquet | Secondary classification batch output. |
| `data.parquet` | Parquet | Consolidated standardized dataset with all political classifications. |
| `political.parquet` | Parquet | Filtered subset containing **16,367 political articles**. |
| `political_annotated.parquet` | Parquet | Primary dataset annotated by `gpt-5.6-luna` (`bias_classification`, `bias_reason`). |
| `gemini_judge_results.jsonl` | JSONL | Checkpoint stream log of `gemini-3.1-flash-lite` verification outputs. |
| `judge_agreement_metrics.json` | JSON | Statistical audit summary (Percent Agreement, Cohen's Kappa, PABAK, Bootstrap CIs). |
| `judge_disagreements.csv` | CSV | Export of all 4,723 articles where Luna and Gemini classifications disagreed. |
| `political_verified.parquet` / `.csv` | Parquet / CSV | **Final Verified Benchmark Dataset** containing all metadata, article texts, Luna annotations, and Gemini judge evaluations. |
| `progress.json` | JSON | Checkpoint state tracker for resumption of batch execution. |

---

## 📑 Dissertation & Report Documents

- `Dissertation_Full_Revised_v5.docx`: Main dissertation document.
- `Dissertation_Full_Revised_v5_backup.docx`: Backup copy of dissertation document.
- `DISSERTATION Draft.docx`: Original dissertation draft document.
- `Kenyan_News_Pipeline_and_Scripts_Analysis.docx`: Comprehensive technical analysis report detailing code architecture, prompts, and verification audit results.

---

## 🚀 Key Statistics & Benchmarks

- **Total Political Benchmark Corpus:** 16,367 articles across 9 major Kenyan news outlets.
- **Inter-Model Raw Agreement:** 71.14% (95% Bootstrap CI: [70.47%, 71.85%]).
- **Chance-Corrected Agreement (PABAK):** 0.423.
- **Confirmed Biased Articles:** 1,165 articles (Agreed by both Luna & Gemini).
- **Confirmed Neutral Articles:** 10,479 articles (Agreed by both Luna & Gemini).
- **Disagreements Flagged for Human Audit:** 4,723 articles (`judge_disagreements.csv`).
