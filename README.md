# Kenyan News Media Bias Framing Pipeline & Model Evaluation

This repository contains the full end-to-end data engineering, machine learning annotation, judge verification, and fine-tuned model evaluation pipeline for constructing and testing the **Kenyan Political News Media Bias Dataset** and its associated fine-tuned LLMs.

---

## 🖥️ Streamlit Test Application (`app.py`)

An interactive Streamlit application (`app.py`) is provided for evaluating fine-tuned Hugging Face models trained on Kenyan political news media bias detection.

### Key Features
- **Test Dataset Article Browser (`test_media_bias.csv`)**: Hardcoded to load and test articles from `test_media_bias.csv` one by one, displaying article details, model predictions, evidence reasoning, and comparison with ground-truth labels.
- **Custom Text Input**: Option for users to paste or type any custom news article text to run real-time bias inference.
- **Structured Prediction Parsing**: Automatically parses model output into `bias_classification` (`biased` vs `not_biased`) and `bias_reason`.
- **Dual Inference Engine Support**:
  - **⚡ Hugging Face API (Cloud)**: Serverless fast inference using `huggingface_hub.InferenceClient` (no local GPU required).
  - **💻 Local PyTorch Transformers**: Loads model weights locally (`transformers`, `torch`) with optional 4-bit quantization.

---

### 📦 Pre-configured Hugging Face Models
The app includes pre-loaded references to fine-tuned model repositories:
- `Mriku/kenyan-media-bias-qwen-continued1` *(Default)*
- `Mriku/kenyan-media-bias-qwen-continued`
- `Mriku/kenyan-media-bias-qwen`
- *Custom HuggingFace Model ID* (User-specifiable repository ID)

---

### 🔒 Setting Up Hugging Face Secrets / Tokens

For private Hugging Face repositories or restricted Inference API access, the application automatically resolves your Hugging Face Access Token (`hf_...`) in priority order:

1. **Manual Sidebar Input**: Type your token directly into the sidebar password field.
2. **Streamlit Secrets (`st.secrets`)**: Automatically reads from `.streamlit/secrets.toml` or platform secrets.
3. **Environment Variables**: Reads `HF_TOKEN`, `HUGGINGFACE_API_KEY`, or `HUGGINGFACE_TOKEN`.

#### Local Secrets Configuration (`.streamlit/secrets.toml`)
Create a `.streamlit/secrets.toml` file in the project root:
```toml
HF_TOKEN = "hf_your_huggingface_access_token_here"
```

---

### 🚀 Running the Streamlit App

1. Install all project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 🏗️ Architecture & Pipeline Overview

The project is structured into **6 key phases**:

```
[1. Scraping] ──> [2. Text Extraction] ──> [3. Political Classification]
                                                        │
[6. Streamlit & HF Models] <── [5. Gemini Judge Verification] <── [4. Bias Framing Annotation]
```

1. **Scraping & Aggregation:** Harvesting article links across leading Kenyan news outlets via KenyaMoja.
2. **Text Extraction & Recovery:** Multi-threaded scraping with headless Edge/Chrome Selenium fallbacks for dynamic JS rendering.
3. **Political Relevance Filtering:** Zero-shot classification using `gpt-5.6-luna` to isolate political news articles.
4. **Media Bias Framing Annotation:** Detailed evidence-based media bias framing analysis via `gpt-5.6-luna`.
5. **Independent Judge Verification:** Secondary evaluation using `gemini-3.1-flash-lite` to calculate inter-model agreement metrics (Cohen's Kappa, PABAK, Bootstrap CIs).
6. **Model Fine-Tuning & Deployment:** Fine-tuning Qwen 2.5 3B Instruct on Kaggle, uploading model weights to Hugging Face, and interactive testing via Streamlit.

---

## 📜 File-by-File Inventory

### 1. Interactive Applications & Web Testers

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `app.py` | Streamlit App | Interactive UI for evaluating fine-tuned Hugging Face models (`Mriku/kenyan-media-bias-qwen-continued1`). Hardcoded to browse `test_media_bias.csv` articles one by one or test custom article text. Supports cloud API and local PyTorch inference. | `test_media_bias.csv` / Custom text | Interactive predictions, bias badges, reasoning |
| `requirements.txt` | Configuration | Master Python dependencies file listing all libraries required for running the web scraper, classifiers, judge verifier, report generator, and Streamlit app. | - | - |

---

### 2. Web Scraping & Text Extraction

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `scraper.py` | Python Script | Primary web scraper harvesting article links, headlines, dates, and source outlets from the KenyaMoja news aggregator. | KenyaMoja web pages | `kenyamoja_news.csv` |
| `scraper2.py` | Python Script | Requests & Selenium article text scraper handling 9 key news sites (Citizen, TV47, Star, Tuko, Standard, Nation, Capital FM, K24, KBC). | `kenyamoja_news.csv` | Extracted article texts in memory / pandas |
| `scraper3.py` | Python Script | Multi-threaded Edge/Selenium scraper targeting JavaScript-heavy sites (`Nation`, `K24`). | URLs list | Extracted DOM paragraph elements |
| `paragraphs.py` | Python Script | Main robust text extraction engine using ThreadPoolExecutor and site-specific BeautifulSoup parsers with fallback handling. | `kenyamoja_news.csv` | `kenyamoja_articles.csv`, `failed_links.csv`, `kenyamoja_skipped.csv` |
| `failed_urls.py` | Python Script | Retries failed extractions logged in `failed_links.csv` using headless Selenium. Merges recovered content back into main dataset. | `failed_links.csv`, `kenyamoja_news_with_articles.csv` | `still_failed_after_selenium.csv`, updated `kenyamoja_news_with_articles.csv` |
| `paragraphs2.py` | Python Script | Prototype parsing script for testing BeautifulSoup extraction of Citizen Digital article containers. | Hardcoded URL | Test stdout output |

---

### 3. Data Cleaning, Merging & Utilities

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `comb.py` | Python Script | Combines multiple scraped Parquet datasets vertically into a single consolidated dataset. | Parquet datasets | `final.parquet` |
| `comb2.py` | Python Script | Standardizes column names, concatenates classification runs, and filters for political news. | Parquet datasets | `data.parquet`, `political.parquet` |
| `change.py` | Python Script | Utility exporting verified Parquet dataset into CSV format for external inspection. | `political_verified.parquet` | `political_verified.csv` |
| `data_analyzer.py` | Python Script | Summary statistics reporter printing article counts by outlet and label distribution. | `political_verified.parquet` | Summary statistics stdout |

---

### 4. Classification & Annotation (LLM Pipelines)

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `classifier.py` | Python Script | Concurrent LLM classifier using `gpt-5.6-luna` to perform zero-shot political relevance classification (`Political` vs `Not Political`). | `final.parquet` | `kenyan_news_political_classified.parquet` |
| `annotator.py` | Python Script | Primary media bias framing annotator using `gpt-5.6-luna`. Requires text evidence before outputting JSON with `bias_classification` and `bias_reason`. | `political.parquet` | `political_annotated.parquet` |
| `judge.py` | Python Script | Independent verification script using `gemini-3.1-flash-lite` (with reasoning) to audit Luna's bias labels. Calculates Cohen's Kappa, PABAK, and bootstrap CIs. | `political_annotated.parquet` | `gemini_judge_results.jsonl`, `judge_agreement_metrics.json`, `judge_disagreements.csv`, `political_verified.parquet` |

---

### 5. Fine-Tuning & Experimentation Notebooks

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `media-bias-continued (1).ipynb` | Jupyter Notebook | Kaggle notebook for fine-tuning Qwen 2.5 3B Instruct on Kenyan political bias classification and uploading model weights to Hugging Face (`Mriku/kenyan-media-bias-qwen-continued`). | Dataset parquet | Fine-tuned LoRA weights on Hugging Face |
| `media-bias-continued (2).ipynb` | Jupyter Notebook | Continued fine-tuning notebook for additional epochs and checkpoint uploading (`Mriku/kenyan-media-bias-qwen-continued1`). | Dataset parquet | Hugging Face model checkpoints & loss curves |
| `mediabias1 (5).ipynb` | Jupyter Notebook | Baseline fine-tuning and evaluation notebook for text generation models. | Dataset parquet | Evaluation metrics & Hugging Face repo setup |

---

### 6. Document & Technical Report Builders

| File | Type | Description | Inputs | Outputs / Artifacts |
|---|---|---|---|---|
| `create_docx.py` | Python Script | Master generator for `Kenyan_News_Pipeline_and_Scripts_Analysis.docx`, generating a technical report covering methodology, prompts, and verification metrics. | Codebase analysis data | `Kenyan_News_Pipeline_and_Scripts_Analysis.docx` |

---

## 🚀 Key Statistics & Benchmarks

- **Total Political Corpus:** 16,367 articles across 9 major Kenyan news outlets.
- **Inter-Model Raw Agreement:** 71.14% (95% Bootstrap CI: [70.47%, 71.85%]).
- **Chance-Corrected Agreement (PABAK):** 0.423.
- **Confirmed Biased Articles:** 1,165 articles (Agreed by both Luna & Gemini).
- **Confirmed Neutral Articles:** 10,479 articles (Agreed by both Luna & Gemini).
- **Disagreements Flagged for Human Audit:** 4,723 articles (`judge_disagreements.csv`).
