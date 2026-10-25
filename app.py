import os
import re
import time
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Kenyan News Media Bias Model Tester",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# TEST DATASET CANDIDATES  (tried in order)
# -----------------------------------------------------------------------------
TEST_DATASET_CANDIDATES = [
    "test_media_bias.csv",
    
]

# -----------------------------------------------------------------------------
# MODEL REPO CONFIGURATION
# -----------------------------------------------------------------------------
# Tokenizer ALWAYS comes from the base Qwen repo.
BASE_TOKENIZER_REPO = "Mriku/kenyan-media-bias-qwen"

# -----------------------------------------------------------------------------
# DEFAULT PROMPT TEMPLATE
# -----------------------------------------------------------------------------
# Model was fine-tuned WITHOUT an instruction prompt: raw article -> labels.
# Therefore default is EMPTY (feed the raw article directly).
DEFAULT_PROMPT_TEMPLATE = ""

# -----------------------------------------------------------------------------
# SAMPLE CURATED ARTICLES
# -----------------------------------------------------------------------------
SAMPLE_ARTICLES = {
    "Sample 1: Political Legislation (Revenue Bill)": {
        "title": "Ruto signs Division of Revenue Act, securing Sh428bn equitable share for counties",
        "text": """NAIROBI, Kenya, Jun 15 — President William Ruto has signed the Division of Revenue (DoR) Bill, 2026 into law, formally allocating Sh428 billion in equitable share funding to county governments for the 2026/27 financial year.

The signing follows a breakthrough agreement reached during a consultative meeting chaired by Deputy President Kithure Kindiki at the official residence in Karen, resolving months of deadlocks between the Senate and the National Assembly over resource sharing.

The new law represents a significant step forward in fiscal decentralization, guaranteeing predictable transfers to county governments to support local development, healthcare, and infrastructure projects across all 47 counties."""
    },
    "Sample 2: Critical / Allegations (Scrutiny)": {
        "title": "Opposition leaders criticize government fuel import scheme as opaque and costly",
        "text": """Opposition MPs have launched a scathing attack on the Ministry of Energy over the ongoing government-to-government oil deal, alleging that the arrangement has failed to deliver promised price relief to Kenyan consumers.

Addressing journalists at Parliament Buildings, the lawmakers claimed that fuel prices in Kenya remain significantly higher than in neighboring Uganda and Tanzania, despite official assurances that the strategic fuel import deal would stabilize the shilling and eliminate middleman markups.

The ministry maintained that the G2G arrangement has reduced foreign exchange pressures and shielded the country from severe international supply disruptions."""
    },
    "Sample 3: Civic & Security Incident": {
        "title": "CCTV shows coordinated attack on All Saints Cathedral budget forum",
        "text": """NAIROBI, Kenya, Jun 14— At exactly 9.39am, a convoy of motorcycles pulled up outside All Saints Cathedral, one of Nairobi’s most prominent places of worship and a long-standing hub of civic engagement, in full view of cameras.

Security footage reviewed by authorities shows individuals dismounting from the vehicles before entering the premises where a public forum discussing national budget proposals was underway.

Police officers arrived shortly thereafter to disperse the crowd and restore order. Church leaders have called for a thorough investigation into the security breach to protect public spaces for civic debate."""
    }
}

# -----------------------------------------------------------------------------
# HF TOKEN RESOLUTION
# -----------------------------------------------------------------------------
def detect_hf_token() -> str:
    token_keys = [
        "HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACE_TOKEN",
        "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"
    ]
    try:
        if hasattr(st, "secrets") and st.secrets:
            for key in token_keys:
                if key in st.secrets and st.secrets[key]:
                    return str(st.secrets[key]).strip()
    except Exception:
        pass
    for key in token_keys:
        env_val = os.environ.get(key)
        if env_val:
            return env_val.strip()
    return ""

def get_effective_hf_token(user_token_input: str) -> str:
    if user_token_input and user_token_input.strip():
        return user_token_input.strip()
    return detect_hf_token()

# -----------------------------------------------------------------------------
# DATASET LOADING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading test dataset...")
def load_test_dataset(candidates):
    for filename in candidates:
        if os.path.exists(filename):
            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(filename)
                    return df, filename
                elif filename.endswith(".parquet"):
                    df = pd.read_parquet(filename)
                    return df, filename
            except Exception as e:
                st.warning(f"Could not read `{filename}`: {e}")
                continue
    return None, None

def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)
    text_cols = [c for c in cols if "article" in c.lower() or "text" in c.lower() or "body" in c.lower()]
    text_col = text_cols[0] if text_cols else cols[0]

    title_cols = [c for c in cols if "title" in c.lower() or "headline" in c.lower()]
    title_col = title_cols[0] if title_cols else None

    truth_cols = [c for c in cols if "bias" in c.lower() or "label" in c.lower() or "classification" in c.lower() or "truth" in c.lower()]
    truth_col = truth_cols[0] if truth_cols else None

    return text_col, title_col, truth_col

def normalize_label(label) -> str:
    """Normalise various label formats to 'biased' / 'not_biased' / 'unknown'."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return "unknown"
    s = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if s in ("biased", "1", "true", "yes", "political_biased"):
        return "biased"
    if s in ("not_biased", "0", "false", "no", "unbiased", "not_political", "neutral"):
        return "not_biased"
    return "unknown"

# -----------------------------------------------------------------------------
# PROMPT + PARSER
# -----------------------------------------------------------------------------
def build_prompt(article_text: str, template: str = "") -> str:
    if template and template.strip():
        return f"""{template}

ARTICLE:
{article_text}

ANSWER:
"""
    return article_text.strip() + "\n\n"

def parse_prediction(response_text: str):
    classification_match = re.search(
        r"bias_classification\s*:\s*(not_biased|biased)\b",
        response_text, flags=re.IGNORECASE
    )
    reason_match = re.search(
        r"bias_reason\s*:\s*(.*)",
        response_text, flags=re.IGNORECASE | re.DOTALL
    )
    classification = classification_match.group(1).lower() if classification_match else "unparsed"
    reason = reason_match.group(1).strip() if reason_match else response_text.strip()
    return classification, reason

# -----------------------------------------------------------------------------
# MODEL LOADER (tokenizer from base repo, weights from selected repo)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading PyTorch model locally...")
def load_local_model(model_id: str, hf_token: str = None, load_in_4bit: bool = False, device_choice: str = "auto"):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        st.error("Missing packages (`torch`, `transformers`). Install via pip.")
        return None, None

    token = hf_token if hf_token else os.environ.get("HF_TOKEN")

    st.info(f"Loading tokenizer from `{BASE_TOKENIZER_REPO}` and weights from `{model_id}`...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_REPO, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    kwargs = {
        "pretrained_model_name_or_path": model_id,
        "token": token,
        "trust_remote_code": True,
    }

    if quantization_config:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"
    else:
        if device_choice == "cuda" and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "cuda"
        elif device_choice == "cpu":
            kwargs["torch_dtype"] = torch.float32
            kwargs["device_map"] = "cpu"
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    model.eval()
    return tokenizer, model

# -----------------------------------------------------------------------------
# INFERENCE BACKENDS
# -----------------------------------------------------------------------------
def query_hf_api(prompt: str, model_id: str, hf_token: str, max_tokens: int, temperature: float, top_p: float):
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise RuntimeError("`huggingface_hub` required. pip install huggingface_hub.")

    token = hf_token.strip() if hf_token else os.environ.get("HF_TOKEN")
    client = InferenceClient(model=model_id, token=token)
    return client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        do_sample=temperature > 0,
        return_full_text=False,
    )

def query_local_transformers(prompt: str, tokenizer, model, max_tokens: int, temperature: float, top_p: float):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    input_device = getattr(model, "device", None)
    if input_device is None and hasattr(model, "get_input_embeddings"):
        input_device = model.get_input_embeddings().weight.device
    if input_device:
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["attention_mask"][0].sum().item()
    new_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def run_inference(prompt, backend, model_id, hf_token, max_tokens, temperature, top_p,
                  tokenizer=None, model=None):
    if backend == "Hugging Face API (Cloud)":
        return query_hf_api(prompt, model_id, hf_token, max_tokens, temperature, top_p)
    return query_local_transformers(prompt, tokenizer, model, max_tokens, temperature, top_p)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Model Configuration")

model_options = [
    "Mriku/kenyan-media-bias-qwen",
    "Mriku/kenyan-media-bias-qwen-continued1",
    "Mriku/kenyan-media-bias-qwen-continued",
    "Custom HuggingFace Model ID",
]
selected_model_option = st.sidebar.selectbox("Select Model Repo", model_options, index=0)
if selected_model_option == "Custom HuggingFace Model ID":
    model_id = st.sidebar.text_input("Custom Model ID", value="Mriku/kenyan-media-bias-qwen")
else:
    model_id = selected_model_option

auto_detected_token = detect_hf_token()
user_hf_token = st.sidebar.text_input(
    "Hugging Face Token",
    value=auto_detected_token, type="password",
    help="HF access token (hf_...). Auto-filled from st.secrets / env vars if found."
)
hf_token = get_effective_hf_token(user_hf_token)
if auto_detected_token and not user_hf_token:
    st.sidebar.caption("🔑 Auto-detected token.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Backend Execution Mode")
backend_choice = st.sidebar.radio(
    "Inference Engine",
    options=["Hugging Face API (Cloud)", "Local Transformers (PyTorch)"],
    index=0,
)

if backend_choice == "Local Transformers (PyTorch)":
    device_choice = st.sidebar.selectbox("Compute Device", ["auto", "cuda", "cpu"], index=0)
    load_in_4bit = st.sidebar.checkbox("Load in 4-bit (bitsandbytes)", value=False)
else:
    device_choice = "auto"
    load_in_4bit = False

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Generation Parameters")
max_tokens = st.sidebar.slider("Max New Tokens", 32, 1024, 200, 16)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
top_p = st.sidebar.slider("Top-P", 0.1, 1.0, 0.9, 0.05)

st.sidebar.markdown("---")
with st.sidebar.expander("📝 Custom Prompt Template (Optional)"):
    prompt_template = st.text_area(
        "Leave EMPTY to feed the raw article (recommended for this model)",
        value=DEFAULT_PROMPT_TEMPLATE, height=250
    )

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("🇰🇪 Kenyan News Media Bias Model Tester")
st.markdown(f"Evaluate fine-tuned Hugging Face models (**`{model_id}`**) for framing bias in Kenyan political news.")

col_status1, col_status2 = st.columns([3, 1])
with col_status1:
    st.caption(f"**Model:** `{model_id}` | **Tokenizer:** `{BASE_TOKENIZER_REPO}` | **Engine:** {backend_choice}")
with col_status2:
    st.caption("🔒 Token Active" if hf_token else "🌐 Anonymous")

tab1, tab2, tab3 = st.tabs([
    "🧪 One-by-One Tester",
    "📊 Batch Evaluation",
    "ℹ️ Model Details",
])

# -----------------------------------------------------------------------------
# TAB 1: ONE-BY-ONE TESTER
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Test Model on a Single Article")

    input_mode = st.radio(
        "Select Testing Mode:",
        [
            "📁 Browse Test Dataset",
            "📝 Enter Custom Article Text",
            "📑 Curated Sample Articles",
        ],
        horizontal=True,
    )

    default_title = ""
    default_text = ""
    ground_truth_label = None

    test_df, loaded_filename = load_test_dataset(tuple(TEST_DATASET_CANDIDATES))

    if input_mode == "📁 Browse Test Dataset":
        if test_df is None or len(test_df) == 0:
            st.warning("No dataset found. Expected one of: " + ", ".join(f"`{c}`" for c in TEST_DATASET_CANDIDATES))
        else:
            st.success(f"Loaded `{loaded_filename}` ({len(test_df):,} rows).")
            text_col, title_col, truth_col = detect_columns(test_df)

            valid_rows = test_df[test_df[text_col].notna() & (test_df[text_col].astype(str).str.len() > 30)].head(500)
            row_options = []
            for idx, r in valid_rows.iterrows():
                t_str = str(r[title_col])[:60] if title_col and pd.notna(r[title_col]) else f"Article {idx}"
                row_options.append(f"Row {idx}: {t_str}")

            selected_option = st.selectbox("Select Article:", row_options)
            if selected_option:
                sel_idx = int(selected_option.split(":")[0].replace("Row ", ""))
                selected_row = test_df.loc[sel_idx]
                default_title = str(selected_row[title_col]) if title_col and pd.notna(selected_row[title_col]) else f"Article {sel_idx}"
                default_text = str(selected_row[text_col])
                if truth_col and pd.notna(selected_row[truth_col]):
                    ground_truth_label = str(selected_row[truth_col]).lower()

    elif input_mode == "📑 Curated Sample Articles":
        sample_choice = st.selectbox("Select Sample Article:", list(SAMPLE_ARTICLES.keys()))
        if sample_choice:
            default_title = SAMPLE_ARTICLES[sample_choice]["title"]
            default_text = SAMPLE_ARTICLES[sample_choice]["text"]

    col_a, col_b = st.columns([1, 1])
    with col_a:
        article_title = st.text_input("Article Title", value=default_title)
        article_text = st.text_area("Article Text", value=default_text, height=340)
        if ground_truth_label:
            st.markdown(f"**Ground Truth:** `{ground_truth_label}`")
        run_button = st.button("🚀 Analyze Framing Bias", type="primary", use_container_width=True)

    with col_b:
        st.markdown("### 📋 Prediction Output")
        if run_button:
            if not article_text.strip():
                st.warning("Please provide article text.")
            else:
                full_prompt = build_prompt(article_text, prompt_template)
                tokenizer = model = None
                if backend_choice == "Local Transformers (PyTorch)":
                    tokenizer, model = load_local_model(model_id, hf_token, load_in_4bit, device_choice)

                if backend_choice == "Local Transformers (PyTorch)" and model is None:
                    st.error("Model could not be loaded.")
                else:
                    with st.spinner("Running inference..."):
                        start = time.time()
                        try:
                            raw_response = run_inference(
                                full_prompt, backend_choice, model_id, hf_token,
                                max_tokens, temperature, top_p,
                                tokenizer=tokenizer, model=model,
                            )
                            error_msg = None
                        except Exception as e:
                            raw_response = ""
                            error_msg = str(e)
                        elapsed = time.time() - start

                    if error_msg:
                        st.error(f"Inference Error: {error_msg}")
                    else:
                        classification, reason = parse_prediction(raw_response)
                        if classification == "biased":
                            st.error("### 🔴 Result: BIASED")
                        elif classification == "not_biased":
                            st.success("### 🟢 Result: NOT BIASED")
                        else:
                            st.warning("### ⚠️ UNPARSED")

                        if ground_truth_label:
                            gt = normalize_label(ground_truth_label)
                            if gt == classification:
                                st.success(f"✅ Matches ground truth (`{ground_truth_label}`)")
                            else:
                                st.warning(f"⚡ Disagrees with ground truth (`{ground_truth_label}`)")

                        st.markdown(f"**Classification:** `{classification}`")
                        st.markdown("**Reasoning:**")
                        st.info(reason or "No reasoning extracted.")
                        st.caption(f"⚡ {elapsed:.2f}s")

                        with st.expander("🔍 Raw Output & Full Prompt"):
                            st.markdown("**Raw Generation:**")
                            st.code(raw_response)
                            st.markdown("**Input Prompt:**")
                            st.code(full_prompt, language="text")

# -----------------------------------------------------------------------------
# TAB 2: BATCH EVALUATION (NEW)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Batch Evaluation on Test Dataset")
    st.markdown(
        "Runs the model across many rows of the CSV and reports accuracy / precision / recall / F1 "
        "against the ground-truth column (if present)."
    )

    test_df, loaded_filename = load_test_dataset(tuple(TEST_DATASET_CANDIDATES))

    if test_df is None or len(test_df) == 0:
        st.error("No dataset found. Place one of these in the project folder: " +
                 ", ".join(f"`{c}`" for c in TEST_DATASET_CANDIDATES))
    else:
        st.success(f"Loaded `{loaded_filename}` — {len(test_df):,} rows.")
        text_col, title_col, truth_col = detect_columns(test_df)

        st.markdown(
            f"**Detected columns** → text: `{text_col}` "
            f"| title: `{title_col or '—'}` "
            f"| ground truth: `{truth_col or '— (metrics disabled)'}`"
        )

        if truth_col is None:
            st.warning("No ground-truth column detected — you'll get predictions only, no metrics.")

        # Preview ground-truth distribution
        if truth_col is not None:
            dist = test_df[truth_col].astype(str).str.lower().value_counts()
            st.caption("Label distribution: " + " | ".join(f"`{k}`: {v}" for k, v in dist.items()))

        max_eval_rows = st.slider(
            "Max rows to evaluate (0 = all)",
            min_value=0, max_value=len(test_df), value=min(50, len(test_df)), step=10,
        )
        n_rows = len(test_df) if max_eval_rows == 0 else max_eval_rows

        st.info(f"Will evaluate **{n_rows}** row(s). "
                f"At ~2s/row on CPU this may take ~{n_rows*2/60:.1f} minutes.")

        if st.button("▶️ Run Batch Evaluation", type="primary", use_container_width=True):
            # Prepare model if local
            tokenizer = model = None
            if backend_choice == "Local Transformers (PyTorch)":
                tokenizer, model = load_local_model(model_id, hf_token, load_in_4bit, device_choice)
                if model is None:
                    st.error("Model failed to load. Aborting.")
                    st.stop()

            eval_df = test_df.head(n_rows).copy()
            results = []
            progress = st.progress(0.0)
            status = st.empty()

            for i, (idx, row) in enumerate(eval_df.iterrows()):
                text = row[text_col]
                if pd.isna(text) or len(str(text).strip()) < 30:
                    progress.progress((i + 1) / n_rows)
                    continue

                prompt = build_prompt(str(text), prompt_template)
                try:
                    raw = run_inference(
                        prompt, backend_choice, model_id, hf_token,
                        max_tokens, temperature, top_p,
                        tokenizer=tokenizer, model=model,
                    )
                    cls, rsn = parse_prediction(raw)
                except Exception as e:
                    cls, rsn, raw = "error", str(e), ""

                results.append({
                    "row_index": idx,
                    "title": (str(row[title_col])[:120] if title_col and pd.notna(row[title_col]) else ""),
                    "true_label_raw": (str(row[truth_col]) if truth_col and pd.notna(row[truth_col]) else None),
                    "true_label_norm": normalize_label(row[truth_col]) if truth_col else "unknown",
                    "predicted": cls,
                    "reason": rsn,
                    "raw_output": raw,
                })

                progress.progress((i + 1) / n_rows)
                status.caption(f"Processed {i+1}/{n_rows}")

            progress.empty()
            status.empty()

            results_df = pd.DataFrame(results)
            st.markdown("### 🎯 Results")
            st.dataframe(results_df, use_container_width=True)

            # Compute metrics
            if truth_col is not None and len(results_df) > 0:
                labelled = results_df[
                    results_df["true_label_norm"].isin(["biased", "not_biased"]) &
                    results_df["predicted"].isin(["biased", "not_biased"])
                ]
                if len(labelled) > 0:
                    y_true = labelled["true_label_norm"]
                    y_pred = labelled["predicted"]

                    tp = int(((y_pred == "biased") & (y_true == "biased")).sum())
                    tn = int(((y_pred == "not_biased") & (y_true == "not_biased")).sum())
                    fp = int(((y_pred == "biased") & (y_true == "not_biased")).sum())
                    fn = int(((y_pred == "not_biased") & (y_true == "biased")).sum())

                    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
                    prec = tp / max(1, (tp + fp))
                    rec = tp / max(1, (tp + fn))
                    f1 = 2 * prec * rec / max(1e-9, (prec + rec))

                    st.markdown("### 📈 Metrics (biased as positive class)")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.2%}")
                    m2.metric("Precision", f"{prec:.2%}")
                    m3.metric("Recall", f"{rec:.2%}")
                    m4.metric("F1", f"{f1:.2f}")

                    st.markdown("### 🧩 Confusion Matrix")
                    cm_df = pd.DataFrame(
                        [[tp, fn], [fp, tn]],
                        index=["Actual: biased", "Actual: not_biased"],
                        columns=["Pred: biased", "Pred: not_biased"],
                    )
                    st.dataframe(cm_df, use_container_width=True)

                    unparsed = int((results_df["predicted"] == "unparsed").sum())
                    errors = int((results_df["predicted"] == "error").sum())
                    if unparsed or errors:
                        st.caption(f"⚠️ {unparsed} unparsed | {errors} errors excluded from metrics.")
                else:
                    st.warning("No rows had both a valid ground-truth label and a parseable prediction.")

            # Download
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv,
                file_name=f"batch_eval_{model_id.split('/')[-1]}_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# -----------------------------------------------------------------------------
# TAB 3: MODEL DETAILS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Model & Prompt Documentation")

    st.markdown(f"""
    ### 📌 Project Context
    Fine-tuned LLMs for **Kenyan Political News Media Framing & Bias**.

    - **Active Model:** `{model_id}`
    - **Tokenizer Repo:** `{BASE_TOKENIZER_REPO}`
    - **Test Datasets searched:** {", ".join(f"`{c}`" for c in TEST_DATASET_CANDIDATES)}
    - **Task:** raw article → `bias_classification` + `bias_reason`

    ---
    ### 🎯 Labels
    - **`biased`**: loaded language, one-sided framing, unequal treatment, selective emphasis, or perspective omission.
    - **`not_biased`**: predominantly factual / balanced, or no sufficient evidence of biased framing.

    ---
    ### 🧠 Prompting Note
    This model was fine-tuned **without an instruction prompt** — it expects **raw article text** and directly emits:
