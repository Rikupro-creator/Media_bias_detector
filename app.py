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
# HARDCODED TEST DATASET FILENAME
# -----------------------------------------------------------------------------
PRIMARY_TEST_DATASET = "test_media_bias.csv"
FALLBACK_TEST_DATASET = "kenyan_news_political_classified1.csv"

# -----------------------------------------------------------------------------
# DEFAULT PROMPT TEMPLATE
# -----------------------------------------------------------------------------
DEFAULT_PROMPT_TEMPLATE = """You are an expert annotator evaluating framing bias in Kenyan political news articles.

TASK
Determine whether the article contains textual evidence of biased framing that favors or disadvantages a political actor, political side, or political perspective.

Evaluate the article's presentation, not whether its political claims are true or whether you agree with them.

ASSESSMENT CRITERIA

Consider the following indicators:
1. Loaded language: emotionally charged or evaluative wording that portrays an actor favorably or unfavorably beyond straightforward reporting.
2. One-sided framing: presenting one political side's interpretation while excluding relevant competing perspectives.
3. Unequal treatment: describing, scrutinizing, or contextualizing comparable political actors differently.
4. Selective emphasis: emphasizing particular facts, allegations, or events in a way that creates a discernible political slant.
5. Perspective omission: excluding a relevant perspective when the omission materially affects the article's framing.

EVIDENCE RULES

- Base your decision exclusively on the supplied article.
- Identify specific words, phrases, descriptions, quotations, or presentation choices that support your decision.
- Distinguish the journalist's framing from claims or opinions attributed to politicians, witnesses, or other sources.
- Do not assume that an article is biased because of the politician, party, institution, topic, or news organization involved.
- Political criticism, allegations, corruption reports, government failures, and opposition claims are not automatically evidence of media bias.
- Do not assume that every article must give equal space to every political side.
- Do not invent quotations, omitted perspectives, or contextual facts.
- If there is insufficient textual evidence of biased framing, classify the article as not_biased.

CLASSIFICATION

Use exactly one label:
- biased: identifiable textual evidence indicates that the article's framing favors or disadvantages a political actor, side, or perspective.
- not_biased: the article is predominantly factual or balanced, or the available text does not provide sufficient evidence of biased framing.

OUTPUT FORMAT

Return exactly these two fields:

bias_classification: biased OR not_biased
bias_reason: A concise explanation grounded in specific evidence from the article. For biased articles, identify the evidence and explain its framing effect. For not_biased articles, explain the relevant features supporting the decision or state that sufficient evidence of biased framing was not found.

Do not add headings, scores, recommendations, or additional fields."""

# -----------------------------------------------------------------------------
# SAMPLE CURATED ARTICLES FOR QUICK TESTING
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
# HF TOKEN RESOLUTION (From UI, st.secrets, or environment variables)
# -----------------------------------------------------------------------------
def detect_hf_token() -> str:
    """Detects Hugging Face API token from st.secrets or os.environ."""
    token_keys = [
        "HF_TOKEN",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HF_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN"
    ]
    
    # Check st.secrets first
    try:
        if hasattr(st, "secrets") and st.secrets:
            for key in token_keys:
                if key in st.secrets and st.secrets[key]:
                    return str(st.secrets[key]).strip()
    except Exception:
        pass
        
    # Check environment variables
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
# DATASET LOADING UTILITY
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading test dataset...")
def load_test_dataset(filename: str):
    if os.path.exists(filename):
        if filename.endswith(".csv"):
            return pd.read_csv(filename), filename
        elif filename.endswith(".parquet"):
            return pd.read_parquet(filename), filename
    # Fallback check
    if os.path.exists(FALLBACK_TEST_DATASET):
        return pd.read_csv(FALLBACK_TEST_DATASET), FALLBACK_TEST_DATASET
    return None, None

def build_prompt(article_text: str, template: str) -> str:
    return f"""{template}

ARTICLE:
{article_text}

ANSWER:
"""

def parse_prediction(response_text: str):
    """Parses bias_classification and bias_reason from the model response."""
    classification_match = re.search(
        r"bias_classification\s*:\s*(not_biased|biased)\b",
        response_text,
        flags=re.IGNORECASE
    )
    
    reason_match = re.search(
        r"bias_reason\s*:\s*(.*)",
        response_text,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    classification = classification_match.group(1).lower() if classification_match else "unparsed"
    reason = reason_match.group(1).strip() if reason_match else response_text.strip()
    
    return classification, reason

# Cached Model Loader for Local PyTorch Transformers
@st.cache_resource(show_spinner="Loading PyTorch model locally...")
def load_local_model(model_id: str, hf_token: str = None, load_in_4bit: bool = False, device_choice: str = "auto"):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        st.error("Missing required packages (`torch`, `transformers`). Install them via pip to use local inference.")
        return None, None

    token = hf_token if hf_token else os.environ.get("HF_TOKEN")
    
    st.info(f"Loading local model weights for `{model_id}`...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    kwargs = {
        "pretrained_model_name_or_path": model_id,
        "token": token,
        "trust_remote_code": True
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

def query_hf_api(prompt: str, model_id: str, hf_token: str, max_tokens: int, temperature: float, top_p: float):
    """Queries Hugging Face Serverless / Inference API."""
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise RuntimeError("`huggingface_hub` package is required for HF API Inference. Install via `pip install huggingface_hub`.")

    token = hf_token.strip() if hf_token else os.environ.get("HF_TOKEN")
    client = InferenceClient(model=model_id, token=token)
    
    response = client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        do_sample=temperature > 0,
        return_full_text=False
    )
    return response

def query_local_transformers(prompt: str, tokenizer, model, max_tokens: int, temperature: float, top_p: float):
    """Executes local PyTorch inference."""
    import torch
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    
    input_device = getattr(model, "device", None)
    if input_device is None and hasattr(model, "get_input_embeddings"):
        input_device = model.get_input_embeddings().weight.device
    if input_device:
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["attention_mask"][0].sum().item()
    new_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Model Configuration")

model_options = [
    "Mriku/kenyan-media-bias-qwen-continued1",
    "Mriku/kenyan-media-bias-qwen-continued",
    "Mriku/kenyan-media-bias-qwen",
    "Custom HuggingFace Model ID"
]

selected_model_option = st.sidebar.selectbox("Select Model Repo", model_options, index=0)

if selected_model_option == "Custom HuggingFace Model ID":
    model_id = st.sidebar.text_input("Custom Model ID", value="Mriku/kenyan-media-bias-qwen-continued1")
else:
    model_id = selected_model_option

# HF Token Auto-Detection & Input
auto_detected_token = detect_hf_token()

user_hf_token = st.sidebar.text_input(
    "Hugging Face Token (Required for Private Repos)",
    value=auto_detected_token,
    type="password",
    help="Provide a HF Access Token (hf_...) if testing a private repository. Automatically populated if found in st.secrets or environment variables."
)

hf_token = get_effective_hf_token(user_hf_token)

if auto_detected_token and not user_hf_token:
    st.sidebar.caption("🔑 Auto-detected token from Secrets/Environment.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Backend Execution Mode")

backend_choice = st.sidebar.radio(
    "Inference Engine",
    options=["Hugging Face API (Cloud)", "Local Transformers (PyTorch)"],
    index=0,
    help="HF API runs inference in the cloud (fast & lightweight). Local Transformers downloads model weights locally."
)

if backend_choice == "Local Transformers (PyTorch)":
    st.sidebar.markdown("#### Local Options")
    device_choice = st.sidebar.selectbox("Compute Device", ["auto", "cuda", "cpu"], index=0)
    load_in_4bit = st.sidebar.checkbox("Load in 4-bit (bitsandbytes)", value=False, help="Requires `bitsandbytes` library")
else:
    device_choice = "auto"
    load_in_4bit = False

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Generation Parameters")
max_tokens = st.sidebar.slider("Max New Tokens", min_value=32, max_value=1024, value=200, step=16)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
top_p = st.sidebar.slider("Top-P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

st.sidebar.markdown("---")
with st.sidebar.expander("📝 Custom Prompt Template"):
    prompt_template = st.text_area(
        "Edit System Prompt",
        value=DEFAULT_PROMPT_TEMPLATE,
        height=250
    )

# -----------------------------------------------------------------------------
# MAIN APP BODY & TABS
# -----------------------------------------------------------------------------
st.title("🇰🇪 Kenyan News Media Bias Model Tester")
st.markdown(f"Evaluate fine-tuned Hugging Face models (**`{model_id}`**) for framing bias in Kenyan political news.")

# Status Bar
col_status1, col_status2 = st.columns([3, 1])
with col_status1:
    st.caption(f"**Active Model:** `{model_id}` | **Engine:** {backend_choice}")
with col_status2:
    if hf_token:
        st.caption("🔒 Token Active (Authenticated)")
    else:
        st.caption("🌐 Public / Anonymous Access")

tab1, tab2 = st.tabs(["🧪 Model Tester (One-by-One)", "ℹ️ Model Details & Prompt"])

# -----------------------------------------------------------------------------
# TAB 1: MODEL TESTER (ONE-BY-ONE & CUSTOM TEXT)
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Test Model on Dataset Articles or Custom Text")
    
    input_mode = st.radio(
        "Select Testing Mode:",
        [
            f"📁 Browse Test Dataset ({PRIMARY_TEST_DATASET})",
            "📝 Enter Custom Article Text",
            "📑 Curated Sample Articles"
        ],
        horizontal=True
    )

    default_title = ""
    default_text = ""
    ground_truth_label = None
    
    test_df, loaded_filename = load_test_dataset(PRIMARY_TEST_DATASET)

    if input_mode.startswith("📁 Browse Test Dataset"):
        if test_df is None or len(test_df) == 0:
            st.warning(f"Could not locate `{PRIMARY_TEST_DATASET}` in workspace directory. Please ensure `{PRIMARY_TEST_DATASET}` is placed in the project folder, or select 'Enter Custom Article Text'.")
        else:
            if loaded_filename != PRIMARY_TEST_DATASET:
                st.info(f"Using fallback dataset `{loaded_filename}` (place `{PRIMARY_TEST_DATASET}` in workspace to override).")
            else:
                st.success(f"Loaded test dataset `{PRIMARY_TEST_DATASET}` ({len(test_df):,} total articles).")

            # Column detection
            cols = list(test_df.columns)
            text_cols = [c for c in cols if "article" in c.lower() or "text" in c.lower() or "body" in c.lower()]
            text_col = text_cols[0] if text_cols else cols[0]
            
            title_cols = [c for c in cols if "title" in c.lower() or "headline" in c.lower()]
            title_col = title_cols[0] if title_cols else None
            
            truth_cols = [c for c in cols if "bias" in c.lower() or "label" in c.lower() or "classification" in c.lower() or "truth" in c.lower()]
            truth_col = truth_cols[0] if truth_cols else None
            
            # Row index selector
            valid_rows = test_df[test_df[text_col].notna() & (test_df[text_col].astype(str).str.len() > 30)].head(500)
            
            row_options = []
            for idx, r in valid_rows.iterrows():
                t_str = str(r[title_col])[:60] if title_col and pd.notna(r[title_col]) else f"Article {idx}"
                row_options.append(f"Row {idx}: {t_str}")
                
            selected_option = st.selectbox("Select Article to Test (One-by-One):", row_options)
            
            if selected_option:
                sel_idx = int(selected_option.split(":")[0].replace("Row ", ""))
                selected_row = test_df.loc[sel_idx]
                default_title = str(selected_row[title_col]) if title_col and pd.notna(selected_row[title_col]) else f"Article Row {sel_idx}"
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
        article_title = st.text_input("Article Title", value=default_title, placeholder="Enter or edit article title...")
        article_text = st.text_area("Article Text / Body", value=default_text, height=340, placeholder="Paste or edit article text...")
        
        if ground_truth_label:
            st.markdown(f"**Ground Truth Label in Dataset:** `{ground_truth_label}`")
            
        run_button = st.button("🚀 Analyze Framing Bias", type="primary", use_container_width=True)

    with col_b:
        st.markdown("### 📋 Prediction Output")
        
        if run_button:
            if not article_text.strip():
                st.warning("Please provide article text to analyze.")
            else:
                full_prompt = build_prompt(article_text, prompt_template)
                
                with st.spinner("Running model inference..."):
                    start_time = time.time()
                    raw_response = ""
                    error_msg = None
                    
                    try:
                        if backend_choice == "Hugging Face API (Cloud)":
                            raw_response = query_hf_api(
                                prompt=full_prompt,
                                model_id=model_id,
                                hf_token=hf_token,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p
                            )
                        else:
                            tokenizer, model = load_local_model(
                                model_id=model_id,
                                hf_token=hf_token,
                                load_in_4bit=load_in_4bit,
                                device_choice=device_choice
                            )
                            if model is not None:
                                raw_response = query_local_transformers(
                                    prompt=full_prompt,
                                    tokenizer=tokenizer,
                                    model=model,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p
                                )
                    except Exception as e:
                        error_msg = str(e)
                    
                    elapsed_time = time.time() - start_time
                
                if error_msg:
                    st.error(f"Inference Error: {error_msg}")
                    st.info("Tip: If the Hugging Face Serverless API is unavailable or returning 404/401, verify your model repo ID is correct and your HF Access Token has permission.")
                else:
                    classification, reason = parse_prediction(raw_response)
                    
                    # Display colored result badge
                    if classification == "biased":
                        st.error("### 🔴 Result: BIASED")
                    elif classification == "not_biased":
                        st.success("### 🟢 Result: NOT BIASED")
                    else:
                        st.warning("### ⚠️ Result: UNPARSED / OTHER")

                    # Ground Truth Comparison if available
                    if ground_truth_label:
                        gt_norm = ground_truth_label.replace("-", "_")
                        is_match = (classification == gt_norm) or (gt_norm in ["political", "biased"] and classification == "biased") or (gt_norm in ["not_political", "not_biased"] and classification == "not_biased")
                        if is_match:
                            st.success(f"✅ Model prediction matches Dataset Ground Truth (`{ground_truth_label}`)!")
                        else:
                            st.warning(f"⚡ Disagreement with Dataset Ground Truth (`{ground_truth_label}`) vs Predicted (`{classification}`).")

                    # Result Summary Card
                    st.markdown(f"**Classification:** `{classification}`")
                    st.markdown("**Reasoning / Evidence:**")
                    st.info(reason if reason else "No reasoning extracted.")
                    st.caption(f"⚡ Inference completed in **{elapsed_time:.2f} seconds**")
                    
                    with st.expander("🔍 View Raw Completion & Full Prompt"):
                        st.markdown("**Raw Generation Output:**")
                        st.code(raw_response)
                        st.markdown("**Full Input Prompt Sent to Model:**")
                        st.code(full_prompt, language="text")

# -----------------------------------------------------------------------------
# TAB 2: MODEL & PROMPT DOCUMENTATION
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Model & Framing Bias Documentation")
    
    st.markdown(f"""
    ### 📌 Project Context
    This application tests fine-tuned Large Language Models (LLMs) specifically trained on the **Kenyan Political News Media Framing & Bias Dataset**.
    
    - **Default Model Repo:** `{model_id}`
    - **Test Dataset:** `{PRIMARY_TEST_DATASET}`
    - **Task:** Causal Text Generation producing structured outputs for `bias_classification` and `bias_reason`.
    
    ---
    ### 🎯 Classification Labels
    1. **`biased`**: The article exhibits loaded language, one-sided framing, unequal treatment of political actors, selective emphasis, or relevant perspective omission favoring or disadvantaging a political side.
    2. **`not_biased`**: The article is predominantly factual, balanced, or lacks sufficient textual evidence of biased framing.
    
    ---
    ### 💻 Python Code Integration
    To use this model directly in your Python code:
    """)
    
    st.code(f"""from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "{model_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

article = "President William Ruto signed the Division of Revenue Bill into law..."

prompt = f\"\"\"{DEFAULT_PROMPT_TEMPLATE}

ARTICLE:
{{article}}

ANSWER:
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""", language="python")
