import time
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kenyan News Media Bias Model Tester",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_TEST_DATASET = "test_media_bias.csv"
FALLBACK_TEST_DATASET = "kenyan_news_political_classified1.csv"

# Anchor all dataset lookups to the directory containing this script, not the CWD.
APP_DIR = Path(__file__).resolve().parent


def _resolve_path(filename: str) -> Path:
    """Resolve a filename relative to the app directory unless it is absolute."""
    p = Path(filename)
    return p if p.is_absolute() else (APP_DIR / p)


MODEL_OPTIONS = [
    "Mriku/kenyan-media-bias-qwen-continued1",
    "Mriku/kenyan-media-bias-qwen-continued",
    "Mriku/kenyan-media-bias-qwen",
    "Custom HuggingFace Model ID",
]

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
    },
}


# -----------------------------------------------------------------------------
# TOKEN VALIDATION
# -----------------------------------------------------------------------------
def _clean_token(token: str) -> str:
    """Strip whitespace and common paste artifacts from a user-supplied token."""
    if token is None:
        return ""
    return str(token).strip().strip('"').strip("'").replace("\r", "").replace("\n", "")


def validate_hf_token(token: str) -> tuple[bool, str]:
    """
    Validate a Hugging Face token by calling whoami().
    Returns (is_valid, message).
    """
    token = _clean_token(token)
    if not token:
        return False, "Token is empty."
    if not token.startswith("hf_"):
        return False, "Token does not look like a Hugging Face token (should start with 'hf_')."

    try:
        from huggingface_hub import HfApi
    except ImportError:
        # Cannot validate without huggingface_hub; let the downstream call surface the error.
        return True, "huggingface_hub not installed; skipping pre-validation."

    try:
        info = HfApi(token=token).whoami()
        username = info.get("name") or info.get("fullname") or "unknown user"
        return True, f"Token valid (authenticated as `{username}`)."
    except Exception as exc:
        return False, f"Token rejected by Hugging Face: {type(exc).__name__}: {exc}"


# -----------------------------------------------------------------------------
# DATASET LOADING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading test dataset...")
def load_test_dataset(filename: str):
    """
    Load the requested dataset, falling back to FALLBACK_TEST_DATASET if the
    primary cannot be found. Paths are resolved relative to the app directory
    (APP_DIR) so this works regardless of the process working directory.

    Returns (dataframe, resolved_path_str) or (None, None).
    """
    if filename and filename.endswith(".parquet"):
        reader = pd.read_parquet
    else:
        reader = pd.read_csv

    candidates = [filename, FALLBACK_TEST_DATASET]
    for candidate in candidates:
        if not candidate:
            continue
        path = _resolve_path(candidate)
        if path.exists() and path.is_file():
            try:
                df = reader(path)
                return df, str(path)
            except Exception as exc:
                st.warning(f"Found `{path}` but failed to read it: {exc}")
                continue

    return None, None


def normalize_label(value):
    """Normalize known bias labels without equating political relevance with bias."""
    if value is None or pd.isna(value):
        return None
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "biased": "biased",
        "not_biased": "not_biased",
        "unbiased": "not_biased",
        "notbiased": "not_biased",
    }
    return aliases.get(label)


def parse_prediction(response_text: str):
    """Parse common label formats, including the expected field format and JSON-like output."""
    import re

    text = (response_text or "").strip()

    match = re.search(
        r"bias_classification\s*['\"]?\s*[:=]\s*['\"]?\s*(not[\s_-]*biased|biased)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        raw_label = match.group(1).lower().replace("-", "_").replace(" ", "_")
        classification = "not_biased" if raw_label in {"not_biased", "not biased"} else "biased"
    else:
        bare = re.fullmatch(r"\s*(not[\s_-]*biased|biased)[.!]?\s*", text, flags=re.IGNORECASE)
        if bare:
            raw_label = bare.group(1).lower().replace("-", "_").replace(" ", "_")
            classification = "not_biased" if raw_label in {"not_biased", "not biased"} else "biased"
        else:
            classification = "unparsed"

    reason_match = re.search(
        r"bias_reason\s*['\"]?\s*[:=]\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    reason = reason_match.group(1).strip().strip('"\'') if reason_match else text
    return classification, reason


# -----------------------------------------------------------------------------
# MODEL LOADING AND INFERENCE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading PyTorch model locally...")
def load_local_model(model_id: str, hf_token: str, load_in_4bit: bool = False, device_choice: str = "auto"):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Missing required packages. Install torch and transformers; "
            "for 4-bit loading also install bitsandbytes."
        ) from exc

    token = _clean_token(hf_token)
    if not token:
        raise ValueError("Enter your Hugging Face token in the sidebar.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise RuntimeError("Tokenizer has no pad token or EOS token.")

    kwargs = {
        "pretrained_model_name_or_path": model_id,
        "token": token,
        "trust_remote_code": True,
    }

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but CUDA is not available.")
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


def query_hf_api(article_text: str, model_id: str, hf_token: str, max_tokens: int, temperature: float, top_p: float):
    """Call Hugging Face hosted text-generation inference using the user-supplied token."""
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub with: pip install -U huggingface_hub"
        ) from exc

    token = _clean_token(hf_token)
    if not token:
        raise ValueError("Enter your Hugging Face token in the sidebar.")

    # Pass token explicitly. Passing it as a keyword argument ensures the client
    # uses it rather than any ambient credentials (HF_TOKEN env var, cached login, etc.).
    client = InferenceClient(model=model_id, token=token)

    # InferenceClient.text_generation returns a plain string in recent
    # huggingface_hub versions. Handle both str and object forms defensively.
    result = client.text_generation(
        article_text,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        do_sample=temperature > 0,
        return_full_text=False,
    )
    if isinstance(result, str):
        return result
    # Fallback for older client versions that return an object with .generated_text
    return getattr(result, "generated_text", str(result))


def query_local_transformers(article_text: str, tokenizer, model, max_tokens: int, temperature: float, top_p: float):
    """Run local inference on the article text, without adding an application prompt."""
    import torch

    inputs = tokenizer(
        article_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# -----------------------------------------------------------------------------
# SIDEBAR: ALL CREDENTIALS MUST BE ENTERED BY THE USER
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Model Configuration")

model_option = st.sidebar.selectbox("Select Model Repository", MODEL_OPTIONS, index=0)
if model_option == "Custom HuggingFace Model ID":
    model_id = st.sidebar.text_input(
        "Custom Model ID",
        value="Mriku/kenyan-media-bias-qwen-continued1",
    ).strip()
else:
    model_id = model_option

hf_token_raw = st.sidebar.text_input(
    "Hugging Face API Token (required)",
    value="",
    type="password",
    help="Enter a token with read access to the private model repository. The app does not read tokens from secrets or environment variables.",
)
hf_token = _clean_token(hf_token_raw)

st.sidebar.caption("Token is supplied manually for this app session; it is not loaded from environment variables or Streamlit secrets.")

# Token status indicator and validation
if hf_token:
    token_ok, token_msg = validate_hf_token(hf_token)
    if token_ok:
        st.sidebar.success(f"🔐 {token_msg}")
    else:
        st.sidebar.error(f"❌ {token_msg}")
    # Show a diagnostic (length only — never the token itself)
    st.sidebar.caption(f"Token length: {len(hf_token)} | Starts with: `{hf_token[:3]}...`")

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Backend Execution Mode")
backend_choice = st.sidebar.radio(
    "Inference Engine",
    ["Hugging Face API (Cloud)", "Local Transformers (PyTorch)"],
    index=0,
)

if backend_choice == "Local Transformers (PyTorch)":
    device_choice = st.sidebar.selectbox("Compute Device", ["auto", "cuda", "cpu"], index=0)
    load_in_4bit = st.sidebar.checkbox(
        "Load in 4-bit (bitsandbytes)",
        value=False,
        help="Requires bitsandbytes. Quantized loading generally uses CUDA.",
    )
else:
    device_choice = "auto"
    load_in_4bit = False

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Generation Parameters")
max_tokens = st.sidebar.slider("Max New Tokens", 32, 1024, 200, 16)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
top_p = st.sidebar.slider("Top-P", 0.1, 1.0, 0.9, 0.05)


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
st.title("🇰🇪 Kenyan News Media Bias Model Tester")
st.markdown(
    f"Test fine-tuned Hugging Face models for framing bias in Kenyan political news. "
    f"**Active model:** `{model_id}`"
)

status1, status2 = st.columns([3, 1])
with status1:
    st.caption(f"**Backend:** {backend_choice}")
with status2:
    st.caption("🔐 Token entered manually" if hf_token else "🔑 Token required")

st.subheader("Test an article")
input_mode = st.radio(
    "Select Testing Mode",
    [
        f"📁 Browse Test Dataset ({PRIMARY_TEST_DATASET})",
        "📝 Enter Custom Article Text",
        "📑 Curated Sample Articles",
    ],
    horizontal=True,
)

default_title = ""
default_text = ""
ground_truth_label = None

test_df, loaded_path = load_test_dataset(PRIMARY_TEST_DATASET)

if input_mode.startswith("📁 Browse Test Dataset"):
    if test_df is None or test_df.empty:
        st.warning(
            f"Could not locate `{PRIMARY_TEST_DATASET}` or `{FALLBACK_TEST_DATASET}` "
            f"in the app directory: `{APP_DIR}`. Add one of these CSV files there and rerun the app."
        )
        with st.expander("🔎 Diagnostics"):
            st.write("**App directory:**", str(APP_DIR))
            try:
                st.write("**Files found in app directory:**")
                st.code("\n".join(sorted(p.name for p in APP_DIR.iterdir())))
            except Exception as exc:
                st.write(f"Could not list app directory: {exc}")
    else:
        loaded_name = Path(loaded_path).name if loaded_path else "unknown"
        if loaded_name != PRIMARY_TEST_DATASET:
            st.info(f"Using fallback dataset `{loaded_name}` ({loaded_path}).")
        else:
            st.success(f"Loaded `{loaded_name}` ({len(test_df):,} rows) from `{loaded_path}`.")

        cols = list(test_df.columns)
        text_cols = [c for c in cols if any(k in c.lower() for k in ("article", "text", "body", "content"))]
        title_cols = [c for c in cols if any(k in c.lower() for k in ("title", "headline"))]
        truth_candidates = [
            c for c in cols
            if c.lower() in {"bias_classification", "bias_label", "ground_truth", "true_label", "label"}
        ]
        text_col = text_cols[0] if text_cols else cols[0]
        title_col = title_cols[0] if title_cols else None
        truth_col = truth_candidates[0] if truth_candidates else None

        valid_rows = test_df[
            test_df[text_col].notna()
            & (test_df[text_col].astype(str).str.len() > 30)
        ].head(500)

        if valid_rows.empty:
            st.warning("No usable article text was found in the selected dataset.")
        else:
            row_options = []
            option_to_index = {}
            for idx, row in valid_rows.iterrows():
                title = str(row[title_col])[:70] if title_col and pd.notna(row[title_col]) else f"Article {idx}"
                option = f"Row {idx}: {title}"
                row_options.append(option)
                option_to_index[option] = idx

            selected_option = st.selectbox("Select Article", row_options)
            if selected_option:
                selected_idx = option_to_index[selected_option]
                selected_row = test_df.loc[selected_idx]
                default_title = (
                    str(selected_row[title_col])
                    if title_col and pd.notna(selected_row[title_col])
                    else f"Article Row {selected_idx}"
                )
                default_text = str(selected_row[text_col])
                if truth_col and pd.notna(selected_row[truth_col]):
                    ground_truth_label = str(selected_row[truth_col]).strip()

elif input_mode == "📑 Curated Sample Articles":
    sample_choice = st.selectbox("Select Sample Article", list(SAMPLE_ARTICLES.keys()))
    if sample_choice:
        default_title = SAMPLE_ARTICLES[sample_choice]["title"]
        default_text = SAMPLE_ARTICLES[sample_choice]["text"]

col_input, col_output = st.columns([1, 1])

with col_input:
    article_title = st.text_input(
        "Article Title",
        value=default_title,
        placeholder="Enter or edit article title...",
        key=f"title_{input_mode}",
    )
    article_text = st.text_area(
        "Article Text / Body",
        value=default_text,
        height=340,
        placeholder="Paste or edit the article text...",
        key=f"text_{input_mode}",
    )

    normalized_truth = normalize_label(ground_truth_label)
    if ground_truth_label:
        st.markdown(f"**Dataset label:** `{ground_truth_label}`")
        if normalized_truth is None:
            st.caption("This dataset label is not a recognized bias label, so it will not be compared with the model prediction.")

    run_button = st.button(
        "🚀 Analyze Framing Bias",
        type="primary",
        use_container_width=True,
    )

with col_output:
    st.markdown("### 📋 Prediction Output")

    if run_button:
        # --- Pre-flight checks ---
        if not hf_token:
            st.error("Enter your Hugging Face token in the sidebar before running inference.")
        elif not model_id:
            st.error("Enter or select a model repository.")
        elif not article_text.strip():
            st.warning("Please provide article text to analyze.")
        else:
            # Validate token against the Hub before attempting inference.
            token_ok, token_msg = validate_hf_token(hf_token)
            if not token_ok:
                st.error(f"🔑 Token check failed: {token_msg}")
                st.info(
                    "Common causes:\n"
                    "1. The token was pasted with extra characters or line breaks.\n"
                    "2. The token has expired or was revoked.\n"
                    "3. The token type (read-only vs. fine-grained) cannot access this repository.\n"
                    "4. The repository is gated and your account has not accepted its terms.\n\n"
                    "Regenerate the token at https://huggingface.co/settings/tokens and paste it again."
                )
            else:
                with st.spinner("Running model inference..."):
                    started = time.time()
                    raw_response = ""
                    error_msg = None

                    try:
                        if backend_choice == "Hugging Face API (Cloud)":
                            raw_response = query_hf_api(
                                article_text=article_text.strip(),
                                model_id=model_id,
                                hf_token=hf_token,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                            )
                        else:
                            tokenizer, model = load_local_model(
                                model_id=model_id,
                                hf_token=hf_token,
                                load_in_4bit=load_in_4bit,
                                device_choice=device_choice,
                            )
                            raw_response = query_local_transformers(
                                article_text=article_text.strip(),
                                tokenizer=tokenizer,
                                model=model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                            )
                    except Exception as exc:
                        error_msg = f"{type(exc).__name__}: {exc}"

                    elapsed = time.time() - started

                if error_msg:
                    st.error(f"Inference failed: {error_msg}")
                    if "401" in error_msg or "RepositoryNotFound" in error_msg or "not found" in error_msg.lower():
                        st.info(
                            "This is an authentication or access error, not a model-output error. Check:\n\n"
                            "1. **Token validity** — run the test in the sidebar; it must show a green check.\n"
                            "2. **Repository access** — the token's account must have read access to "
                            f"`{model_id}`. If the repo is private or gated, visit "
                            f"https://huggingface.co/{model_id} while logged in and accept any terms.\n"
                            "3. **Token type** — a *fine-grained* token may be scoped to specific repos. "
                            "A *read* token covers all public/authorized repos.\n"
                            "4. **Inference provider** — the model may not be available through the "
                            "Hugging Face inference API even if the repo is readable. Try "
                            "'Local Transformers (PyTorch)' instead.\n"
                            "5. **Deployment secrets** — if running on Streamlit Community Cloud, the token "
                            "entered in the sidebar is used; the app does not read from Secrets or env vars."
                        )
                else:
                    classification, reason = parse_prediction(raw_response)

                    if classification == "biased":
                        st.error("### 🔴 Result: BIASED")
                    elif classification == "not_biased":
                        st.success("### 🟢 Result: NOT BIASED")
                    else:
                        st.warning("### ⚠️ Result: UNPARSED / OTHER")

                    if normalized_truth is not None:
                        if classification == normalized_truth:
                            st.success(f"Prediction matches dataset ground truth (`{ground_truth_label}`).")
                        else:
                            st.warning(
                                f"Prediction (`{classification}`) differs from ground truth (`{ground_truth_label}`)."
                            )

                    st.markdown(f"**Article:** {article_title or '(untitled)'}")
                    st.markdown(f"**Classification:** `{classification}`")
                    st.markdown("**Reasoning / Evidence:**")
                    st.info(reason if reason else "No reasoning extracted.")
                    st.caption(f"Inference completed in **{elapsed:.2f} seconds**")

                    with st.expander("🔍 View Raw Model Output"):
                        st.code(raw_response)
