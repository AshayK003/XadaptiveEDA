import json
import time
import urllib.request
import urllib.error
import pandas as pd


DEFAULT_MODEL = "qwen2.5-coder:7b"
LIGHT_MODELS = ["qwen2.5-coder:7b", "qwen2.5:7b", "qwen2.5:3b", "phi3:mini", "llama3.2:3b", "gemma2:2b", "tinyllama"]
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 11434
TIMEOUT = 60
LOCAL_TIMEOUT = 120
MAX_RETRIES = 2
RETRY_DELAYS = [1, 3]

def _urlopen_retry(req, timeout):
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAYS[attempt])
    raise last_err


PROVIDERS = {
    "openrouter": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "models": ["deepseek/deepseek-r1-distill-qwen-7b", "qwen/qwen2.5-7b-instruct", "google/gemma-2-9b-it", "mistralai/mistral-7b-instruct"],
        "default_model": "qwen/qwen2.5-7b-instruct",
    },
    "groq": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it", "llama-3.1-8b-instant"],
        "default_model": "llama-3.3-70b-versatile",
    },
}


def _build_context(profile, df, analysis_type, columns):
    lines = []
    shape = profile.get('shape', (0, 0))
    lines.append(f"Dataset: {shape[0]:,} rows x {shape[1]} columns")

    dtypes = profile.get('dtypes', {})
    type_info = {}
    for col in columns:
        t = dtypes.get(col, str(df[col].dtype) if col in df.columns else "unknown")
        type_info.setdefault(t, []).append(col)
    if type_info:
        parts = [f"{', '.join(cols)} ({t})" for t, cols in sorted(type_info.items())]
        lines.append(f"Column types: {'; '.join(parts)}")

    missing_all = profile.get('missing_percentage', {})
    sel_missing = {c: p for c, p in missing_all.items() if c in columns and p > 0}
    if sel_missing:
        items = sorted(sel_missing.items(), key=lambda x: -x[1])
        lines.append(f"Missing per selected column: {', '.join(f'{c}={p:.1f}%' for c, p in items)}")
    total_missing = {c: p for c, p in missing_all.items() if p > 0}
    if total_missing and not sel_missing:
        top = max(total_missing, key=total_missing.get)
        lines.append(f"Missing data (other cols): '{top}' ({total_missing[top]:.1f}%)")
    if not total_missing:
        lines.append("Missing data: none")

    skew = profile.get('skewness', {})
    sel_skew = {c: s for c, s in skew.items() if c in columns and s is not None}
    if sel_skew:
        items = sorted(sel_skew.items(), key=lambda x: -abs(x[1]))
        labels = []
        for c, s in items:
            tag = "highly skewed" if abs(s) > 1 else "moderately skewed" if abs(s) > 0.5 else "approx symmetric"
            labels.append(f"{c} ({s:.2f}, {tag})")
        lines.append(f"Skewness: {', '.join(labels)}")

    outliers = profile.get('has_outliers', {})
    sel_out = {c: p for c, p in outliers.items() if c in columns}
    if sel_out:
        items = sorted(sel_out.items(), key=lambda x: -x[1])
        lines.append(f"Outlier %: {', '.join(f'{c}={p:.1f}%' for c, p in items)}")
    elif outliers:
        top = max(outliers, key=outliers.get)
        lines.append(f"Outliers (other cols): '{top}' ({outliers[top]:.1f}%)")

    unique = profile.get('unique_counts', {})
    sel_unique = {c: unique.get(c, df[col].nunique()) for c in columns if c in df.columns}
    if sel_unique:
        lines.append(f"Unique values: {', '.join(f'{c}={v}' for c, v in sel_unique.items())}")

    if analysis_type == 'correlation' and len(columns) > 1:
        try:
            corr = df[columns].select_dtypes(include=['number']).corr()
            if not corr.empty:
                strong = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        r = corr.iloc[i, j]
                        if abs(r) > 0.5:
                            strong.append(f"{corr.columns[i]} vs {corr.columns[j]} (r={r:.2f})")
                if strong:
                    lines.append(f"Strong correlations: {'; '.join(strong)}")
        except Exception:
            pass

    lines.append(f"\nAnalysis type: {analysis_type.replace('_', ' ')}")
    lines.append(f"Selected columns: {', '.join(columns)}")

    try:
        sample = df[columns].head(5)
        summary_lines = []
        for col in columns:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                s = sample[col]
                if not s.isnull().all():
                    summary_lines.append(f"  {col}: range [{s.min():.4g}, {s.max():.4g}], mean={s.mean():.4g}")
            elif pd.api.types.is_object_dtype(df[col]):
                vc = sample[col].value_counts()
                if not vc.empty:
                    summary_lines.append(f"  {col}: top value '{vc.index[0]}' ({vc.iloc[0]}/{len(sample)} rows)")
        if summary_lines:
            lines.append(f"\nSample column summary:\n" + "\n".join(summary_lines))
    except Exception:
        pass

    return "\n".join(lines)


ANALYSIS_GUIDES = {
    "distribution": (
        "Focus on: shape (symmetric or skewed), spread (range, variance), central tendency (mean vs median), "
        "and any multi-modal patterns. If skewness data is provided, interpret it. "
        "Comment on data quality (missing values, outliers) and practical implications."
    ),
    "correlation": (
        "Focus on: strength and direction of relationships between variables. "
        "Identify the strongest positive and negative correlations. "
        "Note any unexpected relationships or lack thereof. "
        "Consider multicollinearity if 3+ numerical columns are involved."
    ),
    "missing_values": (
        "Focus on: which columns have gaps and how severe (%). "
        "Identify any patterns — do certain columns tend to be missing together? "
        "Assess the risk: can rows with missing values be dropped, or should you impute? "
        "Suggest a handling strategy (drop, mean/median impute, or flag)."
    ),
    "categorical": (
        "Focus on: category distribution balance vs imbalance, dominant categories, "
        "and rare categories (<5% frequency). Based on unique counts, assess cardinality. "
        "Highlight any low-frequency categories that may need grouping."
    ),
    "outliers": (
        "Focus on: which columns have the most extreme values and what % of rows are flagged. "
        "Assess whether outliers are likely data errors or genuine extreme observations. "
        "Suggest whether to cap, transform, or investigate further."
    ),
    "time_series": (
        "Focus on: trends (upward/downward), seasonality, cyclicity, and any anomalies in the pattern. "
        "Comment on data granularity and coverage gaps. "
        "If the time range is short or sparse, note limitations."
    ),
}

SYSTEM_PROMPT = (
    "You are a senior data analyst. Given a dataset profile and sample rows, "
    "provide 3-5 concise bullet points with specific, actionable insights. "
    "Rules: (1) reference actual column names and values from the context; "
    "(2) be specific — mention numbers, percentages, or categories; "
    "(3) prioritize findings that inform real decisions; "
    "(4) if data is insufficient for meaningful conclusions, state that clearly; "
    "(5) avoid generic advice — anchor every point in the provided data."
)


# ── Local Ollama ─────────────────────────────────────────────

def check_ollama(host=DEFAULT_HOST, port=DEFAULT_PORT):
    try:
        url = f"http://{host}:{port}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def pick_model(models, preferred=DEFAULT_MODEL):
    for m in [preferred] + LIGHT_MODELS:
        if m in models:
            return m
    if models:
        return models[0]
    return preferred


def generate_analysis_local(profile, df, analysis_type, columns,
                            host=DEFAULT_HOST, port=DEFAULT_PORT,
                            model=DEFAULT_MODEL):
    context = _build_context(profile, df, analysis_type, columns)
    guide = ANALYSIS_GUIDES.get(analysis_type, "Focus on key patterns and actionable findings.")
    prompt = (
        f"Analyze this dataset.\n\n"
        f"{context}\n\n"
        f"Guidance for this analysis type:\n{guide}\n\n"
        "Provide 3-5 specific, data-backed insights:"
    )

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"num_predict": 768, "temperature": 0.3}
    }).encode()

    try:
        url = f"http://{host}:{port}/api/generate"
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with _urlopen_retry(req, LOCAL_TIMEOUT) as resp:
            result = json.loads(resp.read())
            return {"ok": True, "text": result.get("response", "").strip()}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace") if e.fp else ""
        msg = f"Ollama returned {e.code}"
        if "memory" in body.lower():
            msg += " — model requires more RAM. Try a smaller model or use a remote provider."
        return {"ok": False, "error": msg}
    except urllib.error.URLError:
        return {"ok": False, "error": "Ollama not reachable — is it running?"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Remote API (OpenAI-compatible) ───────────────────────────

def generate_analysis_remote(profile, df, analysis_type, columns,
                             api_key, endpoint, model):
    context = _build_context(profile, df, analysis_type, columns)
    guide = ANALYSIS_GUIDES.get(analysis_type, "Focus on key patterns and actionable findings.")
    user_prompt = (
        f"Analyze this dataset.\n\n"
        f"{context}\n\n"
        f"Guidance for this analysis type:\n{guide}\n\n"
        "Provide 3-5 specific, data-backed insights:"
    )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 768,
        "temperature": 0.3
    }).encode()

    try:
        req = urllib.request.Request(endpoint, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        with _urlopen_retry(req, TIMEOUT) as resp:
            result = json.loads(resp.read())
            text = result["choices"][0]["message"]["content"].strip()
            return {"ok": True, "text": text}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace") if e.fp else ""
        msg = f"API returned {e.code}"
        if "401" in str(e.code):
            msg = "Invalid API key — check your key and try again"
        elif "402" in str(e.code):
            msg = "API quota exhausted — check your billing"
        elif "insufficient_quota" in body:
            msg = "Free tier quota exceeded"
        return {"ok": False, "error": msg}
    except urllib.error.URLError:
        return {"ok": False, "error": "API not reachable — check your internet connection"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Column naming assistant ──────────────────────────────────

COLUMN_NAMING_SYSTEM_PROMPT = (
    "You are a data engineering assistant. Given sample data from unnamed columns, "
    "suggest short descriptive snake_case names (1-3 words each). "
    "Names should reflect the content, not the position. "
    "Respond ONLY with one mapping per line in this exact format:\n"
    "old_column_name → suggested_name"
)


def suggest_column_names(df, unnamed_cols, model=DEFAULT_MODEL, provider="local",
                         host=DEFAULT_HOST, port=DEFAULT_PORT,
                         api_key="", endpoint=""):
    if not unnamed_cols:
        return {"ok": True, "names": {}}

    lines = []
    for col in unnamed_cols:
        if col not in df.columns:
            continue
        sample = df[col].dropna().head(5).tolist()
        sample_str = ", ".join(repr(v) for v in sample) if sample else "(all empty)"
        lines.append(f'Unnamed column: "{col}"')
        lines.append(f"Sample data: {sample_str}")
        lines.append("")

    user_prompt = (
        "Suggest column names for these unnamed columns.\n\n"
        + "\n".join(lines)
        + "\nRespond ONLY with one mapping per line:"
    )

    if provider == "local":
        payload = json.dumps({
            "model": model,
            "prompt": user_prompt,
            "system": COLUMN_NAMING_SYSTEM_PROMPT,
            "stream": False,
            "options": {"num_predict": 256, "temperature": 0.2}
        }).encode()
        try:
            url = f"http://{host}:{port}/api/generate"
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            with _urlopen_retry(req, LOCAL_TIMEOUT) as resp:
                result = json.loads(resp.read())
                text = result.get("response", "").strip()
        except Exception as e:
            return {"ok": False, "error": str(e)}
    else:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": COLUMN_NAMING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.2
        }).encode()
        try:
            req = urllib.request.Request(endpoint, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with _urlopen_retry(req, TIMEOUT) as resp:
                result = json.loads(resp.read())
                text = result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    names = {}
    for line in text.split("\n"):
        if "→" in line:
            parts = line.split("→", 1)
            old = parts[0].strip().strip('"')
            new = parts[1].strip().strip('"')
            if old in unnamed_cols and new:
                names[old] = new

    if not names:
        return {"ok": False, "error": "Could not parse LLM response"}
    return {"ok": True, "names": names}


# ── Chat with your data ───────────────────────────────────────

CHAT_SYSTEM_PROMPT = (
    "You are a data analysis assistant. You have access to the following dataset. "
    "Answer the user's questions using ONLY the statistics and context provided below. "
    "Be specific — reference actual column names, numbers, and patterns. "
    "If the data doesn't contain enough information, say so clearly. "
    "Do not fabricate data. Keep answers concise (3-5 sentences)."
)


def _build_dataset_context(data_profile, df):
    """Build a text summary of the dataset for the LLM context."""
    parts = []
    shape = data_profile.get('shape', df.shape)
    parts.append(f"Dataset: {shape[0]:,} rows × {shape[1]} columns\n")

    dtypes = data_profile.get('dtypes', {})
    if dtypes:
        type_groups = {}
        for col, t in dtypes.items():
            type_groups.setdefault(t, []).append(col)
        type_lines = []
        for t, cols in sorted(type_groups.items()):
            type_lines.append(f"  {', '.join(cols)} ({t})")
        parts.append("Columns:\n" + "\n".join(type_lines) + "\n")

    missing = data_profile.get('missing_percentage', {})
    missing_with = {c: p for c, p in missing.items() if p > 0}
    if missing_with:
        top = sorted(missing_with.items(), key=lambda x: -x[1])[:5]
        parts.append(f"Missing data: {', '.join(f'{c}={p:.1f}%' for c, p in top)}\n")

    num_cols = data_profile.get('numerical_cols', [])
    if num_cols:
        stats = []
        for col in num_cols[:10]:
            s = df[col].dropna()
            if not s.empty:
                stats.append(f"  {col}: min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}, "
                             f"median={s.median():.4g}, missing={missing.get(col, 0):.1f}%")
        if stats:
            parts.append("Numerical stats:\n" + "\n".join(stats) + "\n")

    cat_cols = data_profile.get('categorical_cols', [])
    if cat_cols:
        cat_stats = []
        for col in cat_cols[:10]:
            n = df[col].nunique()
            top_vals = df[col].value_counts().nlargest(3).index.tolist()
            top_str = ", ".join(repr(v) for v in top_vals)
            cat_stats.append(f"  {col}: {n} unique values, top: {top_str}")
        if cat_stats:
            parts.append("Categorical columns:\n" + "\n".join(cat_stats) + "\n")

    outliers = data_profile.get('has_outliers', {})
    if outliers:
        parts.append(f"Outlier columns: {', '.join(f'{c} ({p:.1f}%)' for c, p in list(outliers.items())[:5])}\n")

    skew = data_profile.get('skewness', {})
    skewed = {c: s for c, s in skew.items() if s is not None and abs(s) > 1}
    if skewed:
        parts.append(f"Skewed columns: {', '.join(f'{c} ({s:.2f})' for c, s in list(skewed.items())[:5])}\n")

    try:
        sample = df.head(3)
        parts.append(f"First 3 rows:\n{sample.to_string(index=False)}\n")
    except Exception:
        pass

    return "\n".join(parts)


def chat_with_data(query, data_profile, df, conversation_history=None,
                   model=DEFAULT_MODEL, provider="local",
                   host=DEFAULT_HOST, port=DEFAULT_PORT,
                   api_key="", endpoint=""):
    """Answer a free-form user question about the dataset using the LLM.

    Args:
        query: User's question string.
        data_profile: Profile dict from DataProcessor.
        df: The DataFrame.
        conversation_history: Optional list of {"role": str, "content": str} dicts.
        provider/model/host/port/api_key/endpoint: LLM connection settings.

    Returns:
        {"ok": True, "text": str} or {"ok": False, "error": str}
    """
    context = _build_dataset_context(data_profile, df)
    user_prompt = f"Dataset context:\n{context}\n\nUser question: {query}"

    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    if conversation_history:
        for msg in conversation_history[-6:]:  # last 6 exchanges
            if msg.get("role") in ("user", "assistant"):
                messages.append(msg)
    messages.append({"role": "user", "content": user_prompt})

    if provider == "local":
        full_prompt = "\n".join(
            (m["content"] if m["role"] == "user" else
             f"System: {m['content']}" if m["role"] == "system" else
             f"Assistant: {m['content']}")
            for m in messages
        )
        payload = json.dumps({
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_predict": 1024, "temperature": 0.3}
        }).encode()
        try:
            url = f"http://{host}:{port}/api/generate"
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            with _urlopen_retry(req, LOCAL_TIMEOUT) as resp:
                result = json.loads(resp.read())
                return {"ok": True, "text": result.get("response", "").strip()}
        except urllib.error.URLError:
            return {"ok": False, "error": "Ollama not reachable — is it running?"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    else:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3
        }).encode()
        try:
            req = urllib.request.Request(endpoint, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with _urlopen_retry(req, TIMEOUT) as resp:
                result = json.loads(resp.read())
                text = result["choices"][0]["message"]["content"].strip()
                return {"ok": True, "text": text}
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace") if e.fp else ""
            msg = f"API returned {e.code}"
            if "401" in str(e.code):
                msg = "Invalid API key"
            elif "402" in str(e.code) or "insufficient_quota" in body:
                msg = "API quota exhausted"
            return {"ok": False, "error": msg}
        except urllib.error.URLError:
            return {"ok": False, "error": "API not reachable"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
