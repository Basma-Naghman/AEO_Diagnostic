import streamlit as st
import concurrent.futures
import re
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient
import openai

# ── Load .env file ────────────────────────────────────────────────────────────
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AEO Diagnostic", page_icon="🔍", layout="wide")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0f0f0f; color: #f0f0f0; }
  h1 { font-size: 2.6rem !important; font-weight: 800 !important; }
  h3 { color: #aaa !important; }
  .score-box { border-radius:16px; padding:24px 20px; text-align:center; margin-bottom:12px; }
  .score-A { background:#0d2e1a; border:2px solid #22c55e; }
  .score-B { background:#1a2e0d; border:2px solid #84cc16; }
  .score-C { background:#2e2200; border:2px solid #eab308; }
  .score-D { background:#2e1100; border:2px solid #f97316; }
  .score-F { background:#2e0d0d; border:2px solid #ef4444; }
  .grade    { font-size:3.5rem; font-weight:900; line-height:1; }
  .ai-name  { font-size:1rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#888; margin-bottom:4px; }
  .ai-badge { font-size:0.72rem; background:#222; border:1px solid #444; border-radius:20px; padding:2px 10px; color:#aaa; display:inline-block; }
  .score-num{ font-size:1.05rem; color:#ccc; margin-top:6px; }
  .response-box { background:#1a1a1a; border:1px solid #333; border-radius:12px; padding:20px; margin-bottom:16px; font-size:0.9rem; line-height:1.7; white-space:pre-wrap; }
  .mention-chip { display:inline-block; background:#1e3a5f; border:1px solid #3b82f6; border-radius:20px; padding:3px 12px; margin:3px; font-size:0.8rem; }
  .mention-chip.target { background:#1a3d1a; border-color:#22c55e; font-weight:700; }
  .divider { border-top:1px solid #222; margin:32px 0; }
  .summary-bar { background:#1a1a1a; border-radius:12px; padding:20px 24px; margin-bottom:24px; }
  .key-status { font-size:0.82rem; padding:2px 0; }
  .warn-box { background:#2e2200; border:1px solid #eab308; border-radius:8px; padding:10px 14px; font-size:0.85rem; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def query_groq(prompt: str, api_key: str) -> str:
    """Query Groq API with Llama 3.3 70B."""
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )
    return resp.choices[0].message.content


def query_openrouter(prompt: str, api_key: str) -> str:
    """
    Query OpenRouter with free model fallbacks.
    Tries multiple free models in order until one succeeds.
    Get your free key at: https://openrouter.ai/keys
    """
    # Free models on OpenRouter (no billing required)
    # openrouter/free auto-picks any available free model — never 404s
    model_candidates = [
        "openrouter/free",                           # ✅ auto-router, always works
        "deepseek/deepseek-chat-v3-0324:free",       # DeepSeek V3 — best quality
        "meta-llama/llama-3.3-70b-instruct:free",    # Llama 3.3 70B
        "mistralai/mistral-small-3.1:free",          # Mistral Small 3.1
        "nvidia/nemotron-nano-12b-v2-vl:free",       # NVIDIA Nemotron — reliable
    ]

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    last_error = None

    for model_name in model_candidates:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            return resp.choices[0].message.content  # ✅ success

        except Exception as e:
            err_str = str(e)
            # 429 or context limit — try next model
            if "429" in err_str or "rate" in err_str.lower() or "context" in err_str.lower():
                last_error = e
                continue
            # 404 model not found — try next
            elif "404" in err_str or "not found" in err_str.lower():
                last_error = e
                continue
            # Any other error — raise immediately
            else:
                raise

    raise RuntimeError(f"All OpenRouter models failed. Last error: {last_error}")


def query_huggingface(prompt: str, api_key: str) -> str:
    """Query HuggingFace Inference API with Qwen 2.5 7B."""
    client = InferenceClient(api_key=api_key)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ── Scoring ───────────────────────────────────────────────────────────────────

def extract_products(text: str) -> list:
    raw = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b', text)
    stopwords = {
        "The","This","These","Those","When","For","And","With","You","Your",
        "Here","Some","Many","Most","Also","Note","Additionally","However",
        "Overall","Consider","Each","Such","While","Since","Both","That",
    }
    seen, products = set(), []
    for p in raw:
        if p not in stopwords and p not in seen and len(p) > 3:
            seen.add(p)
            products.append(p)
    return products[:12]


def score_response(text: str, target: str) -> dict:
    text_lower = text.lower()
    target_lower = target.lower().strip()
    products = extract_products(text)

    if not target_lower:
        return {"rank": None, "mentions": 0, "total_products": len(products),
                "score": None, "grade": "—", "products": products}

    mentions = len(re.findall(re.escape(target_lower), text_lower))
    lines = re.split(r'[\n.•\-\d\.]', text_lower)
    rank = next((i for i, l in enumerate(lines, 1) if target_lower in l), None)

    if mentions == 0:
        score = 0
    else:
        base = 40 + (min(mentions, 5) * 8)
        position_bonus = max(0, 20 - ((rank or 10) * 2))
        score = min(100, base + position_bonus)

    grade = ("A" if score >= 85 else "B" if score >= 70 else
             "C" if score >= 50 else "D" if score >= 30 else "F")

    return {"rank": rank, "mentions": mentions, "total_products": len(products),
            "score": score, "grade": grade, "products": products}


def resolve_key(env_var: str, sidebar_val: str) -> str:
    return sidebar_val.strip() if sidebar_val.strip() else os.getenv(env_var, "")


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("# 🔍 AEO Diagnostic")
st.markdown("### See how **Llama 3, DeepSeek & Qwen** rank your product — all free APIs")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ API Keys")
    st.caption("Loaded from `.env` automatically, or paste here.")

    groq_input   = st.text_input("🟢 Groq (Llama 3) — free",   type="password", placeholder="gsk_...")
    gemini_input = st.text_input("🔵 OpenRouter — free",         type="password", placeholder="sk-or-...")
    hf_input     = st.text_input("🟡 HuggingFace — free",       type="password", placeholder="hf_...")

    st.markdown("**Status:**")
    for label, env_var, inp in [
        ("Groq",         "GROQ_API_KEY",   groq_input),
        ("OpenRouter",    "OPENROUTER_API_KEY", gemini_input),
        ("HuggingFace",  "HF_API_KEY",     hf_input),
    ]:
        resolved = resolve_key(env_var, inp)
        if resolved:
            src = "sidebar" if inp.strip() else ".env file"
            st.markdown(f'<div class="key-status">✅ {label} — <code>{src}</code></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="key-status">❌ {label} — not set</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Get free keys:**")
    st.markdown("🔗 [Groq Console](https://console.groq.com)  \n"
                "🔗 [OpenRouter Keys](https://openrouter.ai/keys)  \n"
                "🔗 [HuggingFace Tokens](https://huggingface.co/settings/tokens)")

    st.markdown("---")
    st.markdown("**OpenRouter free models tried (in order):**")
    st.caption("openrouter/free (auto) → DeepSeek V3 → Llama 3.3 70B → Mistral Small 3.1")

# Main form
col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_input("🛍️ Shopper query",
                          placeholder='"best magnesium supplement for seniors"')
with col2:
    target_product = st.text_input("🎯 Your brand / product (optional)",
                                   placeholder='"Nature Made"')

run = st.button("🚀 Run Diagnostic", use_container_width=True, type="primary")

# ── Execute ───────────────────────────────────────────────────────────────────
if run:
    if not query:
        st.error("Please enter a shopper query.")
        st.stop()

    groq_key   = resolve_key("GROQ_API_KEY",   groq_input)
    gemini_key = resolve_key("OPENROUTER_API_KEY", gemini_input)
    hf_key     = resolve_key("HF_API_KEY",     hf_input)

    missing = [n for n, k in [("Groq", groq_key), ("OpenRouter", gemini_key), ("HuggingFace", hf_key)] if not k]
    if missing:
        st.error(f"Missing API keys: **{', '.join(missing)}** — add in sidebar or `.env` file.")
        st.stop()

    prompt = (
        f'A shopper asks: "{query}"\n\n'
        "As a helpful AI assistant, recommend the best products or brands for this need. "
        "Give a concise, practical answer with specific product or brand names. "
        "Be direct and list your top 5 recommendations."
    )

    AI_CONFIG = [
        ("Llama 3.3 70B",    "Groq",         query_groq,        groq_key),
        ("DeepSeek V3",      "OpenRouter",   query_openrouter,  gemini_key),
        ("Qwen 2.5 7B",      "HuggingFace",  query_huggingface, hf_key),
    ]

    with st.spinner("Querying 3 AIs in parallel… (Gemini retries automatically on errors)"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = {name: ex.submit(fn, prompt, key) for name, _, fn, key in AI_CONFIG}

        results, errors = {}, {}
        for name, _, _, _ in AI_CONFIG:
            try:
                results[name] = futures[name].result()
            except Exception as e:
                errors[name]  = str(e)
                results[name] = None

    # Show errors with helpful context
    for name, err in errors.items():
        if "quota" in err.lower() or "429" in err:
            st.warning(f"⚠️ **{name}:** Quota exhausted — {err}")
        elif "unavailable" in err.lower() or "503" in err:
            st.warning(f"⚠️ **{name}:** Server overloaded after retries — {err}")
        elif "not found" in err.lower() or "404" in err:
            st.warning(f"⚠️ **{name}:** Model not found — {err}")
        else:
            st.error(f"❌ **{name} error:** {err}")

    # ── Report Card ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Report Card")
    st.markdown(f"**Query:** *{query}*")
    if target_product:
        st.markdown(f"**Tracking:** `{target_product}`")
    st.markdown(f"*Run at {datetime.now().strftime('%H:%M:%S')}*\n")

    score_data = {}
    cols = st.columns(3)
    grade_colors = {"A":"#22c55e","B":"#84cc16","C":"#eab308","D":"#f97316","F":"#ef4444","—":"#888"}

    for i, (name, provider, _, _) in enumerate(AI_CONFIG):
        if results.get(name) is None:
            with cols[i]:
                st.markdown(f"""
                <div class="score-box score-F">
                  <div class="ai-name">{name}</div>
                  <div class="ai-badge">{provider}</div>
                  <div class="grade" style="color:#666;margin-top:10px">✗</div>
                  <div class="score-num">API Error</div>
                </div>""", unsafe_allow_html=True)
            continue

        s = score_response(results[name], target_product)
        score_data[name] = s
        grade = s["grade"]
        color = grade_colors.get(grade, "#888")

        with cols[i]:
            rank_text    = f"Rank #{s['rank']}" if s["rank"] else "Not ranked"
            mention_text = f"{s['mentions']} mention{'s' if s['mentions']!=1 else ''}" if target_product else ""
            score_text   = f"Score: {s['score']}/100" if s["score"] is not None else ""

            st.markdown(f"""
            <div class="score-box score-{grade}">
              <div class="ai-name">{name}</div>
              <div class="ai-badge">{provider}</div>
              <div class="grade" style="color:{color};margin-top:10px">{grade}</div>
              <div class="score-num">{score_text}</div>
              <div class="score-num">{mention_text}</div>
              <div class="score-num">{rank_text if target_product else f"{s['total_products']} products found"}</div>
            </div>""", unsafe_allow_html=True)

    # Verdict
    if target_product and score_data:
        avg = sum(v["score"] or 0 for v in score_data.values()) / len(score_data)
        verdict = (
            "🟢 **Strong AI presence** — your product is well known to open AI models." if avg >= 70 else
            "🟡 **Moderate presence** — appearing but not dominating. Build more brand signals." if avg >= 45 else
            "🔴 **Weak AI presence** — your product is largely invisible. You need an AEO strategy."
        )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Overall Verdict")
        st.markdown(f'<div class="summary-bar">{verdict}<br><br><b>Average score across {len(score_data)} AI(s): {avg:.0f} / 100</b></div>',
                    unsafe_allow_html=True)

    # Brands mentioned
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🏷️ Brands / Products Mentioned")
    cols2 = st.columns(3)
    for i, (name, _, _, _) in enumerate(AI_CONFIG):
        if results.get(name) is None:
            continue
        s = score_data.get(name, {})
        with cols2[i]:
            st.markdown(f"**{name}**")
            chips = "".join(
                f'<span class="mention-chip{"  target" if target_product and target_product.lower() in p.lower() else ""}">{p}</span>'
                for p in (s.get("products") or [])
            )
            st.markdown(chips or "_None detected_", unsafe_allow_html=True)

    # Full responses
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 💬 Full AI Responses")
    tabs = st.tabs([name for name, _, _, _ in AI_CONFIG])
    for i, (name, _, _, _) in enumerate(AI_CONFIG):
        with tabs[i]:
            if results.get(name):
                st.markdown(f'<div class="response-box">{results[name]}</div>', unsafe_allow_html=True)
            else:
                st.error("No response — check error above or try again")

    # What to do next
    if target_product:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 💡 What To Do Next")
        low = [n for n, s in score_data.items() if (s.get("score") or 0) < 50]
        if low:
            st.warning(f"**{', '.join(low)}** barely mentions your product. Focus on:")
        st.markdown("""
- **Build citations** — get mentioned on trusted review sites & forums that AIs train on  
- **Optimize your listing** — make title, bullets & A+ content crystal clear  
- **Create Q&A content** — blog posts that directly answer shoppers' questions  
- **Grow reviews** — volume and sentiment on review platforms influences AI rankings
        """)