# 🔍 AEO Diagnostic Tool

> **Answer Engine Optimization** — See how AI models rank your brand across 3 different LLMs instantly.

Python 3.9+ | Streamlit | Groq | OpenRouter | HuggingFace | MIT License

---

## 📌 What is AEO?

**Answer Engine Optimization (AEO)** is the practice of optimizing your brand so that AI models mention and recommend it when users ask shopping or product questions.

Just like SEO helps you rank on Google, **AEO helps you rank inside AI responses** — from ChatGPT to Llama to DeepSeek.

This tool lets you instantly check: *"Does AI know my brand?"*

---

## ✨ Features

- 🤖 **3 AI models queried in parallel** — Llama 3.3 70B, DeepSeek V3, Qwen 2.5 7B
- 📊 **Graded report card** — A to F score based on mentions, position, and frequency
- 🏷️ **Brand detection** — automatically extracts all products/brands mentioned by each AI
- ⚡ **Parallel execution** — all 3 AIs run simultaneously using `concurrent.futures`
- 🔄 **Auto-fallback** — OpenRouter tries multiple free models if one fails
- 💡 **Actionable verdict** — tells you exactly what to do to improve your AI presence
- 🔑 **Flexible API keys** — load from `.env` file or paste directly in the sidebar
- 🌑 **Dark UI** — clean, professional dark theme built with Streamlit

---

## 🖥️ Demo

**Input:** Enter a shopper query + your brand name

```
Query:   "best protein powder for muscle building"
Brand:   "Optimum Nutrition"
```

**Output:** A live report card showing how each AI grades your brand's presence

| Model | Provider | Grade | Score |
|---|---|---|---|
| Llama 3.3 70B | Groq | A | 92/100 |
| DeepSeek V3 | OpenRouter | B | 74/100 |
| Qwen 2.5 7B | HuggingFace | C | 55/100 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Streamlit** | Web UI framework |
| **Groq API** | Llama 3.3 70B — ultra-fast inference (300+ tok/s) |
| **OpenRouter API** | DeepSeek V3 + free model auto-router |
| **HuggingFace Inference API** | Qwen 2.5 7B — open source model |
| **Python `concurrent.futures`** | Parallel API calls |
| **`python-dotenv`** | Secure API key management |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/aeo-diagnostic.git
cd aeo-diagnostic
```

### 2. Install dependencies

```bash
pip install streamlit groq openai huggingface-hub python-dotenv
```

### 3. Set up API keys

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
HF_API_KEY=hf_...
```

### 4. Run the app

```bash
streamlit run aeo_diagnostic.py
```

Open your browser at `http://localhost:8501`

---

## 🔑 Getting Free API Keys

| Provider | Link | Free Tier |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | 14,400 req/day |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | 200 req/day, no card needed |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | 2,000 req/day |

> 💡 All three APIs are completely free — no credit card required.

---

## 📁 Project Structure

```
aeo-diagnostic/
│
├── aeo_diagnostic.py     # Main Streamlit app
├── .env                  # API keys (not committed to git)
├── .env.example          # Template for API keys
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## ⚙️ How It Works

```
User Input (query + brand)
        │
        ▼
 Prompt is constructed
        │
        ▼
┌───────────────────────────────┐
│   concurrent.futures          │
│  ┌─────────┐ ┌────────────┐  │
│  │  Groq   │ │ OpenRouter │  │  ← All 3 run simultaneously
│  └─────────┘ └────────────┘  │
│       ┌──────────────┐        │
│       │  HuggingFace │        │
│       └──────────────┘        │
└───────────────────────────────┘
        │
        ▼
  Scoring Engine
  - Count brand mentions
  - Find rank/position
  - Calculate score (0-100)
  - Assign grade (A-F)
        │
        ▼
  Report Card + Verdict
```

### Scoring Formula

```python
base_score      = 40 + (min(mentions, 5) * 8)   # up to 80 points
position_bonus  = max(0, 20 - (rank * 2))        # up to 20 points
final_score     = min(100, base + position_bonus)
```

| Grade | Score |
|---|---|
| A | 85 – 100 |
| B | 70 – 84 |
| C | 50 – 69 |
| D | 30 – 49 |
| F | 0 – 29 |

---

## 🔄 OpenRouter Fallback Chain

To handle model availability issues, OpenRouter tries models in this order:

```
openrouter/free (auto-router)
    → deepseek/deepseek-chat-v3-0324:free
    → meta-llama/llama-3.3-70b-instruct:free
    → mistralai/mistral-small-3.1:free
    → nvidia/nemotron-nano-12b-v2-vl:free
```

---

## 🗺️ Roadmap

- [ ] Historical tracking — see how AI presence changes over time
- [ ] Competitor comparison — rank yourself vs top 3 competitors
- [ ] More AI models — Claude, GPT-4o, Perplexity
- [ ] PDF export — downloadable report for presentations
- [ ] Batch queries — test multiple queries at once

---

## 👩‍💻 Author

**Basma Naghman**


---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
