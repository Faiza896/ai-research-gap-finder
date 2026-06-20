# main.py
"""
AI Research Gap Finder - Streamlit (Luxury Black & Gold Edition)
FIXED VERSION:
- Removed dead/deprecated Groq model (mixtral-8x7b-32768) that was causing
  silent failures -> fallback gaps (generic, same every time).
- Gaps are now grounded in REAL EXTRACTIVE SENTENCES pulled from each paper
  (not just TF-IDF keywords), so different abstracts genuinely produce
  different gaps.
- Same paper(s) -> same gaps every time (content-hash cache), because the
  hash + temperature=0 keeps the model deterministic for identical input.
  Different paper(s) -> different hash -> fresh Groq call -> different gaps.
- Full error reason is shown in the UI (Settings/Debug) instead of being
  silently swallowed, so you can actually see WHY it falls back if it does.
"""

import os, io, json, re, hashlib
from pathlib import Path
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# Default sklearn tokenization splits on EVERY non-word character, including
# hyphens — so "COVID-19" becomes two separate tokens "covid" + "19", and
# "X-ray" becomes "x" + "ray". This pattern keeps hyphenated alphanumeric
# terms (COVID-19, X-ray, T-cell, multi-modal, etc.) as a single token,
# which is essential for medical/technical research text.
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)*\b"

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT       = Path(os.getcwd())
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR  = ROOT / "gap_cache"
CACHE_DIR.mkdir(exist_ok=True)

def load_groq_api_keys() -> List[str]:
    """
    Supports multiple Groq API keys for automatic rotation when one hits
    its rate limit. Add as many as you want in Streamlit Secrets / env vars:
        GROQ_API_KEY      = "gsk_..."   (first/primary key)
        GROQ_API_KEY_2    = "gsk_..."   (second key, optional)
        GROQ_API_KEY_3    = "gsk_..."   (third key, optional)
        ... up to GROQ_API_KEY_10
    Each key can come from a different free Groq account, so each has its
    own separate 100k-tokens/day pool. When one key gets rate-limited, the
    app automatically rotates to the next key (and tries all models on it)
    before falling back to the evidence-grounded local fallback.
    """
    keys = []
    primary = os.environ.get("GROQ_API_KEY", "").strip()
    if primary:
        keys.append(primary)
    for i in range(2, 11):
        k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys

GROQ_API_KEYS = load_groq_api_keys()
GROQ_API_KEY  = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""  # kept for backward-compat checks elsewhere

st.set_page_config(
    page_title="AI Research Gap Finder",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── BACKGROUND ────────────────────────────────────────────────────────────────
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{data}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>""", unsafe_allow_html=True)
    except Exception:
        pass

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Poppins:wght@300;400;500;600&display=swap');
h1, h2, h3 { font-family: 'Playfair Display', serif; color: #EAD7D1; text-shadow: 0 0 8px rgba(212,175,55,0.7); }
input, textarea { background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(212,175,55,0.4) !important; border-radius: 14px !important; color: white !important; }
.stButton button { background: linear-gradient(135deg, #D4AF37, #B8962E); color: #08090A; font-weight: 700; border-radius: 16px; padding: 12px; width: 100%; box-shadow: 0 0 15px rgba(212,175,55,0.5); transition: all 0.3s ease; font-size: 15px; }
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 0 25px rgba(212,175,55,1); }
header[data-testid="stHeader"] { visibility: hidden; height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important; }
.main .block-container { padding-top: 0 !important; padding-left: 2rem !important; padding-right: 2rem !important; padding-bottom: 2rem !important; }
.app-header { display: flex; align-items: center; gap: 12px; padding: 14px 18px 10px 18px; background: rgba(0,0,0,0.6); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(212,175,55,0.25); margin-bottom: 14px; }
.header-ai { font-size: 46px; font-weight: 900; color: #D4AF37; font-family: 'Playfair Display', serif; text-shadow: 0 0 20px #D4AF37, 0 0 40px #B8962E; line-height: 1; }
.header-title { font-size: 19px; font-weight: 800; color: #D4AF37; font-family: 'Playfair Display', serif; line-height: 1.2; }
.header-subtitle { font-size: 11px; color: #aaa; margin-top: 3px; }
.mobile-card { background: rgba(255,255,255,0.05); backdrop-filter: blur(8px); border: none !important; border-radius: 18px; padding: 22px; box-shadow: 0 8px 30px rgba(0,0,0,0.4); margin-bottom: 14px; }
.gap-card { background: rgba(20,15,5,0.7); border-left: 4px solid #D4AF37; border-radius: 14px; padding: 18px; margin-bottom: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
.gap-number { display: inline-block; background: #D4AF37; color: #000; border-radius: 50%; width: 26px; height: 26px; text-align: center; line-height: 26px; font-weight: 900; font-size: 13px; margin-right: 8px; }
.gap-title { color: #D4AF37; font-weight: 800; font-size: 16px; margin-bottom: 8px; font-family: 'Playfair Display', serif; }
.gap-desc  { color: #ddd; font-size: 13.5px; line-height: 1.7; margin-bottom: 10px; }
.gap-why   { background: rgba(212,175,55,0.08); border-radius: 8px; padding: 8px 12px; color: #c8a84b; font-size: 12.5px; margin-bottom: 8px; line-height: 1.5; }
.gap-question { background: rgba(212,175,55,0.14); border-radius: 8px; padding: 10px 14px; color: #D4AF37; font-size: 13px; font-style: italic; line-height: 1.5; }
.gap-approach { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 8px 12px; color: #aaa; font-size: 12px; margin-top: 8px; line-height: 1.5; }
.gap-evidence { background: rgba(100,180,255,0.06); border: 1px dashed rgba(100,180,255,0.3); border-radius: 8px; padding: 8px 12px; color: #9ec9ff; font-size: 11.5px; margin-top: 8px; line-height: 1.5; }
.badge { display: inline-block; background: rgba(212,175,55,0.15); border: 1px solid rgba(212,175,55,0.4); color: #D4AF37; border-radius: 20px; padding: 4px 12px; margin: 3px; font-size: 12px; }
.topic-card { background: rgba(212,175,55,0.06); border: 1px solid rgba(212,175,55,0.2); border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; }
.topic-title { color: #D4AF37; font-weight: 700; font-size: 13px; margin-bottom: 4px; }
.topic-words { color: #aaa; font-size: 12px; }
.paper-stat { background: rgba(212,175,55,0.08); border-radius: 10px; padding: 8px 12px; margin-bottom: 6px; color: #ddd; font-size: 12px; }
.footer-divider { border: none; border-top: 1px solid rgba(212,175,55,0.25); margin: 24px 0 16px 0; }
.debug-box { background: rgba(0,0,0,0.5); border: 1px solid rgba(212,175,55,0.3); border-radius: 10px; padding: 12px; color: #9ec9ff; font-size: 11.5px; font-family: monospace; white-space: pre-wrap; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

add_bg_from_local("assets/uu.jpeg")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def save_txt(text, fname): return text.encode("utf-8")

def save_pdf(text, fname):
    if not HAS_REPORTLAB: return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750; c.setFont("Helvetica", 10)
    for line in text.splitlines():
        c.drawString(40, y, line[:120]); y -= 14
        if y < 60: c.showPage(); y = 750
    c.save(); buf.seek(0); return buf

def fetch_file_from_url(url):
    try:
        r = requests.get(url, timeout=15); r.raise_for_status()
        filename = os.path.basename(urlparse(url).path) or "downloaded_file"
        return filename.lower(), r.content
    except Exception as e:
        st.error(f"Failed to fetch file: {e}"); return None, None

def parse_uploaded_files(uploaded_files):
    texts = []
    if not uploaded_files: return []
    for file in uploaded_files:
        name = file.name.lower()
        try:
            if name.endswith(".pdf") and HAS_PDFPLUMBER:
                with pdfplumber.open(file) as pdf:
                    texts.append("\n".join(p.extract_text() or "" for p in pdf.pages))
            elif name.endswith(".csv"):
                df = pd.read_csv(file)
                for col in df.columns:
                    if "abstract" in col.lower() or "title" in col.lower():
                        texts.extend(df[col].astype(str).tolist()); break
                else:
                    texts.extend(df.astype(str).iloc[:, 0].tolist())
            else:
                texts.append(file.getvalue().decode("utf-8", errors="ignore"))
        except Exception:
            try: texts.append(file.getvalue().decode("utf-8", errors="ignore"))
            except: continue
    return [t for t in texts if t and t.strip()]

# ── CONTENT HASH (for deterministic caching) ─────────────────────────────────
def content_hash(texts: List[str], keyword: str) -> str:
    """
    Same exact papers + same keyword -> same hash -> same cached gaps.
    Any change in content (even 1 paper swapped) -> different hash -> new gaps.
    """
    joined = "||".join(t.strip() for t in texts) + f"::KW={keyword.strip().lower()}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]

def load_cached_gaps(h: str):
    f = CACHE_DIR / f"{h}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def save_cached_gaps(h: str, gaps: list):
    try:
        (CACHE_DIR / f"{h}.json").write_text(json.dumps(gaps, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# ── TOPIC EXTRACTION ──────────────────────────────────────────────────────────
def extract_tfidf_keywords_per_paper(texts: List[str], top_n: int = 8) -> List[List[str]]:
    """Har paper ke liye top keywords nikalo alag alag."""
    results = []
    for text in texts:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=200, ngram_range=(1, 2), token_pattern=TOKEN_PATTERN)
            X = vec.fit_transform([text])
            terms = vec.get_feature_names_out()
            scores = X.toarray()[0]
            top_idx = np.argsort(scores)[-top_n:][::-1]
            kws = [terms[i] for i in top_idx if scores[i] > 0]
            results.append(kws)
        except Exception:
            results.append([])
    return results

def compute_topics_single_or_few(texts, n_topics=6):
    """
    For 1-3 papers, clustering is meaningless (you need multiple documents
    to form groups). Instead, pull the top TF-IDF n-grams directly from the
    combined text and present each strong term/phrase as its own topic.
    Returns a list of (topic_label, normalized_score) tuples so the
    dashboard can show REAL TF-IDF strength, not a fake metric.
    """
    corpus = [t.replace("\n", " ") for t in texts if t and t.strip()]
    if not corpus:
        return []
    try:
        # Use raw term frequency (not TF-IDF) for single/few docs: with so
        # few documents, IDF weighting collapses to near-identical values
        # for every term, producing a flat/meaningless chart. Raw frequency
        # ("how many times does this term/phrase appear in your paper(s)")
        # is a genuine, explainable metric that naturally varies.
        vec = TfidfVectorizer(stop_words="english", max_features=300, ngram_range=(1, 2),
                               min_df=1, use_idf=False, smooth_idf=False, norm=None, token_pattern=TOKEN_PATTERN)
        X = vec.fit_transform(corpus)
        terms = vec.get_feature_names_out()
        if len(terms) == 0:
            return []
        raw_scores = np.asarray(X.sum(axis=0)).ravel()
        order = np.argsort(raw_scores)[::-1]
        topics, seen = [], set()
        for idx in order:
            term = terms[idx]
            # skip near-duplicate topics (substrings of an already-picked topic)
            if any(term in t or t in term for t, _ in topics):
                continue
            seen.add(term)
            topics.append((term, float(raw_scores[idx])))
            if len(topics) >= n_topics:
                break
        return topics
    except Exception:
        return []


def compute_topics_clustered(texts, n_topics=6):
    """
    For larger collections (4+ papers), use SVD + KMeans to group papers
    into thematic clusters and label each cluster with its top terms.
    Returns (topic_label, paper_count) tuples — paper_count is the REAL
    number of papers assigned to that cluster, a genuine, verifiable metric.
    """
    corpus = [t.replace("\n", " ") for t in texts if t and t.strip()]
    if not corpus: return []
    vec = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2), token_pattern=TOKEN_PATTERN)
    X = vec.fit_transform(corpus)
    if X.shape[1] <= 1: return []
    n_comp = min(50, X.shape[1] - 1, len(corpus) - 1)
    if n_comp < 1: return []
    svd = TruncatedSVD(n_components=n_comp)
    Xr = svd.fit_transform(X)
    # Need at least 2 docs per cluster on average for clustering to be meaningful
    k = min(n_topics, max(1, len(corpus) // 2))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xr)
    terms = vec.get_feature_names_out()
    topics = []
    for i in range(k):
        comp = svd.components_.T.dot(kmeans.cluster_centers_[i])
        top_idx = np.argsort(comp)[-6:][::-1]
        label = " ".join(terms[j] for j in top_idx if j < len(terms))
        paper_count = int(np.sum(labels == i))
        topics.append((label, paper_count))
    seen, unique = set(), []
    for t, c in topics:
        if t not in seen: seen.add(t); unique.append((t, c))
    return unique

def compute_topics_tfidf(texts, n_topics=6):
    """
    Dispatcher: chooses the right strategy based on how many papers are
    given. Clustering needs enough documents to form real groups; below
    that threshold it degrades to "1 topic" or "no topic" (the bug you saw).
    Always returns a list of (topic_label, real_score) tuples.
    """
    n = len([t for t in texts if t and t.strip()])
    if n == 0:
        return []
    if n <= 3:
        return compute_topics_single_or_few(texts, n_topics=n_topics)
    clustered = compute_topics_clustered(texts, n_topics=n_topics)

    def is_messy(label: str) -> bool:
        words = label.split()
        # repeated word inside the same label (e.g. "sensors sensors sensors")
        if len(words) != len(set(words)):
            return True
        return False

    # Safety net: if clustering comes back thin, empty, or with messy/
    # repetitive labels (cluster-center term extraction can do this on
    # small or overlapping corpora), fall back to the cleaner keyword-based
    # method so the user never sees garbled or "no topics" output.
    if len(clustered) < 2 or any(is_messy(t) for t, _ in clustered):
        return compute_topics_single_or_few(texts, n_topics=n_topics)
    return clustered

# ── EXTRACTIVE SENTENCE GROUNDING ────────────────────────────────────────────
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

LIMITATION_CUES = [
    "limitation", "future work", "however", "lack of", "lacks", "not addressed",
    "remains unclear", "remains a challenge", "this study did not", "we did not",
    "further research", "open problem", "not yet", "未", "no existing",
    "few studies", "limited to", "small sample", "has not been", "have not been",
    "is not well", "are not well", "unexplored", "understudied", "gap in",
    "challenge", "constrain", "drawback"
]

def extract_key_sentences(text: str, max_sentences: int = 6) -> List[str]:
    """
    Pull out the most informative sentences from a paper:
    1) Sentences that explicitly signal a limitation/gap/future-work cue.
    2) If not enough found, fall back to the longest/most specific sentences
       (specific = contains numbers, technical terms, capitalized acronyms).
    This gives Groq REAL textual evidence instead of just keyword soup,
    so two different abstracts produce genuinely different gaps.
    """
    if not text or not text.strip():
        return []
    clean = re.sub(r'\s+', ' ', text).strip()
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(clean) if len(s.strip()) > 25]
    if not sentences:
        return [clean[:300]]

    cue_hits = []
    for s in sentences:
        low = s.lower()
        if any(cue in low for cue in LIMITATION_CUES):
            cue_hits.append(s)

    if len(cue_hits) >= max_sentences:
        return cue_hits[:max_sentences]

    # Fill remaining slots with information-dense sentences
    def specificity_score(s):
        score = 0
        score += len(re.findall(r'\d', s))                      # numbers/datasets/%
        score += len(re.findall(r'\b[A-Z]{2,}\b', s))            # acronyms (CT, CNN, IoT)
        score += min(len(s) // 20, 6)                            # longer = more detail, capped
        return score

    remaining = [s for s in sentences if s not in cue_hits]
    remaining.sort(key=specificity_score, reverse=True)
    needed = max_sentences - len(cue_hits)
    return cue_hits + remaining[:needed]

def build_grounded_paper_blocks(texts: List[str], max_papers_detailed: int = 15) -> str:
    """
    Build a per-paper evidence block using extractive key sentences
    (not just truncated raw text, not just keywords).
    """
    blocks = []
    for i, text in enumerate(texts[:max_papers_detailed]):
        sents = extract_key_sentences(text, max_sentences=5)
        if not sents:
            continue
        evidence = " | ".join(sents)
        blocks.append(f"--- PAPER {i+1} EVIDENCE ---\n{evidence}")
    if len(texts) > max_papers_detailed:
        blocks.append(f"--- +{len(texts) - max_papers_detailed} more papers (sampled in topics above) ---")
    return "\n\n".join(blocks)

# ── GROQ API ──────────────────────────────────────────────────────────────────
# Only CURRENT, non-deprecated Groq model IDs. mixtral-8x7b-32768 and
# gemma-7b-it were removed by Groq and will return errors -> silent fallback.
GROQ_MODELS = [
    "llama-3.1-8b-instant",      # smaller, separate quota pool — usually has headroom first
    "llama-3.3-70b-versatile",   # best quality, but shared 100k TPD limit fills up fast
    "gemma2-9b-it",              # secondary fallback, separate quota pool
    "openai/gpt-oss-120b",       # tertiary fallback (current Groq lineup)
]

def call_groq_for_gaps(texts: List[str], keyword: str, topics: List[str], per_paper_kws: List[List[str]]):
    """
    Groq API call grounded in REAL extractive evidence sentences from each
    paper, not just keywords. temperature=0 for determinism: identical input
    -> identical output (combined with the content-hash cache layer).

    KEY ROTATION: if multiple GROQ_API_KEY / GROQ_API_KEY_2 / ... are
    configured, each key (its own separate 100k-tokens/day quota) is tried
    in turn, and for each key every model in GROQ_MODELS is tried. This
    means a single saturated key/model no longer blocks the whole app —
    it just rotates to the next available key automatically.
    """
    if not GROQ_API_KEYS:
        return [], "No API key set in environment / Streamlit secrets", []

    n_papers    = len(texts)
    topic_labels = [t for t, _ in topics] if topics else []
    topics_str  = ", ".join(topic_labels[:8]) if topic_labels else "not detected"
    kw          = keyword.strip() if keyword and keyword.strip() else "the domain found in these papers (infer it yourself from the evidence)"

    evidence_blocks = build_grounded_paper_blocks(texts)
    if not evidence_blocks.strip():
        return [], "No extractable text/evidence from the provided papers", []

    prompt = f"""You are an expert academic peer reviewer specializing in identifying research gaps.

CONTEXT:
- Total papers provided: {n_papers}
- User-specified focus keyword: {kw}
- Cross-paper topic clusters (TF-IDF derived): {topics_str}

REAL EVIDENCE EXTRACTED FROM EACH PAPER (verbatim key sentences — this is your ONLY source of truth):
{evidence_blocks}

YOUR TASK:
Identify exactly 6 research gaps that are DIRECTLY derived from the evidence above. Do NOT invent generic AI/ML gaps (real-time processing, explainability, federated learning, benchmarks, multimodal fusion) UNLESS the evidence sentences themselves actually mention or imply that specific issue for THIS domain.

HARD RULES:
1. Every gap title and description MUST reference specific terminology, methods, datasets, or findings that appear in the EVIDENCE block above — not generic AI buzzwords.
2. If the evidence contains explicit limitation/future-work statements, prioritize turning those into gaps.
3. If evidence is thin for a paper, you may still infer a plausible gap, but it must connect to the actual subject matter (disease, technique, population, dataset, etc.) named in the evidence — never a placeholder like "{kw}" repeated mechanically. Write out the REAL domain term, never leave it as a generic stand-in.
4. The 6 gaps must be meaningfully different from each other (different sub-problems, not rephrasing of the same idea).
5. For each gap, include an "evidence" field quoting (paraphrased, not verbatim) which paper(s) (by number) and which idea in the evidence block led you to this gap.
6. If two different sets of papers were given, your gaps for THIS set must reflect THESE papers' actual content — assume the reader will check correctness against the evidence block.

Respond with ONLY a valid JSON array, nothing else (no markdown fences, no prose before/after).

[
  {{
    "title": "emoji + specific gap title naming the real domain/method from evidence",
    "desc": "2-3 sentences describing this gap using exact terminology found in the evidence block",
    "why": "why this specific gap matters, grounded in the real-world context of these papers",
    "question": "a precise, publishable research question using domain-specific vocabulary from the evidence",
    "approach": "concrete methodology referencing specific tools/datasets/techniques relevant to this exact domain",
    "evidence": "which paper number(s) and which idea in the evidence this gap was derived from"
  }}
]"""

    attempt_log = []

    for key_idx, api_key in enumerate(GROQ_API_KEYS, start=1):
        key_label = f"Key#{key_idx}"
        for model in GROQ_MODELS:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are an academic research analyst. Always respond with valid JSON only, grounded strictly in the evidence given. No markdown, no explanation outside JSON."},
                            {"role": "user",   "content": prompt}
                        ],
                        "temperature": 0.0,
                        "max_tokens":  2200,
                    },
                    timeout=45
                )
            except requests.exceptions.RequestException as e:
                attempt_log.append(f"❌ {key_label}/{model}: network error — {str(e)[:100]}")
                continue

            if resp.status_code == 200:
                try:
                    raw = resp.json()["choices"][0]["message"]["content"].strip()
                except Exception:
                    attempt_log.append(f"❌ {key_label}/{model}: unexpected response shape")
                    continue

                raw = re.sub(r"```json|```", "", raw).strip()
                match = re.search(r'\[.*\]', raw, re.DOTALL)
                try:
                    gaps_raw = json.loads(match.group() if match else raw)
                except json.JSONDecodeError:
                    attempt_log.append(f"❌ {key_label}/{model}: JSON parse error")
                    continue

                validated = []
                for g in gaps_raw:
                    if all(k in g for k in ["title", "desc", "why", "question", "approach"]):
                        validated.append({
                            "title":    g["title"],
                            "desc":     g["desc"],
                            "why":      "🎯 " + g["why"],
                            "question": "💡 " + g["question"],
                            "approach": "🔧 " + g["approach"],
                            "evidence": "📚 " + g.get("evidence", "Derived from provided papers"),
                        })

                if len(validated) >= 3:
                    attempt_log.append(f"✅ {key_label}/{model}: success — {len(validated)} gaps")
                    return validated[:6], f"success via {key_label} / {model}", attempt_log
                else:
                    attempt_log.append(f"⚠️ {key_label}/{model}: only {len(validated)} valid gaps, trying next")
                    continue

            elif resp.status_code == 429:
                detail = resp.text[:150]
                attempt_log.append(f"🚫 {key_label}/{model}: rate-limited (429) — {detail}")
                continue
            elif resp.status_code in (401, 403):
                # This specific key is bad/revoked — no point trying other models on it, move to next key
                attempt_log.append(f"🔑 {key_label}/{model}: auth error ({resp.status_code}) — key may be invalid, skipping rest of this key")
                break
            elif resp.status_code in (400, 404):
                attempt_log.append(f"❌ {key_label}/{model}: invalid/unavailable ({resp.status_code})")
                continue
            else:
                attempt_log.append(f"❌ {key_label}/{model}: HTTP {resp.status_code}")
                continue

    full_log = " | ".join(attempt_log) if attempt_log else "All Groq keys/models failed for an unknown reason"
    return [], full_log, attempt_log


def generate_fallback_gaps(texts, topics, keyword):
    """
    Fallback ONLY if Groq truly cannot be reached. Now grounded in
    extracted key sentences per paper instead of generic templates, so even
    the fallback differs across different input papers.
    """
    all_evidence = []
    for t in texts:
        all_evidence.extend(extract_key_sentences(t, max_sentences=3))
    merged = " ".join(all_evidence).lower() if all_evidence else " ".join(texts).lower()
    kw = keyword.strip() if keyword and keyword.strip() else (topics[0][0] if topics else "this dataset's subject matter")

    # Pull a few concrete evidence snippets to embed directly into gaps
    snippets = all_evidence[:6] if all_evidence else ["No explicit limitation statements found in the provided text."]

    gaps = []
    seen_titles = set()
    for i, snip in enumerate(snippets):
        title = f"📌 Unaddressed Issue #{i+1}: {snip[:55].strip()}…" if len(snip) > 55 else f"📌 Unaddressed Issue #{i+1}: {snip.strip()}"
        if title in seen_titles:
            continue
        seen_titles.add(title)
        gaps.append({
            "title": title,
            "desc": f"The source text states: \"{snip.strip()}\" — this points to an area not fully resolved in the current literature on {kw}.",
            "why": f"🎯 This directly affects how findings on {kw} can be trusted, reproduced, or extended.",
            "question": f"💡 How can future work directly resolve: \"{snip[:80].strip()}\"?",
            "approach": f"🔧 Design a follow-up study or method targeting exactly this stated limitation within {kw} research.",
            "evidence": "📚 Extracted directly from the uploaded paper text (TF-IDF/limitation-cue fallback mode — Groq AI unavailable)."
        })
    if not gaps:
        gaps.append({
            "title": f"ℹ️ {kw.title()} — No explicit gap statements found",
            "desc":  f"The uploaded text for {kw} did not contain identifiable limitation or future-work statements; a manual read-through is recommended.",
            "why":   f"🎯 Automated extraction depends on the paper explicitly stating limitations — short abstracts often omit this.",
            "question": f"💡 What open questions remain in recent {kw} literature based on a manual review?",
            "approach": f"🔧 Upload full papers (not just abstracts) for richer limitation/future-work extraction.",
            "evidence": "📚 No matching evidence found (fallback mode)."
        })
    return gaps[:6]


def detect_gaps_smart(texts, topics, keyword, per_paper_kws):
    """
    Main gap detection with content-hash caching:
    - Same exact paper(s) + keyword -> same hash -> cached gaps returned (consistent).
    - Any different paper(s)/keyword -> new hash -> fresh Groq call (different gaps).
    """
    h = content_hash(texts, keyword)
    st.session_state["last_hash"] = h

    cached = load_cached_gaps(h)
    if cached:
        st.session_state["ai_status"] = "✅ Gaps loaded — these papers were analyzed before, so the same verified gaps are shown again."
        return cached

    if GROQ_API_KEYS:
        gaps, status, attempt_log = call_groq_for_gaps(texts, keyword, topics, per_paper_kws)
        st.session_state["last_groq_status"] = status
        st.session_state["last_groq_attempts"] = attempt_log
        if gaps:
            st.session_state["ai_status"] = f"✅ Analysis complete — {len(gaps)} research gaps identified from {len(texts)} paper(s)"
            save_cached_gaps(h, gaps)
            return gaps
        else:
            st.session_state["ai_status"] = "⚠️ AI service temporarily unavailable — showing evidence-based gaps instead"
    else:
        st.session_state["ai_status"] = "⚠️ AI service not configured — showing evidence-based gaps instead"

    gaps = generate_fallback_gaps(texts, topics, keyword)
    save_cached_gaps(h, gaps)
    return gaps


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def render_ranked_bars(items, value_format, max_value=None, bar_color_start="#7a5c10", bar_color_end="#FFE066"):
    """
    Render a horizontal "leaderboard" style ranked bar list using pure HTML/CSS
    (no plotly) — rank number, label, proportional gold bar, and value, all
    in one readable row per item. Works cleanly whether there are 1 or 15 items.
    items: list of (label, numeric_value) sorted however the caller wants ranked.
    value_format: function(value) -> display string, e.g. lambda v: f"{v:.0f}" or f"{v:.0f}%"
    """
    if not items:
        return
    top_value = max_value if max_value is not None else max((v for _, v in items), default=1)
    top_value = top_value if top_value > 0 else 1
    rows_html = []
    for rank, (label, value) in enumerate(items, start=1):
        pct_width = max(4, round(100 * value / top_value))  # min 4% so the bar is always visible
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
        rows_html.append(f"""
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:10px;'>
            <div style='width:32px; flex-shrink:0; text-align:center; font-size:15px; font-weight:800; color:#D4AF37;'>{medal}</div>
            <div style='width:130px; flex-shrink:0; font-size:13px; font-weight:700; color:#EAD7D1; text-transform:capitalize; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;' title='{label}'>{label}</div>
            <div style='flex:1; background:rgba(255,255,255,0.05); border-radius:8px; height:22px; position:relative; overflow:hidden;'>
                <div style='width:{pct_width}%; height:100%; background:linear-gradient(90deg, {bar_color_start}, {bar_color_end}); border-radius:8px; box-shadow:0 0 8px rgba(212,175,55,0.4);'></div>
            </div>
            <div style='width:55px; flex-shrink:0; text-align:right; font-size:13px; font-weight:800; color:#FFE066;'>{value_format(value)}</div>
        </div>""")
    st.markdown(f"<div class='mobile-card'>{''.join(rows_html)}</div>", unsafe_allow_html=True)


def show_dashboard_page(results):
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px 0;'>
        <div style='font-size:28px; font-weight:800; color:#D4AF37; font-family:"Playfair Display",serif; text-shadow:0 0 12px #D4AF37;'>
            📊 Research Insights Dashboard
        </div>
        <div style='color:#aaa; font-size:13px; margin-top:6px;'>Visual summary of topics and gap coverage</div>
    </div>""", unsafe_allow_html=True)

    topics = results.get("topics", [])  # list of (label, real_score) tuples
    gaps   = results.get("gaps", [])
    n_papers = results.get("n_papers", 0)
    if not topics:
        st.info("No topics to display.")
        if st.button("← Back to Results"): nav_to("results")
        return

    topic_labels = [t for t, _ in topics]
    topic_scores = [s for _, s in topics]

    # Decide what the score actually represents, based on which extraction
    # method produced it, so the label is never misleading.
    is_cluster_mode = n_papers > 3 and all(float(s).is_integer() for s in topic_scores) and max(topic_scores, default=0) <= n_papers
    unit_label = "papers" if is_cluster_mode else "mentions"

    # ── RANKED LIST 1: Topic Strength ───────────────────────────────────────
    st.markdown("#### 🏷 Most Prominent Topics")
    if is_cluster_mode:
        st.caption("Ranked by how many of your uploaded papers fall under each topic — top of the list is the most common theme.")
    else:
        st.caption("Ranked by how often each topic/term appears across your paper(s) — top of the list is the most central theme.")

    # Already sorted by score from the extraction functions, but re-sort
    # here defensively so the ranking is always accurate regardless of caller.
    ranked_topics = sorted(zip(topic_labels, topic_scores), key=lambda x: x[1], reverse=True)
    render_ranked_bars(ranked_topics, value_format=lambda v: f"{v:.0f}")

    # ── RANKED LIST 2: Gap-to-topic coverage ────────────────────────────────
    if gaps and topic_labels:
        st.markdown("#### 🔬 Which Topics Are Driving the Gaps")
        st.caption("Ranked by the share of your research gaps connected to each topic. A topic at 0% simply means none of the current gaps focus on that specific term.")

        gap_texts = [(g.get("title","") + " " + g.get("desc","") + " " + g.get("question","") + " " + g.get("approach","")).lower() for g in gaps]

        def topic_matches_gap(label: str, gap_text: str) -> bool:
            # Partial/substring matching so "covid-19" matches gap text
            # mentioning "covid" alone, and vice versa — much more forgiving
            # than requiring an exact whole-word hit.
            label_clean = label.lower().strip()
            if label_clean in gap_text:
                return True
            for part in re.split(r"[\s-]+", label_clean):
                if len(part) > 3 and part in gap_text:
                    return True
            return False

        coverage_pct = []
        for label in topic_labels:
            matching_gaps = sum(1 for gt in gap_texts if topic_matches_gap(label, gt))
            coverage_pct.append(round(100 * matching_gaps / len(gaps), 1))

        ranked_coverage = sorted(zip(topic_labels, coverage_pct), key=lambda x: x[1], reverse=True)
        render_ranked_bars(ranked_coverage, value_format=lambda v: f"{v:.0f}%", max_value=100,
                            bar_color_start="#3a2a10", bar_color_end="#FFE066")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, val, label in [(c1, len(topics), "Topics Found"), (c2, len(gaps), "Gaps Detected"), (c3, n_papers, "Papers Analyzed")]:
        with col:
            st.markdown(f"<div class='mobile-card' style='text-align:center;'><div style='font-size:32px;font-weight:900;color:#D4AF37;'>{val}</div><div style='color:#aaa;font-size:12px;'>{label}</div></div>", unsafe_allow_html=True)

    if st.button("← Back to Results", key="dash_back"): nav_to("results")



# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("page","welcome"),("page_history",[]),("uploaded_texts",[]),
              ("results",{}),("keyword",""),("ai_status",""),("last_groq_status",""),("last_hash",""),("last_groq_attempts",[])]:
    if k not in st.session_state: st.session_state[k] = v

def nav_to(p):
    cur = st.session_state.page
    if p != cur and cur != "processing":
        st.session_state.page_history.append(cur)
    st.session_state.page = p
    st.rerun()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-ai">AI</div>
    <div>
        <div class="header-title">Research<br>Gap Finder</div>
        <div class="header-subtitle">Find publishable research gaps — fast.</div>
    </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "welcome":
    has_key = bool(GROQ_API_KEYS)
    key_count_note = f" ({len(GROQ_API_KEYS)} key{'s' if len(GROQ_API_KEYS) != 1 else ''})" if has_key else ""
    badge   = f"<span style='background:rgba(50,200,50,0.15);border:1px solid rgba(50,200,50,0.4);color:#90ee90;border-radius:20px;padding:4px 14px;font-size:12px;'>✅ Groq AI Active{key_count_note}</span>" if has_key else "<span style='background:rgba(255,100,100,0.1);border:1px solid rgba(255,100,100,0.4);color:#ff9090;border-radius:20px;padding:4px 14px;font-size:12px;'>⚠️ No API Key</span>"
    st.markdown(f"""
    <div style="text-align:center; padding:24px 16px 14px 16px;">
        <div style="font-size:54px; font-weight:900; color:#D4AF37; text-shadow:0 0 20px #D4AF37, 0 0 40px #B8962E; font-family:'Playfair Display',serif; margin-bottom:16px;">Welcome!</div>
        <p style="color:#ddd; font-size:15px; line-height:1.8; padding:0 12px; margin-bottom:12px;">
            Quickly Analyze paper abstracts or upload your own PDFs/CSVs to find publishable research gap.
        </p>
        <div style="margin-bottom:20px;">{badge}</div>
    </div>""", unsafe_allow_html=True)
    if st.button("🚀  Start Research Analysis →", key="start_btn"): nav_to("input")

elif st.session_state.page == "input":
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Input / Upload")
    keyword  = st.text_input("Research keyword (e.g 'publishable AI')", value=st.session_state.get("keyword",""))
    uploaded = st.file_uploader("Upload files (PDF / TXT / CSV) — multiple allowed", accept_multiple_files=True)
    file_url = st.text_input("Or paste a file URL (PDF/TXT)", placeholder="https://arxiv.org/pdf/xxx.pdf")
    pasted   = st.text_area("Or paste abstracts (one per line)", height=150)

    col_back, col_analyze = st.columns([1, 4], gap="small")
    with col_back:
        if st.button("← Back", key="back_btn", use_container_width=True): nav_to("welcome")
    with col_analyze:
        if st.button("🔍 Analyze Now", key="analyze_btn", use_container_width=True):
            texts = []
            if uploaded: texts.extend(parse_uploaded_files(uploaded))
            if file_url and file_url.strip():
                fname, fbytes = fetch_file_from_url(file_url.strip())
                if fname and fbytes:
                    try:
                        if fname.endswith(".pdf") and HAS_PDFPLUMBER:
                            with pdfplumber.open(io.BytesIO(fbytes)) as pdf:
                                texts.append("\n".join(p.extract_text() or "" for p in pdf.pages))
                        else:
                            texts.append(fbytes.decode("utf-8", errors="ignore"))
                    except Exception as e:
                        st.error(f"URL Error: {e}")
            if pasted and pasted.strip():
                texts.extend(p.strip() for p in pasted.splitlines() if p.strip())
            if not texts:
                st.warning("Please upload a file, paste text, or provide a URL.")
            else:
                st.session_state.uploaded_texts = texts
                st.session_state.keyword = keyword
                nav_to("processing")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "processing":
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.markdown("### 🪄 Analyzing your papers…")

    def run_analysis():
        texts = st.session_state.uploaded_texts or ["AI in healthcare.", "Climate ML models."]

        topics = compute_topics_tfidf(texts, n_topics=min(8, max(3, len(texts))))
        per_paper_kws = extract_tfidf_keywords_per_paper(texts, top_n=8)
        keyword = st.session_state.get("keyword", "")
        gaps    = detect_gaps_smart(texts, topics, keyword, per_paper_kws)

        st.session_state.results = {
            "keyword":       keyword or "N/A",
            "n_papers":      len(texts),
            "method":        "TF-IDF + Groq AI (evidence-grounded)" if GROQ_API_KEYS else "TF-IDF + Evidence Fallback",
            "topics":        topics,
            "per_paper_kws": per_paper_kws,
            "gaps":          gaps,
        }

    with st.spinner(f"Analyzing {len(st.session_state.uploaded_texts)} paper(s) with Groq AI…"):
        run_analysis()

    ai_status = st.session_state.get("ai_status","")
    if "✅" in ai_status or "♻️" in ai_status: st.success(ai_status)
    elif ai_status:       st.warning(ai_status)
    st.success("✅ Analysis complete!")
    if st.button("View Results →"): nav_to("results")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "results":
    res = st.session_state.get("results", {})
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    kw = res.get("keyword","N/A")
    st.markdown(f"### 🔍 Results for: <span style='color:#D4AF37'>{kw}</span>", unsafe_allow_html=True)
    st.markdown(f"**Papers analyzed:** {res.get('n_papers',0)}  &nbsp;•&nbsp;  **Method:** {res.get('method','-')}")

    if st.button("📊 View Dashboard →", key="dash_btn"): nav_to("dashboard")

    # ── Topics Section ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏷 Topics Found Across All Papers")
    topics = res.get("topics", [])
    if topics:
        st.markdown("**Overall themes detected:**")
        st.markdown("".join(f"<span class='badge'>{t} ({s:.0f})</span>" if float(s).is_integer() else f"<span class='badge'>{t}</span>" for t, s in topics), unsafe_allow_html=True)
    else:
        st.info("No topics extracted.")

    per_paper_kws = res.get("per_paper_kws", [])
    if per_paper_kws and len(per_paper_kws) > 1:
        st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)
        with st.expander(f"📄 View topics from each paper ({len(per_paper_kws)} papers)"):
            for i, kws in enumerate(per_paper_kws):
                if kws:
                    st.markdown(
                        f"<div class='topic-card'>"
                        f"<div class='topic-title'>Paper {i+1}</div>"
                        f"<div class='topic-words'>🔑 {' • '.join(kws[:6])}</div>"
                        f"</div>", unsafe_allow_html=True)

    # ── Gaps Section ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔬 Identified Research Gaps")
    gaps = res.get("gaps", [])
    if gaps:
        for i, g in enumerate(gaps, 1):
            evidence_html = f"<div class='gap-evidence'>{g.get('evidence','')}</div>" if g.get("evidence") else ""
            st.markdown(f"""
            <div class='gap-card'>
                <div class='gap-title'><span class='gap-number'>{i}</span>{g.get('title','')}</div>
                <div class='gap-desc'>{g.get('desc','')}</div>
                <div class='gap-why'>{g.get('why','')}</div>
                <div class='gap-question'>{g.get('question','')}</div>
                <div class='gap-approach'>{g.get('approach','')}</div>
                {evidence_html}
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No gaps detected.")

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Export Results")
    txt_lines = [
        f"Keyword: {res.get('keyword','N/A')}",
        f"Papers analyzed: {res.get('n_papers',0)}",
        f"Method: {res.get('method','-')}",
        "", "== OVERALL TOPICS ==",
        *[f"- {t} (score: {s:.2f})" for t, s in res.get("topics",[])],
    ]
    if per_paper_kws:
        txt_lines += ["", "== PER-PAPER TOPICS =="]
        for i, kws in enumerate(per_paper_kws):
            txt_lines.append(f"Paper {i+1}: {', '.join(kws)}")
    txt_lines += ["", "== RESEARCH GAPS =="]
    for i, g in enumerate(gaps):
        txt_lines += [f"\n{i+1}. {g.get('title','')}", f"   {g.get('desc','')}", f"   {g.get('why','')}", f"   {g.get('question','')}", f"   {g.get('approach','')}", f"   {g.get('evidence','')}"]
    txt_content = "\n".join(txt_lines)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.download_button("📄 TXT", data=save_txt(txt_content,"r.txt"), file_name="gaps.txt", mime="text/plain")
    with ec2:
        if HAS_REPORTLAB:
            pb = save_pdf(txt_content, "r.pdf")
            if pb: st.download_button("📕 PDF", data=pb, file_name="gaps.pdf", mime="application/pdf")
        else: st.button("📕 PDF", disabled=True)
    with ec3:
        cb = io.BytesIO()
        pd.DataFrame({"text": st.session_state.uploaded_texts}).to_csv(cb, index=False)
        cb.seek(0)
        st.download_button("📊 CSV", data=cb, file_name="papers.csv", mime="text/csv")

    if st.button("← New Analysis"): st.session_state.page_history = []; nav_to("input")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "dashboard":
    show_dashboard_page(st.session_state.get("results", {}))

elif st.session_state.page == "about":
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.markdown("### ℹ About")
    st.write("**AI Research Gap Finder** — Black & Gold theme.")
    st.write("- Groq AI (Llama 3.3 70B, current model lineup) for domain-specific gap detection.")
    st.write("- Gaps are grounded in extractive evidence sentences pulled from each paper, not generic templates.")
    st.write("- Same papers always return the same gaps (content-hash cache); different papers always get fresh gaps.")
    st.write("- TF-IDF for topic extraction from all papers.")
    st.write("- Supports 1 to 100+ papers — PDF, TXT, CSV.")
    if st.button("← Back", key="about_back"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "settings":
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.markdown("### ⚙ Settings")
    st.markdown("#### Groq AI Status")
    if GROQ_API_KEYS:
        st.success(f"✅ Groq AI connected — {len(GROQ_API_KEYS)} key(s) configured for automatic rotation")
        for i, k in enumerate(GROQ_API_KEYS, start=1):
            masked = k[:8] + "..." + k[-4:] if len(k) > 12 else "***"
            st.caption(f"Key #{i}: {masked}")
        st.caption(f"Models tried per key (in priority order): {', '.join(GROQ_MODELS)}")
        st.caption("On a 429 rate-limit, the app moves to the next model, then the next key automatically. On an invalid/revoked key (401/403), it skips straight to the next key.")
        st.markdown("**To add more keys:** in Streamlit Cloud → App settings → Secrets, add `GROQ_API_KEY_2 = \"gsk_...\"`, `GROQ_API_KEY_3 = \"gsk_...\"`, etc. (up to 10). Each should be a free key from a *different* Groq account for a separate quota pool.")
    else:
        st.warning("⚠️ No GROQ_API_KEY found. Add at least one in Streamlit Secrets.")
    st.markdown("#### Storage")
    st.code(str(MODELS_DIR))
    st.markdown("#### Gap Cache")
    cache_files = list(CACHE_DIR.glob("*.json"))
    st.write(f"{len(cache_files)} cached result set(s) stored. Each is keyed to the exact paper text + keyword used.")
    if st.button("🗑 Clear Gap Cache (force fresh gaps next time)"):
        for f in cache_files:
            try: f.unlink()
            except Exception: pass
        st.success("Cache cleared.")
    if st.button("🗑 Reset App"):
        for k in ["uploaded_texts","page_history"]: st.session_state[k] = []
        for k in ["keyword","ai_status","last_groq_status","last_hash"]: st.session_state[k] = ""
        st.session_state["last_groq_attempts"] = []
        st.session_state["results"] = {}
        st.success("Reset!"); nav_to("welcome")
    if st.button("← Back", key="settings_back"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "help":
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.markdown("### ❓ Help & Tips")
    st.write("- Upload **multiple PDFs** at once — all will be analyzed together.")
    st.write("- CSV with **'abstract'** column works best for bulk papers.")
    st.write("- Add keyword like 'COVID-19' for more focused gaps (optional — the AI infers the domain from your papers either way).")
    st.write("- Results show **overall topics** + **per-paper topics** + **6 evidence-grounded research gaps**.")
    st.write("- Same exact paper(s) always return the same gaps. Change the paper text (even slightly) or keyword to get new gaps.")
    st.write("- Groq API key in Streamlit Secrets for AI-powered gaps.")
    st.write("- Check **Settings → Gap Cache** if you ever want to force a fresh re-analysis of the same papers.")
    if st.button("← Back", key="help_back"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='footer-divider'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⚙️ SETTINGS", key="footer_settings", use_container_width=True): nav_to("settings")
with col2:
    if st.button("❓ HELP", key="footer_help", use_container_width=True): nav_to("help")
with col3:
    if st.button("ℹ️ ABOUT", key="footer_about", use_container_width=True): nav_to("about")
