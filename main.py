# main.py
"""
AI Research Gap Finder - Streamlit (Luxury Black & Gold Edition)
UPDATED: Groq API (Free, Unlimited) for intelligent domain-specific gap detection.
"""

import os, io, json, re
from pathlib import Path
from typing import List
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
import requests
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    HAS_BERTOPIC = True
except Exception:
    HAS_BERTOPIC = False

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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Poppins:wght@300;400;500;600&display=swap');
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #EAD7D1;
    text-shadow: 0 0 8px rgba(212,175,55,0.7);
}
input, textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(212,175,55,0.4) !important;
    border-radius: 14px !important;
    color: white !important;
}
.stButton button {
    background: linear-gradient(135deg, #D4AF37, #B8962E);
    color: #08090A; font-weight: 700; border-radius: 16px;
    padding: 12px; width: 100%;
    box-shadow: 0 0 15px rgba(212,175,55,0.5), inset 0 2px 5px rgba(255,255,255,0.5);
    transition: all 0.3s ease; font-size: 15px;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(212,175,55,1);
}
header[data-testid="stHeader"] {
    visibility: hidden; height: 0 !important;
    min-height: 0 !important; padding: 0 !important; margin: 0 !important;
}
.main .block-container {
    padding-top: 0 !important; padding-left: 2rem !important;
    padding-right: 2rem !important; padding-bottom: 2rem !important;
}
.app-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 18px 10px 18px;
    background: rgba(0,0,0,0.6); backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(212,175,55,0.25); margin-bottom: 14px;
}
.header-ai {
    font-size: 46px; font-weight: 900; color: #D4AF37;
    font-family: 'Playfair Display', serif;
    text-shadow: 0 0 20px #D4AF37, 0 0 40px #B8962E; line-height: 1;
}
.header-title {
    font-size: 19px; font-weight: 800; color: #D4AF37;
    font-family: 'Playfair Display', serif;
    text-shadow: 0 0 10px #D4AF37; line-height: 1.2;
}
.header-subtitle { font-size: 11px; color: #aaa; margin-top: 3px; }
.mobile-card {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(8px);
    border: none !important; border-radius: 18px; padding: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4); margin-bottom: 14px;
}
.gap-card {
    background: rgba(20,15,5,0.7); border-left: 4px solid #D4AF37;
    border-radius: 14px; padding: 18px; margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.gap-number {
    display: inline-block; background: #D4AF37; color: #000;
    border-radius: 50%; width: 26px; height: 26px;
    text-align: center; line-height: 26px;
    font-weight: 900; font-size: 13px; margin-right: 8px;
}
.gap-title { color: #D4AF37; font-weight: 800; font-size: 16px; margin-bottom: 8px;
             font-family: 'Playfair Display', serif; }
.gap-desc  { color: #ddd; font-size: 13.5px; line-height: 1.7; margin-bottom: 10px; }
.gap-why   { background: rgba(212,175,55,0.08); border-radius: 8px; padding: 8px 12px;
             color: #c8a84b; font-size: 12.5px; margin-bottom: 8px; line-height: 1.5; }
.gap-question { background: rgba(212,175,55,0.14); border-radius: 8px; padding: 10px 14px;
                color: #D4AF37; font-size: 13px; font-style: italic; line-height: 1.5; }
.gap-approach { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 8px 12px;
                color: #aaa; font-size: 12px; margin-top: 8px; line-height: 1.5; }
.badge {
    display: inline-block; background: rgba(212,175,55,0.15);
    border: 1px solid rgba(212,175,55,0.4); color: #D4AF37;
    border-radius: 20px; padding: 4px 12px; margin: 3px; font-size: 12px;
}
.footer-divider {
    border: none; border-top: 1px solid rgba(212,175,55,0.25);
    margin: 24px 0 16px 0;
}
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

# ── TOPIC EXTRACTION ──────────────────────────────────────────────────────────
def compute_topics_tfidf(texts, n_topics=6):
    corpus = [t.replace("\n", " ") for t in texts]
    if not corpus: return []
    vec = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    if X.shape[1] <= 1: return []
    svd = TruncatedSVD(n_components=min(50, X.shape[1]-1))
    Xr = svd.fit_transform(X)
    k = min(n_topics, max(1, Xr.shape[0]//2))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit_predict(Xr)
    terms = vec.get_feature_names_out()
    topics = []
    for i in range(k):
        comp = svd.components_.T.dot(kmeans.cluster_centers_[i])
        top_idx = np.argsort(comp)[-6:][::-1]
        topics.append(" ".join(terms[j] for j in top_idx if j < len(terms)))
    seen, unique = set(), []
    for t in topics:
        if t not in seen: seen.add(t); unique.append(t)
    return unique

def compute_topics_bertopic(texts, embedding_model_name="all-MiniLM-L6-v2",
                             model_save_path=MODELS_DIR/"bertopic_model"):
    if not HAS_BERTOPIC: return [], "BERTopic not installed"
    emb = SentenceTransformer(embedding_model_name)
    try:
        if model_save_path.exists():
            t = BERTopic.load(str(model_save_path)); t.transform(texts)
            info = t.get_topic_info()
            top_topics = []
            for t_id in info.topic.values:
                if t_id == -1: continue
                words = t.get_topic(t_id)
                if not words: continue
                top_topics.append(", ".join(w for w, _ in words[:6]))
                if len(top_topics) >= 8: break
            return top_topics, "loaded model"
    except: pass
    model = BERTopic(embedding_model=emb, calculate_probabilities=False, verbose=False)
    model.fit_transform(texts)
    try: model.save(str(model_save_path))
    except: pass
    info = model.get_topic_info()
    top_topics = []
    for t_id in info.topic.values:
        if t_id == -1: continue
        words = model.get_topic(t_id)
        if not words: continue
        top_topics.append(", ".join(w for w, _ in words[:6]))
        if len(top_topics) >= 8: break
    return top_topics, "fitted"

# ══════════════════════════════════════════════════════════════════════════════
# ── GROQ API — INTELLIGENT DYNAMIC GAP DETECTION ─────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def call_groq_for_gaps(text_sample: str, keyword: str, topics: List[str]):
    """Groq API se domain-specific gaps generate karo — free & unlimited!"""
    if not GROQ_API_KEY:
        return [], "No API key"

    topics_str = ", ".join(topics[:5]) if topics else "not detected"
    kw = keyword if keyword else "research domain"

    prompt = f"""You are an expert academic research gap analyst.

PAPER KEYWORD: {kw}
DETECTED TOPICS FROM TF-IDF: {topics_str}

PAPER CONTENT SAMPLE:
{text_sample[:2500]}

TASK: Identify exactly 5 SPECIFIC research gaps from these papers.

STRICT RULES:
- Every gap MUST be 100% specific to THIS paper's domain and content
- Use actual terminology from the paper content above
- Each gap title must mention the specific domain (e.g., "dental caries", "COVID-19", "lung disease")
- Gaps must be different from each other
- Base gaps on what is MISSING or UNEXPLORED in the paper

Respond with ONLY a valid JSON array. No explanation, no markdown backticks.

[
  {{
    "title": "emoji + specific gap title with domain name",
    "desc": "2-3 sentences describing this specific gap using terms from the paper",
    "why": "why this gap matters specifically in this domain",
    "question": "a specific publishable research question for this domain",
    "approach": "concrete methodology using domain-specific methods to address this gap"
  }}
]"""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an academic research analyst. Always respond with valid JSON only. No markdown, no explanation."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.8,
                "max_tokens": 2500,
            },
            timeout=40
        )

        if resp.status_code != 200:
            return [], f"HTTP {resp.status_code}: {resp.text[:300]}"

        data    = resp.json()
        raw     = data["choices"][0]["message"]["content"].strip()

        # Clean JSON
        raw = re.sub(r"```json|```", "", raw).strip()

        # Find JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            gaps_raw = json.loads(match.group())
        else:
            gaps_raw = json.loads(raw)

        validated = []
        for g in gaps_raw:
            if all(k in g for k in ["title","desc","why","question","approach"]):
                validated.append({
                    "title":    g["title"],
                    "desc":     g["desc"],
                    "why":      f"🎯 {g['why']}",
                    "question": f"💡 {g['question']}",
                    "approach": f"🔧 {g['approach']}",
                })

        if len(validated) >= 3:
            return validated[:6], "success"
        else:
            return [], f"Only {len(validated)} valid gaps"

    except json.JSONDecodeError as e:
        return [], f"JSON error: {str(e)[:100]}"
    except Exception as e:
        return [], f"Exception: {str(e)[:100]}"


def generate_fallback_gaps(texts, topics, keyword):
    """Fallback agar Groq fail ho."""
    merged = " ".join(texts).lower()
    kw = keyword or (topics[0] if topics else "this domain")
    gaps = []
    checks = [
        ("real-time" not in merged,
         f"⚡ Real-Time Processing Gap in {kw.title()}",
         f"No real-time approach discussed for {kw}.",
         f"Real-time {kw} systems are critical for deployment.",
         f"How can {kw} models be made real-time capable?",
         f"Apply model pruning and edge deployment for {kw}."),
        ("explain" not in merged,
         f"🧠 Explainability Missing in {kw.title()}",
         f"No XAI methods applied to {kw} models.",
         f"Stakeholders need to trust {kw} AI decisions.",
         f"How can SHAP/LIME be used for {kw} interpretability?",
         f"Integrate Grad-CAM into the {kw} pipeline."),
        ("benchmark" not in merged,
         f"📏 No Benchmark for {kw.title()}",
         f"No standard benchmark for {kw} evaluation.",
         f"Without benchmarks, {kw} progress is unmeasurable.",
         f"What benchmark should the {kw} community adopt?",
         f"Propose a public {kw} leaderboard with fixed splits."),
        ("dataset" not in merged,
         f"📂 No Public Dataset for {kw.title()}",
         f"No open large-scale {kw} dataset available.",
         f"Without open data, {kw} research cannot be replicated.",
         f"How can a public {kw} dataset be created?",
         f"Collaborate with institutions to release {kw} data."),
        ("multimodal" not in merged,
         f"🖼 Multimodal {kw.title()} Not Explored",
         f"Only single data type used for {kw}.",
         f"Multimodal {kw} can capture richer information.",
         f"How can multimodal data improve {kw} accuracy?",
         f"Use fusion architectures for {kw} multimodal learning."),
        ("scalab" not in merged,
         f"📈 {kw.title()} Scalability Untested",
         f"No scalability analysis for {kw} on large datasets.",
         f"Small-scale {kw} models may fail at production scale.",
         f"How does {kw} performance scale with data size?",
         f"Test {kw} with Spark/Ray on progressively larger data."),
    ]
    for cond, title, desc, why, q, approach in checks:
        if cond:
            gaps.append({"title":title,"desc":desc,
                         "why":f"🎯 {why}","question":f"💡 {q}","approach":f"🔧 {approach}"})
    if not gaps:
        gaps.append({"title":f"ℹ️ {kw.title()} Appears Well-Studied",
                     "desc":f"Papers on {kw} seem comprehensive.",
                     "why":f"🎯 Even mature {kw} fields have emerging sub-problems.",
                     "question":f"💡 What are 2024-2025 open challenges in {kw}?",
                     "approach":f"🔧 Do a systematic review of recent {kw} papers."})
    seen, unique = set(), []
    for g in gaps:
        if g["title"] not in seen: seen.add(g["title"]); unique.append(g)
    return unique[:6]


def detect_gaps_smart(texts, topics, keyword):
    """TF-IDF topics + Groq AI gaps = best of both worlds!"""
    text_sample = " ".join(texts)[:3000]

    if GROQ_API_KEY:
        gaps, status = call_groq_for_gaps(text_sample, keyword, topics)
        if gaps and len(gaps) >= 3:
            st.session_state["ai_status"] = f"✅ Groq AI success: {len(gaps)} domain-specific gaps"
            return gaps
        else:
            st.session_state["ai_status"] = f"⚠️ Groq failed: {status} — using fallback"
    else:
        st.session_state["ai_status"] = "⚠️ No Groq API key in .env"

    return generate_fallback_gaps(texts, topics, keyword)


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def show_dashboard_page(results):
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px 0;'>
        <div style='font-size:28px; font-weight:800; color:#D4AF37;
                    font-family:"Playfair Display",serif; text-shadow:0 0 12px #D4AF37;'>
            📊 Research Insights Dashboard
        </div>
        <div style='color:#aaa; font-size:13px; margin-top:6px;'>
            Visual summary of topics and research gap coverage
        </div>
    </div>""", unsafe_allow_html=True)

    topics = results.get("topics", [])
    gaps   = results.get("gaps", [])
    if not topics:
        st.info("No topics to display.")
        if st.button("← Back to Results"): nav_to("results")
        return

    gap_text = " ".join(g.get("title","")+" "+g.get("desc","") for g in gaps).lower()
    scores = [max(1, sum(1 for w in t.lower().split() if len(w)>3 and w in gap_text)) for t in topics]
    short_labels = [t[:25]+"…" if len(t)>25 else t for t in topics]

    fig = go.Figure(go.Bar(
        x=short_labels, y=scores,
        marker=dict(color=scores,
                    colorscale=[[0,"#7a5c10"],[0.5,"#D4AF37"],[1,"#FFE066"]],
                    showscale=False, line=dict(color="#D4AF37", width=1.5)),
        text=scores, textposition="outside",
        textfont=dict(color="#D4AF37", size=13),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", size=12),
        xaxis=dict(title="Research Topics", tickangle=-35,
                   tickfont=dict(size=10, color="#D4AF37"),
                   gridcolor="rgba(212,175,55,0.08)"),
        yaxis=dict(title="Gap Score", tickformat="d",
                   gridcolor="rgba(212,175,55,0.08)", dtick=1),
        margin=dict(t=40, b=100, l=50, r=20), bargap=0.3, height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, val, label in [
        (c1, len(topics), "Topics Found"),
        (c2, len(gaps),   "Gaps Detected"),
        (c3, results.get("n_papers",0), "Papers Analyzed")
    ]:
        with col:
            st.markdown(
                f"<div class='mobile-card' style='text-align:center;'>"
                f"<div style='font-size:32px;font-weight:900;color:#D4AF37;'>{val}</div>"
                f"<div style='color:#aaa;font-size:12px;'>{label}</div></div>",
                unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("← Back to Results", key="dash_back"): nav_to("results")


# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "page"           not in st.session_state: st.session_state.page = "welcome"
if "page_history"   not in st.session_state: st.session_state.page_history = []
if "uploaded_texts" not in st.session_state: st.session_state.uploaded_texts = []
if "results"        not in st.session_state: st.session_state.results = {}
if "keyword"        not in st.session_state: st.session_state.keyword = ""
if "ai_status"      not in st.session_state: st.session_state.ai_status = ""

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
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "welcome":
    st.markdown("""
    <div style="text-align:center; padding:24px 16px 14px 16px;">
        <div style="font-size:54px; font-weight:900; color:#D4AF37;
                    text-shadow:0 0 20px #D4AF37, 0 0 40px #B8962E;
                    font-family:'Playfair Display',serif; margin-bottom:16px;">Welcome!</div>
        <p style="color:#ddd; font-size:15px; line-height:1.8;
                  padding:0 12px; margin-bottom:26px;">
            Quickly analyze paper abstracts or upload your own<br>
            PDFs / CSVs to find publishable research gaps.
        </p>
    </div>""", unsafe_allow_html=True)
    if st.button("🚀  Start Research Analysis →", key="start_btn"): nav_to("input")

elif st.session_state.page == "input":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Input / Upload")
    keyword  = st.text_input("Research keyword (e.g. 'dental caries')",
                              value=st.session_state.get("keyword",""))
    uploaded = st.file_uploader("Upload files (PDF / TXT / CSV)", accept_multiple_files=True)
    file_url = st.text_input("Or paste file URL", placeholder="https://arxiv.org/pdf/xxx.pdf")
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
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### 🪄 Processing — please wait")

    def run_analysis():
        texts = st.session_state.uploaded_texts or [
            "This study explores AI in healthcare diagnosing images.",
            "We propose a climate forecasting model using ML.",
        ]
        topics, method = [], "TF-IDF"
        try:
            if HAS_BERTOPIC and st.session_state.get("use_bertopic", True):
                topics, status = compute_topics_bertopic(texts)
                method = f"BERTopic ({status})"
        except: topics = []
        if not topics:
            topics = compute_topics_tfidf(texts, n_topics=6)
            method = "TF-IDF"

        keyword = st.session_state.get("keyword", "")
        gaps    = detect_gaps_smart(texts, topics, keyword)

        st.session_state.results = {
            "keyword":  keyword or "N/A",
            "n_papers": len(texts),
            "method":   method + (" + Groq AI" if GROQ_API_KEY else ""),
            "topics":   topics,
            "gaps":     gaps,
        }

    with st.spinner("Analyzing with Groq AI… please wait…"):
        run_analysis()

    ai_status = st.session_state.get("ai_status","")
    if ai_status:
        if "success" in ai_status:
            st.success(ai_status)
        else:
            st.warning(ai_status)

    st.success("✅ Analysis complete!")
    if st.button("View Results →"): nav_to("results")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "results":
    res = st.session_state.get("results", {})
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    kw = res.get("keyword","N/A")
    st.markdown(f"### 🔍 Results for: <span style='color:#D4AF37'>{kw}</span>",
                unsafe_allow_html=True)
    st.markdown(f"**Papers:** {res.get('n_papers',0)}  &nbsp;•&nbsp;  **Method:** {res.get('method','-')}")

    if st.button("📊 View Full Dashboard →", key="dash_btn"): nav_to("dashboard")

    st.markdown("#### 🏷 Top Topics Detected")
    topics = res.get("topics",[])
    if topics:
        st.markdown("".join(f"<span class='badge'>{t}</span>" for t in topics), unsafe_allow_html=True)
    else:
        st.info("No topics extracted.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Identified Research Gaps")
    for i, g in enumerate(res.get("gaps",[]), 1):
        st.markdown(f"""
        <div class='gap-card'>
            <div class='gap-title'><span class='gap-number'>{i}</span>{g.get('title','')}</div>
            <div class='gap-desc'>{g.get('desc','')}</div>
            <div class='gap-why'>{g.get('why','')}</div>
            <div class='gap-question'>{g.get('question','')}</div>
            <div class='gap-approach'>{g.get('approach','')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💾 Export Results")
    txt_content = "\n".join([
        f"Keyword: {res.get('keyword','N/A')}",
        f"Papers: {res.get('n_papers',0)}",
        f"Method: {res.get('method','-')}",
        "","== TOPICS ==",
        *["- "+t for t in res.get("topics",[])],
        "","== GAPS ==",
        *[f"\n{i+1}. {g.get('title','')}\n   {g.get('desc','')}\n   {g.get('why','')}\n   {g.get('question','')}\n   {g.get('approach','')}"
          for i,g in enumerate(res.get("gaps",[]))]
    ])
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.download_button("📄 TXT", data=save_txt(txt_content,"r.txt"),
                           file_name="gaps.txt", mime="text/plain")
    with ec2:
        if HAS_REPORTLAB:
            pb = save_pdf(txt_content,"r.pdf")
            if pb: st.download_button("📕 PDF", data=pb, file_name="gaps.pdf", mime="application/pdf")
        else: st.button("📕 PDF", disabled=True)
    with ec3:
        cb = io.BytesIO()
        pd.DataFrame({"text": st.session_state.uploaded_texts}).to_csv(cb, index=False)
        cb.seek(0)
        st.download_button("📊 CSV", data=cb, file_name="papers.csv", mime="text/csv")

    if st.button("← New Analysis"):
        st.session_state.page_history = []; nav_to("input")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "dashboard":
    show_dashboard_page(st.session_state.get("results", {}))

elif st.session_state.page == "about":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### ℹ About")
    st.write("**AI Research Gap Finder** — Black & Gold theme.")
    st.write("- Uses Groq AI (Llama 3) for intelligent domain-specific gap detection.")
    st.write("- TF-IDF + KMeans for topic extraction.")
    st.write("- Supports PDF, TXT, CSV uploads and direct URLs.")
    if st.button("← Back to Home", key="about_back"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "settings":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### ⚙ Settings")
    if HAS_BERTOPIC:
        use_bert = st.checkbox("Use BERTopic", value=st.session_state.get("use_bertopic",True))
        st.session_state.use_bertopic = use_bert
        st.success("BERTopic available ✅")
    else:
        st.checkbox("Use BERTopic", value=False, disabled=True)
        st.warning("BERTopic not installed.")

    st.markdown("#### Groq AI Status")
    if GROQ_API_KEY:
        st.success("✅ Groq AI connected — unlimited free gap detection!")
        ai_status = st.session_state.get("ai_status","")
        if ai_status: st.info(f"Last run: {ai_status}")
    else:
        st.warning("⚠️ GROQ_API_KEY not found in .env file.")

    st.markdown("#### Storage")
    st.code(str(MODELS_DIR))
    if st.button("🗑 Reset App"):
        for k in ["uploaded_texts","page_history"]:
            st.session_state[k] = []
        for k in ["keyword","ai_status"]:
            st.session_state[k] = ""
        st.session_state["results"] = {}
        st.success("Reset done!"); nav_to("welcome")
    if st.button("← Back to Home", key="settings_back"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "help":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### ❓ Help & Tips")
    st.write("- Upload CSV with **'abstract'** column for best results.")
    st.write("- For scanned PDFs, run OCR first.")
    st.write("- Paste ArXiv PDF URL directly.")
    st.write("- More text = better Groq AI gap detection.")
    if st.button("← Back to Home", key="help_back"): nav_to("welcome")
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
