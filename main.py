# main.py
"""
AI Research Gap Finder - Streamlit (Luxury Black & Gold Edition)
FIXED: Footer buttons full width same as textarea, no duplicates, same on all pages.
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
from sklearn.metrics.pairwise import cosine_similarity

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
    from streamlit_lottie import st_lottie
    HAS_LOTTIE = True
except Exception:
    HAS_LOTTIE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

ROOT       = Path(os.getcwd())
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

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

/* ── ALL BUTTONS full width like textarea ── */
.stButton button {
    background: linear-gradient(135deg, #D4AF37, #B8962E);
    color: #08090A;
    font-weight: 700;
    border-radius: 16px;
    padding: 12px;
    width: 100%;
    box-shadow: 0 0 15px rgba(212,175,55,0.5), inset 0 2px 5px rgba(255,255,255,0.5);
    transition: all 0.3s ease;
    font-size: 15px;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(212,175,55,1);
}

/* ── HIDE STREAMLIT HEADER ── */
header[data-testid="stHeader"] {
    visibility: hidden; height: 0 !important;
    min-height: 0 !important; padding: 0 !important; margin: 0 !important;
}

/* ── MAIN CONTAINER — matched to input field width ── */
.main .block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2rem !important;
}

/* ── HEADER ── */
.app-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 18px 10px 18px;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(212,175,55,0.25);
    margin-bottom: 14px;
}
.header-ai {
    font-size: 46px; font-weight: 900; color: #D4AF37;
    font-family: 'Playfair Display', serif;
    text-shadow: 0 0 20px #D4AF37, 0 0 40px #B8962E;
    line-height: 1;
}
.header-title {
    font-size: 19px; font-weight: 800; color: #D4AF37;
    font-family: 'Playfair Display', serif;
    text-shadow: 0 0 10px #D4AF37; line-height: 1.2;
}
.header-subtitle { font-size: 11px; color: #aaa; margin-top: 3px; }

/* ── CARDS ── */
.mobile-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(8px);
    border: none !important;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    margin-bottom: 14px;
}

/* ── GAP CARDS ── */
.gap-card {
    background: rgba(20,15,5,0.7);
    border-left: 4px solid #D4AF37;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.gap-number {
    display: inline-block;
    background: #D4AF37; color: #000;
    border-radius: 50%; width: 26px; height: 26px;
    text-align: center; line-height: 26px;
    font-weight: 900; font-size: 13px; margin-right: 8px;
}
.gap-title    { color: #D4AF37; font-weight: 800; font-size: 16px; margin-bottom: 8px;
                font-family: 'Playfair Display', serif; }
.gap-desc     { color: #ddd; font-size: 13.5px; line-height: 1.7; margin-bottom: 10px; }
.gap-why      { background: rgba(212,175,55,0.08); border-radius: 8px; padding: 8px 12px;
                color: #c8a84b; font-size: 12.5px; margin-bottom: 8px; line-height: 1.5; }
.gap-question { background: rgba(212,175,55,0.14); border-radius: 8px; padding: 10px 14px;
                color: #D4AF37; font-size: 13px; font-style: italic; line-height: 1.5; }
.gap-approach { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 8px 12px;
                color: #aaa; font-size: 12px; margin-top: 8px; line-height: 1.5; }

/* ── BADGES ── */
.badge {
    display: inline-block;
    background: rgba(212,175,55,0.15);
    border: 1px solid rgba(212,175,55,0.4);
    color: #D4AF37; border-radius: 20px;
    padding: 4px 12px; margin: 3px; font-size: 12px;
}

/* ── FOOTER DIVIDER ── */
.footer-divider {
    border: none;
    border-top: 1px solid rgba(212,175,55,0.25);
    margin: 24px 0 16px 0;
}
</style>
""", unsafe_allow_html=True)

add_bg_from_local("assets/uu.jpeg")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def save_txt(text: str, fname: str):
    return text.encode("utf-8")

def save_pdf(text: str, fname: str):
    if not HAS_REPORTLAB:
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    lines = text.splitlines()
    y = 750
    c.setFont("Helvetica", 10)
    for line in lines:
        c.drawString(40, y, line[:120]); y -= 14
        if y < 60:
            c.showPage(); y = 750
    c.save(); buf.seek(0)
    return buf

def fetch_file_from_url(url: str):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        parsed   = urlparse(url)
        filename = os.path.basename(parsed.path) or "downloaded_file"
        return filename.lower(), r.content
    except Exception as e:
        st.error(f"Failed to fetch file: {e}")
        return None, None

def parse_uploaded_files(uploaded_files) -> List[str]:
    texts = []
    if not uploaded_files:
        return []
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
            except Exception: continue
    return [t for t in texts if t and t.strip()]

# ── TOPIC EXTRACTION ──────────────────────────────────────────────────────────
def compute_topics_tfidf(texts, n_topics=6):
    corpus = [t.replace("\n", " ") for t in texts]
    if not corpus: return []
    vec = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2))
    X   = vec.fit_transform(corpus)
    if X.shape[1] <= 1: return []
    svd    = TruncatedSVD(n_components=min(50, X.shape[1]-1))
    Xr     = svd.fit_transform(X)
    k      = min(n_topics, max(1, Xr.shape[0] // 2))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit_predict(Xr)
    terms  = vec.get_feature_names_out()
    topics = []
    for i in range(k):
        comp    = svd.components_.T.dot(kmeans.cluster_centers_[i])
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
            t = BERTopic.load(str(model_save_path))
            t.transform(texts)
            info = t.get_topic_info()
            top_topics = []
            for t_id in info.topic.values:
                if t_id == -1: continue
                words = t.get_topic(t_id)
                if not words: continue
                top_topics.append(", ".join(w for w, _ in words[:6]))
                if len(top_topics) >= 8: break
            return top_topics, "loaded model"
    except Exception: pass
    model = BERTopic(embedding_model=emb, calculate_probabilities=False, verbose=False)
    model.fit_transform(texts)
    try: model.save(str(model_save_path))
    except Exception: pass
    info = model.get_topic_info()
    top_topics = []
    for t_id in info.topic.values:
        if t_id == -1: continue
        words = model.get_topic(t_id)
        if not words: continue
        top_topics.append(", ".join(w for w, _ in words[:6]))
        if len(top_topics) >= 8: break
    return top_topics, "fitted"

# ── GAP LIBRARY ───────────────────────────────────────────────────────────────
GAP_LIBRARY = {
    "real-time": {
        "title":    "⚡ Real-Time Processing Not Addressed",
        "desc":     "The uploaded papers focus on offline or batch processing methods. There is no discussion of real-time or low-latency inference, which is critical for practical deployment in live systems.",
        "why":      "🎯 Why it matters: Real-world applications (e.g., medical monitoring, autonomous vehicles) require decisions in milliseconds — offline models cannot serve these needs.",
        "question": "💡 Research Question: How can the proposed models be optimized for real-time inference without significant loss in accuracy?",
        "approach": "🔧 Suggested Approach: Explore model compression (pruning, quantization), edge deployment, or streaming architectures.",
    },
    "explainable": {
        "title":    "🧠 Explainability & Transparency Missing",
        "desc":     "None of the papers discuss how or why the AI model makes its predictions. The models are treated as black boxes, with no interpretability techniques applied.",
        "why":      "🎯 Why it matters: In high-stakes domains like healthcare or finance, users and regulators need to understand AI decisions before trusting them.",
        "question": "💡 Research Question: How can Explainable AI (XAI) techniques like SHAP or LIME be integrated to make these models interpretable?",
        "approach": "🔧 Suggested Approach: Apply post-hoc explanation methods (SHAP, LIME, Grad-CAM) and evaluate user trust through studies.",
    },
    "benchmark": {
        "title":    "📏 No Standard Benchmark Used",
        "desc":     "The papers evaluate their models on custom or private datasets without comparing against established benchmarks, making it impossible to fairly compare results.",
        "why":      "🎯 Why it matters: Without a common benchmark, the research community cannot measure progress or reproduce results reliably.",
        "question": "💡 Research Question: What standardized benchmark dataset and evaluation protocol should be adopted for fair comparison in this domain?",
        "approach": "🔧 Suggested Approach: Propose a community benchmark with fixed train/test splits, standard metrics, and a public leaderboard.",
    },
    "dataset": {
        "title":    "📂 Lack of Public Dataset",
        "desc":     "The research relies on private or limited datasets. No publicly available, large-scale dataset exists for this specific problem area.",
        "why":      "🎯 Why it matters: Without open data, other researchers cannot validate findings or build upon this work.",
        "question": "💡 Research Question: How can a diverse, representative, and publicly available dataset be collected and curated for this domain?",
        "approach": "🔧 Suggested Approach: Collaborate with institutions, use data augmentation, or apply federated learning to build privacy-preserving datasets.",
    },
    "multimodal": {
        "title":    "🖼 Multimodal Data Not Explored",
        "desc":     "The papers use only a single type of data (e.g., text only or image only). Combining multiple data sources such as text, images, audio, or sensor data is not considered.",
        "why":      "🎯 Why it matters: Real-world problems are inherently multimodal — using only one modality leaves valuable information unused.",
        "question": "💡 Research Question: How does incorporating multimodal inputs (e.g., text + image) improve model accuracy and robustness in this domain?",
        "approach": "🔧 Suggested Approach: Use fusion architectures (early, late, or cross-modal attention) to combine different data types.",
    },
    "scalability": {
        "title":    "📈 Scalability Not Tested",
        "desc":     "The proposed methods are only tested on small datasets. There is no analysis of how performance and efficiency change as data volume or model size increases.",
        "why":      "🎯 Why it matters: A model that works well on 1,000 samples may fail or become too slow with millions of real-world data points.",
        "question": "💡 Research Question: How does the proposed system scale in terms of accuracy, speed, and memory when applied to large-scale, real-world datasets?",
        "approach": "🔧 Suggested Approach: Test with progressively larger datasets, use distributed computing frameworks (Spark, Ray), and profile bottlenecks.",
    },
}

def detect_gaps(texts: List[str], topics: List[str]) -> List[dict]:
    merged        = " ".join(texts).lower()
    merged_topics = " ".join(topics).lower()
    gaps          = []
    future_sents  = []
    for t in texts:
        for s in t.split("."):
            s2 = s.strip().lower()
            if any(k in s2 for k in ["future work","future research","future directions","limitation","in future"]):
                clean = s.strip()
                if len(clean) > 30:
                    future_sents.append(clean[:180])
    for sent in future_sents[:2]:
        gaps.append({
            "title":    "📌 Future Work Identified in Paper",
            "desc":     f'The authors themselves acknowledge: "{sent}..."',
            "why":      "🎯 Why it matters: The authors recognize this gap — it is a validated, unexplored direction.",
            "question": "💡 Research Question: Can you design and implement a study that directly addresses this acknowledged limitation?",
            "approach": "🔧 Suggested Approach: Use the paper's methodology as a baseline and extend it to cover this gap.",
        })
    for term, gap_info in GAP_LIBRARY.items():
        if term not in merged and term not in merged_topics:
            gaps.append(gap_info)
    if not gaps:
        gaps.append({
            "title":    "ℹ️ No Major Gap Detected Automatically",
            "desc":     "The uploaded papers appear comprehensive across common dimensions.",
            "why":      "🎯 Why it matters: Even well-covered topics have niche sub-problems worth exploring.",
            "question": "💡 Research Question: What are the most recent (2024–2025) open challenges in this specific research area?",
            "approach": "🔧 Suggested Approach: Do a systematic literature review of papers from the last 12 months.",
        })
    seen, unique = set(), []
    for g in gaps:
        if g["title"] not in seen:
            seen.add(g["title"]); unique.append(g)
    return unique[:6]

# ── DASHBOARD PAGE ────────────────────────────────────────────────────────────
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
        st.info("No topics to display. Please run analysis first.")
        if st.button("← Back to Results"): nav_to("results")
        return

    gap_titles = " ".join(g.get("title","") + " " + g.get("desc","") for g in gaps).lower()
    scores = []
    for topic in topics:
        words = [w for w in topic.lower().split() if len(w) > 3]
        score = sum(1 for w in words if w in gap_titles)
        scores.append(max(1, score))

    short_labels = [t[:25]+"…" if len(t) > 25 else t for t in topics]
    df_chart = pd.DataFrame({"Topic": short_labels, "Gap Coverage Score": scores})
    fig = go.Figure(go.Bar(
        x=df_chart["Topic"], y=df_chart["Gap Coverage Score"],
        marker=dict(color=scores, colorscale=[[0,"#7a5c10"],[0.5,"#D4AF37"],[1,"#FFE066"]],
                    showscale=False, line=dict(color="#D4AF37", width=1.5)),
        text=df_chart["Gap Coverage Score"], textposition="outside",
        textfont=dict(color="#D4AF37", size=13),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", size=12),
        xaxis=dict(title="Research Topics Detected", tickangle=-35,
                   tickfont=dict(size=10, color="#D4AF37"), gridcolor="rgba(212,175,55,0.08)"),
        yaxis=dict(title="Gap Coverage Score", tickformat="d",
                   gridcolor="rgba(212,175,55,0.08)", dtick=1),
        margin=dict(t=40, b=100, l=50, r=20), bargap=0.3, height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='mobile-card' style='text-align:center;'>"
                    f"<div style='font-size:32px;font-weight:900;color:#D4AF37;'>{len(topics)}</div>"
                    f"<div style='color:#aaa;font-size:12px;'>Topics Found</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='mobile-card' style='text-align:center;'>"
                    f"<div style='font-size:32px;font-weight:900;color:#D4AF37;'>{len(gaps)}</div>"
                    f"<div style='color:#aaa;font-size:12px;'>Gaps Detected</div></div>", unsafe_allow_html=True)
    with c3:
        n = results.get("n_papers", 0)
        st.markdown(f"<div class='mobile-card' style='text-align:center;'>"
                    f"<div style='font-size:32px;font-weight:900;color:#D4AF37;'>{n}</div>"
                    f"<div style='color:#aaa;font-size:12px;'>Papers Analyzed</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("← Back to Results", key="dash_back"):
        nav_to("results")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "page"           not in st.session_state: st.session_state.page = "welcome"
if "page_history"   not in st.session_state: st.session_state.page_history = []
if "uploaded_texts" not in st.session_state: st.session_state.uploaded_texts = []
if "results"        not in st.session_state: st.session_state.results = {}
if "keyword"        not in st.session_state: st.session_state.keyword = ""

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
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ── WELCOME ───────────────────────────────────────────────────────────────────
if st.session_state.page == "welcome":
    st.markdown("""
    <div style="text-align:center; padding:24px 16px 14px 16px;">
        <div style="font-size:54px; font-weight:900; color:#D4AF37;
                    text-shadow:0 0 20px #D4AF37, 0 0 40px #B8962E;
                    font-family:'Playfair Display',serif; margin-bottom:16px;">
            Welcome!
        </div>
        <p style="color:#ddd; font-size:15px; line-height:1.8;
                  padding:0 12px; margin-bottom:26px;">
            Quickly analyze paper abstracts or upload your own<br>
            PDFs / CSVs to find publishable research gaps.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀  Start Research Analysis →", key="start_btn"):
        nav_to("input")

# ── INPUT ─────────────────────────────────────────────────────────────────────
elif st.session_state.page == "input":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Input / Upload")
    keyword  = st.text_input("Research keyword (e.g. 'explainable AI')",
                              value=st.session_state.get("keyword",""))
    uploaded = st.file_uploader("Upload files (PDF / TXT / CSV)", accept_multiple_files=True)
    file_url = st.text_input("Or paste file URL", placeholder="https://example.com/paper.pdf")
    pasted   = st.text_area("Or paste abstracts (one per line)", height=150)

    # Naya Layout: Back chota ho jayega aur Analyze Now bada ho jayega
    col_back, col_analyze = st.columns([1, 4], gap="small")

    with col_back:
        if st.button("← Back", key="back_btn", use_container_width=True): 
            nav_to("welcome")

    with col_analyze:
        if st.button("🔍 Analyze Now", key="analyze_btn", use_container_width=True):
            
            texts = []
            if uploaded:
                texts.extend(parse_uploaded_files(uploaded))
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
                full = " ".join(texts).lower()
                if keyword.lower().strip() and keyword.lower().strip() not in full:
                    st.warning("⚠️ Keyword not found — results may be broad.")
                nav_to("processing")
    st.markdown('</div>', unsafe_allow_html=True)

# ── PROCESSING ────────────────────────────────────────────────────────────────
elif st.session_state.page == "processing":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### 🪄 Processing — please wait")

    def run_analysis():
        texts = st.session_state.uploaded_texts or [
            "This study explores AI in healthcare diagnosing images.",
            "We propose a climate forecasting model using ML.",
            "Deep learning models for image recognition are evaluated.",
        ]
        topics, method = [], "TF-IDF"
        try:
            if HAS_BERTOPIC and st.session_state.get("use_bertopic", True):
                topics, status = compute_topics_bertopic(texts)
                method = f"BERTopic ({status})"
        except Exception:
            topics = []
        if not topics:
            topics = compute_topics_tfidf(texts, n_topics=6)
            method = "TF-IDF"
        gaps = detect_gaps(texts, topics)
        st.session_state.results = {
            "keyword":  st.session_state.get("keyword","N/A"),
            "n_papers": len(texts),
            "method":   method,
            "topics":   topics,
            "gaps":     gaps,
        }

    with st.spinner("Analyzing… may take a moment on first run…"):
        run_analysis()
    st.success("✅ Analysis complete!")
    if st.button("View Results →"): nav_to("results")
    st.markdown('</div>', unsafe_allow_html=True)

# ── RESULTS ───────────────────────────────────────────────────────────────────
elif st.session_state.page == "results":
    res = st.session_state.get("results", {})
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    kw = res.get("keyword","N/A")
    st.markdown(f"### 🔍 Results for: <span style='color:#D4AF37'>{kw}</span>", unsafe_allow_html=True)
    st.markdown(f"**Papers analyzed:** {res.get('n_papers',0)}  &nbsp;•&nbsp;  **Method:** {res.get('method','-')}")

    if st.button("📊 View Full Dashboard →", key="dash_btn"): nav_to("dashboard")

    st.markdown("#### 🏷 Top Topics Detected")
    topics = res.get("topics",[])
    if topics:
        st.markdown("".join(f"<span class='badge'>{t}</span>" for t in topics), unsafe_allow_html=True)
    else:
        st.info("No topics extracted.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Identified Research Gaps")
    gaps = res.get("gaps",[])
    if gaps:
        for i, g in enumerate(gaps, 1):
            st.markdown(f"""
            <div class='gap-card'>
                <div class='gap-title'><span class='gap-number'>{i}</span>{g.get('title','')}</div>
                <div class='gap-desc'>{g.get('desc','')}</div>
                <div class='gap-why'>{g.get('why','')}</div>
                <div class='gap-question'>{g.get('question','')}</div>
                <div class='gap-approach'>{g.get('approach','')}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No gaps detected.")

    st.markdown("---")
    st.markdown("#### 💾 Export Results")
    txt_content = "\n".join([
        f"Keyword: {res.get('keyword','N/A')}",
        f"Papers analyzed: {res.get('n_papers',0)}",
        f"Method: {res.get('method','-')}",
        "", "== TOPICS ==",
        *["- "+t for t in res.get("topics",[])],
        "", "== RESEARCH GAPS ==",
        *[f"\n{i+1}. {g.get('title','')}\n"
          f"   Description: {g.get('desc','')}\n"
          f"   Why it matters: {g.get('why','')}\n"
          f"   Research Question: {g.get('question','')}\n"
          f"   Approach: {g.get('approach','')}"
          for i,g in enumerate(res.get("gaps",[]))]
    ])
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.download_button("📄 TXT", data=save_txt(txt_content,"r.txt"),
                           file_name="research_gap_results.txt", mime="text/plain")
    with ec2:
        if HAS_REPORTLAB:
            pdf_buf = save_pdf(txt_content,"r.pdf")
            if pdf_buf:
                st.download_button("📕 PDF", data=pdf_buf,
                                   file_name="research_gap_results.pdf", mime="application/pdf")
        else:
            st.button("📕 PDF", disabled=True)
    with ec3:
        csv_buf = io.BytesIO()
        pd.DataFrame({"text": st.session_state.uploaded_texts}).to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button("📊 CSV", data=csv_buf,
                           file_name="uploaded_papers.csv", mime="text/csv")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("← New Analysis"):
        st.session_state.page_history = []
        nav_to("input")
    st.markdown('</div>', unsafe_allow_html=True)

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
elif st.session_state.page == "dashboard":
    show_dashboard_page(st.session_state.get("results", {}))

# ── ABOUT ─────────────────────────────────────────────────────────────────────
elif st.session_state.page == "about":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### ℹ About")
    st.write("**AI Research Gap Finder** — Black & Gold theme.")
    st.write("- Detects research gaps from uploaded papers using NLP.")
    st.write("- Supports PDF, TXT, CSV uploads and direct URLs.")
    st.write("- Built with Python, Streamlit, scikit-learn & Plotly.")
    if st.button("← Back to Home", key="about_back"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
elif st.session_state.page == "settings":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### ⚙ Settings")
    st.markdown("#### Analysis Options")
    if HAS_BERTOPIC:
        use_bert = st.checkbox("Use Advanced Topic Modeling (BERTopic)",
                               value=st.session_state.get("use_bertopic",True))
        st.session_state.use_bertopic = use_bert
        st.success("BERTopic available ✅")
    else:
        st.checkbox("Use Advanced Topic Modeling (BERTopic)", value=False, disabled=True)
        st.warning("BERTopic not installed — using TF-IDF fallback.")
    st.markdown("#### Storage")
    st.code(str(MODELS_DIR))
    st.markdown("#### App Controls")
    if st.button("🗑 Reset App"):
        st.session_state.uploaded_texts = []
        st.session_state.keyword = ""
        st.session_state.results = {}
        st.session_state.page_history = []
        st.success("App reset!")
        nav_to("welcome")
    st.write("**App:** AI Research Gap Finder | **Theme:** Black & Gold")
    if st.button("← Back to Home", key="settings_back"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)

# ── HELP ──────────────────────────────────────────────────────────────────────
elif st.session_state.page == "help":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### ❓ Help & Tips")
    st.write("- Upload a CSV with an **'abstract'** column for best results.")
    st.write("- For scanned PDFs, run OCR first then upload the text.")
    st.write("- Paste a direct file URL (PDF/TXT/CSV) in the Input page.")
    st.write("- More papers uploaded = better and more accurate gap detection.")
    st.write("- Use the Dashboard page to see a visual summary of your results.")
    if st.button("← Back to Home", key="help_back"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── FOOTER — Har page par same, textarea jitni puri width ────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='footer-divider'>", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⚙️ SETTINGS", key="footer_settings", use_container_width=True): 
        nav_to("settings")

with col2:
    if st.button("❓ HELP", key="footer_help", use_container_width=True): 
        nav_to("help")

with col3:
    if st.button("ℹ️ ABOUT", key="footer_about", use_container_width=True): 
        nav_to("about")
