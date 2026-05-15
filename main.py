# main.py
"""
AI Research Gap Finder - Streamlit (Luxury Black & Gold / Glassmorphism Edition)
Visual Goal: Match the stunning, polished aesthetic of high-end black and gold website designs.
"""

import os, io, time, json, threading
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
from sklearn.metrics.pairwise import cosine_similarity
# Optional heavy libraries
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

# Lottie
try:
    from streamlit_lottie import st_lottie
    HAS_LOTTIE = True
except Exception:
    HAS_LOTTIE = False

# For PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# Basic NLP / fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# App dirs
ROOT = Path(os.getcwd())
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Theme colors (user requested)
ROSE_GOLD_1 = "#B1705C"
ROSE_GOLD_2 = "#DD856A"
SHINY_BLACK_1 = "#08090A"
SHINY_BLACK_2 = "#0f1113"
ANTUM_GOLD_1 = "#B6931E"   # Primary Shiny Gold
ANTUM_GOLD_2 = "#937604"   # Secondary Gold
NEW_GOLD_DARK = "#CE9A18"  # Dark Goldenrod

st.set_page_config(page_title="AI Research Gap Finder", layout="centered", initial_sidebar_state="collapsed")
# --- BACKGROUND FUNCTION (Alternative Method) ---
import base64
def add_bg_from_local(image_file):
    # This reads the image file and encodes it to base64
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    
    # Now, we use st.markdown to inject the CSS style using that base64 data
    st.markdown(
        f"""
        <style>
        .welcome-gradient {{
        color: {ANTUM_GOLD_1};  /* Solid gold color */
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;

        text-shadow: 0 0 5px {ANTUM_GOLD_1}, 0 0 10px {ANTUM_GOLD_2}; /* shine effect */
    }}

    .mobile-card {{
        background: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(8px); 
        -webkit-backdrop-filter: blur(8px);
        border: none !important; 
        border-radius: 18px;
        padding: 24px; 
        box-shadow: 0 8px 30px rgba(0,0,0,0.4), 0 0 10px rgba(102, 85, 43, 0.1); 
    }}
        .stApp {{
            background-image: url("data:image/png;base64,{data}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;  
        }}
                  
        </style>
        """,
        unsafe_allow_html=True
    )

# --- CALL THE FUNCTION AT THE VERY TOP OF THE APP ---
# --- CUSTOM LUXURY CSS STYLING ---
st.markdown("""
<style>

/* ===== PREMIUM FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Poppins:wght@300;400;500;600&display=swap');


/* ===== MAIN APP CONTAINER ===== */


/* ===== HEADINGS (GOLD LUXURY) ===== */
h1, h2, h3, .large-gold-heading {
    font-family: 'Playfair Display', serif;
    color: #EAD7D1;
    text-shadow:
        0 0 8px rgba(212,175,55,0.7),
        0 0 20px rgba(212,175,55,0.25);
}

/* ===== SUBTEXT ===== */
.subtitle-gold {
    color: #D4AF37;
    font-size: 13px;
    opacity: 0.9;
}

/* ===== INPUTS ===== */
input, textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(212,175,55,0.4) !important;
    border-radius: 14px !important;
    color: white !important;
}

/* ===== BUTTONS (PREMIUM GOLD) ===== */
.stButton button {
    background: linear-gradient(135deg, #D4AF37, #B8962E);
    color: #08090A;
    font-weight: 600;
    border-radius: 16px;
    padding: 12px;
    width: 100%;
    box-shadow:
        0 0 15px rgba(212,175,55,0.6),
        inset 0 2px 5px rgba(255,255,255,0.6);
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 0 25px rgba(212,175,55,1);
}

/* ===== GAP RESULT CARD ===== */
.gap-card {
    background: rgba(212,175,55,0.08);
    border-left: 4px solid #D4AF37;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
}
header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0px !important;
    min-height: 0px !important;
    padding: 0px !important;
    margin: 0px !important;
}
.css-18e3th9 {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
   margin-top: 0rem !important;          
}
.welcome-gradient {
    color: #D4AF37;
    font-size: 32px;
    font-weight: 800;

    margin-top: -30px;   
    margin-bottom: 10px;

   
}
.welcome-gradient {
    color: #D4AF37;
    font-size: 39px;
    font-weight: 500;
             text-shadow:
        0 0 2px #D4AF37,
        0 0 2px #937604;
margin-left:20px;
    transform: translateY(-60px);   /* 🔥 REAL MOVE */
    margin-bottom: 10px;

    text-shadow:
        0.1 0 0px #D4AF37,
        0 0.1 0px #937604;
} 
  .welcome-gradient {
    font-size: 45px !important;
}          
.main .block-container {
    padding-top: 0rem !important;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 0rem !important;
    overflow-y:auto !important;        
} 
.mobile-wrap.mobile-card {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
}            
/* ===== MOBILE FEEL ===== */
@media (max-width: 600px) {
    .mobile-wrap {
        max-width: 100%;
    }
}
/* Naya CSS: Simple Fixed Footer Container */
.bottom-nav-fixed {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #000000;
    padding: 15px 0; 
    z-index: 1000;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.5); 
    display: flex;
    justify-content: space-around;
    align-items: center; 
}

/* Naya CSS: Golden Bold Clickable Text Style (Anchor tag use karenge) */
.nav-link-text-anchor {
    color: #FFC300 !important;
    font-weight: bold !important;
    font-size: 16px !important;
    text-decoration: none !important; /* Underline hatao */
    cursor: pointer;
    line-height: 1.2;
}

.nav-link-text-anchor:hover {
    color: #D4AF37 !important;
}
</style>
""", unsafe_allow_html=True)

add_bg_from_local("assets/uu.jpeg")
# ----------- helper utilities (No changes) -----------
def check_keyword_relevance(keyword: str, texts: list[str]) -> float:
    if not keyword or not texts:
        return 0.0

    corpus = [keyword] + texts
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)

    similarity = cosine_similarity(tfidf[0:1], tfidf[1:])
    return similarity.mean()
def load_lottie_file(path: str):
    if not HAS_LOTTIE:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_txt(text: str, fname: str):
    b = text.encode("utf-8")
    return b

def save_pdf(text: str, fname: str):
    if not HAS_REPORTLAB:
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    lines = text.splitlines()
    y = 750
    c.setFont("Helvetica", 10)
    for line in lines:
        c.drawString(40, y, line[:1000])
        y -= 14
        if y < 60:
            c.showPage(); y = 750
    c.save()
    buf.seek(0)
    return buf
def fetch_file_from_url(url: str):
    """
    Download file from internet URL and return (filename, bytes)
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)

        if not filename:
            filename = "downloaded_file"

        return filename.lower(), response.content

    except Exception as e:
        st.error(f"Failed to fetch file from URL: {e}")
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
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    texts.append("\n".join(pages))
            elif name.endswith(".csv"):
                df = pd.read_csv(file)
                # heuristics for abstract/title
                for col in df.columns:
                    if "abstract" in col.lower() or "title" in col.lower():
                        texts.extend(df[col].astype(str).tolist())
                        break
                else:
                    texts.extend(df.astype(str).iloc[:, 0].astype(str).tolist())
            else:
                # read as text
                raw = file.getvalue().decode("utf-8", errors="ignore")
                texts.append(raw)
        except Exception as e:
            # fallback read
            try:
                raw = file.getvalue().decode("utf-8", errors="ignore")
                texts.append(raw)
            except Exception:
                continue
    return [t for t in texts if t and t.strip()]
def show_visual_dashboard(results):
    if not results:
        return
    st.markdown("### 📊 Research Insights Dashboard")
    topics = results.get("topics", [])
    gaps = results.get("gaps", [])
    data = {
        'Topic': topics,
        'Gaps Found': [len(gaps)] * len(topics)
    }
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Topic', y='Gaps Found', color_discrete_sequence=['#D4AF37'])
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
# Topic extraction functions (No changes)
def compute_topics_tfidf(texts, n_topics=6):
    corpus = [t.replace("\n", " ") for t in texts]
    if len(corpus) == 0:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    if X.shape[1] <= 1:
        return []
    svd = TruncatedSVD(n_components=min(50, X.shape[1]-1))
    Xr = svd.fit_transform(X)
    k = min(n_topics, max(1, Xr.shape[0] // 2))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xr)
    centers = kmeans.cluster_centers_
    terms = vec.get_feature_names_out()
    topics = []
    for i in range(k):
        center = centers[i]
        comp = svd.components_.T.dot(center) if hasattr(svd, "components_") else center
        top_idx = np.argsort(comp)[-6:][::-1]
        topic_terms = [terms[j] for j in top_idx if j < len(terms)]
        topics.append(" ".join(topic_terms))
    # dedupe
    seen = []
    for t in topics:
        if t not in seen:
            seen.append(t)
    return seen

def compute_topics_bertopic(texts, embedding_model_name="all-MiniLM-L6-v2", model_save_path=MODELS_DIR/"bertopic_model"):
    if not HAS_BERTOPIC:
        return [], "BERTopic not installed"
    # Load or create embedding model
    emb = SentenceTransformer(embedding_model_name)
    # load existing BERTopic model if available
    try:
        if model_save_path.exists():
            t = BERTopic.load(str(model_save_path))
            topics, probs = t.transform(texts)
            info = t.get_topic_info()
            top_topics = []
            for t_id in info.topic.values:
                if t_id == -1: continue
                words = t.get_topic(t_id)
                if not words: continue
                top_topics.append(", ".join([w for w,_ in words[:6]]))
                if len(top_topics) >= 8: break
            return top_topics, "loaded model"
    except Exception:
        pass
    # fit new model
    model = BERTopic(embedding_model=emb, calculate_probabilities=False, verbose=False)
    topics, probs = model.fit_transform(texts)
    try:
        model.save(str(model_save_path))
    except Exception:
        pass
    info = model.get_topic_info()
    top_topics = []
    for t_id in info.topic.values:
        if t_id == -1: continue
        words = model.get_topic(t_id)
        if not words: continue
        top_topics.append(", ".join([w for w,_ in words[:6]]))
        if len(top_topics) >= 8: break
    return top_topics, "fitted"

# Gap detection heuristics (No changes)
def detect_gaps(texts: List[str], topics: List[str]) -> List[str]:
    merged = " ".join(texts).lower()
    gaps = []
    # future work detection simple: look for typical sentences
    future_sents = []
    for t in texts:
        for s in t.split("."):
            s2 = s.strip().lower()
            if any(k in s2 for k in ["future work", "future research", "future directions", "limit", "limitation"]):
                future_sents.append(s.strip())
    for s in future_sents[:5]:
        gaps.append(f"Found future-work hint: \"{s[:160]}\"")
    # topic-based heuristics
    merged_topics = " ".join(topics).lower()
    for term in ["real-time", "explainable", "benchmark", "dataset", "multimodal", "scalability"]:
        if term not in merged and term not in merged_topics:
            gaps.append(f"Little-to-no work detected on '{term}'")
    if not gaps:
        gaps.append("No explicit gap detected — add more documents or refine keyword.")
    # rank and unique
    unique = []
    for g in gaps:
        if g not in unique: unique.append(g)
    return unique[:6]

# ---------- App navigation / pages ----------
if "page" not in st.session_state:
    st.session_state.page = "welcome"
#... (doosre session_state initializations) ...
if "page_history" not in st.session_state: # <-- NAYA: Page History ko shuru karein
    st.session_state.page_history = []     # <-- NAYA
#... (rest of the code) ...
if "uploaded_texts" not in st.session_state:
    st.session_state.uploaded_texts = []
if "results" not in st.session_state:
    st.session_state.results = {}

# Aapki existing nav_to function ko isse replace karein:
def nav_to(p):
    # Yahaan hum pichla page 'page_history' mein store karte hain,
    # bas agar naya page pichle page se alag ho.
    current_page = st.session_state.page
    if p != current_page: 
        # 'processing' page ko history mein store na karein, kyunki ye bas ek chota step hai
        if current_page != "processing": 
            st.session_state.page_history.append(current_page)
    st.session_state.page = p
    st.rerun() 

# Naya nav_back function daalein:
def nav_back(): 
    if st.session_state.page_history:
        prev_page = st.session_state.page_history.pop() # Pichla page nikaal kar wahan jaayein
        st.session_state.page = prev_page
        st.rerun()
    else:
        # Agar koi history nahi hai, toh 'welcome' par le jaao (fallback)
        st.session_state.page = "welcome"
        st.rerun() 

# header (consistent)
with st.container():
    st.markdown('<div class="mobile-wrap mobile-card sticky-header" style="border:none !important;" >', unsafe_allow_html=True)
    cols = st.columns([1,6,1])
    with cols[1]:
        # Using custom large-gold-heading for the main title for maximum impact
        st.markdown('<div class="large-gold-heading" style="font-size:30px; height:140px; margin-bottom: 5px;">🤯AI Research Gap Finder</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-gold" style="height:130px;margin-top: -90px;" > Find publishable research gaps — fast.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# MAIN NAV
if st.session_state.page == "welcome":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    
            
    # --- Luxury Welcome Text ---
    st.markdown('<div class="welcome-gradient">Welcome! </div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Quickly analyze a set of paper abstracts or upload your own PDFs/CSVs to find research gaps.</p>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("Start Research Analysis →", key="start_btn"):
        nav_to("input")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "input":
    # ... (input fields remains the same) ...
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown('<div style="margin-top: -150px !important; padding: 0 !important;"></div>', unsafe_allow_html=True)
    st.markdown("### Input / Upload")
    st.markdown(
        f'<div style="display:flex; align-items:center;">'
        f'</div>',
        unsafe_allow_html=True
    )
    keyword = st.text_input("Enter research keyword (e.g., 'explainable AI') ",value=st.session_state.get("keyword", ""))
    uploaded = st.file_uploader("Upload files (PDF / TXT / CSV) — you can select multiple", accept_multiple_files=True, help="PDFs best with pdfplumber installed")
    file_url = st.text_input(
    "Or paste file URL (PDF / CSV / TXT)",
    placeholder="https://example.com/research-paper.pdf"
)
    pasted = st.text_area("Or paste abstracts (one per line)", height=150)
    st.markdown('<div style="display:flex; gap:8px;">', unsafe_allow_html=True)
      # Use st.columns ko 3 mein badle
    col1, col2= st.columns(2) 
    
    with col1:
        # <-- YAHAN NAYA BACK BUTTON AAYEGA
        if st.button("← Back", key="back_btn"):
            nav_to("welcome")
             
    with col2:
        if st.button("Analyze Now", key="analyze_now_final"):
            texts = []

            # 1. Files
            if uploaded:
                texts.extend(parse_uploaded_files(uploaded))

            # 2. URL
            if file_url and file_url.strip():
                fname, file_bytes = fetch_file_from_url(file_url.strip())
                if fname and file_bytes:
                    try:
                        if fname.endswith(".pdf") and HAS_PDFPLUMBER:
                            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                                texts.append("\n".join([p.extract_text() or "" for p in pdf.pages]))
                        else:
                            texts.append(file_bytes.decode("utf-8", errors="ignore"))
                    except Exception as e:
                        st.error(f"URL Error: {e}")

            # 3. Pasted
            if pasted and pasted.strip():
                texts.extend([p.strip() for p in pasted.splitlines() if p.strip()])

            if not texts:
                st.warning("Please upload a file, paste text, or provide a valid URL.")
            else:
                st.session_state.uploaded_texts = texts
                st.session_state.keyword = keyword
                
                # Keyword warning (Stop nahi karega)
                full_text = " ".join(texts).lower()
                if keyword.lower().strip() and keyword.lower().strip() not in full_text:
                    st.warning("⚠️ Keyword not found, results might be broad.")
                
                nav_to("processing") 
                
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "processing":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### 🪄Processing — please wait", unsafe_allow_html=True)
    
    def run_analysis():
        texts = st.session_state.uploaded_texts or []
        if len(texts) == 0:
            # demo tiny texts
            texts = [
                "This study explores AI in healthcare diagnosing images.",
                "We propose a climate forecasting model using ML.",
                "Deep learning models for image recognition are evaluated."
            ]
        topics = []
        method = "fallback (TF-IDF)"
        try:
            if HAS_BERTOPIC and st.session_state.get("use_bertopic", True):
                topics, status = compute_topics_bertopic(texts)
                method = f"BERTopic ({status})"
        except Exception as e:
            topics = []
        if not topics:
            topics = compute_topics_tfidf(texts, n_topics=6)
        gaps = detect_gaps(texts, topics)
        results = {
            "keyword": st.session_state.get("keyword", "N/A"),
            "n_papers": len(texts),
            "method": method,
            "topics": topics,
            "gaps": gaps,
            "suggestions":[f"How can we address: {g}?" for g in gaps]
        }
        st.session_state.results = results
        
    with st.spinner("Analyzing... this may take some time on first run (model download)..."):
        run_analysis()
        
    st.success("Analysis finished")
    if st.button("Show Results"):
        nav_to("results")
    st.markdown("</div>", unsafe_allow_html=True)
elif st.session_state.page == "results":
    # Is line ke shuru mein 4 spaces hain
    st.markdown('<div class="mobile-wrap">', unsafe_allow_html=True)
    
    # --- YE HAI WO CODE ---
    # Iske shuru mein bhi 4 spaces dein:
    show_visual_dashboard(st.session_state.get("results", {}))
    
    # Iske baad aapka purana title aur baaki code aayega
    st.title("🔍 Analysis Results")
    
    if st.button("← New Analysis / Back to Input"):
        st.session_state.page_history = [] # History ko saaf (clear) kar dein
        nav_to("input")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    res = st.session_state.get("results", {})
    st.markdown(f"### Results for: <span style='color: var(--antum-light);'>{res.get('keyword','N/A')}</span>", unsafe_allow_html=True)
    st.markdown(f"Papers analyzed: {res.get('n_papers',0)}  •  Method: {res.get('method','-')}")
    st.markdown("#### Top Topics")
    topics = res.get("topics", [])
    if topics:
        for t in topics:
            st.markdown(f"<span class='badge'>{t}</span>", unsafe_allow_html=True)
    else:
        st.info("No topics extracted.")
        
    st.markdown("#### Identified Research Gaps")
    gaps = res.get("gaps", [])
    suggestions = res.get("suggestions", [])
    
    # Use Expander for Questions (more interactive)
    for i,g in enumerate(gaps, 1):
        st.markdown(f"<div class='gap-card'><b>{i}.</b> {g}</div>", unsafe_allow_html=True)
        if suggestions and i-1 < len(suggestions):
            with st.expander(f"✨ Suggested Research Question for Gap {i}"):
                st.markdown(f"Q: <span style='color: var(--antum-light);'>{suggestions[i-1]}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    # Export Options Section
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    txt = "\n".join([
        f"Keyword: {res.get('keyword','N/A')}",
        f"Papers analyzed: {res.get('n_papers',0)}",
        "Topics:",
        *["- "+t for t in res.get("topics",[])],
        "",
        "Gaps:",
        *[f"{i+1}. {g}" for i,g in enumerate(res.get("gaps",[]))]
    ])
    
    with col_dl1:
        st.download_button("TXT", data=save_txt(txt, "results.txt"), file_name="research_gap_results.txt", mime="text/plain")
    
    if HAS_REPORTLAB:
        pdf_buf = save_pdf(txt, "results.pdf")
        if pdf_buf:
         with col_dl2:
                st.download_button("PDF", data=pdf_buf, file_name="research_gap_results.pdf", mime="application/pdf")
    
    csv_buf = io.BytesIO()
    df = pd.DataFrame({"text": st.session_state.uploaded_texts})
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    with col_dl3:
        st.download_button("CSV", data=csv_buf, file_name="uploaded_papers.csv", mime="text/csv")
        


# 'about' page mein badlav:
elif st.session_state.page=="about":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### About")
    st.write("AI Research Gap Finder —  Black-gold theme.")
    st.write("- Designed for quick detection of research gaps.")
    st.write("- Supports file uploads and URLs.")
    if st.button("Back to Home"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)
    
# 'settings' page mein badlav:
# --- SETTINGS PAGE ---
elif st.session_state.page == "settings":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### ⚙ Settings")

    # --- Analysis Settings ---
    st.markdown("#### Analysis Options")

    # Use BERTopic toggle
    if HAS_BERTOPIC:
        use_bertopic = st.checkbox(
            "Use Advanced Topic Modeling (BERTopic)",
            value=st.session_state.get("use_bertopic", True),
            help="Gives better topics but may be slower on first run"
        )
        st.session_state.use_bertopic = use_bertopic
        st.success("BERTopic is available on this system ✅")
    else:
        st.checkbox(
            "Use Advanced Topic Modeling (BERTopic)",
            value=False,
            disabled=True
        )
        st.warning("BERTopic is not installed. App will use basic TF-IDF.")

    # --- Storage / Model Info ---
    st.markdown("#### Storage Information")
    st.write("**Model storage folder:**")
    st.code(str(MODELS_DIR))

    # --- App Controls ---
    st.markdown("#### App Controls")

    if st.button("🗑 Reset App (Clear Inputs & Results)"):
        st.session_state.uploaded_texts = []
        st.session_state.keyword = ""
        st.session_state.results = {}
        st.session_state.page_history = []
        st.success("App reset successfully.")
        nav_to("welcome")

    # --- App Info ---
    st.markdown("#### App Information")
    st.write("**Application:** AI Research Gap Finder")
    st.write("**Mode:** Beginner Friendly")
    st.write("**Theme:** Black & Gold")

    # --- Back Button ---
    if st.button("← Back to Home"):
        nav_to("welcome")

    st.markdown('</div>', unsafe_allow_html=True)

    
# 'help' page mein badlav:
elif st.session_state.page=="help":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### Help & Tips")
    st.write("- Upload CSV with an 'abstract' column for best results.")
    st.write("- For scanned PDFs, use OCR externally then upload text.")
    st.write("- Or paste direct file URLs (PDF/TXT/CSV) in Input page.")
    if st.button("Back to Home"): nav_to("welcome")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "export":
    st.markdown('<div class="mobile-wrap mobile-card">', unsafe_allow_html=True)
    st.markdown("### Export (Use Results screen for quick export)")
    if st.button("Back"):
        nav_to("welcome")
    st.markdown("</div>", unsafe_allow_html=True)


# footer nav (mobile-like)
st.markdown(
    """
    <div class="mobile-wrap" style="text-align:center; padding:10px 0;">
    </div>
    """,
    unsafe_allow_html=True,
)
# --- Yeh code saare 'elif st.session_state.page == "..."' blocks ke baad aayega ---

# BOTTOM NAVIGATION BAR


# --- Yeh code saare 'elif st.session_state.page == "..."' blocks ke baad aayega ---

# BOTTOM NAVIGATION BAR
# Purana Error-wala Code:
# with col_hidden[0]:
#     st.button("hidden_settings_btn", key="hidden_settings", on_click=nav_to, args=("settings",), aria_label="hidden_nav_btn") 
# ...
# --- Bottom Navigation ---
st.markdown('<div class="bottom-nav-fixed">', unsafe_allow_html=True)
nav_cols=st.columns(3)
with nav_cols[0]:
    if st.button("⚙ SETTINGS"): nav_to("settings")
with nav_cols[1]:
    if st.button("❓ HELP"): nav_to("help")
with nav_cols[2]:
    if st.button("ℹ ABOUT"): nav_to("about")
st.markdown('</div>', unsafe_allow_html=True)