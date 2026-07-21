import streamlit as st
import requests
import google.generativeai as genai
import os
import time
import pandas as pd
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Literature Gap Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE
if "papers" not in st.session_state: st.session_state.papers = None
if "gemini_result" not in st.session_state: st.session_state.gemini_result = None
if "fetch_time" not in st.session_state: st.session_state.fetch_time = 0.0
if "gemini_time" not in st.session_state: st.session_state.gemini_time = 0.0
if "approx_cost" not in st.session_state: st.session_state.approx_cost = 0.0
if "last_keyword" not in st.session_state: st.session_state.last_keyword = ""
if "last_domain" not in st.session_state: st.session_state.last_domain = ""
if "api_sources" not in st.session_state: st.session_state.api_sources = {}

# ────────────────────────────────────────────────────────────────────────────
# IMPROVED CSS — Minimal, clean, professional
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background-color: #ffffff !important;
        color: #1a1a1a;
    }
    .stApp { background-color: #ffffff !important; }
    #MainMenu, footer { visibility: hidden; }

    /* ── Sidebar (collapsed by default) ── */
    [data-testid="stSidebar"] { background-color: #f5f5f5 !important; }

    /* ── Main container padding ── */
    .main { padding: 0 2rem; }

    /* ── Typography ── */
    h1, h2, h3 { color: #1a1a1a; font-weight: 600; letter-spacing: -0.01em; }
    p { color: #4a4a4a; line-height: 1.6; }

    /* ── Header section ── */
    .header-section {
        text-align: center;
        padding: 2.5rem 0 2rem;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .header-section h1 {
        font-size: 2.2rem;
        margin: 0.5rem 0 0.3rem;
        color: #1a1a1a;
    }
    .header-section p {
        font-size: 0.95rem;
        color: #6a6a6a;
        max-width: 600px;
        margin: 0.5rem auto;
    }
    .provider-badges {
        display: flex;
        gap: 0.8rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .badge {
        display: inline-block;
        background: #f0f0f0;
        border: 1px solid #d0d0d0;
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        font-size: 0.75rem;
        font-weight: 500;
        color: #666;
    }

    /* ── Input container ── */
    .input-container {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
    }
    .input-container h3 {
        font-size: 1.1rem;
        margin: 0 0 1.5rem;
        color: #1a1a1a;
    }

    /* ── Form inputs ── */
    .stTextInput input, .stSelectbox select {
        border: 1px solid #000000 !important;
        border-radius: 8px !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 0.95rem !important;
        background: white !important;
    }
    .stTextInput input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        background: #f5f5f5 !important;
        color: #1a1a1a !important;
        border: 1px solid #d0d0d0 !important;
        border-radius: 8px !important;
    }
    .stDownloadButton > button:hover {
        background: #efefef !important;
    }

    /* ── Results section ── */
    .results-container {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
    }
    .results-container h2 {
        font-size: 1.3rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 0.75rem;
    }

    /* ── Metrics cards ── */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb;
    }

    /* ── Status messages ── */
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2563eb;
    }
    .status-box.success {
        background: #f0fdf4;
        border-left-color: #16a34a;
        color: #15803d;
    }
    .status-box.info {
        background: #f0f9ff;
        border-left-color: #2563eb;
        color: #1e40af;
    }
    .status-box.warning {
        background: #fffbeb;
        border-left-color: #f59e0b;
        color: #92400e;
    }

    /* ── Paper table ── */
    .stMarkdown table {
        font-size: 0.9rem !important;
    }
    .stMarkdown th {
        background: #f0f0f0 !important;
        font-weight: 600 !important;
    }
    .stMarkdown tr:hover {
        background: #f9f9f9 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        background: white !important;
    }

    /* ── Gap analysis content ── */
    .gap-content h3 { margin-top: 1.5rem; font-size: 1.1rem; }
    .gap-content h4 { margin-top: 1rem; font-size: 0.95rem; color: #2563eb; }
    .gap-content p { color: #4a4a4a; }
    .gap-content a { color: #2563eb; text-decoration: none; }
    .gap-content a:hover { text-decoration: underline; }

    /* ── Divider ── */
    .divider { border: none; border-top: 1px solid #e0e0e0; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# HEADER
st.markdown("""
<div class="header-section">
    <h1>Literature Gap Analyzer</h1>
    <p>Automated research gap identification using dual API retrieval and AI-powered Chain-of-Thought analysis</p>
    <div class="provider-badges">
        <span class="badge">Semantic Scholar</span>
        <span class="badge">OpenAlex</span>
        <span class="badge">Google Gemini</span>
        <span class="badge">Chain-of-Thought</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# API KEY CHECK
if not GEMINI_API_KEY:
    st.markdown("""
    <div class="status-box warning">
    <strong>⚠️ API Key Missing</strong><br>
    Please create a <code>.env</code> file with: <code>GEMINI_API_KEY=your_key_here</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
st.markdown("### Step 1: Configure Your Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    keyword = st.text_input(
        "Research keyword",
        placeholder="e.g., digital transformation SME, cloud computing security",
        help="Enter your research topic in English for best results",
    )

with col2:
    domain = st.selectbox(
        "Analysis domain",
        options=[
            "SME and Small Business Context",
            "Policy and Regulatory Framework",
            "User Behavior and Adoption",
            "Technology and Infrastructure",
        ],
        help="Select the domain most relevant to your research",
    )

col3, col4 = st.columns([1, 1])
with col3:
    max_papers = st.slider("Papers per API", 5, 25, 10, 5, help="Total papers will be ~2x after merging sources")

with col4:
    min_year = st.number_input("Min. year", 2000, 2026, 2020, help="Only retrieve papers from this year onwards")

st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# RUN BUTTON
run = st.button("🚀 Analyze Research Gaps", use_container_width=True, type="primary")

# ────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_paper_id(paper: dict) -> str:
    if "externalIds" in paper and paper["externalIds"]:
        if "DOI" in paper["externalIds"]:
            return f"DOI:{paper['externalIds']['DOI'].lower()}"
    if "paperId" in paper:
        return f"SS:{paper['paperId']}"
    return f"TITLE:{paper.get('title', '').lower()}"

def is_duplicate(paper1: dict, paper2: dict) -> bool:
    doi1 = paper1.get("externalIds", {}).get("DOI", "").lower() if isinstance(paper1.get("externalIds"), dict) else ""
    doi2 = paper2.get("externalIds", {}).get("DOI", "").lower() if isinstance(paper2.get("externalIds"), dict) else ""
    
    if doi1 and doi2 and doi1 == doi2:
        return True
    
    if not doi1 and not doi2:
        title1 = paper1.get("title", "").lower()
        title2 = paper2.get("title", "").lower()
        year1 = paper1.get("year")
        year2 = paper2.get("year")
        title_sim = string_similarity(title1, title2)
        if title_sim > 0.85 and year1 == year2:
            return True
    
    return False

def deduplicate_papers(papers1: list[dict], papers2: list[dict]) -> tuple[list[dict], dict]:
    merged = []
    seen_ids = set()
    api_sources = {}
    
    for p in papers1:
        merged.append(p)
        pid = get_paper_id(p)
        seen_ids.add(pid)
        api_sources[pid] = ["Semantic Scholar"]
    
    for p2 in papers2:
        is_dup = False
        for p1 in papers1:
            if is_duplicate(p1, p2):
                is_dup = True
                pid = get_paper_id(p1)
                if "OpenAlex" not in api_sources[pid]:
                    api_sources[pid].append("OpenAlex")
                break
        
        if not is_dup:
            merged.append(p2)
            pid = get_paper_id(p2)
            api_sources[pid] = ["OpenAlex"]
    
    return merged, api_sources

def fetch_papers_semantic_scholar(query: str, limit: int, min_year: int) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "title,abstract,year,authors,citationCount,url,externalIds,paperId"}
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"User-Agent": "LiteratureGapAnalyzer/2.0"}
    if s2_key:
        headers["x-api-key"] = s2_key
    
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=25)
            if resp.status_code in [429, 500, 502, 503]:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            raw = resp.json().get("data", [])
            return [p for p in raw if p.get("abstract") and p.get("year") and p["year"] >= min_year]
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(5)
            else:
                raise
    raise Exception("Semantic Scholar API error")

def fetch_papers_openalex(query: str, limit: int, min_year: int) -> list[dict]:
    url = "https://api.openalex.org/works"
    params = {"search": query, "per_page": limit, "sort": "cited_by_count:desc"}
    headers = {"User-Agent": "LiteratureGapAnalyzer/2.0"}
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=25)
        resp.raise_for_status()
        raw = resp.json().get("results", [])
        
        converted = []
        for work in raw:
            if not work.get("abstract_inverted_index"):
                continue
            year = work.get("publication_year")
            if not year or year < min_year:
                continue
            
            abstract_dict = work.get("abstract_inverted_index", {})
            abstract_words = [""] * len(abstract_dict)
            for word, positions in abstract_dict.items():
                for pos in positions:
                    if pos < len(abstract_words):
                        abstract_words[pos] = word
            abstract = " ".join(abstract_words).strip()
            
            authors = []
            for au in work.get("authorships", [])[:5]:
                name = au.get("author", {}).get("display_name")
                if name:
                    authors.append({"name": name})
            
            external_ids = {}
            doi = work.get("doi")
            if doi:
                external_ids["DOI"] = doi.replace("https://doi.org/", "")
            
            paper = {
                "title": work.get("title", ""),
                "abstract": abstract[:1000] if abstract else "",
                "year": year,
                "authors": authors,
                "citationCount": work.get("cited_by_count", 0),
                "url": work.get("landing_page_url") or work.get("doi"),
                "externalIds": external_ids,
                "paperId": work.get("id", ""),
            }
            converted.append(paper)
        
        return converted
    except Exception as e:
        st.warning(f"OpenAlex error: {e}")
        return []

def fetch_papers_dual(query: str, limit: int, min_year: int) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    papers_ss = fetch_papers_semantic_scholar(query, limit, min_year)
    time.sleep(0.5)
    papers_oa = fetch_papers_openalex(query, limit, min_year)
    fetch_time = time.perf_counter() - t0
    merged, api_sources = deduplicate_papers(papers_ss, papers_oa)
    return merged, api_sources, fetch_time

def build_prompt(papers: list[dict], domain: str) -> str:
    corpus = ""
    for i, p in enumerate(papers, 1):
        authors_list = p.get("authors", [])
        authors_disp = ", ".join(a["name"] for a in authors_list[:2])
        if len(authors_list) > 2:
            authors_disp += " et al."
        year = p.get("year", "n/a")
        abstract = p["abstract"][:800] + ("..." if len(p["abstract"]) > 800 else "")
        url = p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
        corpus += f"\n[{i}] {p['title']} ({year}) — {authors_disp}\n{abstract}\nURL: {url}\n"

    prompt = f"""Anda adalah profesor riset senior di bidang Sistem Informasi.

Tugas: Identifikasi research gaps dari paper berikut dalam konteks: {domain}

DATA PAPER:
{corpus}

INSTRUKSI KETAT:
1. Hanya gunakan informasi dari paper yang diberikan
2. Setiap claim harus mendapat hyperlink ke sumber: ([Author, Year](URL))
3. Jangan karang informasi yang tidak ada di abstrak
4. Jika data tidak cukup, tulis: "Data tidak cukup"
5. Gunakan Bahasa Indonesia formal

FORMAT RESPONS:

### 1. Analisis State of the Art
[Ringkas kontribusi & keterbatasan per paper. Gunakan hyperlink ke setiap claim.]

### 2. Research Gaps (Max 3)
Untuk setiap gap:
- **Deskripsi:** Apa gap-nya?
- **Bukti:** Dari paper mana? ([Author, Year](URL))
- **Tipe:** Empiris/Metodologis/Teoretis
- **Penting karena:** Mengapa harus diteliti?

### 3. Usulan Penelitian Baru
- **Judul (English):** ...
- **Judul (Indonesia):** ...
- **Novelty:** Kebaruan dibanding paper yang ada
- **Metrik:** 3 metrik evaluasi terukur
- **Target jurnal:** Rekomendasi Q2-Q3

---

DAFTAR REFERENSI:
"""
    
    for i, p in enumerate(papers, 1):
        authors_list = p.get("authors", [])
        authors = ", ".join(a["name"] for a in authors_list[:2])
        if len(authors_list) > 2:
            authors += " et al."
        year = p.get("year", "n/a")
        title_short = p["title"][:60] + "..." if len(p["title"]) > 60 else p["title"]
        url = p.get("url") or ""
        prompt += f"\n[{i}] {authors} ({year}). {title_short}. {url}"
    
    return prompt

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.4,
            top_p=0.92,
            max_output_tokens=65536,
        ),
    )
    return response.text

def make_csv(papers: list[dict]) -> bytes:
    df = pd.DataFrame([{
        "No": i, "Year": p.get("year"), "Title": p["title"],
        "Authors": ", ".join(a["name"] for a in p.get("authors", [])[:3]),
        "Citations": p.get("citationCount", 0), "Abstract": p["abstract"][:200],
    } for i, p in enumerate(papers, 1)])
    return df.to_csv(index=False).encode("utf-8")

# ────────────────────────────────────────────────────────────────────────────
# MAIN LOGIC
if run:
    if not keyword.strip():
        st.error("⚠️ Please enter a research keyword")
        st.stop()

    if keyword != st.session_state.last_keyword or domain != st.session_state.last_domain:
        st.session_state.papers = None
        st.session_state.gemini_result = None
        st.session_state.api_sources = {}

    # ── STEP 1: Fetch Papers ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Step 2: Retrieving Papers")
    
    step1_container = st.container()
    with step1_container:
        if st.session_state.papers is None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔍 Searching Semantic Scholar...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("📊 Searching OpenAlex...")
                progress_bar.progress(50)
                
                t0 = time.perf_counter()
                papers, api_sources, fetch_time = fetch_papers_dual(keyword, max_papers, min_year)
                progress_bar.progress(100)
                
                if not papers:
                    st.error("❌ No papers found. Try a different keyword or year range.")
                    st.stop()
                
                st.session_state.papers = papers
                st.session_state.api_sources = api_sources
                st.session_state.fetch_time = fetch_time
                st.session_state.last_keyword = keyword
                st.session_state.last_domain = domain
                
                status_text.empty()
                progress_bar.empty()
                
                st.markdown(f"""
                <div class="status-box success">
                ✅ Retrieved <strong>{len(papers)} papers</strong> in {fetch_time:.2f}s
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()
        else:
            st.markdown(f"""
            <div class="status-box info">
            📖 Loaded from cache: {len(st.session_state.papers)} papers
            </div>
            """, unsafe_allow_html=True)

    # ── Display Paper Metrics ──
    papers = st.session_state.papers
    year_range = f"{min(p['year'] for p in papers)}-{max(p['year'] for p in papers)}"
    avg_cit = sum(p.get("citationCount", 0) for p in papers) / len(papers) if papers else 0
    ss_count = sum(1 for p in papers if st.session_state.api_sources.get(get_paper_id(p), [""])[0] == "Semantic Scholar")
    oa_count = sum(1 for p in papers if st.session_state.api_sources.get(get_paper_id(p), [""])[0] == "OpenAlex")
    both_count = sum(1 for p in papers if len(st.session_state.api_sources.get(get_paper_id(p), [])) == 2)

    st.markdown("""<div class="metrics-grid">""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Papers</div><div class="metric-value">{len(papers)}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Year Range</div><div class="metric-value">{year_range}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Citations</div><div class="metric-value">{avg_cit:.0f}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Dual Source</div><div class="metric-value">{both_count}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Paper Details Expander ──
    with st.expander(f"📋 View {len(papers)} papers", expanded=False):
        table_md = "| # | Title | Authors | Year | Citations | Source |\n|:--:|--------|---------|:--:|:--:|:--:|\n"
        for i, p in enumerate(papers, 1):
            authors = ", ".join(a["name"] for a in p.get("authors", [])[:2])
            if len(p.get("authors", [])) > 2:
                authors += " et al."
            url = p.get("url") or ""
            title_short = p["title"][:50] + "..." if len(p["title"]) > 50 else p["title"]
            source = ", ".join(st.session_state.api_sources.get(get_paper_id(p), ["Unknown"]))
            table_md += f"| {i} | [{title_short}]({url}) | {authors} | {p.get('year')} | {p.get('citationCount', 0)} | {source} |\n"
        st.markdown(table_md)
        
        st.download_button("📥 Download papers (CSV)", make_csv(papers),
                          file_name=f"papers_{keyword.replace(' ', '_')}.csv", mime="text/csv")

    # ── STEP 2: Gemini Analysis ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Step 3: Analyzing with AI")

    if st.session_state.gemini_result is None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        prompt = build_prompt(papers, domain)
        status_text.text("⏳ Analyzing gaps with Gemini 2.5 Flash...")
        progress_bar.progress(50)
        
        try:
            t1 = time.perf_counter()
            result = call_gemini(prompt)
            gemini_time = time.perf_counter() - t1
            
            approx_input = len(prompt.split()) * 1.3
            approx_output = len(result.split()) * 1.3
            cost = (approx_input / 1_000_000 * 0.075) + (approx_output / 1_000_000 * 0.30)
            
            st.session_state.gemini_result = result
            st.session_state.gemini_time = gemini_time
            st.session_state.approx_cost = cost
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            st.markdown(f"""
            <div class="status-box success">
            ✅ Analysis complete in {gemini_time:.1f}s
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Gemini error: {e}")
            st.stop()
    else:
        st.markdown("""
        <div class="status-box info">
        📖 Loaded from cache
        </div>
        """, unsafe_allow_html=True)

    # ── Display Results ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Step 4: Research Gap Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Analysis Time</div><div class="metric-value">{st.session_state.gemini_time:.1f}s</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Est. Cost</div><div class="metric-value">${st.session_state.approx_cost:.4f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sources</div><div class="metric-value">2 APIs</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="gap-content">', unsafe_allow_html=True)
    st.markdown(st.session_state.gemini_result)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Download Results ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Gap Analysis (Markdown)",
                          st.session_state.gemini_result.encode("utf-8"),
                          file_name=f"gap_analysis_{keyword.replace(' ', '_')}.md",
                          mime="text/markdown", use_container_width=True)
    with col2:
        st.download_button("📥 Full Report (TXT)",
                          f"Keyword: {keyword}\nDomain: {domain}\nPapers: {len(papers)}\nAnalysis Time: {st.session_state.gemini_time:.1f}s\n\n{st.session_state.gemini_result}".encode("utf-8"),
                          file_name=f"report_{keyword.replace(' ', '_')}.txt",
                          mime="text/plain", use_container_width=True)

    st.markdown(f"""
    <div class="status-box success" style="margin-top: 2rem;">
    ✅ Analysis complete! {len(papers)} papers analyzed in {st.session_state.fetch_time + st.session_state.gemini_time:.1f}s total.
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# FOOTER
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem 0;">
Literature Gap Analyzer v2.1 · Semantic Scholar + OpenAlex · Google Gemini · Chain-of-Thought
</div>
""", unsafe_allow_html=True)
