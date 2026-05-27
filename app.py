import streamlit as st
import requests
import google.generativeai as genai
import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Literature Gap Analyzer",
    page_icon=None,
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "papers"        not in st.session_state: st.session_state.papers        = None
if "gemini_result" not in st.session_state: st.session_state.gemini_result = None
if "fetch_time"    not in st.session_state: st.session_state.fetch_time    = 0.0
if "gemini_time"   not in st.session_state: st.session_state.gemini_time   = 0.0
if "approx_cost"   not in st.session_state: st.session_state.approx_cost   = 0.0
if "last_keyword"  not in st.session_state: st.session_state.last_keyword  = ""
if "last_domain"   not in st.session_state: st.session_state.last_domain   = ""

# ── CSS — white background, neutral academic palette ─────────────────────────
st.markdown("""
<style>
    /* ── Base & background ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #ffffff !important;
        color: #1f2937;
    }
    .stApp { background-color: #ffffff !important; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Streamlit sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] * { color: #495057 !important; }

    /* Streamlit buttons */
    .stButton > button {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 7px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.2rem !important;
        transition: background 0.15s;
    }
    .stButton > button:hover { background-color: #374151 !important; }

    /* Download buttons */
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #374151 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 7px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #f9fafb !important;
        border-color: #9ca3af !important;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 7px !important;
        color: #1f2937 !important;
        font-size: 0.88rem !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #e9ecef !important;
        border-radius: 8px !important;
        background: #ffffff !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }

    /* Streamlit metric */
    [data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #9ca3af !important; }
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; color: #1f2937 !important; font-weight: 700 !important; }

    /* ── App header ── */
    .app-header {
        display: flex;
        align-items: flex-start;
        gap: 1.2rem;
        padding: 1.8rem 0 1.4rem;
        border-bottom: 1.5px solid #e9ecef;
        margin-bottom: 2rem;
    }
    .header-logo img { width: 34px; height: 34px; border-radius: 6px; margin-top: 3px; }
    .header-text h1 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        letter-spacing: -0.02em;
        margin: 0 0 0.2rem;
        line-height: 1;
    }
    .header-text p { font-size: 0.84rem; color: #6b7280; margin: 0; }

    /* Provider badges */
    .provider-bar { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.6rem; flex-wrap: wrap; }
    .provider-badge {
        display: inline-flex; align-items: center; gap: 5px;
        font-size: 0.69rem; font-weight: 500; color: #6b7280;
        background: #f3f4f6; border: 1px solid #e5e7eb;
        border-radius: 20px; padding: 2px 9px;
    }
    .provider-badge img { width: 12px; height: 12px; border-radius: 2px; }
    .badge-sep { color: #d1d5db; font-size: 0.75rem; }

    /* Section label */
    .section-label {
        font-size: 0.67rem; font-weight: 600; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 0.75rem;
    }

    /* Metric row (custom HTML cards) */
    .metric-row { display: flex; gap: 0.75rem; margin: 0.75rem 0 1.25rem; }
    .metric-card {
        flex: 1; background: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 8px; padding: 0.85rem 1.1rem;
    }
    .metric-card .m-label {
        font-size: 0.67rem; font-weight: 500; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.28rem;
    }
    .metric-card .m-value { font-size: 1.3rem; font-weight: 700; color: #1f2937; line-height: 1; }

    /* Result box */
    .result-wrap {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 2rem 2.4rem;
        margin-top: 0.75rem;
    }
    .result-wrap h3 {
        font-size: 0.96rem !important; font-weight: 700 !important;
        color: #1f2937 !important; border-bottom: 1px solid #f3f4f6;
        padding-bottom: 0.35rem; margin-top: 1.8rem !important;
    }
    .result-wrap h4 {
        font-size: 0.87rem !important; font-weight: 600 !important;
        color: #374151 !important; margin-top: 1.1rem !important;
    }
    .result-wrap p, .result-wrap li {
        font-size: 0.86rem !important; color: #374151 !important; line-height: 1.8 !important;
    }
    .result-wrap a { color: #2563eb !important; text-decoration: none !important; }
    .result-wrap a:hover { text-decoration: underline !important; }
    .result-wrap table {
        font-size: 0.8rem !important; border-collapse: collapse; width: 100%; margin-top: 1rem;
    }
    .result-wrap th {
        background: #f8f9fa; color: #374151; font-weight: 600;
        text-align: left; padding: 0.55rem 0.75rem; border: 1px solid #e9ecef;
    }
    .result-wrap td {
        padding: 0.5rem 0.75rem; border: 1px solid #f3f4f6;
        color: #4b5563; vertical-align: top;
    }
    .result-wrap blockquote {
        border-left: 3px solid #d1d5db; margin: 0.5rem 0;
        padding: 0.3rem 1rem; color: #6b7280 !important; font-style: italic;
        background: #f9fafb; border-radius: 0 6px 6px 0;
    }

    /* Status bar */
    .status-bar {
        display: flex; align-items: center; gap: 1.2rem;
        font-size: 0.74rem; color: #9ca3af; margin-bottom: 0.75rem;
    }
    .status-dot {
        display: inline-block; width: 6px; height: 6px;
        background: #10b981; border-radius: 50%;
        margin-right: 4px; vertical-align: middle;
    }

    /* Cached notice */
    .cached-notice {
        display: inline-flex; align-items: center; gap: 5px;
        font-size: 0.71rem; color: #059669;
        background: #f0fdf4; border: 1px solid #bbf7d0;
        border-radius: 20px; padding: 2px 10px; margin-bottom: 1rem;
    }

    /* Warning */
    .warn-plain {
        background: #fffbeb; border: 1px solid #fde68a;
        border-radius: 8px; padding: 0.75rem 1rem;
        font-size: 0.84rem; color: #92400e;
    }

    /* Markdown table links */
    .stMarkdown a { color: #2563eb !important; text-decoration: none !important; }
    .stMarkdown a:hover { text-decoration: underline !important; }

    /* Divider */
    .clean-divider { border: none; border-top: 1px solid #f3f4f6; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Header with Semantic Scholar logo ────────────────────────────────────────
S2_LOGO = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Semantic_Scholar_logo.svg/120px-Semantic_Scholar_logo.svg.png"

st.markdown(f"""
<div class="app-header">
    <div class="header-logo">
        <img src="{S2_LOGO}" alt="Semantic Scholar">
    </div>
    <div class="header-text">
        <h1>Literature Gap Analyzer</h1>
        <p>Automated research gap identification from academic literature using Chain-of-Thought AI analysis</p>
        <div class="provider-bar">
            <span class="provider-badge">
                <img src="{S2_LOGO}" alt="S2"> Semantic Scholar
            </span>
            <span class="badge-sep">·</span>
            <span class="provider-badge">Google Gemini 2.5 Flash</span>
            <span class="badge-sep">·</span>
            <span class="provider-badge">Chain-of-Thought Prompting</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Configuration**")
    max_papers = st.slider("Papers to retrieve", min_value=5, max_value=25, value=5, step=5)
    min_year = st.number_input("Minimum publication year", min_value=2000, max_value=2024, value=2018)

    if st.session_state.papers is not None:
        st.markdown("---")
        st.markdown("**Session**")
        st.caption(f"Keyword: `{st.session_state.last_keyword}`")
        st.caption(f"Papers cached: {len(st.session_state.papers)}")
        if st.button("Clear session", use_container_width=True):
            for key in ["papers", "gemini_result", "fetch_time", "gemini_time",
                        "approx_cost", "last_keyword", "last_domain"]:
                st.session_state[key] = None if key in ["papers", "gemini_result"] else (0.0 if key in ["fetch_time", "gemini_time", "approx_cost"] else "")
            st.rerun()

    st.markdown("---")
    st.caption("PoC · Academic Research Tool · v1.0")

# ── API key check ─────────────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    st.markdown("""<div class="warn-plain">
    <strong>GEMINI_API_KEY not found.</strong> Create a <code>.env</code> file:<br>
    <code>GEMINI_API_KEY=your_key_here</code></div>""", unsafe_allow_html=True)
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ── Input ─────────────────────────────────────────────────────────────────────
col_input, col_domain = st.columns([2, 1])
with col_input:
    keyword = st.text_input(
        "Research keyword",
        placeholder="e.g. digital transformation SME, e-government adoption, knowledge management",
        help="Use English keywords for best results.",
    )
with col_domain:
    domain = st.selectbox(
        "Analysis domain focus",
        options=[
            "SME and Small Business Context",
            "Policy and Regulatory Framework",
            "User Behavior and Adoption",
            "Technology and Infrastructure",
        ],
    )

run = st.button("Run Analysis", type="primary", use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_papers(query: str, limit: int, min_year: int) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "title,abstract,year,authors,citationCount,url,externalIds"}
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"User-Agent": "LiteratureGapAnalyzer/1.0"}
    if s2_key:
        headers["x-api-key"] = s2_key
    wait_seconds = [5, 15, 30, 60]
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=25)
            if resp.status_code in [429, 500, 502, 503]:
                wait = wait_seconds[attempt]
                st.warning(f"API rate limit — retrying in {wait}s (attempt {attempt+1}/4)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            raw = resp.json().get("data", [])
            return [p for p in raw if p.get("abstract") and p.get("year") and p["year"] >= min_year]
        except requests.exceptions.Timeout:
            if attempt < 3:
                st.warning(f"Timeout — retrying ({attempt+2}/4)")
                time.sleep(wait_seconds[attempt])
            else:
                raise
    raise Exception("Semantic Scholar API continues to return errors. Wait 1–2 minutes and retry.")


def build_prompt(papers: list[dict], domain: str) -> str:
    corpus = ""
    ref_rows = ""

    for i, p in enumerate(papers, 1):
        authors_list = p.get("authors", [])
        surname      = authors_list[0]["name"].split()[-1] if authors_list else "Unknown"
        authors_disp = ", ".join(a["name"] for a in authors_list[:2])
        if len(authors_list) > 2:
            authors_disp += " et al."
        year     = p.get("year", "n/a")
        abstract = p["abstract"][:800] + ("..." if len(p["abstract"]) > 800 else "")
        url      = p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"

        corpus += (
            f"\n---\n"
            f"[{i}] | Judul: {p['title']} | Penulis: {authors_disp} | "
            f"Tahun: {year} | URL: {url}\n"
            f"Abstrak: {abstract}\n"
        )

        title_short = p["title"][:60] + "..." if len(p["title"]) > 60 else p["title"]
        ref_rows += (
            f"| {i} | [{title_short}]({url}) | {authors_disp} | {year} | "
            f"{p.get('citationCount', 0)} | [Buka]({url}) |\n"
        )

    ref_table = (
        "\n\n---\n"
        "### Daftar Referensi Paper\n\n"
        "| No | Judul Paper | Penulis | Tahun | Sitasi | Tautan |\n"
        "|:--:|------------|---------|:----:|:------:|:------:|\n"
        + ref_rows
    )

    prompt = f"""Anda adalah seorang Profesor Riset senior dalam bidang Sistem Informasi, spesialis systematic literature review dan publikasi jurnal Scopus Q1–Q2.

Tugas Anda adalah menemukan Research Gap yang relevan dengan konteks: **{domain}** dari kumpulan data paper berikut.

DATA PAPER UNTUK DIANALISIS:
{corpus}

---

ATURAN KETAT ANTI-HALUSINASI — patuhi tanpa pengecualian:
1. Anda DILARANG KERAS mengarang informasi, teori, nama penulis, atau kelemahan yang tidak dapat dibuktikan langsung dari teks abstrak yang diberikan di atas.
2. Jika abstrak tidak memberikan informasi yang cukup untuk bagian tertentu, nyatakan secara eksplisit: "Data tidak mencukupi untuk menarik kesimpulan pada bagian ini."
3. Setiap argumen, klaim, atau penyebutan kelemahan WAJIB disertai kutipan sebaris dalam format Markdown Hyperlink: ([Penulis, Tahun](URL Paper)). Contoh: ...hanya berfokus pada efisiensi teknis tanpa mempertimbangkan faktor adopsi pengguna ([Smith, 2023](https://semanticscholar.org/...)).
4. URL yang digunakan dalam hyperlink harus berasal dari data URL yang telah disediakan di atas, bukan URL yang dikarang sendiri.
5. Jangan gunakan emoji. Gunakan Bahasa Indonesia ilmiah yang formal dan netral.
6. Jangan sertakan tabel referensi — sudah disiapkan oleh sistem secara otomatis.

---

### 1. Analisis State of the Art (SOTA) dan Limitasi

Berdasarkan bukti yang dapat diverifikasi dari abstrak yang diberikan, jelaskan sejauh mana penelitian saat ini telah berjalan dalam konteks **{domain}**. Kemudian petakan kelemahan atau keterbatasan spesifik dari masing-masing paper secara objektif.

Untuk setiap paper yang memiliki bukti limitasi yang cukup, gunakan format:

#### Paper [N]: [Judul Singkat]
- **Kontribusi utama:** Apa yang sudah dicapai oleh penelitian ini berdasarkan abstraknya?
- **Keterbatasan teridentifikasi:** Apa yang secara eksplisit atau implisit tidak dicakup? Sertakan hyperlink: ([Penulis, Tahun](URL)).
- **Relevansi terhadap {domain}:** Apakah temuan ini relevan atau justru mengabaikan konteks ini?

---

### 2. Formulasi Research Gaps

Berdasarkan analisis SOTA di atas, rumuskan maksimal **3 Research Gap** (dapat berupa Gap Empiris, Metodologis, atau Teoretis). Hanya masukkan gap yang didukung bukti nyata dari abstrak.

#### Gap 1: [Nama Gap — spesifik dan deskriptif]
- **Konteks:** Apa celahnya secara konkret?
- **Bukti dari Literatur:** Buktikan bahwa paper yang diberikan mengabaikan hal ini. Sertakan hyperlink pada setiap kutipan: ([Penulis, Tahun](URL)).
- **Tipe Gap:** Empiris / Metodologis / Teoretis — pilih satu dan jelaskan.
- **Mengapa penting:** Dampak akademis dan praktis jika celah ini tidak diteliti dalam konteks {domain}.

#### Gap 2: [Nama Gap]
- **Konteks:** ...
- **Bukti dari Literatur:** ...
- **Tipe Gap:** ...
- **Mengapa penting:** ...

#### Gap 3: [Nama Gap]
- **Konteks:** ...
- **Bukti dari Literatur:** ...
- **Tipe Gap:** ...
- **Mengapa penting:** ...

---

### 3. Usulan Novelty dan Ide Riset Baru

Berikan **1 ide judul penelitian** yang menawarkan kebaruan (Novelty) untuk mengisi salah satu gap di atas.

**Judul (Bahasa Inggris):**
> [tulis judul]

**Judul (Bahasa Indonesia):**
> [tulis judul]

**Gap yang dijawab:** [nama gap dari Bagian 2]

**Letak Kebaruan (Novelty):** Jelaskan secara konkret di mana letak kebaruannya dibandingkan paper-paper yang telah dianalisis. Gunakan hyperlink saat menyebut paper pembanding: ([Penulis, Tahun](URL)).

**Metrik Evaluasi (terukur tanpa responden manusia):**
- Metrik 1: [nama metrik dan cara pengukurannya]
- Metrik 2: [nama metrik dan cara pengukurannya]
- Metrik 3: [nama metrik dan cara pengukurannya]

**Metodologi yang Disarankan:** [Design Science Research / Comparative Experiment / Systematic Experiment — pilih satu dan jelaskan alasannya]

**Target Jurnal Scopus:** [Rekomendasikan 1–2 jurnal dengan kuartil, misal: Journal of Information Systems Q2]
"""
    return prompt + ref_table


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
        "Citations": p.get("citationCount", 0), "Abstract": p["abstract"],
    } for i, p in enumerate(papers, 1)])
    return df.to_csv(index=False).encode("utf-8")


def make_full_report(keyword, domain, papers, result, fetch_time, gemini_time, cost) -> bytes:
    year_range = f"{min(p['year'] for p in papers)}-{max(p['year'] for p in papers)}"
    body = (
        f"# Research Gap Analysis Report\n\n"
        f"Keyword: {keyword}\nDomain: {domain}\nPapers: {len(papers)}\n"
        f"Year range: {year_range}\nFetch time: {fetch_time:.2f}s\n"
        f"Analysis time: {gemini_time:.2f}s\nEst. cost: ${cost:.5f}\n\n---\n\n"
        f"{result}\n\n---\n\n## Papers Analyzed\n\n"
        + "\n".join(f"{i}. ({p.get('year')}) {p['title']}" for i, p in enumerate(papers, 1))
    )
    return body.encode("utf-8")


# ── DISPLAY HELPER — renders results without re-calling APIs ──────────────────
def display_papers(papers, fetch_time):
    year_range = f"{min(p['year'] for p in papers)}-{max(p['year'] for p in papers)}"
    avg_cit    = sum(p.get("citationCount", 0) for p in papers) / len(papers)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><div class="m-label">Papers retrieved</div><div class="m-value">{len(papers)}</div></div>
        <div class="metric-card"><div class="m-label">Year range</div><div class="m-value">{year_range}</div></div>
        <div class="metric-card"><div class="m-label">Avg. citations</div><div class="m-value">{avg_cit:.0f}</div></div>
        <div class="metric-card"><div class="m-label">Fetch time</div><div class="m-value">{fetch_time:.2f}s</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"View {len(papers)} retrieved papers", expanded=True):
        # Build markdown table with hyperlinks
        table_md = (
            "| No | Judul Paper | Penulis | Tahun | Sitasi |\n"
            "|:--:|------------|---------|:----:|:------:|\n"
        )
        for i, p in enumerate(papers, 1):
            authors_list = p.get("authors", [])
            authors      = ", ".join(a["name"] for a in authors_list[:2])
            if len(authors_list) > 2:
                authors += " et al."
            year     = p.get("year", "N/A")
            cit      = p.get("citationCount", 0)
            url      = p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
            title_md = p["title"][:70] + "..." if len(p["title"]) > 70 else p["title"]
            table_md += f"| {i} | [{title_md}]({url}) | {authors or 'Unknown'} | {year} | {cit} |\n"

        st.markdown(table_md)

        st.download_button(
            "Download paper list (CSV)",
            data=make_csv(st.session_state.papers),
            file_name=f"papers_{st.session_state.last_keyword.replace(' ', '_')}.csv",
            mime="text/csv",
            key="dl_csv",
        )


def display_result(result, gemini_time, cost):
    st.markdown(
        f'<div class="status-bar">'
        f'<span><span class="status-dot"></span>gemini-2.5-flash</span>'
        f'<span>Domain: {st.session_state.last_domain}</span>'
        f'<span>Chain-of-Thought Prompting</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Analysis time", f"{gemini_time:.2f}s")
    col_m2.metric("Est. tokens", f"~{int((len(result.split())*1.3)*2):,}")
    col_m3.metric("Est. API cost", f"${cost:.5f}")

    st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
    st.markdown(result)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download gap report (Markdown)",
            data=st.session_state.gemini_result.encode("utf-8"),
            file_name=f"gap_analysis_{st.session_state.last_keyword.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
            key="dl_md",
        )
    with col_dl2:
        st.download_button(
            "Download full report (TXT)",
            data=make_full_report(
                st.session_state.last_keyword,
                st.session_state.last_domain,
                st.session_state.papers,
                st.session_state.gemini_result,
                st.session_state.fetch_time,
                st.session_state.gemini_time,
                st.session_state.approx_cost,
            ),
            file_name=f"full_report_{st.session_state.last_keyword.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_txt",
        )


# ── Main logic ────────────────────────────────────────────────────────────────
if run:
    if not keyword.strip():
        st.error("Please enter a research keyword.")
        st.stop()

    # Clear previous session if keyword/domain changed
    if keyword != st.session_state.last_keyword or domain != st.session_state.last_domain:
        st.session_state.papers        = None
        st.session_state.gemini_result = None

    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)

    # ── Step 1: Fetch papers ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 1 — Data Retrieval · Semantic Scholar</div>', unsafe_allow_html=True)

    if st.session_state.papers is None:
        with st.spinner("Retrieving papers from Semantic Scholar..."):
            try:
                t0 = time.perf_counter()
                papers = fetch_papers(keyword, max_papers, min_year)
                fetch_time = time.perf_counter() - t0
            except Exception as e:
                st.error(f"Failed to retrieve papers: {e}")
                st.stop()

        if not papers:
            st.error("No papers with abstracts found. Try a broader or different keyword.")
            st.stop()

        # Save to session
        st.session_state.papers       = papers
        st.session_state.fetch_time   = fetch_time
        st.session_state.last_keyword = keyword
        st.session_state.last_domain  = domain
    else:
        st.markdown('<div class="cached-notice">Loaded from session — no API call made</div>', unsafe_allow_html=True)

    display_papers(st.session_state.papers, st.session_state.fetch_time)

    # ── Step 2: Gemini ────────────────────────────────────────────────────────
    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Step 2 — AI Gap Analysis · Gemini</div>', unsafe_allow_html=True)

    if st.session_state.gemini_result is None:
        prompt = build_prompt(st.session_state.papers, domain)
        with st.spinner("Analyzing — this may take 15-30 seconds"):
            try:
                t1 = time.perf_counter()
                result = call_gemini(prompt)
                gemini_time = time.perf_counter() - t1
            except Exception as e:
                st.error(f"Gemini API error: {e}")
                st.stop()

        approx_input  = len(prompt.split()) * 1.3
        approx_output = len(result.split()) * 1.3
        cost = (approx_input / 1_000_000 * 0.075) + (approx_output / 1_000_000 * 0.30)

        # Save to session
        st.session_state.gemini_result = result
        st.session_state.gemini_time   = gemini_time
        st.session_state.approx_cost   = cost
    else:
        st.markdown('<div class="cached-notice">Loaded from session — no API call made</div>', unsafe_allow_html=True)

    display_result(
        st.session_state.gemini_result,
        st.session_state.gemini_time,
        st.session_state.approx_cost,
    )

    st.success(
        f"Analysis complete. {len(st.session_state.papers)} papers · "
        f"{st.session_state.fetch_time + st.session_state.gemini_time:.2f}s total."
    )

# ── If session already has results (no button press needed) ───────────────────
elif st.session_state.papers is not None:
    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Step 1 — Data Retrieval · Semantic Scholar</div>', unsafe_allow_html=True)
    st.markdown('<div class="cached-notice">Loaded from session</div>', unsafe_allow_html=True)
    display_papers(st.session_state.papers, st.session_state.fetch_time)

    if st.session_state.gemini_result:
        st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Step 2 — AI Gap Analysis · Gemini</div>', unsafe_allow_html=True)
        st.markdown('<div class="cached-notice">Loaded from session</div>', unsafe_allow_html=True)
        display_result(
            st.session_state.gemini_result,
            st.session_state.gemini_time,
            st.session_state.approx_cost,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
st.caption("Literature Gap Analyzer · Semantic Scholar API · Google Gemini · Information Systems Research PoC")
