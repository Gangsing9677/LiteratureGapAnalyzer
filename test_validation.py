"""
Test Validation Script - Dual API Integration
Runs 10-15 automated sessions to validate:
1. System works correctly with dual API retrieval
2. Deduplication logic functions properly
3. Cross-source consistency metrics
4. Performance benchmarks
"""

import requests
import google.generativeai as genai
import os
import time
import pandas as pd
from dotenv import load_dotenv
from difflib import SequenceMatcher
import json
from datetime import datetime

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# TEST CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    # SME and Small Business Context (6 sessions)
    {"keyword": "Digital transformation small medium enterprises", "domain": "SME and Small Business Context", "papers": 10},
    {"keyword": "E-commerce adoption local vendors", "domain": "SME and Small Business Context", "papers": 10},
    {"keyword": "Cloud computing benefits small businesses", "domain": "SME and Small Business Context", "papers": 10},
    {"keyword": "Social media marketing impact SME sales", "domain": "SME and Small Business Context", "papers": 10},
    {"keyword": "ERP implementation challenges micro enterprises", "domain": "SME and Small Business Context", "papers": 10},
    {"keyword": "Digital payment system traditional markets", "domain": "SME and Small Business Context", "papers": 10},
    
    # Policy and Regulatory Framework (8 sessions)
    {"keyword": "E-government data privacy regulations", "domain": "Policy and Regulatory Framework", "papers": 10},
    {"keyword": "Cyber law electronic transaction security", "domain": "Policy and Regulatory Framework", "papers": 10},
    {"keyword": "AI ethics governance public sector", "domain": "Policy and Regulatory Framework", "papers": 10},
    {"keyword": "Open government data policy implementation", "domain": "Policy and Regulatory Framework", "papers": 10},
    {"keyword": "Digital tax regulations global e-commerce", "domain": "Policy and Regulatory Framework", "papers": 10},
    {"keyword": "National cybersecurity strategy framework", "domain": "Policy and Regulatory Framework", "papers": 10},
    {"keyword": "Intellectual property rights digital age", "domain": "Policy and Regulatory Framework", "papers": 10},
    {"keyword": "Business resilience SMEs economic crisis", "domain": "Policy and Regulatory Framework", "papers": 10},
    
    # User Behavior and Adoption (8 sessions)
    {"keyword": "Mobile banking user trust factors", "domain": "User Behavior and Adoption", "papers": 10},
    {"keyword": "Gamification student engagement learning management system", "domain": "User Behavior and Adoption", "papers": 10},
    {"keyword": "Telemedicine adoption elderly users", "domain": "User Behavior and Adoption", "papers": 10},
    {"keyword": "Perceived risk online shopping behavior", "domain": "User Behavior and Adoption", "papers": 10},
    {"keyword": "User resistance new enterprise systems", "domain": "User Behavior and Adoption", "papers": 10},
    {"keyword": "Digital literacy online safety behavior", "domain": "User Behavior and Adoption", "papers": 10},
    {"keyword": "Influencer impact consumer purchasing intent", "domain": "User Behavior and Adoption", "papers": 10},
    {"keyword": "Mental health social media usage", "domain": "User Behavior and Adoption", "papers": 10},
    
    # Technology and Infrastructure (8 sessions)
    {"keyword": "Blockchain supply chain transparency", "domain": "Technology and Infrastructure", "papers": 10},
    {"keyword": "IoT smart agriculture monitoring system", "domain": "Technology and Infrastructure", "papers": 10},
    {"keyword": "Microservices architecture performance cloud", "domain": "Technology and Infrastructure", "papers": 10},
    {"keyword": "5G network infrastructure smart cities", "domain": "Technology and Infrastructure", "papers": 10},
    {"keyword": "Machine learning phishing website detection", "domain": "Technology and Infrastructure", "papers": 10},
    {"keyword": "Edge computing autonomous vehicle sensors", "domain": "Technology and Infrastructure", "papers": 10},
    {"keyword": "Big data analytics traffic management", "domain": "Technology and Infrastructure", "papers": 10},
    {"keyword": "Wireless sensor networks disaster early warning", "domain": "Technology and Infrastructure", "papers": 10},
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def string_similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings (0-1)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_paper_id(paper: dict) -> str:
    """Extract a unique ID from paper"""
    if "externalIds" in paper and paper["externalIds"]:
        if "DOI" in paper["externalIds"]:
            return f"DOI:{paper['externalIds']['DOI'].lower()}"
    if "paperId" in paper:
        return f"SS:{paper['paperId']}"
    return f"TITLE:{paper.get('title', '').lower()}"


def is_duplicate(paper1: dict, paper2: dict) -> bool:
    """Check if two papers are duplicates"""
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
    """Merge and deduplicate papers from two sources"""
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


def fetch_papers_semantic_scholar(query: str, limit: int, min_year: int = 2018) -> list[dict]:
    """Fetch from Semantic Scholar"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,authors,citationCount,url,externalIds,paperId"
    }
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"User-Agent": "TestValidation/2.0"}
    if s2_key:
        headers["x-api-key"] = s2_key
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=25)
        if resp.status_code in [429, 500, 502, 503]:
            print(f"  ⚠️  Semantic Scholar rate limit, retrying...")
            time.sleep(5)
            return fetch_papers_semantic_scholar(query, limit, min_year)
        resp.raise_for_status()
        raw = resp.json().get("data", [])
        return [p for p in raw if p.get("abstract") and p.get("year") and p["year"] >= min_year]
    except Exception as e:
        print(f"  ❌ Semantic Scholar error: {e}")
        return []


def fetch_papers_openalex(query: str, limit: int, min_year: int = 2018) -> list[dict]:
    """Fetch from OpenAlex and convert to unified format"""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": limit,
        "sort": "cited_by_count:desc",
    }
    headers = {"User-Agent": "TestValidation/2.0"}
    
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
            
            # Reconstruct abstract
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
        print(f"  ⚠️  OpenAlex error: {e}")
        return []


def fetch_papers_dual(query: str, limit: int, min_year: int = 2018) -> tuple[list[dict], dict, float]:
    """Fetch from both APIs and deduplicate"""
    t0 = time.perf_counter()
    
    print(f"  📚 Fetching from Semantic Scholar...", end=" ", flush=True)
    papers_ss = fetch_papers_semantic_scholar(query, limit, min_year)
    print(f"✓ ({len(papers_ss)} papers)")
    
    time.sleep(0.5)  # Rate limit courtesy
    
    print(f"  📚 Fetching from OpenAlex...", end=" ", flush=True)
    papers_oa = fetch_papers_openalex(query, limit, min_year)
    print(f"✓ ({len(papers_oa)} papers)")
    
    fetch_time = time.perf_counter() - t0
    
    merged, api_sources = deduplicate_papers(papers_ss, papers_oa)
    return merged, api_sources, fetch_time


def build_prompt_simple(papers: list[dict], domain: str) -> str:
    """Build simplified prompt for faster testing"""
    corpus = ""
    
    for i, p in enumerate(papers[:10], 1):  # Limit to first 10 for speed
        authors_list = p.get("authors", [])
        authors_disp = ", ".join(a["name"] for a in authors_list[:1])
        year = p.get("year", "n/a")
        abstract = p["abstract"][:400]  # Shorter for testing
        url = p.get("url") or f"https://example.com/{i}"
        
        corpus += f"\n[{i}] {p['title']} ({year})\n{authors_disp}\n{abstract}\n"
    
    prompt = f"""Analyze these papers for research gaps in: {domain}

{corpus}

Identify 1-2 key research gaps with evidence from the papers above. Be concise."""
    
    return prompt


def call_gemini(prompt: str, max_tokens: int = 8000) -> tuple[str, float]:
    """Call Gemini and measure time"""
    t0 = time.perf_counter()
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                top_p=0.92,
                max_output_tokens=max_tokens,
            ),
        )
        gemini_time = time.perf_counter() - t0
        return response.text, gemini_time
    except Exception as e:
        print(f"  ❌ Gemini error: {e}")
        return "", 0.0


def run_test_session(test_case: dict, session_num: int) -> dict:
    """Run a single test session"""
    keyword = test_case["keyword"]
    domain = test_case["domain"]
    max_papers = test_case["papers"]
    
    print(f"\n{'='*80}")
    print(f"Session {session_num} | Keyword: '{keyword}' | Domain: '{domain}'")
    print(f"{'='*80}")
    
    result = {
        "session": session_num,
        "keyword": keyword,
        "domain": domain,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "papers_retrieved": 0,
        "papers_ss": 0,
        "papers_oa": 0,
        "duplicates_found": 0,
        "fetch_time": 0.0,
        "gemini_time": 0.0,
        "total_time": 0.0,
    }
    
    t_start = time.perf_counter()
    
    try:
        # Step 1: Retrieve papers
        print("\n[Step 1] Retrieving papers from dual APIs...")
        papers, api_sources, fetch_time = fetch_papers_dual(keyword, max_papers, min_year=2018)
        
        if not papers:
            print("  ❌ No papers found!")
            result["status"] = "failed"
            return result
        
        # Count sources
        papers_ss = sum(1 for p in papers if api_sources.get(get_paper_id(p), [""])[0] == "Semantic Scholar")
        papers_oa = sum(1 for p in papers if api_sources.get(get_paper_id(p), [""])[0] == "OpenAlex")
        both = sum(1 for p in papers if len(api_sources.get(get_paper_id(p), [])) == 2)
        
        result["papers_retrieved"] = len(papers)
        result["papers_ss"] = papers_ss
        result["papers_oa"] = papers_oa
        result["duplicates_found"] = both
        result["fetch_time"] = fetch_time
        
        print(f"  ✓ Retrieved {len(papers)} papers (SS: {papers_ss}, OA: {papers_oa}, Both: {both})")
        
        # Step 2: Call Gemini for gap analysis
        print("\n[Step 2] Analyzing gaps with Gemini...")
        prompt = build_prompt_simple(papers, domain)
        gap_result, gemini_time = call_gemini(prompt)
        
        if not gap_result:
            print("  ❌ Gemini analysis failed!")
            result["status"] = "failed"
            return result
        
        result["gemini_time"] = gemini_time
        print(f"  ✓ Analysis complete ({gemini_time:.2f}s)")
        
        # Summary
        total_time = time.perf_counter() - t_start
        result["total_time"] = total_time
        result["status"] = "success"
        
        print(f"\n[Summary]")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Fetch time: {fetch_time:.2f}s")
        print(f"  Gemini time: {gemini_time:.2f}s")
        print(f"  Papers analyzed: {len(papers)}")
        print(f"  Verified in multiple sources: {both} ({both/len(papers)*100:.1f}%)")
        
    except Exception as e:
        print(f"  ❌ Session failed: {e}")
        result["status"] = "error"
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TEST VALIDATION - DUAL API INTEGRATION")
    print("="*80)
    print(f"Total sessions to run: {len(TEST_CASES)}")
    print(f"Tests: {[tc['keyword'] for tc in TEST_CASES]}")
    print("="*80)
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        result = run_test_session(test_case, i)
        results.append(result)
        
        # Courtesy rate limit
        if i < len(TEST_CASES):
            time.sleep(1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE REPORT
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n\n{'='*80}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(results)
    
    # Success rate
    success_count = (df_results["status"] == "success").sum()
    print(f"✓ Successful sessions: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count > 0:
        print(f"\n[Performance Metrics]")
        successful = df_results[df_results["status"] == "success"]
        print(f"  Avg. total time: {successful['total_time'].mean():.2f}s (±{successful['total_time'].std():.2f}s)")
        print(f"  Avg. fetch time: {successful['fetch_time'].mean():.2f}s")
        print(f"  Avg. gemini time: {successful['gemini_time'].mean():.2f}s")
        print(f"  Avg. papers retrieved: {successful['papers_retrieved'].mean():.1f}")
        
        print(f"\n[Cross-Source Validation]")
        avg_both = successful['duplicates_found'].mean()
        avg_total = successful['papers_retrieved'].mean()
        if avg_total > 0:
            cross_coverage = (avg_both / avg_total) * 100
            print(f"  Avg. papers verified by 2+ sources: {avg_both:.1f} ({cross_coverage:.1f}% of papers)")
        
        print(f"\n[API Source Distribution]")
        print(f"  Avg. from Semantic Scholar: {successful['papers_ss'].mean():.1f}")
        print(f"  Avg. from OpenAlex: {successful['papers_oa'].mean():.1f}")
    
    # Save detailed results
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    # Save JSON for easy reference
    json_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ JSON results saved to: {json_file}")
    
    print(f"\n{'='*80}\n")
