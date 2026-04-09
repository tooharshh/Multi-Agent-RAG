"""
Live URL Fetcher — Fetch and extract clean text from article URLs.

Multi-strategy approach:
  1. Primary: requests + trafilatura (fastest, works for most sites).
  2. Fallback A (PMC only): NCBI BioC JSON API — bypasses captcha/rate-limits
     by fetching structured full-text directly from NCBI's open-access API.
  3. Fallback B: curl_cffi with Chrome TLS-fingerprint impersonation
     (bypasses Cloudflare / bot-detection on WEF, ScienceDirect, Hastings).
  4. Fallback C: For academic papers behind paywalls (ACS, Elsevier),
     look up PubMed Central mirror via Semantic Scholar API.
  5. PMC rate-limit handling: retry after delay when extraction fails
     despite a 200 response during sequential fetching.
"""

import json as _json
import re
import time
import logging
import requests

try:
    import trafilatura
except ImportError:
    trafilatura = None

logger = logging.getLogger(__name__)

# Minimum character threshold — below this, we consider the fetch a failure
MIN_TEXT_LENGTH = 500

# Request timeout in seconds
REQUEST_TIMEOUT = 30

# Headers to mimic a real browser (avoids 403s from simple bot detection)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Domains known to require curl_cffi for TLS fingerprint bypass
_CURL_CFFI_DOMAINS = {"weforum.org", "sciencedirect.com", "thehastingscenter.org"}

# Domains known to block all automated access — use PMC mirror lookup
_PMC_MIRROR_DOMAINS = {"pubs.acs.org"}


def _normalise_pmc_url(url: str) -> str:
    """
    If the URL is a PMC article, rewrite to the full-text HTML version for
    more complete content extraction.
    """
    pmc_match = re.search(r"pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC\d+)", url)
    if pmc_match:
        pmc_id = pmc_match.group(1)
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
    return url


def _extract_doi(url: str) -> str | None:
    """Extract a DOI from an ACS, ScienceDirect, or Nature URL."""
    # ACS: pubs.acs.org/doi/10.1021/...
    m = re.search(r"pubs\.acs\.org/doi(?:/full)?/(10\.\d{4,9}/[^\s?#]+)", url)
    if m:
        return m.group(1)
    # ScienceDirect: doi embedded in redirect or use PII
    m = re.search(r"doi\.org/(10\.\d{4,9}/[^\s?#]+)", url)
    if m:
        return m.group(1)
    return None


def _fetch_html(url: str) -> str | None:
    """Download raw HTML from a URL with retries (standard requests)."""
    for attempt in range(3):
        try:
            resp = requests.get(
                url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(
                f"  Attempt {attempt + 1}/3 failed for {url}: {e}"
            )
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def _fetch_html_curl_cffi(url: str, retries: int = 1) -> str | None:
    """Download HTML using curl_cffi with Chrome TLS fingerprint."""
    try:
        from curl_cffi import requests as curl_requests
        for attempt in range(1 + retries):
            resp = curl_requests.get(url, impersonate="chrome", timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and len(resp.text) > 1000:
                return resp.text
            logger.warning(f"  curl_cffi attempt {attempt+1} got status {resp.status_code} for {url}")
            if attempt < retries:
                time.sleep(3)
    except Exception as e:
        logger.warning(f"  curl_cffi failed for {url}: {e}")
    return None


def _fetch_pmc_mirror(url: str, doc_id: str, title: str = "") -> str | None:
    """
    Look up PubMed Central mirror for an academic paper via Semantic Scholar API,
    then fetch the full text from PMC. Falls back to title-based search if no DOI.
    """
    try:
        from curl_cffi import requests as curl_requests

        # Try DOI-based lookup first
        doi = _extract_doi(url)
        ext_ids = None
        if doi:
            api_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=externalIds"
            resp = curl_requests.get(api_url, timeout=15)
            if resp.status_code == 200:
                ext_ids = resp.json().get("externalIds", {})

        # Fallback: title-based search
        if not ext_ids and title:
            search_url = (
                "https://api.semanticscholar.org/graph/v1/paper/search"
                f"?query={requests.utils.quote(title)}&fields=externalIds&limit=1"
            )
            time.sleep(1)  # rate-limit courtesy
            resp = curl_requests.get(search_url, timeout=15)
            if resp.status_code == 200:
                data_list = resp.json().get("data", [])
                if data_list:
                    ext_ids = data_list[0].get("externalIds", {})

        if not ext_ids:
            logger.info(f"  No Semantic Scholar result for {doc_id}")
            return None

        pmc_id = ext_ids.get("PubMedCentral")
        if not pmc_id:
            logger.info(f"  No PMC mirror found for {doc_id}")
            return None

        pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/"
        logger.info(f"  Found PMC mirror for {doc_id}: PMC{pmc_id}")

        pmc_html = curl_requests.get(pmc_url, impersonate="chrome", timeout=REQUEST_TIMEOUT)
        if pmc_html.status_code == 200:
            return pmc_html.text

    except Exception as e:
        logger.warning(f"  PMC mirror lookup failed for {doc_id}: {e}")
    return None


def _extract_text_trafilatura(html: str, url: str) -> str | None:
    """Use trafilatura to extract main article text from HTML."""
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=False,
        favor_recall=True,
        deduplicate=True,
    )
    if text and len(text) >= MIN_TEXT_LENGTH:
        return text.strip()
    return None


def _domain_of(url: str) -> str:
    """Extract the base domain from a URL."""
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else ""


def _needs_curl_cffi(url: str) -> bool:
    domain = _domain_of(url)
    return any(d in domain for d in _CURL_CFFI_DOMAINS)


def _needs_pmc_mirror(url: str) -> bool:
    domain = _domain_of(url)
    return any(d in domain for d in _PMC_MIRROR_DOMAINS)


def _is_pmc_url(url: str) -> bool:
    return "pmc.ncbi.nlm.nih.gov" in url


def _extract_pmc_id(url: str) -> str | None:
    """Extract the PMC ID (e.g. 'PMC12202002') from a PMC URL."""
    m = re.search(r"(PMC\d+)", url)
    return m.group(1) if m else None


def _fetch_pmc_bioc_api(url: str, doc_id: str = "") -> str | None:
    """
    Fetch full-text from PMC via NCBI's BioC JSON API.
    This bypasses captcha/rate-limit pages that block normal HTML fetches.
    Only works for open-access PMC articles.

    Returns cleaned article text or None if the API call fails.
    """
    pmc_id = _extract_pmc_id(url)
    if not pmc_id:
        return None

    bioc_url = (
        f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
        f"pmcoa.cgi/BioC_json/{pmc_id}/unicode"
    )
    try:
        resp = requests.get(bioc_url, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"  BioC API returned {resp.status_code} for {doc_id}")
            return None

        data = _json.loads(resp.text)

        # BioC format: list[0]['documents'][0]['passages']
        if not isinstance(data, list) or len(data) == 0:
            return None
        documents = data[0].get("documents", [])
        if not documents:
            return None
        passages = documents[0].get("passages", [])

        # Skip reference, supplementary, author contributions, competing interests,
        # acknowledgements sections — keep article body only
        skip_sections = {"REF", "SUPPL", "AUTH_CONT", "COMP_INT", "ACK_FUND"}
        texts = []
        for passage in passages:
            section_type = passage.get("infons", {}).get("section_type", "")
            text = passage.get("text", "").strip()
            if text and section_type not in skip_sections:
                texts.append(text)

        full_text = "\n\n".join(texts)
        if len(full_text) >= MIN_TEXT_LENGTH:
            logger.info(
                f"  ✓ {doc_id}: {len(full_text):,} chars via BioC API"
            )
            return full_text
        else:
            logger.warning(
                f"  BioC API returned only {len(full_text)} chars for {doc_id}"
            )
            return None

    except Exception as e:
        logger.warning(f"  BioC API failed for {doc_id}: {e}")
        return None


def fetch_article(url: str, doc_id: str = "", title: str = "") -> dict:
    """
    Fetch a single article URL and extract clean text.
    Uses a multi-strategy cascade:
      1. Standard requests + trafilatura
      2. curl_cffi with Chrome TLS fingerprint (for anti-bot sites)
      3. PMC mirror via Semantic Scholar (for paywalled ACS/Elsevier)
      4. PMC rate-limit retry after delay

    Returns:
        {
            "url": str,
            "doc_id": str,
            "text": str | None,
            "status": "success" | "partial" | "failed",
            "char_count": int,
            "method": str,
            "error": str | None,
        }
    """
    result = {
        "url": url,
        "doc_id": doc_id,
        "text": None,
        "status": "failed",
        "char_count": 0,
        "method": "none",
        "error": None,
    }

    fetch_url = _normalise_pmc_url(url)
    logger.info(f"  Fetching {doc_id}: {fetch_url}")

    # ── Strategy 1: Standard requests ─────────────────────────────────
    html = _fetch_html(fetch_url)
    text = None

    if html:
        text = _extract_text_trafilatura(html, fetch_url)

    if text:
        result["text"] = text
        result["char_count"] = len(text)
        result["method"] = "trafilatura"
        result["status"] = "success" if len(text) >= 2000 else "partial"
        return result

    # ── Strategy 1b: PMC rate-limit retry ─────────────────────────────
    # If we got HTML (200) from PMC but extraction failed, it may be a
    # rate-limited thin page. Wait and retry once.
    if html and _is_pmc_url(fetch_url):
        logger.info(f"  {doc_id}: PMC extraction failed — retrying after 5s delay...")
        time.sleep(5)
        html = _fetch_html(fetch_url)
        if html:
            text = _extract_text_trafilatura(html, fetch_url)
            if text:
                result["text"] = text
                result["char_count"] = len(text)
                result["method"] = "trafilatura+pmc_retry"
                result["status"] = "success" if len(text) >= 2000 else "partial"
                return result

    # ── Strategy 1c: NCBI BioC JSON API (PMC only) ───────────────────
    # Bypasses captcha/rate-limit pages by fetching structured full-text
    # directly from NCBI's open-access BioC API.
    if _is_pmc_url(fetch_url):
        logger.info(f"  {doc_id}: Trying NCBI BioC JSON API...")
        bioc_text = _fetch_pmc_bioc_api(fetch_url, doc_id)
        if bioc_text:
            result["text"] = bioc_text
            result["char_count"] = len(bioc_text)
            result["method"] = "bioc_api"
            result["status"] = "success" if len(bioc_text) >= 2000 else "partial"
            return result

    # ── Strategy 2: curl_cffi with Chrome TLS fingerprint ─────────────
    if _needs_curl_cffi(url) or not html:
        extra_retries = 2 if "sciencedirect.com" in url else 1
        logger.info(f"  {doc_id}: Trying curl_cffi (Chrome TLS fingerprint)...")
        cffi_html = _fetch_html_curl_cffi(fetch_url, retries=extra_retries)
        if cffi_html:
            text = _extract_text_trafilatura(cffi_html, fetch_url)
            if text:
                result["text"] = text
                result["char_count"] = len(text)
                result["method"] = "curl_cffi+trafilatura"
                result["status"] = "success" if len(text) >= 2000 else "partial"
                return result

    # ── Strategy 3: PMC mirror via Semantic Scholar ───────────────────
    # Try for ACS, ScienceDirect, and any other DOI-bearing URL
    if _needs_pmc_mirror(url) or "sciencedirect.com" in url or (not text and _extract_doi(url)):
        logger.info(f"  {doc_id}: Looking up PMC mirror via Semantic Scholar...")
        pmc_html = _fetch_pmc_mirror(url, doc_id, title=title)
        if pmc_html:
            pmc_text = _extract_text_trafilatura(pmc_html, url)
            if pmc_text:
                result["text"] = pmc_text
                result["char_count"] = len(pmc_text)
                result["method"] = "pmc_mirror+trafilatura"
                result["status"] = "success" if len(pmc_text) >= 2000 else "partial"
                return result

    # All strategies exhausted
    result["error"] = (
        f"All fetch strategies failed (requests={bool(html)}, "
        f"bioc_api={'tried' if _is_pmc_url(fetch_url) else 'skipped'}, "
        f"curl_cffi={'tried' if _needs_curl_cffi(url) or not html else 'skipped'}, "
        f"pmc_mirror={'tried' if _needs_pmc_mirror(url) else 'skipped'})"
    )
    return result


def fetch_all_articles(documents: list[dict]) -> list[dict]:
    """
    Fetch all articles from their URLs using multi-strategy cascade.

    Args:
        documents: List of article metadata dicts, each with at least
                   'url' and 'doc_id' keys.

    Returns:
        List of fetch result dicts (same order as input).
    """
    results = []
    success = 0
    partial = 0
    failed = 0

    logger.info(f"[Fetcher] Starting live fetch of {len(documents)} articles...")

    for i, doc in enumerate(documents, 1):
        url = doc.get("url", "")
        doc_id = doc.get("doc_id", f"DOC-{i:03d}")
        doc_title = doc.get("title", "")
        logger.info(f"[{i}/{len(documents)}] {doc_id}: {url[:80]}")

        result = fetch_article(url, doc_id, title=doc_title)
        results.append(result)

        if result["status"] == "success":
            success += 1
            logger.info(
                f"  ✓ {doc_id}: {result['char_count']:,} chars via {result['method']}"
            )
        elif result["status"] == "partial":
            partial += 1
            logger.warning(
                f"  ~ {doc_id}: {result['char_count']:,} chars (partial) via {result['method']}"
            )
        else:
            failed += 1
            logger.error(f"  ✗ {doc_id}: {result['error']}")

        # Polite delay between requests
        if i < len(documents):
            time.sleep(1.0)

    logger.info(
        f"[Fetcher] Complete: {success} success, {partial} partial, {failed} failed"
    )
    return results
