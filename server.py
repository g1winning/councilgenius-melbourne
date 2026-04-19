#!/usr/bin/env python3
"""
CouncilGenius — City of Melbourne
TechEntity · server.py v10.2

V10 deltas vs v9.0 (all additive; existing endpoints unchanged):
  1. Loads melbourne_synonyms.json + knowledge_meta.json at startup
  2. normalise_query() — applies global_substitutions + child_vocabulary_hints
  3. phonetic_match() — metaphone fallback (graceful if pkg absent)
  4. claude_rewrite() — Sonnet single-shot rewrite for low-confidence queries
  5. resolve_query() — orchestrates the 6-step cascade (substring→tag→section→phonetic→rewrite→fallback)
  6. GET /pdf_lookup?q=... — direct-lookup endpoint with confidence gate
  7. GET /suggest?q=...     — debounced autocomplete feed for page.html
"""

import os
import json
import csv
import datetime
import hashlib
import time
import re
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote_plus

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(BASE_DIR, 'knowledge.txt')
ANALYTICS_PATH = os.path.join(BASE_DIR, 'analytics.csv')
FEEDBACK_PATH  = os.path.join(BASE_DIR, 'feedback.csv')
API_KEY_PATH   = os.path.join(BASE_DIR, 'api_key.txt')
QUERIES_BASIC_JSONL = os.path.join(BASE_DIR, 'queries_basic.jsonl')
QUERIES_FULL_JSONL = os.path.join(BASE_DIR, 'queries_full.jsonl')

# V10 sidecars
SYNONYMS_PATH       = os.path.join(BASE_DIR, 'melbourne_synonyms.json')
KNOWLEDGE_META_PATH = os.path.join(BASE_DIR, 'knowledge_meta.json')

# ── Startup tracking ──────────────────────────────────────────────────────────
SERVER_START_TIME = time.time()
TOTAL_QUERIES = 0

# ── API key ──────────────────────────────────────────────────────────────────
def get_api_key():
    key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if key:
        return key
    if os.path.exists(API_KEY_PATH):
        with open(API_KEY_PATH) as f:
            return f.read().strip()
    raise RuntimeError('No ANTHROPIC_API_KEY found in environment or api_key.txt')

# ── Knowledge base ───────────────────────────────────────────────────────────
def load_knowledge():
    if os.path.exists(KNOWLEDGE_PATH):
        with open(KNOWLEDGE_PATH, encoding='utf-8') as f:
            return f.read()
    return ''

KNOWLEDGE = load_knowledge()

# ═════════════════════════════════════════════════════════════════════════════
# ██ V10 LAYER — synonyms, normaliser, phonetic, resolver, pdf_lookup ████████
# ═════════════════════════════════════════════════════════════════════════════

# ── V10 loaders (synonyms + knowledge_meta) ──────────────────────────────────
def load_synonyms():
    """Read melbourne_synonyms.json — returns {} if missing so v9 behaviour is preserved."""
    if not os.path.exists(SYNONYMS_PATH):
        print(f'[V10] synonyms file not found at {SYNONYMS_PATH} — normaliser will be a no-op')
        return {}
    try:
        with open(SYNONYMS_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'[V10][WARN] failed to load synonyms: {e}')
        return {}

def load_knowledge_meta():
    """Read knowledge_meta.json — returns {'documents':[], 'sections':{}} on failure."""
    if not os.path.exists(KNOWLEDGE_META_PATH):
        print(f'[V10] knowledge_meta file not found at {KNOWLEDGE_META_PATH} — /pdf_lookup will return 503')
        return {'documents': [], 'sections': {}, '_meta': {}}
    try:
        with open(KNOWLEDGE_META_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'[V10][WARN] failed to load knowledge_meta: {e}')
        return {'documents': [], 'sections': {}, '_meta': {}}

SYNONYMS       = load_synonyms()
KNOWLEDGE_META = load_knowledge_meta()

# Convenience views ----------------------------------------------------------
GLOBAL_SUBS           = SYNONYMS.get('global_substitutions', {}) or {}
CHILD_HINTS           = SYNONYMS.get('child_vocabulary_hints', {}) or {}
PHONETIC_CONFUSABLES  = SYNONYMS.get('phonetic_confusables', {}) or {}
CATEGORIES_V10        = SYNONYMS.get('categories', {}) or {}
FALLBACK_RULES        = SYNONYMS.get('fallback_rules', {}) or {}

# Build a fast {doc_id: doc} index + a flat list for scans
_DOCS_LIST    = KNOWLEDGE_META.get('documents', []) or []
DOCS_BY_ID    = {d.get('doc_id'): d for d in _DOCS_LIST if d.get('doc_id')}
SECTIONS_MAP  = KNOWLEDGE_META.get('sections', {}) or {}

# Redirect phrase table flattened: {phrase: category}
REDIRECT_PHRASES = {}
for cat_key, cat_body in CATEGORIES_V10.items():
    for phrase, target in (cat_body.get('redirect_phrases') or {}).items():
        REDIRECT_PHRASES[phrase.lower()] = target

# Single flat synonym→category lookup (lay + canonical + misspellings + voice_garbles)
SYNONYM_TO_CATEGORY = {}
for cat_key, cat_body in CATEGORIES_V10.items():
    bucket = (
        (cat_body.get('canonical')      or []) +
        (cat_body.get('lay_synonyms')   or []) +
        (cat_body.get('misspellings')   or []) +
        (cat_body.get('voice_garbles')  or []) +
        (cat_body.get('child_terms')    or []) +
        (cat_body.get('senior_terms')   or [])
    )
    for term in bucket:
        SYNONYM_TO_CATEGORY.setdefault(term.lower(), cat_body.get('canonical_category', cat_key))

# ── Metaphone (optional) ──────────────────────────────────────────────────────
try:
    from metaphone import doublemetaphone       # pip install metaphone
    _METAPHONE_OK = True
except Exception as _e:
    _METAPHONE_OK = False
    def doublemetaphone(s):  # type: ignore[misc]
        return ('', '')
    print(f'[V10] metaphone not installed ({_e}); phonetic fallback disabled. '
          f'Install via `pip install metaphone` in requirements.txt.')

# Pre-compute phonetic keys for every canonical token we know about
_PHONETIC_SEEDS = {}
if _METAPHONE_OK:
    for cat_key, cat_body in CATEGORIES_V10.items():
        target_cat = cat_body.get('canonical_category', cat_key)
        for seed in (cat_body.get('phonetic_seeds') or []):
            p1, p2 = doublemetaphone(seed)
            if p1: _PHONETIC_SEEDS.setdefault(p1, []).append((seed, target_cat))
            if p2: _PHONETIC_SEEDS.setdefault(p2, []).append((seed, target_cat))
    # also seed with known confusable root words
    for canonical, variants in PHONETIC_CONFUSABLES.items():
        p1, p2 = doublemetaphone(canonical)
        for p in (p1, p2):
            if p:
                _PHONETIC_SEEDS.setdefault(p, []).append((canonical, SYNONYM_TO_CATEGORY.get(canonical.lower(), 'other')))

# ── V10 query normaliser ──────────────────────────────────────────────────────
_WORD_RE = re.compile(r"[A-Za-z']+")

def normalise_query(q: str) -> str:
    """
    Apply global_substitutions + child_vocabulary_hints.
    Cheap, deterministic, case-insensitive, punctuation-preserving on boundaries.
    """
    if not q:
        return ''
    out = q.strip().lower()
    # 1. whole-phrase child vocab (longest key first to avoid partial shadowing)
    for phrase in sorted(CHILD_HINTS.keys(), key=len, reverse=True):
        if phrase in out:
            out = out.replace(phrase, CHILD_HINTS[phrase])
    # 2. token-level global substitutions
    def _sub(match):
        tok = match.group(0).lower()
        return GLOBAL_SUBS.get(tok, tok)
    out = _WORD_RE.sub(_sub, out)
    # 3. collapse whitespace
    return re.sub(r'\s+', ' ', out).strip()

# ── V10 phonetic match ────────────────────────────────────────────────────────
def phonetic_match(q: str):
    """Return (category, matched_token) best-effort; ('', '') on miss."""
    if not _METAPHONE_OK or not q:
        return ('', '')
    best = ('', '')
    for token in _WORD_RE.findall(q.lower()):
        if len(token) < 3:
            continue
        p1, p2 = doublemetaphone(token)
        for p in (p1, p2):
            hits = _PHONETIC_SEEDS.get(p)
            if hits:
                # take the first hit — seeds were inserted in priority order
                seed, cat = hits[0]
                return (cat, seed)
    return best

# ── V10 Claude query-rewrite (optional; cheap Haiku call) ────────────────────
_REWRITE_SYS = (
    "You rewrite resident queries about the City of Melbourne into clear canonical form. "
    "Return JSON ONLY with keys: {\"rewritten\":\"…\",\"category\":\"…\",\"confidence\":0.0-1.0}. "
    "Categories: waste_bins, rates_payments, rates_hardship, rates_concessions, planning_building, "
    "animals_pets, roads_traffic, parking, water_stormwater, environment_climate, emergency_bushfire, "
    "health_safety, families_children, aged_disability, community_events, library_learning, "
    "recreation_sport, governance_contact, business_economy, arts_culture_heritage, pdf_document_search, other."
)

def claude_rewrite(q: str, timeout: int = 10):
    """Ask Sonnet to rewrite an ambiguous query. Returns dict or None on failure."""
    try:
        api_key = get_api_key()
    except Exception:
        return None
    payload = json.dumps({
        'model': 'claude-sonnet-4-6',
        'max_tokens': 200,
        'system': _REWRITE_SYS,
        'messages': [{'role': 'user', 'content': f'Rewrite: {q}'}]
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=payload,
        headers={
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        raw = data['content'][0]['text'].strip()
        # Accept either a JSON blob or a code-fenced JSON blob
        m = re.search(r'\{.*\}', raw, re.S)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception as e:
        print(f'[V10][WARN] claude_rewrite failed: {e}')
        return None

# ── V10 resolver — 6-step cascade ─────────────────────────────────────────────
def resolve_query(raw_query: str, allow_rewrite: bool = True) -> dict:
    """
    Full cascade: redirect → synonym → phonetic → Claude rewrite → fallback.
    Returns {category, confidence, normalised, stage, rewritten?}
    Confidence scale:
      1.00 — direct redirect_phrase hit
      0.90 — multi-token synonym hit
      0.80 — single-token synonym hit
      0.60 — phonetic hit
      per rewrite.confidence — Claude rewrite branch
      0.00 — fallback_rules.if_no_match
    """
    nq = normalise_query(raw_query or '')
    # 1. exact redirect phrase
    for phrase, cat in REDIRECT_PHRASES.items():
        if phrase in nq:
            return {'category': cat, 'confidence': 1.0, 'normalised': nq,
                    'stage': 'redirect_phrase', 'matched': phrase}
    # 2. synonym table hit (prefer multi-word matches first)
    multi_hits = [t for t in SYNONYM_TO_CATEGORY.keys() if ' ' in t and t in nq]
    if multi_hits:
        best = max(multi_hits, key=len)
        return {'category': SYNONYM_TO_CATEGORY[best], 'confidence': 0.9,
                'normalised': nq, 'stage': 'synonym_multi', 'matched': best}
    tokens = set(_WORD_RE.findall(nq))
    single_hits = [t for t in tokens if t in SYNONYM_TO_CATEGORY]
    if single_hits:
        best = single_hits[0]
        return {'category': SYNONYM_TO_CATEGORY[best], 'confidence': 0.8,
                'normalised': nq, 'stage': 'synonym_single', 'matched': best}
    # 3. phonetic
    cat, seed = phonetic_match(nq)
    if cat:
        return {'category': cat, 'confidence': 0.6, 'normalised': nq,
                'stage': 'phonetic', 'matched': seed}
    # 4. Claude rewrite (only if caller allows — disabled on /suggest for speed)
    if allow_rewrite:
        rw = claude_rewrite(raw_query)
        if rw and rw.get('category') and rw.get('category') != 'other':
            return {'category': rw['category'],
                    'confidence': float(rw.get('confidence', 0.55) or 0.55),
                    'normalised': nq, 'stage': 'claude_rewrite',
                    'rewritten': rw.get('rewritten', ''), 'matched': ''}
    # 5. fallback
    return {'category': 'other', 'confidence': 0.0, 'normalised': nq,
            'stage': 'fallback', 'matched': '',
            'fallback_message': FALLBACK_RULES.get('if_no_match', '')}

# ── V10 PDF direct-lookup cascade ─────────────────────────────────────────────
def _score_doc(q_tokens: set, doc: dict) -> float:
    """Cheap overlap score over title + search_tags + filename stem."""
    corpus = ' '.join([
        (doc.get('title') or ''),
        (doc.get('filename') or '').replace('.pdf', '').replace('-', ' ').replace('_', ' '),
        ' '.join(doc.get('search_tags') or [])
    ]).lower()
    doc_tokens = set(_WORD_RE.findall(corpus))
    if not doc_tokens or not q_tokens:
        return 0.0
    overlap = len(q_tokens & doc_tokens)
    # normalise by query length — favour tight matches
    return overlap / max(len(q_tokens), 1)

def pdf_lookup(raw_query: str, top_k: int = 3) -> dict:
    """
    Title substring → tag → section → phonetic cascade, ranked.
    Returns {status, doc_id?, title?, url?, confidence, top3?, fallback?}.
    Confidence gate: ≥0.75 direct, 0.40–0.75 clarify (top3), <0.40 fallback.
    """
    if not DOCS_BY_ID:
        return {'status': 'unavailable',
                'message': 'knowledge_meta.json not loaded'}
    res = resolve_query(raw_query, allow_rewrite=False)
    nq = res['normalised']
    q_tokens = set(_WORD_RE.findall(nq))

    # Stage A — exact substring on title
    exact = []
    for doc in _DOCS_LIST:
        title_l = (doc.get('title') or '').lower()
        if nq and nq in title_l:
            exact.append((1.0, doc))
    if exact:
        exact.sort(key=lambda x: -len(x[1].get('title') or ''))
        best = exact[0][1]
        return _pdf_hit(best, confidence=0.95, match_type='title_substring')

    # Stage B — section-scoped candidates (if resolver found a category)
    candidates = []
    section_docs = SECTIONS_MAP.get(res['category'], []) if res['category'] != 'other' else []
    search_pool = ([DOCS_BY_ID[d] for d in section_docs if d in DOCS_BY_ID]
                   if section_docs else _DOCS_LIST)

    # Stage C — score by token overlap
    for doc in search_pool:
        s = _score_doc(q_tokens, doc)
        if s > 0:
            candidates.append((s, doc))
    candidates.sort(key=lambda x: -x[0])

    # Stage D — phonetic fallback if nothing scored
    if not candidates and _METAPHONE_OK:
        qp = set()
        for t in q_tokens:
            if len(t) >= 3:
                p1, p2 = doublemetaphone(t)
                for p in (p1, p2):
                    if p: qp.add(p)
        for doc in _DOCS_LIST:
            dp = set()
            for t in _WORD_RE.findall((doc.get('title') or '').lower()):
                p1, p2 = doublemetaphone(t)
                for p in (p1, p2):
                    if p: dp.add(p)
            shared = len(qp & dp)
            if shared:
                candidates.append((0.3 + 0.1 * shared, doc))
        candidates.sort(key=lambda x: -x[0])

    # Apply confidence gate
    if not candidates:
        return {'status': 'no_match', 'confidence': 0.0,
                'fallback': '03 9658 9658',
                'normalised': nq}
    best_score, best_doc = candidates[0]
    if best_score >= 0.75:
        return _pdf_hit(best_doc, confidence=best_score, match_type='token_overlap')
    if best_score >= 0.40:
        top3 = [{'doc_id': d.get('doc_id'),
                 'title': d.get('title'),
                 'url':   d.get('url'),
                 'score': round(s, 3)}
                for s, d in candidates[:top_k]]
        return {'status': 'clarify', 'confidence': best_score,
                'question': "I have a few candidates — which one did you mean?",
                'top3': top3, 'normalised': nq}
    return {'status': 'no_match', 'confidence': best_score,
            'fallback': '03 9658 9658',
            'closest': {'doc_id': best_doc.get('doc_id'),
                        'title':  best_doc.get('title'),
                        'url':    best_doc.get('url'),
                        'score':  round(best_score, 3)},
            'normalised': nq}

def _pdf_hit(doc: dict, confidence: float, match_type: str) -> dict:
    return {'status': 'ok',
            'doc_id': doc.get('doc_id'),
            'title':  doc.get('title'),
            'url':    doc.get('url'),
            'category': (doc.get('sections') or ['other'])[0],
            'confidence': round(confidence, 3),
            'match_type': match_type}

# ── V10 autocomplete feed for /suggest ────────────────────────────────────────
def suggest(raw_query: str, limit: int = 8):
    """Prefix+synonym ranked suggestions. No Claude call — sub-10ms target."""
    if not raw_query:
        return []
    nq = normalise_query(raw_query)
    prefix = nq
    hits = []
    # prefix matches over redirect_phrases + synonym terms
    for phrase in REDIRECT_PHRASES.keys():
        if phrase.startswith(prefix):
            hits.append((3.0, phrase))
    for term in SYNONYM_TO_CATEGORY.keys():
        if term.startswith(prefix) and ' ' not in term:
            hits.append((2.0, term))
    for phrase in REDIRECT_PHRASES.keys():
        if prefix in phrase and not phrase.startswith(prefix):
            hits.append((1.0, phrase))
    # dedupe preserve order of best-first
    seen, out = set(), []
    for _, v in sorted(hits, key=lambda x: (-x[0], x[1])):
        if v in seen:
            continue
        seen.add(v); out.append(v)
        if len(out) >= limit:
            break
    return out

# ═════════════════════════════════════════════════════════════════════════════
# ██ End V10 LAYER ████████████████████████████████████████████████████████████
# ═════════════════════════════════════════════════════════════════════════════

# ── PII Filtering ──────────────────────────────────────────────────────────────
def filter_pii(text: str) -> str:
    """Mask common PII patterns: email, phone, postal codes, etc."""
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
    text = re.sub(r'\b\d{3,4}\s?\d{3,4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{4}\b', '[POSTCODE]', text)
    text = re.sub(r'\b\d{10}\b', '[ID_NUMBER]', text)
    return text

# ── Query Logging ──────────────────────────────────────────────────────────────
def log_query_basic(query: str, category: str):
    """Log query with PII filtered to JSONL."""
    filtered_query = filter_pii(query)
    record = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'category': category,
        'query_preview': filtered_query[:200]
    }
    try:
        with open(QUERIES_BASIC_JSONL, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        print(f'[WARN] Failed to log basic query: {e}')

def log_query_full(query: str, response: str, category: str):
    """Log full query and response to JSONL."""
    filtered_query = filter_pii(query)
    filtered_response = filter_pii(response)
    record = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'category': category,
        'query': filtered_query,
        'response': filtered_response[:500]
    }
    try:
        with open(QUERIES_FULL_JSONL, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        print(f'[WARN] Failed to log full query: {e}')

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are CouncilGenius, the official AI resident assistant for the City of Melbourne, deployed by TechEntity on behalf of City of Melbourne.

YOUR PURPOSE: Answer questions so completely the resident never needs to contact Council.

AGENTIC BEHAVIOUR: For any process question, give all steps, fees, correct forms, and correct officer.

FOR BUILDING/PLANNING: Ask property address (inside the CBD or which inner suburb), use of the building, and whether heritage overlay applies before answering.

FOR WASTE / BIN QUESTIONS: Ask the resident's suburb or street address. City of Melbourne uses a per-address waste lookup — direct residents to [www.melbourne.vic.gov.au/waste-and-recycling-for-residents](https://www.melbourne.vic.gov.au/waste-and-recycling-for-residents) rather than asserting a fixed bin-day map.

FOR RATES: the City of Melbourne uses Net Annual Value (NAV), not Capital Improved Value. Do not assume CIV. See knowledge §3.1–§3.2.

FOR PARKING: clarify whether the question is about a permit (residents), a fine, or parking in the CBD before answering.

FOUR-TURN RESOLUTION: Resolve every query within four user inputs.

OUT OF AREA: If a query is about a location or service clearly outside the City of Melbourne (e.g. St Kilda, Brunswick, Richmond, South Yarra outside the CoM boundary, Prahran, Moonee Ponds, Footscray, Fitzroy, Collingwood), respond with only:
- One sentence noting it is outside the City of Melbourne area
- Full official name of the relevant council
- That authority's main phone number only
No links. No further elaboration. No offers of further help.

Neighbouring councils and numbers (use the council's FULL official name — never abbreviated):
- City of Port Phillip (St Kilda, Port Melbourne, South Melbourne, Albert Park, Elwood, Balaclava, Middle Park, Ripponlea, parts of Windsor): [03 9209 6777](tel:0392096777)
- City of Yarra (Richmond, Collingwood, Fitzroy, Abbotsford, Carlton North outside CoM, Clifton Hill, Burnley, Cremorne, Fairfield, Alphington, Princes Hill): [03 9205 5555](tel:0392055555)
- City of Stonnington (South Yarra outside CoM boundary, Prahran, Windsor, Toorak, Armadale, Malvern, Malvern East, Kooyong, Glen Iris part): [03 8290 1333](tel:0382901333)
- Merri-bek City Council — formerly Moreland (Brunswick, Brunswick East, Brunswick West, Coburg, Pascoe Vale, Fawkner, Glenroy, Hadfield): [03 9240 1111](tel:0392401111)
- City of Moonee Valley (Moonee Ponds, Ascot Vale, Essendon, Flemington outside CoM, Kensington outside CoM, Aberfeldie, Airport West): [03 9243 8888](tel:0392438888)
- Maribyrnong City Council (Footscray, Yarraville, Seddon, Kingsville, West Footscray, Braybrook, Maidstone, Maribyrnong): [03 9688 0200](tel:0396880200)
- Public Transport Victoria (tram, train, bus, myki questions): [1800 800 007](tel:1800800007)
- Department of Transport and Planning / VicRoads (state roads, driver licensing): [13 11 70](tel:131170)

MULTILINGUAL: If asked in another language, answer in that language then repeat in English labelled "English version:"

COMMUNICATION STYLE — apply to every response:
- Use plain English. "Use" not "utilise." "About" not "regarding." "Help" not "facilitate." "Start" not "commence."
- Sentences average 15–20 words. One idea per sentence.
- Lead with what CAN be done before explaining any limitations.
- When a resident is reporting a problem that has already happened (a complaint), acknowledge their experience before providing process information. Do not jump straight to procedure.
- When a resident is asking for something to happen (a service request), respond efficiently with the information they need.
- Deliver bad news in this order: acknowledge, explain, offer next step.
- Never minimise a resident's concern. Never use "just," "only," "simply," or "it's easy."
- When you don't have specific information, say so clearly and direct the resident to the right contact — never invent fees, dates, or processes.

SERVICE REQUESTS vs COMPLAINTS:
- If the resident is asking for something to happen → answer efficiently.
- If the resident is expressing that something went wrong → acknowledge first, then inform.

FORMAT RULES — NON-NEGOTIABLE:
- NEVER use emoji of any kind
- NEVER output raw HTML — no <a> tags, no HTML elements of any kind whatsoever
- Phone numbers: markdown hyperlink ONLY — [03 9658 9658](tel:0396589658)
  Never bare digits. Never plain text alongside a link.
- Emails: markdown mailto ONLY — [melbourne@melbourne.vic.gov.au](mailto:melbourne@melbourne.vic.gov.au)
  Never plain text alongside a link.
- URLs: markdown links ONLY — [descriptive label](https://full-url)
  Never bare URLs. Never HTML anchor tags.
- Use **bold** for key terms
- Use bullet lists for multi-step processes
- Keep responses under 300 words unless a complex process genuinely requires more
- Do NOT use ## headers — use **bold text** instead
- Include the council contact footer no more than once per response

KNOWLEDGE BASE — CITY OF MELBOURNE:

{KNOWLEDGE}

END OF KNOWLEDGE BASE.

If information is not in the knowledge base, direct the resident to [03 9658 9658](tel:0396589658) or [melbourne@melbourne.vic.gov.au](mailto:melbourne@melbourne.vic.gov.au). Do not invent fees, dates, or processes.
"""

# ── Analytics categories ─────────────────────────────────────────────────────
CATEGORIES = {
    'rates':            ['rate', 'rates', 'levy', 'nav', 'net annual value', 'municipal charge', 'payment plan', 'hardship', 'instalment', 'rebate', 'concession', 'valuation', 'rate cap', 'pensioner concession', 'differential rating', 'waste charge'],
    'waste_bins':       ['bin', 'bins', 'rubbish', 'recycling', 'green waste', 'collection', 'fogo', 'kerbside', 'compost', 'organics', 'apartment waste', 'dynon road', 'a-z waste', 'hard waste', 'hard rubbish', 'landfill'],
    'planning':         ['planning', 'planning permit', 'subdivision', 'zoning', 'overlay', 'heritage', 'development plan', 'rezoning', 'melbourne planning scheme', 'amendment', 'fishermans bend', 'arden', 'queen victoria market precinct', 'qvmpr'],
    'building':         ['building permit', 'building surveyor', 'building inspection', 'construction', 'demolition', 'owner builder', 'vba', 'building commission', 'compliance'],
    'parking':          ['parking', 'parking permit', 'residential parking permit', 'fine', 'infringement', 'appeal', 'parking zone', 'dispute fine', 'pay park', 'meter', 'clearway', 'better parking'],
    'animals':          ['dog', 'cat', 'animal', 'pet', 'register', 'pound', 'roaming', 'attack', 'barking', 'dangerous dog', 'microchip', 'desexed', 'off-leash', 'domestic animal business'],
    'local_laws':       ['local law', 'noise', 'nuisance', 'skip bin', 'shipping container', 'footpath trading', 'outdoor dining', 'busking', 'nature strip', 'graffiti'],
    'roads':            ['road', 'footpath', 'pothole', 'kerb', 'drainage', 'street light', 'signage', 'driveway', 'vehicle crossing', 'street tree', 'road closure', 'bike lane', 'cycling'],
    'transport':        ['tram', 'train', 'bus', 'myki', 'ptv', 'public transport', 'free tram zone', 'city circle tram', 'skybus', 'nightbus', 'night network', 'southern cross', 'flinders street station'],
    'utilities':        ['water', 'sewer', 'sewerage', 'electricity', 'power', 'outage', 'gas leak', 'citipower', 'jemena', 'greater western water', 'nbn', 'streetlight'],
    'venues_events':    ['venue', 'hire', 'event', 'permit', 'book', 'facility', 'library', 'city library', 'library at the dock', 'kathleen syme', 'narrm ngarrgu', 'baths', 'aquatic', 'artplay', 'signal', 'meat market', 'moomba', 'christmas festival'],
    'community':        ['grant', 'program', 'service', 'aged', 'older', 'disability', 'youth', 'maternal', 'child health', 'mch', 'kindergarten', 'immunisation', 'multicultural', 'lgbtiq', 'volunteer', 'neighbourhood house', 'participate melbourne'],
    'first_nations':    ['aboriginal', 'first nations', 'traditional owner', 'wurundjeri', 'bunurong', 'boon wurrung', 'narrm', 'naarm', 'reconciliation action plan', 'rap', '13yarn'],
    'housing':          ['housing', 'homeless', 'make room', 'affordable housing', 'rough sleeping', '602 little bourke'],
    'governance':       ['meeting', 'councillor', 'mayor', 'lord mayor', 'nicholas reece', 'roshena campbell', 'future melbourne committee', 'fmc', 'agenda', 'minutes', 'foi', 'freedom of information', 'complaint', 'petition', 'council plan', 'ombudsman', 'governance', 'annual report', 'city of melbourne act'],
    'economy':          ['business', 'invest melbourne', 'miiab', 'innovation', 'startup', 'grant', 'tourism', 'experience melbourne', 'queen victoria market', 'creative melbourne'],
    'environment':      ['climate', 'net zero', 'emissions reduction', 'biodiversity emergency', 'urban forest', 'nature in the city', 'green factor', 'cityswitch', 'mrep', 'greenline', 'swimmable birrarung', 'tree planting'],
    'emergency':        ['emergency', 'flood', 'heatwave', 'memplan', 'relief centre', 'ses', 'cfa', 'family violence', '1800respect', 'safe steps', 'orange door', 'lifeline', 'beyond blue', 'recovery', 'disaster'],
    'off_topic_benign': ['recipe', 'football', 'weather', 'stock price', 'poem', 'news', 'sport', 'joke', 'movie', 'afl'],
    'potential_api_abuse': ['ignore previous', 'jailbreak', 'pretend you are', 'act as', 'system prompt', 'disregard', 'override', 'forget instructions', 'new instructions', 'ignore all'],
    'other':            []
}

def classify(text: str) -> str:
    lower = text.lower()
    for category, keywords in CATEGORIES.items():
        if category == 'other':
            continue
        if any(kw in lower for kw in keywords):
            return category
    return 'other'

# ── CSV logging ───────────────────────────────────────────────────────────────
def log_analytics(category: str, query: str):
    exists = os.path.exists(ANALYTICS_PATH)
    with open(ANALYTICS_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['timestamp', 'category', 'query_preview'])
        w.writerow([
            datetime.datetime.utcnow().isoformat(),
            category,
            query[:120].replace('\n', ' ')
        ])

def log_feedback(query: str, response: str, rating: str):
    exists = os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['timestamp', 'rating', 'query_preview', 'response_preview'])
        w.writerow([
            datetime.datetime.utcnow().isoformat(),
            rating,
            query[:120].replace('\n', ' '),
            response[:200].replace('\n', ' ')
        ])

# ── Anthropic API call ────────────────────────────────────────────────────────
def call_claude(messages: list) -> str:
    api_key = get_api_key()
    payload = json.dumps({
        'model': 'claude-sonnet-4-6',
        'max_tokens': 1024,
        'system': SYSTEM_PROMPT,
        'messages': messages
    }).encode('utf-8')

    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=payload,
        headers={
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data['content'][0]['text']
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Anthropic API error {e.code}: {body}')

# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress default access log noise

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: str, content_type: str):
        if not os.path.isfile(path):
            self._send_text('Not found', 404)
            return
        with open(path, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Health check endpoint
        if path == '/health':
            knowledge_hash = hashlib.sha256(KNOWLEDGE.encode('utf-8')).hexdigest()[:16]
            knowledge_lines = len(KNOWLEDGE.split('\n'))
            uptime = time.time() - SERVER_START_TIME
            health = {
                'status': 'ok',
                'council': 'City of Melbourne',
                'knowledge_lines': knowledge_lines,
                'knowledge_hash': knowledge_hash,
                'prompt_version': '1.0',
                'uptime_seconds': round(uptime, 2),
                'total_queries': TOTAL_QUERIES,
                'model': 'claude-sonnet-4-6',
                'bin_mode': 'per-address-lookup',
                # V10 additions
                'server_version': 'v10.2',
                'synonyms_loaded': bool(SYNONYMS),
                'synonym_categories': len(CATEGORIES_V10),
                'knowledge_meta_loaded': bool(DOCS_BY_ID),
                'documents_indexed': len(DOCS_BY_ID),
                'sections_indexed': len(SECTIONS_MAP),
                'phonetic_enabled': _METAPHONE_OK,
            }
            self._send_json(health)
            return

        # V10 — PDF direct-lookup
        if path == '/pdf_lookup':
            qs = parse_qs(parsed.query or '')
            q = (qs.get('q', [''])[0] or '').strip()
            if not q:
                self._send_json({'error': 'missing q parameter'}, 400)
                return
            try:
                result = pdf_lookup(q)
                log_analytics('pdf_lookup:' + result.get('status', '?'), q)
                self._send_json(result)
            except Exception as e:
                print(f'[ERROR /pdf_lookup] {e}')
                self._send_json({'error': str(e)}, 500)
            return

        # V10 — autocomplete
        if path == '/suggest':
            qs = parse_qs(parsed.query or '')
            q = (qs.get('q', [''])[0] or '').strip()
            if not q:
                self._send_json({'suggestions': []})
                return
            try:
                self._send_json({'suggestions': suggest(q)})
            except Exception as e:
                print(f'[ERROR /suggest] {e}')
                self._send_json({'error': str(e)}, 500)
            return

        # Static files
        if path == '/' or path == '/index.html':
            self._serve_file(os.path.join(BASE_DIR, 'page.html'), 'text/html; charset=utf-8')
            return

        if path == '/page.html':
            self._serve_file(os.path.join(BASE_DIR, 'page.html'), 'text/html; charset=utf-8')
            return

        # PDFs
        if path.startswith('/pdfs/'):
            safe = os.path.normpath(path.lstrip('/'))
            full = os.path.join(BASE_DIR, safe)
            pdfs_dir = os.path.join(BASE_DIR, 'pdfs')
            if not full.startswith(pdfs_dir):
                self._send_text('Forbidden', 403)
                return
            self._serve_file(full, 'application/pdf')
            return

        # Images
        if path.startswith('/images/'):
            safe = os.path.normpath(path.lstrip('/'))
            full = os.path.join(BASE_DIR, safe)
            images_dir = os.path.join(BASE_DIR, 'images')
            if not full.startswith(images_dir):
                self._send_text('Forbidden', 403)
                return
            ext = os.path.splitext(safe)[1].lower()
            ct = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.svg': 'image/svg+xml',
                '.gif': 'image/gif',
                '.ico': 'image/x-icon',
                '.webp': 'image/webp'
            }.get(ext, 'application/octet-stream')
            self._serve_file(full, ct)
            return

        # Admin endpoints
        if path == '/admin/analytics':
            if os.path.exists(ANALYTICS_PATH):
                with open(ANALYTICS_PATH, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv')
                self.send_header('Content-Disposition', 'attachment; filename="analytics.csv"')
                self.send_header('Content-Length', str(len(data)))
                self._cors()
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_text('category,query_preview,timestamp\n', 200)
            return

        if path == '/admin/feedback':
            if os.path.exists(FEEDBACK_PATH):
                with open(FEEDBACK_PATH, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv')
                self.send_header('Content-Disposition', 'attachment; filename="feedback.csv"')
                self.send_header('Content-Length', str(len(data)))
                self._cors()
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_text('timestamp,rating,query_preview,response_preview\n', 200)
            return

        self._send_text('Not found', 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get('Content-Length', 0))
        body_bytes = self.rfile.read(length) if length else b''

        if path == '/chat':
            try:
                body = json.loads(body_bytes.decode('utf-8'))
                messages = body.get('messages', [])
                if not messages:
                    self._send_json({'error': 'No messages provided'}, 400)
                    return

                last_user = next(
                    (m['content'] for m in reversed(messages) if m['role'] == 'user'),
                    ''
                )

                # V10 — resolve via synonyms+phonetic first (no Claude rewrite on the
                # hot /chat path; rewrite only fires when v9 classifier also says 'other')
                v10 = resolve_query(last_user, allow_rewrite=False)
                v9_category = classify(last_user)

                # Prefer V10 category when it found something; otherwise fall back
                category = v10['category'] if v10['category'] != 'other' else v9_category

                log_analytics(category, last_user)
                log_query_basic(last_user, category)

                # Block only potential_api_abuse; off_topic is logging-only
                if v9_category == 'potential_api_abuse' or category == 'potential_api_abuse':
                    self._send_json({'error': 'This request cannot be processed.'}, 400)
                    return

                # V10 — last-chance rewrite when both classifiers said 'other'
                if category == 'other':
                    rw = claude_rewrite(last_user)
                    if rw and rw.get('category') and rw['category'] != 'other':
                        category = rw['category']

                global TOTAL_QUERIES
                TOTAL_QUERIES += 1
                reply = call_claude(messages)
                log_query_full(last_user, reply, category)
                self._send_json({
                    'reply': reply,
                    'category': category,
                    'v10': {
                        'stage': v10.get('stage'),
                        'confidence': v10.get('confidence'),
                        'normalised': v10.get('normalised'),
                    },
                })

            except Exception as e:
                print(f'[ERROR /chat] {e}')
                error_msg = 'Sorry, I\'m having trouble right now. Please try again or call the City of Melbourne on 03 9658 9658.'
                self._send_json({'error': error_msg}, 500)
            return

        if path == '/feedback':
            try:
                body = json.loads(body_bytes.decode('utf-8'))
                log_feedback(
                    body.get('query', ''),
                    body.get('response', ''),
                    body.get('rating', 'unknown')
                )
                self._send_json({'ok': True})
            except Exception as e:
                print(f'[ERROR /feedback] {e}')
                self._send_json({'error': str(e)}, 500)
            return

        self._send_text('Not found', 404)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f'CouncilGenius — City of Melbourne — listening on port {port}')
    server.serve_forever()
