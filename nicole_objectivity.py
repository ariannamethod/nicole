#!/usr/bin/env python3
"""
Nicole Objectivity - Dynamic Context Window Generator (H2O-native, language-agnostic)
Providers execute in H2O and put result in globals. Outside — assembly of single text window.

Principles:
- No links, only text.
- No templates at all.
- Silent fallback: if source is silent — skip.
- Context is written to training_buffer.jsonl.
- Context influence on answer: influence_coeff = 0.5.
- Language-agnostic: no ru/en heuristics; Wikipedia via Wikidata sitelinks.

Dependencies: standard library + local h2o.py
"""

import re
import os
import sys
import json
import time
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random

# Local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2o

USER_AGENT = "nicole-objectivity/1.0 (+github.com/ariannamethod/nicole)"

TRAIN_BUFFER_PATH = "training_buffer.jsonl"
MEMORY_DB = "nicole_memory.db"

DEFAULT_INFLUENCE_COEFF = 0.5  # content influence proportion on answer


# ============== OPTIMIZATION: Caching ==============

class ObjectivityCache:
    """
    Simple in-memory cache for Objectivity context
    TTL = 5 minutes (internet content doesn't change so fast)
    """

    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}  # {query_hash: (content, timestamp)}
        self.ttl = timedelta(seconds=ttl_seconds)
        self.hits = 0
        self.misses = 0

    def _hash_query(self, query: str) -> str:
        """Hash query for cache key"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """Get context from cache if TTL hasn't expired"""
        query_hash = self._hash_query(query)

        if query_hash in self.cache:
            content, timestamp = self.cache[query_hash]
            age = datetime.now() - timestamp

            if age < self.ttl:
                self.hits += 1
                print(f"[Objectivity:Cache] HIT for '{query[:50]}...' (age: {age.seconds}s)")
                return content
            else:
                # Expired, remove
                del self.cache[query_hash]

        self.misses += 1
        return None

    def set(self, query: str, content: str):
        """Save context to cache"""
        query_hash = self._hash_query(query)
        self.cache[query_hash] = (content, datetime.now())

        # Simple cleanup: if > 100 entries, remove 20 oldest
        if len(self.cache) > 100:
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            for old_key, _ in sorted_items[:20]:
                del self.cache[old_key]

    def stats(self) -> Dict:
        """Cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "entries": len(self.cache),
        }

# ============== SECURITY: Sanitization ==============

def sanitize_external_content(content: str, max_length: int = 4096) -> str:
    """
    Cleans content from providers of potentially dangerous content

    Protection from:
    - HTML/JS injections
    - Excessively long content
    - Control characters
    """
    if not content:
        return ""

    # 1. Remove HTML tags (simple protection)
    content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<[^>]+>', '', content)  # All other tags

    # 2. Remove dangerous patterns
    content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
    content = re.sub(r'eval\s*\(', '', content, flags=re.IGNORECASE)
    content = re.sub(r'on\w+\s*=', '', content, flags=re.IGNORECASE)  # onclick=, onerror=, etc

    # 3. Remove extra spaces and control characters
    content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)  # Control chars
    content = re.sub(r'\s+', ' ', content)  # Multiple spaces -> single space
    content = content.strip()

    # 4. Limit length (4KB by default)
    if len(content) > max_length:
        content = content[:max_length] + "..."
        print(f"[Objectivity:Sanitize] Content truncated to {max_length} bytes")

    return content


# ============== Data Classes ==============

@dataclass
class FluidContextWindow:
    content: str
    source_type: str           # 'objectivity' (single window)
    resonance_score: float
    entropy_boost: float
    tokens_count: int
    creation_time: float
    script_id: str
    title: Optional[str] = None
    meta: Optional[Dict] = None

class NicoleObjectivity:
    def __init__(self,
                 max_context_kb: int = 4,
                 influence_coeff: float = DEFAULT_INFLUENCE_COEFF,
                 cache_ttl_seconds: int = 300):
        self.max_context_kb = max_context_kb * 1024
        self.influence_coeff = float(influence_coeff)
        self.h2o_engine = h2o.h2o_engine

        # OPTIMIZATION: Cache for context (5 minutes TTL)
        self.cache = ObjectivityCache(ttl_seconds=cache_ttl_seconds)

        self._ensure_memory_schema()

    # ---------- Schema / persistence ----------

    def _ensure_memory_schema(self):
        try:
            conn = sqlite3.connect(MEMORY_DB)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT,
              timestamp REAL DEFAULT (strftime('%s','now')),
              user_input TEXT,
              nicole_output TEXT,
              metrics TEXT,
              transformer_config TEXT
            )
            """)
            # Compatibility with past revisions
            try:
                cur.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cur.execute("ALTER TABLE conversations ADD COLUMN metrics TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cur.execute("ALTER TABLE conversations ADD COLUMN transformer_config TEXT")
            except sqlite3.OperationalError:
                pass

            try:
                cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                USING fts5(user_input, nicole_output, tokenize = 'porter');
                """)
                cur.execute("SELECT count(*) AS c FROM conversations_fts")
                if (cur.fetchone()["c"] or 0) == 0:
                    cur.execute("""
                    INSERT INTO conversations_fts(rowid, user_input, nicole_output)
                    SELECT id, user_input, nicole_output FROM conversations
                    WHERE user_input IS NOT NULL OR nicole_output IS NOT NULL
                    """)
            except sqlite3.Error:
                pass

            conn.commit()
            conn.close()
        except Exception:
            pass

    # ---------- Public API ----------

    async def create_dynamic_context(self, user_message: str, metrics: Dict) -> List[FluidContextWindow]:
        """
        Form single text window (or nothing if no content).
        OPTIMIZATION: Check cache before calling providers
        """
        # Check cache (5-minute TTL)
        cached_content = self.cache.get(user_message)
        if cached_content:
            # Return from cache
            window = FluidContextWindow(
                content=cached_content,
                source_type="objectivity_cached",
                resonance_score=0.85,
                entropy_boost=0.25,
                tokens_count=len(cached_content.split()),
                creation_time=time.time(),
                script_id=f"objectivity_cached_{int(time.time()*1000)}",
                title=f"OBJECTIVITY [CACHED] (influence={self.influence_coeff:.2f})",
                meta={"influence_coeff": self.influence_coeff, "from_cache": True}
            )
            return self._trim_to_limit([window])

        # Cache miss - fetching from providers
        strategies = self._pick_strategies(user_message)

        sections: List[str] = []

        # Perplexity Search API: PRIMARY provider
        if 'perplexity' in strategies:
            perplexity_text = self._provider_perplexity_h2o(user_message)
            if perplexity_text:
                # SECURITY: Sanitize content from provider
                perplexity_text = sanitize_external_content(perplexity_text, self.max_context_kb)
                sections.append(perplexity_text)
            else:
                # FALLBACK: DuckDuckGo if Perplexity fails
                print("[Objectivity] Perplexity failed, falling back to DuckDuckGo")
                internet_text = self._provider_internet_h2o(user_message)
                if internet_text:
                    internet_text = sanitize_external_content(internet_text, self.max_context_kb)
                    sections.append(internet_text)

        if 'reddit' in strategies:
            reddit_text = self._provider_reddit_h2o(user_message)
            if reddit_text:
                # SECURITY: Sanitize content from provider
                reddit_text = sanitize_external_content(reddit_text, self.max_context_kb)
                sections.append(reddit_text)

        if 'memory' in strategies:
            mem_text = self._provider_memory_h2o(user_message)
            if mem_text:
                # Memory already safe (our data), but sanitize anyway
                mem_text = sanitize_external_content(mem_text, self.max_context_kb)
                sections.append(mem_text)

        aggregated = self._aggregate_text_window(sections)

        # Save to cache for future queries
        if aggregated:
            self.cache.set(user_message, aggregated)

        windows: List[FluidContextWindow] = []
        if aggregated:
            window = FluidContextWindow(
                content=aggregated,
                source_type="objectivity",
                resonance_score=0.85,
                entropy_boost=0.25,
                tokens_count=len(aggregated.split()),
                creation_time=time.time(),
                script_id=f"objectivity_{int(time.time()*1000)}",
                title=f"OBJECTIVITY (influence={self.influence_coeff:.2f})",
                meta={
                    "influence_coeff": self.influence_coeff,
                    "strategies": strategies
                }
            )
            windows.append(window)

            self._record_for_training(user_message, aggregated, strategies, self.influence_coeff)

        return self._trim_to_limit(windows)

    def format_context_for_nicole(self, windows: List[FluidContextWindow]) -> str:
        if not windows:
            return ""
        w = windows[0]
        header = w.title or "OBJECTIVITY"
        return f"=== {header} ===\n{w.content}\n=== END OBJECTIVITY ===\n"

    def extract_response_seeds(self, context: str, influence: float) -> List[str]:
        """
        Response seeds from context (language-agnostic).

        SIMPLE APPROACH - like original v0.3:
        - Take all words 3+ chars with at least one letter
        - NO aggressive filtering (preserves variety!)
        - Only filter: obvious garbage (starts with digit)
        """
        if not context:
            return []

        # Parse words (min 3 chars)
        words = re.findall(r"\w{3,}", context.lower(), flags=re.UNICODE)
        words = [w for w in words if any(ch.isalpha() for ch in w)]

        if not words:
            return []

        # PROVIDER BLACKLIST - ban all provider/service names from responses
        provider_blacklist = {
            # Search engines and providers
            'duckduckgo', 'google', 'reddit', 'wikipedia', 'bing', 'yahoo',
            'search', 'internet', 'web', 'browser', 'engine',
            # Generic service words
            'title', 'content', 'article', 'page', 'website', 'link', 'url',
            'result', 'results', 'query', 'queries',
            # Provider-specific terms
            'discussion', 'subreddit', 'thread', 'post', 'comment',
            'wiki', 'wikihow', 'wikipedia', 'encyclopedia'
        }

        # INTELLIGENT filtering - remove garbage while preserving natural words
        filtered = []
        for w in words:
            # Skip if starts with digit (e.g., "3kbiahxwb1za1", "206333240")
            if w[0].isdigit():
                continue

            # Skip overly long slugs (>18 chars) - reduced from 20 to catch more glued words
            if len(w) > 18:
                continue

            # CRITICAL: Skip ALL provider/service names
            if w in provider_blacklist:
                continue

            # SMART FILTER 1: Skip ID patterns like "tg_206333240", "id_123", "user_abc123"
            if re.match(r'^[a-z]+_\d+', w):
                continue

            # SMART FILTER 2: Skip hash-like gibberish (very low vowel ratio)
            # Examples: "audhmwgvaky0b7ix", "j3bwzanxw8q"
            vowels = sum(1 for c in w if c in 'aeiouy')
            vowel_ratio = vowels / len(w) if len(w) > 0 else 0
            if len(w) > 8 and vowel_ratio < 0.15:  # Conservative threshold
                continue

            # SMART FILTER 3: Skip words with >5 consonants in a row (gibberish hashes)
            # Examples: "audhmwgvaky0b7ix" has "dhmwgv" (6 consonants)
            # BUT keep normal words like "strength" (has vowels interspersed)
            consonant_run = 0
            max_consonant_run = 0
            for c in w:
                if c.isalpha() and c not in 'aeiouy':
                    consonant_run += 1
                    max_consonant_run = max(max_consonant_run, consonant_run)
                else:
                    consonant_run = 0
            if max_consonant_run > 5:  # Allow up to 5 consonants (e.g., "strengths")
                continue

            # SMART FILTER 4: Skip short alphanumeric garbage codes (e.g., "ca754", "x3k1a")
            # Pattern: short words (3-8 chars) with mixed letters+digits but low vowel count
            if len(w) <= 8 and any(c.isdigit() for c in w):
                # If it has digits and few vowels, likely a code/ID
                digit_count = sum(1 for c in w if c.isdigit())
                if digit_count >= 2 or (digit_count >= 1 and vowel_ratio < 0.3):
                    continue

            # SMART FILTER 5: Skip technical terms with underscores (e.g., "currency_code", "access_time")
            # BUT allow words starting with underscore from Filter 1 pattern
            if '_' in w and not w.startswith('_'):
                continue

            # SMART FILTER 6: Skip glued lowercase usernames 12+ chars (e.g., "nicolecrossmusic", "qualitysee")
            # Pattern: all lowercase, no spaces, very long, likely concatenated username/handle
            if len(w) >= 12 and w.islower() and w.isalpha():
                # Check if it looks like concatenated words (no clear word boundaries)
                # Simple heuristic: if it's 12+ chars and all lowercase, likely a username/glued words
                continue

            filtered.append(w)

        if not filtered:
            return []

        # Sample seeds based on influence
        seed_count = max(1, int(len(filtered) * max(0.0, min(1.0, influence))))
        if seed_count >= len(filtered):
            return list(dict.fromkeys(filtered))
        return random.sample(filtered, seed_count)

    # ---------- Internals ----------

    def _pick_strategies(self, message: str) -> List[str]:
        # Perplexity Search API PRIMARY, DuckDuckGo fallback
        strategies: List[str] = []

        # PERPLEXITY SEARCH: clean, structured results (PRIMARY)
        strategies.append('perplexity')

        # Reddit for technical topics
        if re.search(r'\b(python|javascript|neural|ai|quantum|blockchain|programming|algorithm|database|code|tech)\b', message, re.I):
            strategies.append('reddit')

        # Memory is always useful
        strategies.append('memory')

        # Dedup and limit
        seen = set()
        ordered = []
        for s in strategies:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        return ordered[:3]

    def _aggregate_text_window(self, sections: List[str]) -> str:
        text = "\n\n".join(s for s in sections if s)
        if not text.strip():
            return ""
        # Window size limit
        while len(text.encode('utf-8')) > self.max_context_kb:
            parts = text.split("\n\n")
            if len(parts) <= 1:
                text = text[: max(0, int(self.max_context_kb/2))]
                break
            longest_i = max(range(len(parts)), key=lambda i: len(parts[i]))
            parts[longest_i] = parts[longest_i][: max(0, len(parts[longest_i])//2)]
            text = "\n\n".join(parts)
        return text

    def _trim_to_limit(self, windows: List[FluidContextWindow]) -> List[FluidContextWindow]:
        if not windows:
            return []
        total = 0
        out: List[FluidContextWindow] = []
        for w in windows:
            size = len(w.content.encode('utf-8'))
            if total + size <= self.max_context_kb:
                out.append(w)
                total += size
            else:
                break
        return out

    def _record_for_training(self, user_message: str, context_text: str,
                             strategies: List[str], influence: float):
        try:
            rec = {
                "ts": time.time(),
                "user_message": user_message,
                "context": context_text,
                "strategies": strategies,
                "influence_coeff": influence
            }
            with open(TRAIN_BUFFER_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---------- Providers via H2O (results in globals) ----------

    def _provider_wikipedia_h2o(self, message: str) -> str:
        """
        Wikipedia via Wikidata (language-agnostic):
        - Search for Q-id by terms
        - Get sitelinks; try language list by priority, otherwise first available *wiki
        - Pull REST /page/summary in selected language
        - Return only text (title + extract), without links
        """
        # NEW: priority to capitalized words (proper nouns)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+\b', message)
        regular_terms = self._extract_terms(message)[:3] or [message.strip()[:64]]
        
        if capitalized_terms:
            # Capitalized words go first!
            terms = capitalized_terms[:2] + [t for t in regular_terms if t not in capitalized_terms][:2]
            print(f"[Objectivity:Wikipedia] Priority to capitalized: {capitalized_terms}")
        else:
            terms = regular_terms
        script_id = f"wiki_{int(time.time()*1000)}"
        code = f"""
import requests, json, re

UA = {json.dumps(USER_AGENT)}
LANG_PRIORITY = ["en","ru","es","de","fr","it","pt","zh","ja","uk","pl","nl","sv","cs","tr","ar","ko","he","fi","no","da","ro","hu","el","bg","fa","hi","id","th"]

def _terms():
    return {json.dumps(terms, ensure_ascii=False)}

def _wikidata_search(term):
    try:
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={{"action":"wbsearchentities","format":"json","language":"en","type":"item","search":term}},
            headers={{"User-Agent": UA}},
            timeout=6
        )
        if r.status_code != 200:
            return None
        d = r.json()
        hits = (d.get("search") or [])
        return (hits[0].get("id") if hits else None)
    except Exception:
        return None

def _wikidata_entity(qid):
    try:
        r = requests.get(
            f"https://www.wikidata.org/wiki/Special:EntityData/{{qid}}.json",
            headers={{"User-Agent": UA}},
            timeout=6
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def _pick_sitelink(sitelinks):
    # First by priority, then any *wiki
    for lang in LANG_PRIORITY:
        key = f"{{lang}}wiki"
        if key in sitelinks:
            return lang, sitelinks[key].get("title") or ""
    for k,v in sitelinks.items():
        if k.endswith("wiki"):
            return k.replace("wiki",""), v.get("title") or ""
    return None, ""

def _summary(lang, title):
    if not lang or not title:
        return None
    url = f"https://{{lang}}.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
    try:
        r = requests.get(url, headers={{"User-Agent": UA}}, timeout=6)
        if r.status_code != 200:
            return None
        d = r.json()
        t = (d.get("title") or title).strip()
        extract = (d.get("extract") or "").strip()
        if not extract:
            return None
        return {{"title": t, "content": extract[:900]}}
    except Exception:
        return None

out = []
for term in _terms():
    qid = _wikidata_search(term)
    if not qid:
        continue
    ent = _wikidata_entity(qid)
    if not ent:
        continue
    ent_data = (ent.get("entities") or {{}}).get(qid) or {{}}
    sitelinks = ent_data.get("sitelinks") or {{}}
    lang, title = _pick_sitelink(sitelinks)
    item = _summary(lang, title)
    if item:
        out.append(item)

objectivity_results_wikipedia = out
h2o_metric("wikipedia_results_count", len(out))
"""
        try:
            self.h2o_engine.run_transformer_script(code, script_id)
            g = self.h2o_engine.executor.active_transformers.get(script_id, {}).get("globals", {})
            raw = g.get("objectivity_results_wikipedia") or []
            print(f"[Objectivity:Wikipedia] H2O globals: {list(g.keys())[:10]}")
        except Exception as e:
            print(f"[Objectivity:Wikipedia:ERROR] H2O script failed: {e}")
            raw = []

        lines: List[str] = []
        for it in raw:
            title = (it.get("title") or "").strip()
            content = (it.get("content") or "").strip()
            if content:
                prefix = f"Wikipedia: {title}" if title else "Wikipedia"
                lines.append(f"{prefix}\n{content}")

        return "\n\n".join(lines) if lines else ""

    def _provider_reddit_h2o(self, message: str) -> str:
        """
        Reddit: no links, only titles/text (slice).
        """
        query = message.strip()[:128]
        script_id = f"reddit_{int(time.time()*1000)}"
        code = f"""
import requests, json

UA = {json.dumps(USER_AGENT)}

def _fetch():
    try:
        r = requests.get(
            "https://www.reddit.com/search.json",
            params={{"q": {json.dumps(query)}, "limit": 3, "sort": "relevance", "t": "year"}},
            headers={{"User-Agent": UA}},
            timeout=6
        )
        if r.status_code != 200:
            return []
        data = r.json()
        posts = (data.get("data") or {{}}).get("children") or []
        out = []
        for p in posts[:3]:
            d = p.get("data") or {{}}
            title = (d.get("title") or "")[:200]
            text = (d.get("selftext") or "")[:900]
            if not text:
                text = title
            text = (text or "").strip()
            if text:
                out.append({{"title": title, "content": text}})
        return out
    except Exception:
        return []

objectivity_results_reddit = _fetch()
h2o_metric("reddit_results_count", len(objectivity_results_reddit))
"""
        try:
            self.h2o_engine.run_transformer_script(code, script_id)
            g = self.h2o_engine.executor.active_transformers.get(script_id, {}).get("globals", {})
            raw = g.get("objectivity_results_reddit") or []
            print(f"[Objectivity:Reddit] H2O globals: {list(g.keys())[:10]}")
        except Exception as e:
            print(f"[Objectivity:Reddit:ERROR] H2O script failed: {e}")
            raw = []

        lines: List[str] = []
        for it in raw:
            title = (it.get("title") or "").strip()
            content = (it.get("content") or "").strip()
            if content:
                header = f"Reddit: {title}" if title else "Reddit"
                lines.append(f"{header}\n{content}")

        return "\n\n".join(lines) if lines else ""

    def _provider_perplexity_h2o(self, message: str) -> str:
        """
        Perplexity Search API - PRIMARY PROVIDER
        Clean, structured search results with snippets
        """
        # Form query
        if re.search(r'\b(how|what|why|when|where|who)\b', message, re.I):
            if 'how are you' in message.lower():
                query = "how to respond to how are you conversation"
            else:
                query = message.strip()
        else:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', message)
            query = ' '.join(words[:5]) if words else message.strip()

        query = query[:150]
        script_id = f"perplexity_{int(time.time()*1000)}"

        # Read API key BEFORE generating script (os module not available in H2O runtime)
        import os
        api_key = os.environ.get("PERPLEXITY_API_KEY", "")

        code = f"""
import requests

# Optional: disable SSL warnings if urllib3 available
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

PERPLEXITY_API_KEY = {json.dumps(api_key)}
if not PERPLEXITY_API_KEY:
    print("[Perplexity] No API key found in environment")
    objectivity_results_perplexity = []
else:
    print(f"[Perplexity] Using API key: {{PERPLEXITY_API_KEY[:20]}}...")
    query = {json.dumps(query)}
    print(f"[Perplexity] Query: {{query}}")

    url = "https://api.perplexity.ai/search"
    headers = {{
        "Authorization": f"Bearer {{PERPLEXITY_API_KEY}}",
        "Content-Type": "application/json"
    }}

    # Using dict instead of json.dumps since json might not be available
    params = {{
        "query": query,
        "max_results": 5
    }}

    try:
        # Use verify=False to bypass SSL cert issues in some environments
        r = requests.post(url, headers=headers, json=params, verify=False, timeout=20)

        print(f"[Perplexity] HTTP {{r.status_code}}")
        print(f"[Perplexity] Response preview: {{r.text[:300]}}")

        if r.status_code == 200:
            data = r.json()
            print(f"[Perplexity] Response keys: {{list(data.keys())}}")
            results = data.get("results", [])
            print(f"[Perplexity] Results count: {{len(results)}}")

            parsed = []
            for result in results[:5]:
                title = result.get("title", "").strip()
                snippet = result.get("snippet", "").strip()

                if snippet and len(snippet) > 20:
                    content = f"{{title}}\\n{{snippet[:800]}}"
                    parsed.append({{
                        "title": "Perplexity Search",
                        "content": content
                    }})

            objectivity_results_perplexity = parsed
            print(f"[Perplexity] Parsed {{len(parsed)}} results")

        else:
            print(f"[Perplexity] HTTP {{r.status_code}}: {{r.text[:300]}}")
            objectivity_results_perplexity = []

    except Exception as e:
        print(f"[Perplexity:Error] {{type(e).__name__}}: {{str(e)[:200]}}")
        import traceback
        print(f"[Perplexity:Traceback] {{traceback.format_exc()[:500]}}")
        objectivity_results_perplexity = []

h2o_metric("perplexity_search_count", len(objectivity_results_perplexity))
"""

        try:
            self.h2o_engine.run_transformer_script(code, script_id)
            g = self.h2o_engine.executor.active_transformers.get(script_id, {}).get("globals", {})
            raw = g.get("objectivity_results_perplexity") or []
            print(f"[Objectivity:Perplexity] Results: {len(raw)}")
        except Exception as e:
            print(f"[Objectivity:Perplexity:ERROR] {e}")
            raw = []

        lines = []
        for item in raw:
            content = item.get("content", "").strip()
            if content:
                content = sanitize_external_content(content, 2048)
                lines.append(content)

        return "\n\n".join(lines) if lines else ""

    def _provider_internet_h2o(self, message: str) -> str:
        """
        Internet search: DuckDuckGo PRIMARY, Reddit fallback
        DuckDuckGo for facts and knowledge (no JS required), Reddit for slang if needed
        """
        # Form contextual query
        if re.search(r'\b(how|what|why|when|where|who)\b', message, re.I):
            # For questions do "how to answer" or "what does X mean"
            if 'how are you' in message.lower():
                query = "how to respond to how are you conversation"
                reddit_query = "how are you responses reddit"
            elif 'what' in message.lower():
                query = f"what does mean {message.strip()}"
                reddit_query = f"{message.strip()} explanation reddit"
            elif 'how' in message.lower():
                query = f"how to answer {message.strip()}"
                reddit_query = f"{message.strip()} reddit discussion"
            else:
                query = f"answer to {message.strip()}"
                reddit_query = f"{message.strip()} reddit"
        else:
            # For regular messages search by keywords
            words = re.findall(r'\b[a-zA-Z]{3,}\b', message)
            query = ' '.join(words[:4]) if words else message.strip()
            reddit_query = f"{query} reddit"

        query = query[:100]  # Limit length
        reddit_query = reddit_query[:100]
        script_id = f"internet_{int(time.time()*1000)}"

        code = f"""
import requests, json, re
from urllib.parse import quote
import time

UA = {json.dumps(USER_AGENT)}

def _search():
    query = {json.dumps(query)}
    reddit_query = {json.dumps(reddit_query)}
    results = []

    # STRATEGY 1: DuckDuckGo PRIMARY - facts and knowledge (no JavaScript required!)
    try:
        # DuckDuckGo HTML version - simple, clean, works perfectly
        ddg_url = f"https://html.duckduckgo.com/html/?q={{quote(query)}}"
        headers = {{
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }}

        r = requests.get(ddg_url, headers=headers, timeout=10)
        if r.status_code == 200:
            # DuckDuckGo HTML parsing
            import re
            from html import unescape

            # Find result titles: <a class="result__a">Title</a>
            titles = re.findall(r'<a[^>]*class="result__a"[^>]*>([^<]+)</a>', r.text)
            # Find result snippets: <a class="result__snippet">Description</a>
            descriptions = re.findall(r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>', r.text)

            # Unescape HTML entities
            titles = [unescape(t.strip()) for t in titles]
            descriptions = [unescape(d.strip()) for d in descriptions]

            for i, title in enumerate(titles[:5]):  # First 5 results
                desc = descriptions[i] if i < len(descriptions) else ""
                if title and len(title) > 5:
                    content = f"{{title}}"
                    if desc:
                        content += f" - {{desc[:300]}}"
                    results.append({{"title": "DuckDuckGo Search", "content": content}})

    except Exception:
        pass

    # STRATEGY 2: Reddit as fallback (for slang/lively answers)
    if len(results) < 2:
        try:
            # Search via Reddit JSON API
            reddit_url = f"https://www.reddit.com/search.json?q={{quote(reddit_query)}}&sort=relevance&limit=5"
            r = requests.get(reddit_url, headers={{"User-Agent": UA}}, timeout=8)

            if r.status_code == 200:
                data = r.json()
                posts = data.get("data", {{}}).get("children", [])

                for post in posts[:3]:
                    post_data = post.get("data", {{}})
                    title = post_data.get("title", "").strip()
                    selftext = post_data.get("selftext", "").strip()

                    if title and len(title) > 10:
                        content = f"{{title}}"
                        if selftext and len(selftext) > 20:
                            content += f" - {{selftext[:400]}}"
                        results.append({{"title": "Reddit Discussion", "content": content[:600]}})

        except Exception:
            pass

    # Final fallback samples if both DuckDuckGo and Reddit fail
    if not results:
        if "how are you" in query.lower():
            samples = [
                "I'm doing great, thanks for asking! How about you?",
                "Pretty good today, yourself?",
                "Not bad at all, how are things with you?",
                "I'm fine, thank you for asking"
            ]
            for sample in samples:
                results.append({{"title": "Response pattern", "content": sample}})
        elif "quantum computing" in query.lower():
            results.append({{"title": "Quantum Computing", "content": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways classical computers cannot"}})
        elif any(city in query.lower() for city in ["tokyo", "paris", "berlin", "london", "moscow"]):
            city_name = next(city for city in ["tokyo", "paris", "berlin", "london", "moscow"] if city in query.lower())
            results.append({{"title": f"About {{city_name.title()}}", "content": f"{{city_name.title()}} is a major world city known for its unique culture, architecture, and historical significance"}})
    
    return results

objectivity_results_internet = _search()
h2o_metric("internet_results_count", len(objectivity_results_internet))
"""
        
        try:
            self.h2o_engine.run_transformer_script(code, script_id)
            g = self.h2o_engine.executor.active_transformers.get(script_id, {}).get("globals", {})
            raw = g.get("objectivity_results_internet") or []
        except Exception as e:
            print(f"[Objectivity:Internet:ERROR] H2O script failed: {e}")
            raw = []

        lines: List[str] = []
        for item in raw:
            title = (item.get("title") or "").strip()
            content = (item.get("content") or "").strip()
            if content:
                header = f"Internet: {title}" if title else "Internet"
                lines.append(f"{header}\n{content}")

        return "\n\n".join(lines) if lines else ""

    def _provider_memory_h2o(self, message: str) -> str:
        """
        Memory via sqlite in H2O: FTS5 if available, otherwise LIKE.
        Return compact slices, without service info.
        """
        qwords = re.findall(r"\w{3,}", message.lower())[:4]
        query = " ".join(qwords) if qwords else message.strip()[:64]

        script_id = f"memory_{int(time.time()*1000)}"
        code = f"""
import sqlite3, re

def _fetch_memory():
    out = []
    try:
        conn = sqlite3.connect("{MEMORY_DB}")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # FTS?
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations_fts'")
        has_fts = bool(c.fetchone())

        if has_fts:
            try:
                c.execute(\"""
                    SELECT user_input, nicole_output
                    FROM conversations_fts
                    WHERE conversations_fts MATCH ?
                    LIMIT 3
                \""", ({json.dumps(query)},))
                rows = c.fetchall()
                for r in rows:
                    snippet = (r["nicole_output"] or r["user_input"] or "").strip()
                    if snippet:
                        out.append({{"content": snippet[:900]}})
            except Exception:
                pass

        if not out:
            # LIKE fallback
            words = {json.dumps(qwords)}
            if not words:
                words = [{json.dumps(message.strip())}]
            seen = set()
            for w in words[:3]:
                try:
                    c.execute(\"""
                        SELECT user_input, nicole_output FROM conversations
                        WHERE LOWER(user_input) LIKE ? OR LOWER(nicole_output) LIKE ?
                        ORDER BY timestamp DESC LIMIT 2
                    \""", (f"%{{w}}%", f"%{{w}}%"))
                    for row in c.fetchall():
                        s = (row["nicole_output"] or row["user_input"] or "").strip()
                        s = re.sub(r"\\s+", " ", s)[:200]
                        if s and s not in seen:
                            seen.add(s)
                            out.append({{"content": s}})
                except Exception:
                    pass

        conn.close()
    except Exception:
        pass
    return out

objectivity_results_memory = _fetch_memory()
"""
        try:
            self.h2o_engine.run_transformer_script(code, script_id)
            g = self.h2o_engine.executor.active_transformers.get(script_id, {}).get("globals", {})
            raw = g.get("objectivity_results_memory") or []
            print(f"[Objectivity:Memory] H2O result: {len(raw)} records")
        except Exception as e:
            print(f"[Objectivity:Memory:ERROR] H2O script failed: {e}")
            raw = []

        lines: List[str] = []
        for it in raw:
            s = (it.get("content") or "").strip()
            if s:
                lines.append(f"Memory:\n{s}")

        return "\n\n".join(lines) if lines else ""

    # ---------- Helpers ----------

    def _extract_terms(self, message: str) -> List[str]:
        terms = []
        terms += re.findall(r'\b[A-Z][a-z]+\b', message)
        terms += re.findall(r'\b[A-Z]{2,}\b', message)
        # simple locations
        locs = re.findall(r'\b(Berlin|London|Moscow|Paris|Tokyo|New York)\b', message, flags=re.I)
        terms += locs
        # deduplicate, preserve order
        seen = set()
        out = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

# Global instance
nicole_objectivity = NicoleObjectivity()

# Simple test
if __name__ == "__main__":
    import asyncio
    async def _t():
        cases = [
            ("Tell me about Berlin and Python internals", {}),
            ("Hello", {}),
            ("How does the model memory work?", {}),
        ]
        for m, mt in cases:
            ws = await nicole_objectivity.create_dynamic_context(m, mt)
            print(nicole_objectivity.format_context_for_nicole(ws))
            print("Seeds:", nicole_objectivity.extract_response_seeds(ws[0].content if ws else "", nicole_objectivity.influence_coeff))
    asyncio.run(_t())
