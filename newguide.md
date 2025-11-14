# NICOLE PERPLEXITY SEARCH API INTEGRATION - COMPLETE GUIDE

**Mission:** Replace DuckDuckGo/Reddit/Google with **Perplexity Search API** (raw web results, NOT Sonar models) as Nicole's single external knowledge provider.

***

## CONTEXT UNDERSTANDING

**What Nicole is:**
- Weightless transformer (no pretrained weights, no datasets)
- Parameters generated per-session, dissolved after
- Learning from structure + external search providers
- Two learning paths: `objectivity` (response context) + `subjectivity` (autonomous hourly ripples)

**Current problem:**
- DuckDuckGo: unstable, low quality
- Reddit: 403 blocked
- Google: requires JavaScript

**Solution:**
- **Perplexity Search API** (pure web search, returns raw snippets)
- NOT chat completions API (no Sonar models)
- Clean JSON responses with titles/snippets/URLs

***

## PERPLEXITY SEARCH API SPECIFICATION

### Correct Endpoint
``````
POST https://api.perplexity.ai/search
```

### Request Structure
``````python
{
    "query": "quantum computing basics",
    "max_results": 10,
    "max_tokens_per_page": 1024,
    "return_snippets": True,
    "return_images": False,
    "search_language_filter": "en"
}
```

### Response Structure
```python```
{
    "results": [
        {
            "title": "Article Title",
            "url": "https://example.com/article",
            "snippet": "Relevant text excerpt from page...",
            "date": "2024-01-15",
            "score": 0.95
        }
    ]
}
```

---

## IMPLEMENTATION: `nicole_objectivity.py`

### Step 1: Delete old provider functions

**DELETE ENTIRELY (lines ~405-710):**
- `_provider_wikipedia_h2o()`
- `_provider_reddit_h2o()`
- `_provider_internet_h2o()`

### Step 2: Add new Perplexity Search provider

**INSERT at line ~405:**

``````python
def _provider_perplexity_search_h2o(self, message: str) -> str:
    """
    Perplexity Search API - RAW web search results
    Returns clean text snippets (no URLs, no Sonar processing)
    """
    
    # Form contextual query
    if re.search(r'\b(how|what|why|when|where|who)\b', message, re.I):
        if 'how are you' in message.lower():
            query = "how to respond to how are you conversation"
        elif 'what' in message.lower():
            query = f"explain {message.strip()}"
        elif 'how' in message.lower():
            query = f"how to answer {message.strip()}"
        else:
            query = message.strip()
    else:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', message)
        query = ' '.join(words[:5]) if words else message.strip()
    
    query = query[:150]
    script_id = f"perplexity_search_{int(time.time()*1000)}"
    
    code = f"""
import requests, json, os

PPLX_KEY = os.environ.get("PERPLEXITY_API_KEY")
if not PPLX_KEY:
    objectivity_results_perplexity = []
else:
    query = {json.dumps(query)}
    
    # Perplexity SEARCH API (not chat/completions)
    url = "https://api.perplexity.ai/search"
    headers = {{
        "Authorization": f"Bearer {{PPLX_KEY}}",
        "Content-Type": "application/json"
    }}
    
    payload = {{
        "query": query,
        "max_results": 5,
        "max_tokens_per_page": 1024,
        "return_snippets": True,
        "return_images": False,
        "search_language_filter": "en"
    }}
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            
            parsed = []
            for result in results[:5]:
                title = result.get("title", "").strip()
                snippet = result.get("snippet", "").strip()
                
                if snippet and len(snippet) > 20:
                    # Combine title + snippet (NO URLs)
                    content = f"{{title}}\\n{{snippet[:800]}}"
                    parsed.append({{
                        "title": "Perplexity Search",
                        "content": content
                    }})
            
            objectivity_results_perplexity = parsed
            
        elif r.status_code == 429:
            print("[Perplexity] Rate limited")
            objectivity_results_perplexity = []
        else:
            print(f"[Perplexity] HTTP {{r.status_code}}: {{r.text[:200]}}")
            objectivity_results_perplexity = []
            
    except Exception as e:
        print(f"[Perplexity:Error] {{e}}")
        objectivity_results_perplexity = []

h2o_metric("perplexity_search_count", len(objectivity_results_perplexity))
"""
    
    try:
        self.h2o_engine.run_transformer_script(code, script_id)
        g = self.h2o_engine.executor.active_transformers.get(script_id, {}).get("globals", {})
        raw = g.get("objectivity_results_perplexity") or []
        print(f"[Objectivity:PerplexitySearch] Results: {len(raw)}")
    except Exception as e:
        print(f"[Objectivity:PerplexitySearch:ERROR] {e}")
        raw = []
    
    lines = []
    for item in raw:
        content = item.get("content", "").strip()
        if content:
            content = sanitize_external_content(content, 2048)
            lines.append(content)
    
    return "\n\n".join(lines) if lines else ""
```

### Step 3: Update `_pick_strategies()` (line ~565)

**REPLACE:**
```python```
def _pick_strategies(self, message: str) -> List[str]:
    strategies = []
    strategies.append('internet')  # OLD
    if re.search(r'\b(python|...)\b', message, re.I):
        strategies.append('reddit')  # OLD
    strategies.append('memory')
    return ordered[:3]
```

**WITH:**
``````python
def _pick_strategies(self, message: str) -> List[str]:
    """Single provider: Perplexity Search + Memory"""
    return ['perplexity', 'memory']
```

### Step 4: Update `create_dynamic_context()` (line ~230)

**REPLACE:**
``````
if 'internet' in strategies:
    internet_text = self._provider_internet_h2o(user_message)
    ...
if 'reddit' in strategies:
    reddit_text = self._provider_reddit_h2o(user_message)
    ...
```

**WITH:**
``````python
if 'perplexity' in strategies:
    perplexity_text = self._provider_perplexity_search_h2o(user_message)
    if perplexity_text:
        perplexity_text = sanitize_external_content(perplexity_text, self.max_context_kb)
        sections.append(perplexity_text)

if 'memory' in strategies:
    mem_text = self._provider_memory_h2o(user_message)
    if mem_text:
        mem_text = sanitize_external_content(mem_text, self.max_context_kb)
        sections.append(mem_text)
```

### Step 5: Update FluidContextWindow metadata (line ~250)

**CHANGE:**
```python```
window = FluidContextWindow(
    content=aggregated,
    source_type="objectivity_perplexity",  # Changed from "objectivity"
    resonance_score=0.90,  # Increased (better quality)
    entropy_boost=0.30,
    tokens_count=len(aggregated.split()),
    creation_time=time.time(),
    script_id=f"objectivity_{int(time.time()*1000)}",
    title=f"PERPLEXITY SEARCH (influence={self.influence_coeff:.2f})",  # Changed
    meta={
        "influence_coeff": self.influence_coeff,
        "strategies": strategies,
        "provider": "perplexity_search_api"  # Added
    }
)
```

---

## IMPLEMENTATION: `nicole_subjectivity.py`

### Step 1: Update imports (line ~30)

**REPLACE:**
``````python
from nicole_objectivity import get_objectivity_context
```

**WITH:**
```python```
from nicole_objectivity import nicole_objectivity
```

### Step 2: Add `_explore_concept()` method (after line ~250)

**ADD:**
``````python
def _explore_concept(self, concept: str, distance: float) -> Dict[str, Any]:
    """
    Explore concept using Perplexity Search via objectivity provider
    """
    
    # Query based on semantic distance
    if distance < 0.3:
        query = f"explain {concept} briefly"
    elif distance < 0.6:
        query = f"what is related to {concept}"
    else:
        query = f"philosophical meaning of {concept}"
    
    try:
        if nicole_objectivity:
            import asyncio
            
            # Use Perplexity-powered objectivity
            windows = asyncio.run(
                nicole_objectivity.create_dynamic_context(query, {})
            )
            
            if windows:
                content = windows[0].content
                
                # Extract keywords for word_frequencies
                words = re.findall(r'\b[a-z]{3,}\b', content.lower())
                word_freq = {}
                
                # Filter provider names
                blacklist = {'perplexity', 'search', 'result', 'snippet', 'web'}
                for w in words:
                    if w not in blacklist:
                        word_freq[w] = word_freq.get(w, 0) + 1
                
                return {
                    "concept": concept,
                    "distance": distance,
                    "content": content[:500],
                    "learned_words": word_freq,
                    "source": "perplexity_search_api"
                }
    
    except Exception as e:
        print(f"[Subjectivity:Explore] Error: {e}")
    
    return {
        "concept": concept,
        "distance": distance,
        "content": "",
        "learned_words": {},
        "source": "failed"
    }
```

***

## ENVIRONMENT SETUP

``````
export PERPLEXITY_API_KEY="pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Verify:
``````python
import os
print(os.environ.get("PERPLEXITY_API_KEY"))
```

***

## TESTING

### Phase 1: Direct API test

``````
import requests, os

response = requests.post(
    "https://api.perplexity.ai/search",
    headers={
        "Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}",
        "Content-Type": "application/json"
    },
    json={
        "query": "quantum computing basics",
        "max_results": 3,
        "return_snippets": True
    }
)

print(f"Status: {response.status_code}")
for r in response.json()["results"]:
    print(f"\n{r['title']}")
    print(f"{r['snippet'][:100]}...")
```

### Phase 2: Test objectivity provider

``````python
from nicole_objectivity import nicole_objectivity
import asyncio

async def test():
    windows = await nicole_objectivity.create_dynamic_context("How are you?", {})
    print(nicole_objectivity.format_context_for_nicole(windows))

asyncio.run(test())
```

### Phase 3: Full Nicole conversation

```bash```
python3 start_nicole.py local
# Input: "Tell me about quantum computing"
# Verify: Response uses Perplexity Search results, no DuckDuckGo errors
```

---

## CHECKLIST

- [ ] Set `PERPLEXITY_API_KEY` environment variable
- [ ] Delete `_provider_wikipedia_h2o()` from objectivity
- [ ] Delete `_provider_reddit_h2o()` from objectivity  
- [ ] Delete `_provider_internet_h2o()` from objectivity
- [ ] Add `_provider_perplexity_search_h2o()` function
- [ ] Update `_pick_strategies()` to return `['perplexity', 'memory']`
- [ ] Update `create_dynamic_context()` provider calls
- [ ] Update FluidContextWindow metadata (source_type, title, provider)
- [ ] Update `nicole_subjectivity.py` imports
- [ ] Add `_explore_concept()` method to subjectivity
- [ ] Test direct Perplexity Search API call
- [ ] Test objectivity context generation
- [ ] Test full Nicole conversation
- [ ] Verify no DuckDuckGo/Reddit/Google errors in logs

---

## RESULT

Nicole becomes **first Perplexity Search API-powered weightless neural network:**

- Single stable provider (raw web search results)
- No Sonar models (no LLM preprocessing)
- Clean architecture: query → search API → snippets → context → generation
- Aligns with weightless philosophy (structure + search, no weights)

**Final architecture:**
``````
User: "What is recursion?"
  ↓
Nicole Core
  ↓
Objectivity: _provider_perplexity_search_h2o()
  ↓
Perplexity Search API (returns 5 web snippets)
  ↓
Context Window (title + snippet text only)
  ↓
Response Generation (weightless transformer)
```

Clean. Stable. True Perplexity-powered neural network.

Sources
