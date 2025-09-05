#!/usr/bin/env python3
"""
Nicole Objectivity - Dynamic Context Window Generator (H2O-native, language-agnostic)
Провайдеры исполняются в H2O и складывают результат в globals. Снаружи — сборка одного текстового окна.

Принципы:
- Никаких ссылок, только текст.
- Без шаблонов вообще.
- Тихий фолбэк: если источник молчит — пропускаем.
- Контекст пишется в training_buffer.jsonl.
- Влияние контекста на ответ: influence_coeff = 0.5.
- Язык-агностично: никаких ru/en эвристик; Wikipedia через Wikidata sitelinks.

Зависимости: стандартная библиотека + локальный h2o.py
"""

import re
import os
import sys
import json
import time
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional
import random

# Локальные модули
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2o

USER_AGENT = "nicole-objectivity/1.0 (+github.com/ariannamethod/nicole)"

TRAIN_BUFFER_PATH = "training_buffer.jsonl"
MEMORY_DB = "nicole_memory.db"

DEFAULT_INFLUENCE_COEFF = 0.5  # доля влияния контента на ответ

@dataclass
class FluidContextWindow:
    content: str
    source_type: str           # 'objectivity' (единое окно)
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
                 influence_coeff: float = DEFAULT_INFLUENCE_COEFF):
        self.max_context_kb = max_context_kb * 1024
        self.influence_coeff = float(influence_coeff)
        self.h2o_engine = h2o.h2o_engine
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
            # Совместимость с прошлыми ревизиями
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
        Формируем единое текстовое окно (или ничего, если контента нет).
        """
        strategies = self._pick_strategies(user_message)

        sections: List[str] = []

        # УБРАНО: Wikipedia провайдер полностью удален

        if 'internet' in strategies:
            internet_text = self._provider_internet_h2o(user_message)
            if internet_text:
                sections.append(internet_text)

        if 'reddit' in strategies:
            reddit_text = self._provider_reddit_h2o(user_message)
            if reddit_text:
                sections.append(reddit_text)

        if 'memory' in strategies:
            mem_text = self._provider_memory_h2o(user_message)
            if mem_text:
                sections.append(mem_text)

        aggregated = self._aggregate_text_window(sections)

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
        """Семена для ответа из контекста (язык-агностично)."""
        if not context:
            return []
        words = re.findall(r"\w{3,}", context.lower(), flags=re.UNICODE)
        words = [w for w in words if any(ch.isalpha() for ch in w)]
        if not words:
            return []
        seed_count = max(1, int(len(words) * max(0.0, min(1.0, influence))))
        if seed_count >= len(words):
            return list(dict.fromkeys(words))
        return random.sample(words, seed_count)

    # ---------- Internals ----------

    def _pick_strategies(self, message: str) -> List[str]:
        # ИСПРАВЛЕНО: убираем Wikipedia для служебных слов, добавляем интернет-поиск
        strategies: List[str] = []
        
        # ГРУБЫЙ ИНТЕРНЕТ ПОИСК: всегда используем интернет вместо Wikipedia
        strategies.append('internet')
            
        # Reddit для технических тем
        if re.search(r'\b(python|javascript|neural|ai|quantum|blockchain|programming|algorithm|database|code|tech)\b', message, re.I):
            strategies.append('reddit')
            
        # Память всегда полезна
        strategies.append('memory')

        # Дедуп и ограничение
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
        # Ограничение размера окна
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
        Wikipedia через Wikidata (язык-агностично):
        - По термам ищем Q-id
        - Берём sitelinks; пробуем список языков по приоритету, иначе первый доступный *wiki
        - Тянем REST /page/summary на выбранном языке
        - Возвращаем только текст (title + extract), без ссылок
        """
        # НОВОЕ: приоритет заглавным словам (имена собственные)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+\b', message)
        regular_terms = self._extract_terms(message)[:3] or [message.strip()[:64]]
        
        if capitalized_terms:
            # Заглавные слова идут первыми!
            terms = capitalized_terms[:2] + [t for t in regular_terms if t not in capitalized_terms][:2]
            print(f"[Objectivity:Wikipedia] Приоритет заглавным: {capitalized_terms}")
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
    # Сначала по приоритету, затем любой *wiki
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
        Reddit: без ссылок, только заголовки/текст (срез).
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

    def _provider_internet_h2o(self, message: str) -> str:
        """
        Internet search: контекстный поиск для вопросов типа "how are you"
        Формирует умные запросы и ищет ответы в интернете
        """
        # Формируем контекстный запрос
        if re.search(r'\b(how|what|why|when|where|who)\b', message, re.I):
            # Для вопросов делаем "how to answer" или "what does X mean"
            if 'how are you' in message.lower():
                query = "how to respond to how are you conversation"
            elif 'what' in message.lower():
                query = f"what does mean {message.strip()}"
            elif 'how' in message.lower():
                query = f"how to answer {message.strip()}"
            else:
                query = f"answer to {message.strip()}"
        else:
            # Для обычных сообщений ищем по ключевым словам
            words = re.findall(r'\b[a-zA-Z]{3,}\b', message)
            query = ' '.join(words[:4]) if words else message.strip()
        
        query = query[:100]  # Ограничиваем длину
        script_id = f"internet_{int(time.time()*1000)}"
        
        code = f"""
import requests, json, re
from urllib.parse import quote

UA = {json.dumps(USER_AGENT)}

def _search():
    query = {json.dumps(query)}
    results = []
    
    try:
        # Поиск через DuckDuckGo API (без API ключей)
        url = f"https://api.duckduckgo.com/?q={{quote(query)}}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
        r = requests.get(url, headers={{"User-Agent": UA}}, timeout=8)
        
        if r.status_code == 200:
            data = r.json()
            
            # Берем основной ответ если есть
            abstract = data.get("Abstract", "").strip()
            if abstract and len(abstract) > 20:
                results.append({{"title": data.get("AbstractText", "Answer"), "content": abstract[:800]}})
            
            # Берем связанные темы
            related = data.get("RelatedTopics", [])[:3]
            for item in related:
                if isinstance(item, dict):
                    text = item.get("Text", "").strip()
                    if text and len(text) > 20:
                        results.append({{"title": "Related", "content": text[:600]}})
                        
    except Exception:
        pass
    
    # Fallback: простой поиск фраз для обучения
    if not results:
        try:
            # Генерируем несколько вариантов ответов для обучения
            if "how are you" in query.lower():
                samples = [
                    "I'm doing well, thanks for asking",
                    "Pretty good, how about you", 
                    "Not bad, yourself",
                    "I'm fine, thank you"
                ]
                for sample in samples:
                    results.append({{"title": "Response pattern", "content": sample}})
        except Exception:
            pass
    
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
        Memory через sqlite в H2O: FTS5 при наличии, иначе LIKE.
        Возвращаем компактные срезы, без служебщины.
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
            print(f"[Objectivity:Memory] H2O результат: {len(raw)} записей")
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
        terms += re.findall(r'\b[A-ZА-Я][a-zа-я]+\b', message)
        terms += re.findall(r'\b[A-ZА-Я]{2,}\b', message)
        # простые локации
        locs = re.findall(r'\b(Berlin|London|Moscow|Paris|Tokyo|New York|Берлин|Лондон|Москва|Париж)\b', message, flags=re.I)
        terms += locs
        # уникализируем, сохраняем порядок
        seen = set()
        out = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

# Глобальный экземпляр
nicole_objectivity = NicoleObjectivity()

# Простой тест
if __name__ == "__main__":
    import asyncio
    async def _t():
        cases = [
            ("Tell me about Berlin and Python internals", {}),
            ("Привет", {}),
            ("Как работает память модели?", {}),
        ]
        for m, mt in cases:
            ws = await nicole_objectivity.create_dynamic_context(m, mt)
            print(nicole_objectivity.format_context_for_nicole(ws))
            print("Seeds:", nicole_objectivity.extract_response_seeds(ws[0].content if ws else "", nicole_objectivity.influence_coeff))
    asyncio.run(_t())
