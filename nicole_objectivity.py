#!/usr/bin/env python3
"""
Nicole Objectivity - Dynamic Context Window Generator (H2O-native)
Вариант, где провайдеры полностью исполняются в H2O и складывают результат в globals.
Снаружи мы только забираем их объектные результаты и собираем одно текстовое окно.

Принципы:
- Никаких ссылок. Это контекст для генерации/обучения, не справочный режим.
- Шаблоны разрешены только для первых N сообщений после инициализации (persist через nicole_state.json).
- Тихий фолбэк: если источник молчит — просто пропускаем без служебных сообщений.
- Контекст записывается в training_buffer.jsonl для последующей выборки/обучения.
- Влияние контекста на ответ: influence_coeff = 0.5.

Зависимости: стандартная библиотека + локальный h2o.py
"""

import re
import os
import sys
import json
import time
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

# Локальные модули
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2o

USER_AGENT = "nicole-objectivity/1.0 (+github.com/ariannamethod/nicole)"

STATE_PATH = "nicole_state.json"
TRAIN_BUFFER_PATH = "training_buffer.jsonl"
MEMORY_DB = "nicole_memory.db"

# Тонкая настройка
DEFAULT_TEMPLATE_WARMUP = 3      # шаблоны можно во время первых N сообщений после старта
DEFAULT_INFLUENCE_COEFF = 0.5    # доля влияния контента на ответ

@dataclass
class FluidContextWindow:
    content: str
    source_type: str           # здесь используем 'objectivity' (единое окно)
    resonance_score: float     # опционально используем как интегральный скор
    entropy_boost: float       # опционально
    tokens_count: int
    creation_time: float
    script_id: str
    title: Optional[str] = None
    template_used: bool = False
    meta: Optional[Dict] = None

class NicoleObjectivity:
    def __init__(self,
                 max_context_kb: int = 4,
                 template_warmup_messages: int = DEFAULT_TEMPLATE_WARMUP,
                 influence_coeff: float = DEFAULT_INFLUENCE_COEFF):
        self.max_context_kb = max_context_kb * 1024
        self.template_warmup_messages = max(0, int(template_warmup_messages))
        self.influence_coeff = float(influence_coeff)
        self.h2o_engine = h2o.h2o_engine
        self._message_index = self._load_state_message_index()
        self._ensure_memory_schema()

    # ---------- State / persistence ----------

    def _load_state_message_index(self) -> int:
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    st = json.load(f)
                return int(st.get("message_index", 0))
        except Exception:
            pass
        return 0

    def _save_state_message_index(self, idx: int):
        try:
            st = {"message_index": int(idx)}
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False)
        except Exception:
            # Тихо игнорируем: никакой болтовни наружу
            pass

    def _ensure_memory_schema(self):
        try:
            conn = sqlite3.connect(MEMORY_DB)
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp REAL DEFAULT (strftime('%s','now')),
              user_input TEXT,
              nicole_output TEXT
            )
            """)
            try:
                cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                USING fts5(user_input, nicole_output, tokenize = 'porter');
                """)
                # первичное наполнение FTS, если пусто
                cur.execute("SELECT count(*) FROM conversations_fts")
                if cur.fetchone()[0] == 0:
                    cur.execute("""
                    INSERT INTO conversations_fts(rowid, user_input, nicole_output)
                    SELECT id, user_input, nicole_output FROM conversations
                    WHERE user_input IS NOT NULL OR nicole_output IS NOT NULL
                    """)
            except sqlite3.Error:
                # ок, без fts5
                pass
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ---------- Public API ----------

    async def create_dynamic_context(self, user_message: str, metrics: Dict) -> List[FluidContextWindow]:
        """
        Главная точка: формируем единое текстовое окно (или ничего, если контента нет).
        """
        # Решаем, допускаем ли шаблоны на этом шаге
        templates_allowed = (self._message_index < self.template_warmup_messages)

        # Анализ языка (минимально, без лишних зависимостей)
        lang = self._lang_heuristic(user_message)

        # Выбираем стратегии (минимально, без «злобных» эвристик)
        strategies = self._pick_strategies(user_message, lang)

        # Гоним каждый провайдер через H2O и читаем globals
        sections: List[str] = []
        template_used_any = False

        if 'wikipedia' in strategies:
            wiki_text, wiki_template = self._provider_wikipedia_h2o(user_message, lang, templates_allowed)
            if wiki_text:
                sections.append(wiki_text)
            template_used_any = template_used_any or wiki_template

        if 'reddit' in strategies:
            reddit_text, reddit_template = self._provider_reddit_h2o(user_message, templates_allowed)
            if reddit_text:
                sections.append(reddit_text)
            template_used_any = template_used_any or reddit_template

        if 'memory' in strategies:
            mem_text, mem_template = self._provider_memory_h2o(user_message, templates_allowed)
            if mem_text:
                sections.append(mem_text)
            template_used_any = template_used_any or mem_template

        # Агрегируем в одно окно
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
                template_used=template_used_any,
                meta={
                    "influence_coeff": self.influence_coeff,
                    "strategies": strategies,
                    "lang": lang
                }
            )
            windows.append(window)

            # Записываем в training-буфер
            self._record_for_training(user_message, aggregated, strategies, lang, self.influence_coeff)

        # Увеличиваем счётчик сообщений и сохраняем
        self._message_index += 1
        self._save_state_message_index(self._message_index)

        # Триммим по размеру (единичное окно — но на всякий случай)
        return self._trim_to_limit(windows)

    def format_context_for_nicole(self, windows: List[FluidContextWindow]) -> str:
        if not windows:
            return ""
        # Одно окно. Без ссылок, без служебщины.
        w = windows[0]
        header = w.title or "OBJECTIVITY"
        return f"=== {header} ===\n{w.content}\n=== END OBJECTIVITY ===\n"

    # ---------- Internals ----------

    def _lang_heuristic(self, text: str) -> str:
        latin = len(re.findall(r'[A-Za-z]', text))
        cyr = len(re.findall(r'[А-Яа-я]', text))
        return 'ru' if cyr > latin else 'en'

    def _pick_strategies(self, message: str, lang: str) -> List[str]:
        # Минималистично: если есть имена/локации → wikipedia, техтермины → reddit, smalltalk → memory
        strategies: List[str] = []
        if re.search(r'\b([A-ZА-Я][a-zа-я]+|[A-ZА-Я]{2,})\b', message):
            strategies.append('wikipedia')
        if re.search(r'\b(python|javascript|neural|ai|quantum|blockchain|programming|algorithm|database)\b', message, re.I):
            strategies.append('reddit')
        if len(message.split()) < 8:
            strategies.append('memory')

        # Всегда полезно смотреть память, но без навязчивости
        if 'memory' not in strategies:
            strategies.append('memory')

        # Удаляем дубли, сохраняем порядок
        seen = set()
        ordered = []
        for s in strategies:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        # Ограничим 2-3 источниками
        return ordered[:3]

    def _aggregate_text_window(self, sections: List[str]) -> str:
        text = "\n\n".join(s for s in sections if s)
        if not text.strip():
            return ""
        # ограничение размера окна
        while len(text.encode('utf-8')) > self.max_context_kb:
            # грубое усечение: обрезаем самый длинный абзац
            parts = text.split("\n\n")
            if len(parts) <= 1:
                text = text[: max(0, int(self.max_context_kb/2))]  # совсем грубо
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
                             strategies: List[str], lang: str, influence: float):
        try:
            rec = {
                "ts": time.time(),
                "user_message": user_message,
                "context": context_text,
                "strategies": strategies,
                "lang": lang,
                "influence_coeff": influence
            }
            with open(TRAIN_BUFFER_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # тихо
            pass

    # ---------- Providers via H2O (results in globals) ----------

    def _provider_wikipedia_h2o(self, message: str, lang: str, templates_allowed: bool) -> Tuple[str, bool]:
        """
        Возвращает (текст-секция, template_used).
        Все HTTP — внутри H2O; здесь только чтение globals.
        Никаких ссылок, только title + extract.
        """
        base = f"https://{'ru' if lang=='ru' else 'en'}.wikipedia.org/api/rest_v1/page/summary"
        terms = self._extract_terms(message)[:3] or [message.strip()[:64]]

        script_id = f"wiki_{int(time.time()*1000)}"
        code = f"""
import requests, json, re

def _wiki_terms():
    return {json.dumps(terms, ensure_ascii=False)}

def _fetch(term):
    url = "{base}/" + term.replace(" ", "_")
    try:
        r = requests.get(url, headers={{"User-Agent": "{USER_AGENT}"}}, timeout=5)
        if r.status_code != 200:
            return None
        d = r.json()
        title = d.get("title") or term
        extract = (d.get("extract") or "")[:900]
        if not extract:
            return None
        # Никаких ссылок
        return {{"title": title, "content": extract}}
    except Exception:
        return None

res = []
for t in _wiki_terms():
    item = _fetch(t)
    if item:
        res.append(item)

objectivity_results_wikipedia = res
h2o_metric("wikipedia_results_count", len(res))
"""
        self.h2o_engine.run_transformer_script(code, script_id)
        g = self.h2o_engine.executor.active_transformers.get(script_id, {}).get("globals", {})
        raw = g.get("objectivity_results_wikipedia") or []

        lines: List[str] = []
        for it in raw:
            title = (it.get("title") or "").strip()
            content = (it.get("content") or "").strip()
            if title and content:
                lines.append(f"Wikipedia: {title}\n{content}")

        if lines:
            return "\n\n".join(lines), False

        # Пусто — никаких болтовен
        if templates_allowed:
            # Мягкий шаблон: просто молчим, не добавляем секцию
            return "", True
        return "", False

    def _provider_reddit_h2o(self, message: str, templates_allowed: bool) -> Tuple[str, bool]:
        """
        Reddit: без ссылок, только заголовки/текст срезанный.
        """
        query = message.strip()[:128]
        script_id = f"reddit_{int(time.time()*1000)}"
        code = f"""
import requests, json, re

def _fetch():
    try:
        r = requests.get(
            "https://www.reddit.com/search.json",
            params={{"q": {json.dumps(query)}, "limit": 3, "sort": "relevance", "t": "year"}},
            headers={{"User-Agent": "{USER_AGENT}"}},
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
            text = text.strip()
            if text:
                out.append({{"title": title, "content": text}})
        return out
    except Exception:
        return []

objectivity_results_reddit = _fetch()
h2o_metric("reddit_results_count", len(objectivity_results_reddit))
"""
        self.h2o_engine.run_transformer_script(code, script_id)
        g = self.h2o_engine.executor.active_transformers.get(script_id, {}).get("globals", {})
        raw = g.get("objectivity_results_reddit") or []

        lines: List[str] = []
        for it in raw:
            title = (it.get("title") or "").strip()
            content = (it.get("content") or "").strip()
            if content:
                header = f"Reddit: {title}" if title else "Reddit"
                lines.append(f"{header}\n{content}")

        if lines:
            return "\n\n".join(lines), False

        if templates_allowed:
            return "", True
        return "", False

    def _provider_memory_h2o(self, message: str, templates_allowed: bool) -> Tuple[str, bool]:
        """
        Memory через sqlite в H2O: FTS5 при наличии, иначе LIKE.
        Возвращаем компактные срезы, без служебщины.
        """
        # простая токенизация запроса
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
                    SELECT highlight(conversations_fts, 1, '', '') AS n_h,
                           highlight(conversations_fts, 0, '', '') AS u_h
                    FROM conversations_fts
                    WHERE conversations_fts MATCH ?
                    LIMIT 3
                \""", ({json.dumps(query)},))
                rows = c.fetchall()
                for r in rows:
                    snippet = (r["n_h"] or r["u_h"] or "").strip()
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
                        s = re.sub(r"\\s+", " ", s)[:900]
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
        self.h2o_engine.run_transformer_script(code, script_id)
        g = self.h2o_engine.executor.active_transformers.get(script_id, {}).get("globals", {})
        raw = g.get("objectivity_results_memory") or []

        lines = []
        for it in raw:
            s = (it.get("content") or "").strip()
            if s:
                lines.append(f"Memory:\n{s}")

        if lines:
            return "\n\n".join(lines), False

        if templates_allowed:
            return "", True
        return "", False

    # ---------- Helpers ----------

    def _extract_terms(self, message: str) -> List[str]:
        terms = []
        terms += re.findall(r'\b[A-ZА-Я][a-zа-я]+\b', message)
        terms += re.findall(r'\b[A-ZА-Я]{2,}\b', message)
        # локации (простые)
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
    asyncio.run(_t())
