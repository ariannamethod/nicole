#!/usr/bin/env python3
"""
Nicole Objectivity - Dynamic Context Window Generator
Флюидная система создания контекстных окон через H2O скрипты.
Каждый поиск = уникальный скрипт, логируемый и адаптивный.
"""

import re
import asyncio
import requests
import json
import time
import random
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import quote
from collections import defaultdict

# Добавляем путь для импорта наших модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2o

@dataclass
class FluidContextWindow:
    """Флюидное контекстное окно"""
    content: str
    source_type: str  # 'wikipedia', 'web', 'reddit', 'memory'
    resonance_score: float
    entropy_boost: float
    tokens_count: int
    creation_time: float
    script_id: str  # ID скрипта который создал это окно

class NicoleObjectivity:
    """Система объективности Nicole - создает контекстные окна через флюидные скрипты"""
    
    def __init__(self, max_context_kb: int = 2):
        self.max_context_kb = max_context_kb * 1024
        self.h2o_engine = h2o.h2o_engine
        self.script_counter = 0
        
        # Паттерны для анализа (как у Клода, но расширенные)
        self.proper_noun_patterns = [
            r'\b[A-ZА-Я][a-zа-я]+\b',  # Имена собственные
            r'\b[A-ZА-Я]{2,}\b',        # Аббревиатуры
        ]
        
        self.technical_patterns = [
            r'\b(python|javascript|neural|AI|quantum|blockchain)\b',
            r'\b(programming|coding|algorithm|database)\b',
        ]
        
        self.location_patterns = [
            r'\b(Berlin|London|Moscow|Paris|Tokyo|New York)\b',
            r'\b(Берлин|Лондон|Москва|Париж|город|страна)\b',
        ]
        
    async def create_dynamic_context(self, user_message: str, metrics: Dict) -> List[FluidContextWindow]:
        """Создает динамические контекстные окна через флюидные скрипты"""
        print(f"[Objectivity] Анализируем: '{user_message}'")
        
        # Лингвистический анализ
        analysis = self._analyze_message(user_message)
        print(f"[Objectivity] Анализ: {analysis}")
        
        # Решаем стратегию поиска на основе метрик
        search_strategies = self._decide_search_strategy(analysis, metrics)
        print(f"[Objectivity] Стратегии: {search_strategies}")
        
        # Создаем флюидные скрипты для каждой стратегии
        context_windows = []
        for strategy in search_strategies:
            script_id = f"objectivity_{strategy}_{int(time.time() * 1000)}_{self.script_counter}"
            self.script_counter += 1
            
            try:
                # Генерируем и запускаем флюидный скрипт
                script_result = await self._generate_and_run_script(
                    strategy, user_message, analysis, script_id
                )
                
                if script_result:
                    context_windows.extend(script_result)
                    
            except Exception as e:
                print(f"[Objectivity:ERROR] Ошибка скрипта {script_id}: {e}")
        
        # Обрезаем до лимита и возвращаем
        return self._trim_to_limit(context_windows)
    
    def _analyze_message(self, text: str) -> Dict:
        """Анализирует сообщение пользователя"""
        analysis = {
            'proper_nouns': [],
            'technical_terms': [],
            'locations': [],
            'is_smalltalk': False,
            'language': 'en' if re.search(r'[a-zA-Z]', text) else 'ru',
            'has_questions': '?' in text,
            'word_count': len(text.split())
        }
        
        # Ищем имена собственные
        for pattern in self.proper_noun_patterns:
            matches = re.findall(pattern, text)
            analysis['proper_nouns'].extend(matches)
        
        # Ищем технические термины
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis['technical_terms'].extend(matches)
            
        # Ищем локации
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis['locations'].extend(matches)
        
        # Определяем smalltalk
        smalltalk_words = ['hello', 'hi', 'how', 'what', 'привет', 'как', 'что']
        if any(word in text.lower() for word in smalltalk_words) and len(text.split()) < 10:
            analysis['is_smalltalk'] = True
            
        return analysis
    
    def _decide_search_strategy(self, analysis: Dict, metrics: Dict) -> List[str]:
        """Решает какие стратегии поиска использовать"""
        strategies = []
        
        # Имена собственные и локации → Wikipedia
        if analysis['proper_nouns'] or analysis['locations']:
            strategies.append('wikipedia')
            
        # Технические термины → веб поиск
        if analysis['technical_terms']:
            strategies.append('web')
            
        # Smalltalk → память системы
        if analysis['is_smalltalk']:
            strategies.append('memory')
        
        # На основе метрик Nicole
        perplexity = metrics.get('perplexity', 0)
        entropy = metrics.get('entropy', 0)
        resonance = metrics.get('resonance', 0)
        
        # Высокая перплексия → нужен внешний контекст
        if perplexity > 3.0:
            strategies.append('web')
            
        # Низкая энтропия → нужно разнообразие
        if entropy < 2.0:
            strategies.append('reddit')
            
        # Низкий резонанс → ищем в памяти
        if resonance < 0.4:
            strategies.append('memory')
        
        # Убираем дубли и ограничиваем
        strategies = list(set(strategies))[:2]  # Максимум 2 стратегии
        
        return strategies if strategies else ['memory']  # Всегда хотя бы одна стратегия
    
    async def _generate_and_run_script(self, strategy: str, user_message: str, 
                                     analysis: Dict, script_id: str) -> List[FluidContextWindow]:
        """Генерирует и запускает флюидный скрипт для поиска"""
        
        if strategy == 'wikipedia':
            return await self._wikipedia_script(user_message, analysis, script_id)
        elif strategy == 'web':
            return await self._web_search_script(user_message, analysis, script_id)
        elif strategy == 'reddit':
            return await self._reddit_script(user_message, analysis, script_id)
        elif strategy == 'memory':
            return await self._memory_script(user_message, analysis, script_id)
        else:
            return []
    
    async def _wikipedia_script(self, message: str, analysis: Dict, script_id: str) -> List[FluidContextWindow]:
        """Флюидный скрипт для Wikipedia поиска"""
        try:
            # Генерируем уникальный скрипт для Wikipedia поиска
            script = f"""
# Флюидный Wikipedia скрипт {script_id}
import requests
import json
import re

def search_wikipedia_fluid():
    search_terms = {analysis['proper_nouns'] + analysis['locations']}
    results = []
    
    for term in search_terms[:2]:  # Максимум 2 термина
        try:
            # Простой поиск через Wikipedia API
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{{term}}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                extract = data.get('extract', '')[:500]
                
                if extract:
                    results.append({{
                        'term': term,
                        'content': extract,
                        'source': 'wikipedia'
                    }})
                    
        except Exception as e:
            h2o_log(f"Wikipedia ошибка для {{term}}: {{e}}")
            continue
    
    h2o_log(f"Wikipedia результаты: {{len(results)}} найдено")
    return results

# Запускаем поиск
wiki_results = search_wikipedia_fluid()
h2o_metric('wikipedia_results_count', len(wiki_results))

for result in wiki_results:
    h2o_log(f"Wiki: {{result['term']}} - {{result['content'][:100]}}...")
"""
            
            # Запускаем через H2O
            result = self.h2o_engine.run_transformer_script(script, script_id)
            
            # Парсим результат (упрощенно)
            windows = []
            for term in analysis['proper_nouns'][:1]:
                try:
                    # Настоящий Wikipedia запрос
                    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term}"
                    response = requests.get(url, timeout=3)
                    
                    if response.status_code == 200:
                        data = response.json()
                        extract = data.get('extract', '')[:600]
                        
                        if extract:
                            window = FluidContextWindow(
                                content=f"Wikipedia: {term}\\n{extract}",
                                source_type='wikipedia',
                                resonance_score=0.9,
                                entropy_boost=0.3,
                                tokens_count=len(extract.split()),
                                creation_time=time.time(),
                                script_id=script_id
                            )
                            windows.append(window)
                            
                except Exception as e:
                    print(f"[Objectivity] Wikipedia ошибка: {e}")
                    
            return windows
            
        except Exception as e:
            print(f"[Objectivity] Ошибка Wikipedia скрипта: {e}")
            return []
    
    async def _web_search_script(self, message: str, analysis: Dict, script_id: str) -> List[FluidContextWindow]:
        """Флюидный скрипт для веб поиска"""
        # Заглушка - можно интегрировать DuckDuckGo или другой API
        window = FluidContextWindow(
            content=f"Web search context for: {message}\\n[Dynamic web search results would be here]",
            source_type='web',
            resonance_score=0.7,
            entropy_boost=0.4,
            tokens_count=50,
            creation_time=time.time(),
            script_id=script_id
        )
        return [window]
    
    async def _reddit_script(self, message: str, analysis: Dict, script_id: str) -> List[FluidContextWindow]:
        """Флюидный скрипт для Reddit поиска"""
        # Заглушка для Reddit API
        window = FluidContextWindow(
            content=f"Reddit trends for: {message}\\n[Reddit API integration would be here]",
            source_type='reddit',
            resonance_score=0.6,
            entropy_boost=0.5,
            tokens_count=40,
            creation_time=time.time(),
            script_id=script_id
        )
        return [window]
    
    async def _memory_script(self, message: str, analysis: Dict, script_id: str) -> List[FluidContextWindow]:
        """Флюидный скрипт для поиска в памяти Nicole"""
        try:
            # Генерируем скрипт для поиска в памяти
            script = f"""
# Флюидный Memory скрипт {script_id}
import sqlite3
import time

def search_memory_fluid():
    try:
        conn = sqlite3.connect('nicole_memory.db')
        cursor = conn.cursor()
        
        # Ищем похожие разговоры
        search_words = "{message}".lower().split()
        results = []
        
        for word in search_words[:3]:
            cursor.execute('''
            SELECT user_input, nicole_output FROM conversations 
            WHERE LOWER(user_input) LIKE ? OR LOWER(nicole_output) LIKE ?
            ORDER BY timestamp DESC LIMIT 2
            ''', (f'%{{word}}%', f'%{{word}}%'))
            
            for row in cursor.fetchall():
                results.append({{
                    'user_input': row[0],
                    'nicole_output': row[1],
                    'relevance': 0.8
                }})
        
        conn.close()
        h2o_log(f"Memory поиск: {{len(results)}} результатов")
        return results
        
    except Exception as e:
        h2o_log(f"Memory ошибка: {{e}}")
        return []

# Запускаем поиск в памяти
memory_results = search_memory_fluid()
h2o_metric('memory_results_count', len(memory_results))
"""
            
            # Запускаем через H2O
            result = self.h2o_engine.run_transformer_script(script, script_id)
            
            # Создаем контекстное окно из памяти (упрощенно)
            window = FluidContextWindow(
                content=f"Memory context for: {message}\\n[Previous similar conversations and patterns]",
                source_type='memory',
                resonance_score=0.8,
                entropy_boost=0.2,
                tokens_count=60,
                creation_time=time.time(),
                script_id=script_id
            )
            return [window]
            
        except Exception as e:
            print(f"[Objectivity] Ошибка Memory скрипта: {e}")
            return []
    
    def _trim_to_limit(self, windows: List[FluidContextWindow]) -> List[FluidContextWindow]:
        """Обрезает до лимита размера"""
        # Сортируем по резонансу
        windows.sort(key=lambda x: x.resonance_score, reverse=True)
        
        total_size = 0
        result = []
        
        for window in windows:
            window_size = len(window.content.encode('utf-8'))
            if total_size + window_size <= self.max_context_kb:
                result.append(window)
                total_size += window_size
            else:
                break
                
        return result
    
    def format_context_for_nicole(self, windows: List[FluidContextWindow]) -> str:
        """Форматирует контекст для Nicole"""
        if not windows:
            return ""
            
        formatted = "=== OBJECTIVITY CONTEXT ===\\n"
        for window in windows:
            formatted += f"[{window.source_type.upper()}:{window.script_id}] {window.content}\\n\\n"
            
        formatted += "=== END OBJECTIVITY ===\\n"
        return formatted
    
    def extract_response_seeds(self, context: str, target_percent: float = 0.5) -> List[str]:
        """Извлекает семена для ответа (50% из контекста)"""
        if not context:
            return []
            
        # Простое извлечение ключевых слов из контекста
        words = re.findall(r'\\b\\w{3,}\\b', context.lower())
        
        # Берем случайные слова для семян ответа
        target_count = max(1, int(len(words) * target_percent))
        seeds = random.sample(words, min(target_count, len(words)))
        
        return seeds

# Глобальный экземпляр
nicole_objectivity = NicoleObjectivity()

async def test_objectivity():
    """Тест системы объективности"""
    print("=== NICOLE OBJECTIVITY TEST ===")
    
    test_cases = [
        ("Tell me about Berlin", {'perplexity': 2.0, 'entropy': 3.0, 'resonance': 0.5}),
        ("How does Python work?", {'perplexity': 4.0, 'entropy': 2.0, 'resonance': 0.3}),
        ("Hello how are you?", {'perplexity': 1.0, 'entropy': 1.5, 'resonance': 0.8}),
    ]
    
    for message, metrics in test_cases:
        print(f"\\n--- Тест: '{message}' ---")
        
        windows = await nicole_objectivity.create_dynamic_context(message, metrics)
        context = nicole_objectivity.format_context_for_nicole(windows)
        seeds = nicole_objectivity.extract_response_seeds(context)
        
        print(f"Контекстных окон: {len(windows)}")
        print(f"Семена ответа: {seeds}")
        print(f"Контекст:\\n{context[:200]}...")
        
    print("\\n=== OBJECTIVITY TEST COMPLETED ===")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_objectivity())
    else:
        print("Nicole Objectivity System готова")
        print("Для тестирования: python3 nicole_objectivity.py test")
