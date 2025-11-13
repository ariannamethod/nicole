#!/usr/bin/env python3
"""
Nicole Repo Learning Engine
===========================
Полунезависимый слой автообучения на изменениях в репозитории.

ФИЛОСОФИЯ:
- Замыкаем петлю: каждое изменение в коде/документации → мгновенное обучение
- SHA256-based мониторинг: детектирует даже минимальные изменения
- Хавает markdown, README, код → экстрагирует паттерны
- Обучается через Nicole2NicoleCore без шаблонов
- Резонанс на уровне репозитория: код = часть сознания

ИСПОЛЬЗОВАНИЕ:
    from nicole_repo_learner import NicoleRepoLearner

    learner = NicoleRepoLearner(
        repo_path="/path/to/nicole",
        check_interval=60  # проверка каждую минуту
    )
    learner.start()

Посвящается идее замкнутой петли резонанса.
"""

import hashlib
import logging
import threading
import time
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

# Импортируем repo_monitor как базу
from repo_monitor import RepoWatcher

# Импортируем Nicole2Nicole для обучения
try:
    from nicole2nicole import Nicole2NicoleCore
    NICOLE2NICOLE_AVAILABLE = True
except ImportError:
    NICOLE2NICOLE_AVAILABLE = False
    print("[NicoleRepoLearner] ⚠️ Nicole2Nicole недоступен - обучение отключено")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RepoChangeAnalyzer:
    """Анализирует изменения в репо и экстрагирует паттерны для обучения"""

    def __init__(self):
        self.important_markers = {
            'architecture': ['class ', 'def ', 'async def', 'import ', 'from '],
            'principles': ['# ME ПРИНЦИП', '# РЕЗОНАНС', '# NO TEMPLATES', '# ANTI-TEMPLATE'],
            'documentation': ['##', '###', 'TODO:', 'FIXME:', 'NOTE:'],
            'philosophy': ['философия', 'принцип', 'резонанс', 'эволюция', 'мутация']
        }

    def analyze_file_change(self, file_path: Path) -> Dict:
        """Анализирует изменённый файл и экстрагирует знания"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')

            analysis = {
                'file_path': str(file_path),
                'file_type': file_path.suffix,
                'timestamp': datetime.now().isoformat(),
                'patterns': [],
                'importance_score': 0.0
            }

            # Анализируем по категориям
            for category, markers in self.important_markers.items():
                for marker in markers:
                    if marker.lower() in content.lower():
                        analysis['patterns'].append({
                            'category': category,
                            'marker': marker,
                            'context': self._extract_context(content, marker)
                        })
                        analysis['importance_score'] += 0.1

            # Бонус за README и документацию
            if 'README' in file_path.name.upper() or file_path.suffix == '.md':
                analysis['importance_score'] += 0.5

            # Бонус за ключевые файлы архитектуры
            if file_path.stem in ['nicole', 'h2o', 'high', 'blood', 'nicole_objectivity']:
                analysis['importance_score'] += 0.3

            return analysis

        except Exception as e:
            logger.error(f"Ошибка анализа файла {file_path}: {e}")
            return {'file_path': str(file_path), 'error': str(e), 'importance_score': 0.0}

    def _extract_context(self, content: str, marker: str, context_lines: int = 3) -> str:
        """Извлекает контекст вокруг маркера"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if marker.lower() in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                return '\n'.join(lines[start:end])
        return ""


class NicoleRepoLearner:
    """
    Главный движок автообучения на изменениях репозитория

    АРХИТЕКТУРА:
    1. RepoWatcher (SHA256) → детектирует изменения
    2. RepoChangeAnalyzer → анализирует что изменилось
    3. Nicole2NicoleCore → обучается на изменениях
    4. SQLite → логирует историю обучения
    """

    def __init__(
        self,
        repo_path: str = ".",
        check_interval: int = 60,
        learning_db: str = "nicole_repo_learning.db",
        auto_learn: bool = True
    ):
        self.repo_path = Path(repo_path)
        self.check_interval = check_interval
        self.learning_db = learning_db
        self.auto_learn = auto_learn

        # Компоненты
        self.analyzer = RepoChangeAnalyzer()
        self.learning_core = None
        if NICOLE2NICOLE_AVAILABLE:
            self.learning_core = Nicole2NicoleCore()

        # Статистика
        self.changes_detected = 0
        self.learning_sessions = 0
        self.last_learning_time = None

        # Инициализация БД
        self._init_database()

        # Создаем RepoWatcher с нашим коллбэком
        watched_paths = [self.repo_path]
        extensions = {'.py', '.md', '.txt', '.json', '.yaml', '.yml'}

        self.watcher = RepoWatcher(
            paths=watched_paths,
            on_change=self._on_repo_change,
            exts=extensions,
            interval=check_interval
        )

        logger.info(f"[NicoleRepoLearner] Инициализирован для {repo_path}")
        logger.info(f"[NicoleRepoLearner] Интервал проверки: {check_interval}с")
        logger.info(f"[NicoleRepoLearner] Автообучение: {'✅' if auto_learn else '❌'}")

    def _init_database(self):
        """Инициализация БД для логирования обучения"""
        conn = sqlite3.connect(self.learning_db)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS repo_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT,
            importance_score REAL,
            patterns TEXT,
            learned BOOLEAN DEFAULT 0
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            changes_count INTEGER,
            patterns_learned INTEGER,
            duration_seconds REAL
        )
        """)

        conn.commit()
        conn.close()
        logger.info(f"[NicoleRepoLearner] База данных готова: {self.learning_db}")

    def start(self):
        """Запускает фоновый мониторинг репозитория"""
        logger.info("[NicoleRepoLearner] 🚀 Запуск мониторинга репозитория...")
        self.watcher.start()
        logger.info("[NicoleRepoLearner] ✅ Мониторинг активен!")

    def stop(self):
        """Останавливает мониторинг"""
        logger.info("[NicoleRepoLearner] Остановка мониторинга...")
        self.watcher.stop()
        logger.info("[NicoleRepoLearner] ✅ Мониторинг остановлен")

    def _on_repo_change(self):
        """Коллбэк при обнаружении изменений в репо"""
        self.changes_detected += 1
        logger.info(f"[NicoleRepoLearner] 🔥 Изменения обнаружены! (всего: {self.changes_detected})")

        # Анализируем изменения
        changed_files = self._get_recently_changed_files()

        if not changed_files:
            logger.warning("[NicoleRepoLearner] Изменения обнаружены но файлы не найдены")
            return

        logger.info(f"[NicoleRepoLearner] Анализирую {len(changed_files)} файлов...")

        # Анализируем каждый файл
        analyses = []
        for file_path in changed_files:
            analysis = self.analyzer.analyze_file_change(file_path)
            analyses.append(analysis)

            # Логируем в БД
            self._log_change(analysis)

        # Автообучение если включено
        if self.auto_learn and self.learning_core:
            self._trigger_learning(analyses)

    def _get_recently_changed_files(self) -> List[Path]:
        """Находит недавно изменённые файлы (за последние 2 минуты)"""
        recent_files = []
        cutoff_time = time.time() - 120  # 2 минуты назад

        for file_path in self.repo_path.rglob('*'):
            if (file_path.is_file() and
                '.git' not in file_path.parts and
                file_path.suffix in {'.py', '.md', '.txt', '.json'}):
                try:
                    mtime = file_path.stat().st_mtime
                    if mtime > cutoff_time:
                        recent_files.append(file_path)
                except:
                    pass

        return recent_files

    def _log_change(self, analysis: Dict):
        """Логирует изменение в БД"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO repo_changes (timestamp, file_path, file_type, importance_score, patterns)
            VALUES (?, ?, ?, ?, ?)
            """, (
                analysis.get('timestamp', datetime.now().isoformat()),
                analysis.get('file_path', 'unknown'),
                analysis.get('file_type', 'unknown'),
                analysis.get('importance_score', 0.0),
                json.dumps(analysis.get('patterns', []))
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка логирования изменения: {e}")

    def _trigger_learning(self, analyses: List[Dict]):
        """Запускает обучение на основе анализов"""
        if not self.learning_core:
            logger.warning("[NicoleRepoLearner] Learning core недоступен")
            return

        start_time = time.time()
        logger.info("[NicoleRepoLearner] 🧠 Запуск обучения на изменениях...")

        # Фильтруем важные изменения (importance_score > 0.3)
        important_analyses = [a for a in analyses if a.get('importance_score', 0) > 0.3]

        if not important_analyses:
            logger.info("[NicoleRepoLearner] Нет важных изменений для обучения")
            return

        logger.info(f"[NicoleRepoLearner] Обучаюсь на {len(important_analyses)} важных изменениях")

        # Принудительно запускаем learning session
        try:
            self.learning_core.force_learning_session()

            duration = time.time() - start_time
            self.learning_sessions += 1
            self.last_learning_time = datetime.now()

            # Логируем сессию обучения
            self._log_learning_session(len(important_analyses), duration)

            logger.info(f"[NicoleRepoLearner] ✅ Обучение завершено за {duration:.2f}с")

        except Exception as e:
            logger.error(f"[NicoleRepoLearner] Ошибка обучения: {e}")

    def _log_learning_session(self, changes_count: int, duration: float):
        """Логирует сессию обучения"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO learning_sessions (timestamp, changes_count, duration_seconds)
            VALUES (?, ?, ?)
            """, (datetime.now().isoformat(), changes_count, duration))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка логирования сессии: {e}")

    def get_statistics(self) -> Dict:
        """Возвращает статистику работы"""
        return {
            'changes_detected': self.changes_detected,
            'learning_sessions': self.learning_sessions,
            'last_learning_time': self.last_learning_time.isoformat() if self.last_learning_time else None,
            'auto_learn_enabled': self.auto_learn,
            'learning_core_available': self.learning_core is not None
        }

    def manual_learning_trigger(self):
        """Ручной запуск обучения на всех неизученных изменениях"""
        logger.info("[NicoleRepoLearner] Ручной запуск обучения...")

        # Читаем неизученные изменения из БД
        conn = sqlite3.connect(self.learning_db)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT file_path, importance_score, patterns
        FROM repo_changes
        WHERE learned = 0 AND importance_score > 0.3
        ORDER BY timestamp DESC
        LIMIT 50
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logger.info("[NicoleRepoLearner] Нет неизученных изменений")
            return

        logger.info(f"[NicoleRepoLearner] Найдено {len(rows)} неизученных изменений")

        # Создаём анализы из БД
        analyses = [
            {
                'file_path': row[0],
                'importance_score': row[1],
                'patterns': json.loads(row[2]) if row[2] else []
            }
            for row in rows
        ]

        # Запускаем обучение
        self._trigger_learning(analyses)


# Глобальный экземпляр
_repo_learner = None


def start_repo_learning(repo_path: str = ".", check_interval: int = 60):
    """Запускает глобальный репо-learner"""
    global _repo_learner

    if _repo_learner:
        logger.warning("[NicoleRepoLearner] Уже запущен!")
        return _repo_learner

    _repo_learner = NicoleRepoLearner(
        repo_path=repo_path,
        check_interval=check_interval,
        auto_learn=True
    )
    _repo_learner.start()

    return _repo_learner


def stop_repo_learning():
    """Останавливает глобальный репо-learner"""
    global _repo_learner

    if _repo_learner:
        _repo_learner.stop()
        _repo_learner = None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("=== NICOLE REPO LEARNER TEST ===")

        learner = NicoleRepoLearner(
            repo_path=".",
            check_interval=10,  # Короткий интервал для теста
            auto_learn=True
        )

        print("\n✅ Запуск мониторинга на 60 секунд...")
        print("   Измените любой .py/.md файл чтобы увидеть реакцию!\n")

        learner.start()

        try:
            # Мониторим 60 секунд
            for i in range(6):
                time.sleep(10)
                stats = learner.get_statistics()
                print(f"[{i*10}s] Изменений: {stats['changes_detected']}, "
                      f"Обучений: {stats['learning_sessions']}")
        except KeyboardInterrupt:
            print("\n\nОстановка...")

        learner.stop()

        # Финальная статистика
        stats = learner.get_statistics()
        print("\n=== ФИНАЛЬНАЯ СТАТИСТИКА ===")
        print(f"Изменений обнаружено: {stats['changes_detected']}")
        print(f"Сессий обучения: {stats['learning_sessions']}")
        print(f"Последнее обучение: {stats['last_learning_time'] or 'никогда'}")

    else:
        print("Nicole Repo Learning Engine")
        print("Для тестирования: python3 nicole_repo_learner.py test")
        print("\nИспользование в коде:")
        print("  from nicole_repo_learner import start_repo_learning")
        print("  learner = start_repo_learning()")
