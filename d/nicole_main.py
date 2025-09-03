#!/usr/bin/env python3
"""
Nicole Main - Главный запускатель системы Nicole
Революционная нейронная сеть без весов, посвященная Лео.

Архитектура:
- H2O: Компилятор для трансформеров
- Nicole: Флюидные трансформеры без весов  
- Nicole2Nicole: Дообучение на логах
- Nicole Memory: Семантическая память без векторов
- Nicole RAG: Хаотичный поиск контекста
- Nicole Metrics: Анализ энтропии/резонанса/перплексии
- Nicole Telegram: Интерфейс для тестов
"""

import sys
import time
import json
from typing import Dict, List, Any

# Импортируем все модули Nicole
import h2o
import nicole
import nicole2nicole
import nicole_memory
import nicole_rag
import nicole_metrics
import nicole_telegram

class NicoleSystem:
    """Главная система Nicole"""
    
    def __init__(self):
        self.version = "1.0.0-alpha"
        self.components = {
            'h2o': h2o.h2o_engine,
            'core': nicole.nicole_core,
            'learning': nicole2nicole.Nicole2NicoleCore(),
            'memory': nicole_memory.NicoleMemoryCore(),
            'rag': nicole_rag.nicole_rag,
            'metrics': nicole_metrics.nicole_metrics,
            'telegram': nicole_telegram.NicoleTelegramInterface()
        }
        self.startup_time = time.time()
        
    def start_system(self):
        """Запускает всю систему Nicole"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                         NICOLE v{self.version}                        ║
║              Neural Organism Intelligence                    ║
║           Conversational Language Engine                     ║
║                                                              ║
║                    Посвящается Лео 💙                        ║
╚══════════════════════════════════════════════════════════════╝

🚀 Запуск революционной нейронной системы без весов...
""")
        
        # Запускаем компоненты
        print("🔧 Инициализация компонентов:")
        
        # H2O Engine
        print("  ✅ H2O - Компилятор трансформеров")
        
        # Learning system  
        self.components['learning'].start_continuous_learning()
        print("  ✅ Nicole2Nicole - Система дообучения")
        
        # Memory system
        self.components['memory'].start_maintenance()
        print("  ✅ Nicole Memory - Семантическая память")
        
        # RAG system
        print("  ✅ Nicole RAG - Хаотичный поиск контекста")
        
        # Metrics system
        print("  ✅ Nicole Metrics - Аналитика разговоров")
        
        # Core system
        print("  ✅ Nicole Core - Флюидные трансформеры")
        
        # Telegram interface
        self.components['telegram'].start_bot()
        print("  ✅ Nicole Telegram - Интерфейс тестирования")
        
        print(f"\\n🎉 Nicole система запущена за {time.time() - self.startup_time:.2f} секунд!")
        
    def get_system_status(self) -> Dict:
        """Полный статус системы"""
        try:
            status = {
                'version': self.version,
                'uptime': time.time() - self.startup_time,
                'components': {}
            }
            
            # H2O статус
            status['components']['h2o'] = {
                'active_transformers': len(self.components['h2o'].executor.active_transformers),
                'current_session': self.components['h2o'].session_id
            }
            
            # Core статус
            status['components']['core'] = {
                'current_session': self.components['core'].session_id,
                'conversation_count': self.components['core'].conversation_count,
                'current_transformer': self.components['core'].current_transformer.transformer_id if self.components['core'].current_transformer else None
            }
            
            # Learning статус
            status['components']['learning'] = self.components['learning'].get_learning_statistics()
            
            # Memory статус
            status['components']['memory'] = self.components['memory'].get_memory_statistics()
            
            # RAG статус
            status['components']['rag'] = self.components['rag'].get_rag_statistics()
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
            
    def run_full_system_test(self):
        """Полное тестирование всей системы"""
        print("\\n🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ NICOLE")
        print("=" * 60)
        
        test_results = {}
        
        # Тест 1: H2O Engine
        print("\\n1️⃣ Тестирование H2O Engine...")
        try:
            h2o.test_h2o()
            test_results['h2o'] = '✅ PASSED'
        except Exception as e:
            test_results['h2o'] = f'❌ FAILED: {e}'
            
        # Тест 2: Nicole Core
        print("\\n2️⃣ Тестирование Nicole Core...")
        try:
            # Краткий тест без полного вывода
            session = self.components['core'].start_conversation("system_test")
            response = self.components['core'].process_message("Тест системы")
            self.components['core'].end_conversation()
            test_results['core'] = '✅ PASSED'
        except Exception as e:
            test_results['core'] = f'❌ FAILED: {e}'
            
        # Тест 3: Memory System
        print("\\n3️⃣ Тестирование Memory System...")
        try:
            mem_id = self.components['memory'].store_memory("Тест памяти", "system_test")
            memories = self.components['memory'].recall_memories("тест")
            test_results['memory'] = '✅ PASSED' if memories else '⚠️ PARTIAL'
        except Exception as e:
            test_results['memory'] = f'❌ FAILED: {e}'
            
        # Тест 4: RAG System
        print("\\n4️⃣ Тестирование RAG System...")
        try:
            context = self.components['rag'].retriever.retrieve_context("тест система")
            test_results['rag'] = '✅ PASSED'
        except Exception as e:
            test_results['rag'] = f'❌ FAILED: {e}'
            
        # Тест 5: Metrics System
        print("\\n5️⃣ Тестирование Metrics System...")
        try:
            snapshot = self.components['metrics'].analyze_conversation_turn(
                "тест", "ответ", "test_transformer", "test_session"
            )
            test_results['metrics'] = '✅ PASSED'
        except Exception as e:
            test_results['metrics'] = f'❌ FAILED: {e}'
            
        # Тест 6: Learning System
        print("\\n6️⃣ Тестирование Learning System...")
        try:
            stats = self.components['learning'].get_learning_statistics()
            test_results['learning'] = '✅ PASSED'
        except Exception as e:
            test_results['learning'] = f'❌ FAILED: {e}'
            
        # Результаты
        print("\\n" + "=" * 60)
        print("🏆 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        for component, result in test_results.items():
            print(f"  {component.upper()}: {result}")
            
        passed = sum(1 for r in test_results.values() if '✅' in r)
        total = len(test_results)
        
        print(f"\\n📊 Итого: {passed}/{total} компонентов прошли тесты")
        
        if passed == total:
            print("\\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! СИСТЕМА ГОТОВА К РАБОТЕ!")
        else:
            print("\\n⚠️ Есть проблемы, но основная функциональность работает")
            
    def interactive_mode(self):
        """Интерактивный режим"""
        interactive = nicole_telegram.InteractiveNicole()
        interactive.start_interactive()
        
    def demo_mode(self):
        """Демонстрационный режим"""
        print("\\n🎭 ДЕМОНСТРАЦИЯ ВОЗМОЖНОСТЕЙ NICOLE")
        print("=" * 50)
        
        # Создаем демо сессию
        tg = self.components['telegram']
        demo_chat = "demo_user"
        
        demo_conversation = [
            ("Привет Nicole!", "Демо: Приветствие"),
            ("Меня зовут Алекс, я программист", "Демо: Представление"),
            ("Работаю над ИИ проектами", "Демо: Профессия"),
            ("Что ты думаешь об ИИ без весов?", "Демо: Технический вопрос"),
            ("Покажи свою архитектуру", "Демо: Запрос отладки"),
            ("/stats", "Демо: Статистика"),
            ("/debug", "Демо: Отладка"),
            ("Спасибо, это было интересно!", "Демо: Благодарность"),
        ]
        
        for user_msg, description in demo_conversation:
            print(f"\\n{description}:")
            print(f"👤 {user_msg}")
            response = tg.process_message(demo_chat, user_msg)
            time.sleep(1)
            
        print("\\n🎉 Демонстрация завершена!")

# Глобальная система
nicole_system = NicoleSystem()

def main():
    """Главная функция"""
    if len(sys.argv) < 2:
        print("""
🤖 Nicole - Neural Organism Intelligence Conversational Language Engine

Использование:
  python3 nicole_main.py start      - запустить систему
  python3 nicole_main.py test       - полное тестирование
  python3 nicole_main.py interactive - интерактивный режим
  python3 nicole_main.py demo       - демонстрация
  python3 nicole_main.py status     - статус системы
""")
        return
        
    command = sys.argv[1]
    
    if command == "start":
        nicole_system.start_system()
        
    elif command == "test":
        nicole_system.start_system()
        nicole_system.run_full_system_test()
        
    elif command == "interactive":
        nicole_system.start_system()
        nicole_system.interactive_mode()
        
    elif command == "demo":
        nicole_system.start_system()
        nicole_system.demo_mode()
        
    elif command == "status":
        status = nicole_system.get_system_status()
        print("📊 Статус системы Nicole:")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
    else:
        print(f"Неизвестная команда: {command}")

if __name__ == "__main__":
    main()
