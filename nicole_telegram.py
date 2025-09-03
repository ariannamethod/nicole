#!/usr/bin/env python3
"""
Nicole Telegram - Telegram client for testing Nicole
Fluid neural network interface.
"""

import asyncio
import json
import time
import sys
import threading
import random
import os
from typing import Dict, Any, Optional
import sqlite3

# Импортируем все компоненты Nicole
import sys
import os
# Добавляем текущую директорию в путь для импорта наших модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h2o
import nicole
import nicole2nicole  
import nicole_memory
import nicole_rag
import nicole_metrics

# Загружаем переменные окружения
try:
    from load_env import load_env
    load_env()  # Автоматически загружаем .env если есть
except ImportError:
    pass

# Telegram Bot API (если доступен)
try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("[TelegramBot] python-telegram-bot not installed, using mock")

class MockTelegramBot:
    """Mock telegram bot for testing without API"""
    
    def __init__(self):
        self.chat_sessions = {}
        self.message_history = []
        self.is_running = False
        
    def start_polling(self):
        """Starts polling (simulation)"""
        self.is_running = True
        print("[TelegramBot] Bot started (simulation)")
        
    def stop_polling(self):
        """Stops polling"""
        self.is_running = False
        print("[TelegramBot] Bot stopped")
        
    def send_message(self, chat_id: str, text: str):
        """Sends message (simulation)"""
        message = {
            'chat_id': chat_id,
            'text': text,
            'timestamp': time.time(),
            'type': 'bot_message'
        }
        self.message_history.append(message)
        print(f"[Bot -> {chat_id}] {text}")
        
    def simulate_user_message(self, chat_id: str, text: str):
        """Simulates user message"""
        message = {
            'chat_id': chat_id,
            'text': text,
            'timestamp': time.time(),
            'type': 'user_message'
        }
        self.message_history.append(message)
        print(f"[{chat_id} -> Bot] {text}")
        return message

class RealTelegramBot:
    """Real Telegram bot for production"""
    
    def __init__(self, token: str):
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot not installed")
            
        self.token = token
        self.application = Application.builder().token(token).build()
        self.chat_sessions = {}
        self.message_history = []
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command /start"""
        chat_id = str(update.effective_chat.id)
        welcome_msg = """🧠 Hello! I'm NICOLE - Neural Intelligent Conversational Organism Language Engine.

I work without pre-trained weights, creating unique transformers for each dialogue.
I use Method Engine principles for proper speech and resonance.

Commands:
/newconvo - start new conversation"""
        
        await update.message.reply_text(welcome_msg)
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command /help"""
        help_text = """🤖 NICOLE: Neural Intelligent Conversational Organism Language Engine

/newconvo - start new conversation

Just write me messages - I will learn and adapt!"""
        
        await update.message.reply_text(help_text)
        
    async def message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Regular message handler"""
        try:
            chat_id = str(update.effective_chat.id)
            user_input = update.message.text
            
            # Log message
            self.message_history.append({
                'chat_id': chat_id,
                'text': user_input,
                'timestamp': time.time(),
                'type': 'user_message'
            })
            
            # Create Nicole session if none exists
            if chat_id not in self.chat_sessions:
                nicole_core = nicole.NicoleCore()
                nicole_core.start_conversation(f"tg_{chat_id}")
                self.chat_sessions[chat_id] = nicole_core
                print(f"[RealTelegramBot] Created Nicole session for {chat_id}")
            
            # Process through Nicole with ME principles
            nicole_session = self.chat_sessions[chat_id]
            response = nicole_session.process_message(user_input)
            
            # Log response
            self.message_history.append({
                'chat_id': chat_id,
                'text': response,
                'timestamp': time.time(),
                'type': 'bot_message'
            })
            
            await update.message.reply_text(response)
            
        except Exception as e:
            error_msg = f"Nicole Error: {str(e)}"
            print(f"[RealTelegramBot:ERROR] {error_msg}")
            await update.message.reply_text(error_msg)
    
    async def newconvo_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command /newconvo - new conversation"""
        chat_id = str(update.effective_chat.id)
        
        # End current session but preserve memory
        if chat_id in self.chat_sessions:
            old_session = self.chat_sessions[chat_id]
            # Memory stays in SQLite, just create new session
            del self.chat_sessions[chat_id]
            
        await update.message.reply_text("⚡ New conversation started. Memory preserved.")
    
    def setup_handlers(self):
        """Sets up command handlers"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("newconvo", self.newconvo_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler))
        
    def run_bot(self):
        """Runs the bot"""
        self.setup_handlers()
        print(f"[RealTelegramBot] Starting bot with token: {self.token[:10]}...")
        self.application.run_polling()

class NicoleTelegramInterface:
    """Интерфейс Nicole для Telegram"""
    
    def __init__(self):
        self.bot = MockTelegramBot()
        self.nicole_sessions = {}
        self.enhanced_nicole = self._setup_enhanced_nicole()
        self.command_handlers = {
            '/start': self._cmd_start,
            '/help': self._cmd_help,
            '/newconvo': self._cmd_reset
        }
        
    def _setup_enhanced_nicole(self):
        """Настраивает полную Nicole систему"""
        # Создаем интегрированную систему со всеми модулями
        class FullNicole:
            def __init__(self):
                self.core = nicole.nicole_core
                self.learning = nicole2nicole.Nicole2NicoleCore()
                self.memory = nicole_memory.NicoleMemoryCore()
                self.rag = nicole_rag.nicole_rag
                self.metrics = nicole_metrics.nicole_metrics
                
                # Запускаем фоновые процессы
                self.learning.start_continuous_learning()
                self.memory.start_maintenance()
                
            def process_message(self, user_input: str, chat_id: str) -> str:
                """Обрабатывает сообщение через всю систему"""
                
                # Если нет активной сессии - создаем
                if not self.core.session_id:
                    self.core.start_conversation(f"tg_{chat_id}")
                    
                # Получаем контекст из памяти
                memory_context = self.memory.get_conversation_context(user_input)
                
                # Дополняем контекст через RAG
                enhanced_response, rag_context = self.rag.generate_augmented_response(
                    user_input, 
                    strategy=self.rag.get_best_strategy()
                )
                
                # Обрабатываем через основную Nicole
                base_response = self.core.process_message(user_input)
                
                # Улучшаем ответ
                if enhanced_response and enhanced_response != "Программирование это круто":
                    final_response = enhanced_response
                else:
                    final_response = base_response
                    
                # Анализируем метрики
                if self.core.current_transformer:
                    self.metrics.analyze_conversation_turn(
                        user_input,
                        final_response, 
                        self.core.current_transformer.transformer_id,
                        self.core.session_id
                    )
                    
                # Сохраняем в память
                self.memory.learn_from_conversation(user_input, final_response)
                
                return final_response
                
            def get_system_status(self) -> Dict:
                """Статус всей системы"""
                return {
                    'h2o_active_transformers': len(self.core.h2o_engine.executor.active_transformers),
                    'current_session': self.core.session_id,
                    'conversation_count': self.core.conversation_count,
                    'learning_stats': self.learning.get_learning_statistics(),
                    'memory_stats': self.memory.get_memory_statistics(),
                    'rag_stats': self.rag.get_rag_statistics(),
                    'current_transformer': self.core.current_transformer.transformer_id if self.core.current_transformer else None
                }
                
        return FullNicole()
        
    def start_bot(self):
        """Запускает телеграм бота"""
        self.bot.start_polling()
        print("[NicoleTelegram] Интерфейс готов к работе!")
        
    def handle_message(self, chat_id: str, text: str) -> str:
        """Обрабатывает входящее сообщение"""
        try:
            # Проверяем команды
            if text.startswith('/'):
                command = text.split()[0]
                if command in self.command_handlers:
                    return self.command_handlers[command](chat_id, text)
                else:
                    return f"Неизвестная команда: {command}"
                    
            # Обычное сообщение - передаем Nicole
            response = self.enhanced_nicole.process_message(text, chat_id)
            return response
            
        except Exception as e:
            error_msg = f"Ошибка обработки: {e}"
            print(f"[NicoleTelegram:ERROR] {error_msg}")
            return "Извини, произошла ошибка. Попробуй еще раз."
            
    def _cmd_start(self, chat_id: str, text: str) -> str:
        """Команда /start"""
        return """🧠 Hello! I'm NICOLE - Neural Intelligent Conversational Organism Language Engine.

I work without pre-trained weights, creating unique transformers for each dialogue.
I use Method Engine principles for proper speech and resonance.

Commands:
/newconvo - start new conversation

Just write me messages - I will learn and adapt!"""

    def _cmd_help(self, chat_id: str, text: str) -> str:
        """Команда /help"""
        return """🤖 NICOLE: Neural Intelligent Conversational Organism Language Engine

/newconvo - start new conversation

Just write me messages - I will learn and adapt!"""

    def _cmd_stats(self, chat_id: str, text: str) -> str:
        """Команда /stats"""
        try:
            status = self.enhanced_nicole.get_system_status()
            
            stats_text = f"""📊 Статистика Nicole:

🤖 H2O Engine:
• Активных трансформеров: {status['h2o_active_transformers']}
• Текущая сессия: {status['current_session']}
• Сообщений в сессии: {status['conversation_count']}

🧠 Обучение:
• Изученных паттернов: {status['learning_stats'].get('learned_patterns', 0)}
• Предпочтений архитектуры: {status['learning_stats'].get('architecture_preferences', 0)}

💾 Память:
• Всего воспоминаний: {status['memory_stats'].get('total_memories', 0)}
• Ассоциаций: {status['memory_stats'].get('total_associations', 0)}

🔍 RAG:
• Запросов: {status['rag_stats'].get('total_queries', 0)}
• Фактор хаоса: {status['rag_stats'].get('chaos_factor', 0):.3f}

🎯 Текущий трансформер: {status['current_transformer'] or 'Нет'}"""

            return stats_text
            
        except Exception as e:
            return f"Ошибка получения статистики: {e}"
            
    def _cmd_reset(self, chat_id: str, text: str) -> str:
        """Команда /newconvo"""
        try:
            if self.enhanced_nicole.core.session_id:
                self.enhanced_nicole.core.end_conversation()
                
            return "⚡ New conversation started. Memory preserved."
            
        except Exception as e:
            return f"Error: {e}"

    def process_message(self, chat_id: str, message_text: str) -> str:
        """Обрабатывает сообщение от пользователя"""
        # Логируем входящее сообщение
        user_message = self.bot.simulate_user_message(chat_id, message_text)
        
        # Обрабатываем сообщение
        response = self.handle_message(chat_id, message_text)
        
        # Отправляем ответ
        self.bot.send_message(chat_id, response)
        
        return response

def test_telegram_interface():
    """Тестирование телеграм интерфейса"""
    print("=== NICOLE TELEGRAM INTERFACE TEST ===")
    
    # Создаем интерфейс
    tg_interface = NicoleTelegramInterface()
    tg_interface.start_bot()
    
    # Симулируем пользователя
    test_chat_id = "test_user_123"
    
    # Тест команд
    print("\\n--- Тест команд ---")
    commands_to_test = [
        "/start",
        "/help",
        "/newconvo"
    ]
    
    for cmd in commands_to_test:
        print(f"\\n> {cmd}")
        response = tg_interface.process_message(test_chat_id, cmd)
        time.sleep(0.2)
        
    # Тест обычного разговора
    print("\\n--- Тест разговора ---")
    conversation = [
        "Привет Nicole! Меня зовут Тестер",
        "Я изучаю нейронные сети",
        "Расскажи о себе",
        "Как ты работаешь без весов?",
        "Это очень интересно!",
        "Покажи свою эволюцию"
    ]
    
    for msg in conversation:
        print(f"\\n> {msg}")
        response = tg_interface.process_message(test_chat_id, msg)
        time.sleep(0.3)
        
    # Финальная статистика
    print("\\n--- Финальная статистика ---")
    final_stats = tg_interface.process_message(test_chat_id, "/stats")
    
    print("\\n--- Тест эволюции ---")
    evolution_result = tg_interface.process_message(test_chat_id, "/evolve")
    
    print("\\n--- Память после разговора ---")
    memory_result = tg_interface.process_message(test_chat_id, "/memory")
    
    print("\\n=== TELEGRAM TEST COMPLETED ===")

class InteractiveNicole:
    """Интерактивный режим для консоли"""
    
    def __init__(self):
        self.tg_interface = NicoleTelegramInterface()
        self.chat_id = "console_user"
        
    def start_interactive(self):
        """Запускает интерактивный режим"""
        print("🤖 Nicole Interactive Mode")
        print("Введите 'quit' для выхода")
        print("-" * 40)
        
        self.tg_interface.start_bot()
        
        # Приветствие
        welcome = self.tg_interface.process_message(self.chat_id, "/start")
        
        while True:
            try:
                user_input = input("\\n👤 Ты: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'выход']:
                    print("👋 До свидания!")
                    break
                    
                if not user_input:
                    continue
                    
                # Обрабатываем сообщение
                response = self.tg_interface.process_message(self.chat_id, user_input)
                
            except KeyboardInterrupt:
                print("\\n\\n👋 До свидания!")
                break
            except Exception as e:
                print(f"Ошибка: {e}")

def run_production_bot():
    """Запускает продакшен бота с настоящим Telegram API"""
    token = os.getenv('TELEGRAM_TOKEN')
    if not token:
        print("❌ TELEGRAM_TOKEN не найден в переменных окружения!")
        print("Создайте .env файл или установите переменную окружения")
        return
        
    if not TELEGRAM_AVAILABLE:
        print("❌ python-telegram-bot не установлен!")
        print("Установите: pip install python-telegram-bot")
        return
        
    print("🚀 Запускаем Nicole Production Telegram Bot...")
    bot = RealTelegramBot(token)
    bot.run_bot()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_telegram_interface()
        elif sys.argv[1] == "interactive":
            interactive = InteractiveNicole()
            interactive.start_interactive()
        elif sys.argv[1] == "bot":
            run_production_bot()
    else:
        print("Nicole Telegram Interface")
        print("Команды:")
        print("  python3 nicole_telegram.py test - тестирование")
        print("  python3 nicole_telegram.py interactive - интерактивный режим") 
        print("  python3 nicole_telegram.py bot - продакшен бот")
