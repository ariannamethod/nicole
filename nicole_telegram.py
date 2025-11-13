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

# Import all Nicole components
import sys
import os
# Add current directory to path for importing our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h2o

# CRITICAL: import high and blood with error handling
try:
    import high
    print("[TELEGRAM] High imported successfully")
except ImportError as e:
    print(f"[TELEGRAM] High NOT imported: {e}")

try:
    import blood
    print("[TELEGRAM] Blood imported successfully")
except ImportError as e:
    print(f"[TELEGRAM] Blood NOT imported: {e}")

import nicole
import nicole2nicole
import nicole_memory
import nicole_rag
import nicole_metrics

# NEW: Enable repo learning for automatic training on changes
try:
    from nicole_repo_learner import start_repo_learning
    REPO_LEARNING_AVAILABLE = True
    print("[TELEGRAM] Repo learning imported successfully")
except ImportError as e:
    REPO_LEARNING_AVAILABLE = False
    print(f"[TELEGRAM] Repo learning NOT imported: {e}")

# Load environment variables
try:
    from load_env import load_env
    load_env()  # Automatically load .env if exists
except ImportError:
    pass

# Telegram Bot API (if available)
try:
    from telegram import Update, Bot, BotCommand
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

        # NEW: Start repo learning for automatic training
        if REPO_LEARNING_AVAILABLE:
            try:
                print("[RealTelegramBot] 🧠 Starting repo learning...")
                self.repo_learner = start_repo_learning(
                    repo_path=".",
                    check_interval=300  # Check every 5 minutes
                )
                print("[RealTelegramBot] ✅ Repo learning activated!")

                # INITIAL TRAINING: Immediately consume all existing markdown
                print("[RealTelegramBot] 📚 Starting initial training on markdown...")
                self._initial_markdown_learning()

            except Exception as e:
                print(f"[RealTelegramBot] ⚠️ Failed to start repo learning: {e}")
                self.repo_learner = None
        else:
            print("[RealTelegramBot] ⚠️ Repo learning unavailable")
            self.repo_learner = None

    def _initial_markdown_learning(self):
        """Initial training on all existing markdown files"""
        try:
            from pathlib import Path
            import re

            # Find all markdown files in repo
            repo_path = Path(".")
            markdown_files = list(repo_path.glob("**/*.md"))

            if not markdown_files:
                print("[RealTelegramBot] Markdown files not found")
                return

            print(f"[RealTelegramBot] Found {len(markdown_files)} markdown files")

            # Read and learn
            learned_words = set()
            for md_file in markdown_files:
                try:
                    # Skip huge files
                    if md_file.stat().st_size > 100000:  # > 100KB
                        continue

                    content = md_file.read_text(encoding='utf-8', errors='ignore')

                    # Extract words (minimum 3 chars, only Latin)
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())

                    # Add to memory word_frequencies
                    for word in words:
                        if len(word) > 15:  # Skip very long words
                            continue
                        learned_words.add(word)
                        # Update frequencies through Nicole memory
                        nicole.nicole_core.memory.update_word_frequencies(word)

                except Exception as e:
                    print(f"[RealTelegramBot] Error reading {md_file}: {e}")
                    continue

            print(f"[RealTelegramBot] ✅ Initial training: {len(learned_words)} unique words from {len(markdown_files)} files")

        except Exception as e:
            print(f"[RealTelegramBot] Error in initial training: {e}")
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command /start"""
        chat_id = str(update.effective_chat.id)

        # Set menu button on first /start
        try:
            commands = [BotCommand("newconvo", "RESTART")]
            await self.application.bot.set_my_commands(commands)
        except Exception as e:
            print(f"[RealTelegramBot] Menu setup failed: {e}")
        
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
            
            # FIXED: DO NOT restart session on every message!
            if chat_id not in self.chat_sessions:
                # Create session ONLY ONCE for new chat
                chat_session_id = f"tg_{chat_id}"
                nicole.nicole_core.start_conversation(chat_session_id)
                self.chat_sessions[chat_id] = True  # Mark session created
                print(f"[RealTelegramBot] CREATED new session for {chat_id} - High: {nicole.nicole_core.high_enabled}")
            else:
                # Session already created - DON'T TOUCH IT!
                print(f"[RealTelegramBot] Using existing session {nicole.nicole_core.session_id}")

            # DIAGNOSTICS: check system state before processing
            print(f"[DIAGNOSTICS] High enabled: {nicole.nicole_core.high_enabled}")
            print(f"[DIAGNOSTICS] High is_active: {nicole.nicole_core.high_core.is_active if nicole.nicole_core.high_core else 'None'}")
            print(f"[DIAGNOSTICS] Message length: {len(user_input)} characters")
            
            # Process through Nicole with ME principles
            response = nicole.nicole_core.process_message(user_input)

            # DIAGNOSTICS: check result
            print(f"[DIAGNOSTICS] Response length: {len(response)} characters")
            print(f"[DIAGNOSTICS] Response: {response[:100]}...")
            
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

        # Start polling - menu button will be configured automatically
        self.application.run_polling()

class NicoleTelegramInterface:
    """Nicole interface for Telegram"""
    
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
        """FIXED: Telegram simply uses main Nicole system"""
        class TelegramNicole:
            def __init__(self):
                # Use global instance, don't create own modules
                self.core = nicole.nicole_core

            def process_message(self, user_input: str, chat_id: str) -> str:
                """Simple message forwarding to main Nicole system"""

                # Check/create session for this chat
                expected_session = f"tg_{chat_id}"
                if not self.core.session_id or self.core.session_id != expected_session:
                    print(f"[TelegramInterface] Creating session {expected_session}")
                    self.core.start_conversation(expected_session)
                else:
                    print(f"[TelegramInterface] Session {self.core.session_id} active")

                # Pass to main system - it will do everything
                return self.core.process_message(user_input)
                
            def get_system_status(self) -> Dict:
                """Status of core Nicole system"""
                return {
                    'h2o_active_transformers': len(self.core.h2o_engine.executor.active_transformers),
                    'current_session': self.core.session_id,
                    'conversation_count': self.core.conversation_count,
                    'high_enabled': self.core.high_enabled,
                    'current_transformer': self.core.current_transformer.transformer_id if self.core.current_transformer else None
                }
                
        return TelegramNicole()
        
    def start_bot(self):
        """Starts telegram bot"""
        self.bot.start_polling()
        print("[NicoleTelegram] Interface ready!")
        
    def handle_message(self, chat_id: str, text: str) -> str:
        """Handles incoming message"""
        try:
            # Check commands
            if text.startswith('/'):
                command = text.split()[0]
                if command in self.command_handlers:
                    return self.command_handlers[command](chat_id, text)
                else:
                    return f"Unknown command: {command}"
                    
            # Regular message - pass to Nicole
            response = self.enhanced_nicole.process_message(text, chat_id)
            return response
            
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"[NicoleTelegram:ERROR] {error_msg}")
            return "Sorry, an error occurred. Please try again."
            
    def _cmd_start(self, chat_id: str, text: str) -> str:
        """Command /start"""
        return """🧠 Hello! I'm NICOLE - Neural Intelligent Conversational Organism Language Engine.

I work without pre-trained weights, creating unique transformers for each dialogue.
I use Method Engine principles for proper speech and resonance.

Commands:
/newconvo - start new conversation

Just write me messages - I will learn and adapt!"""

    def _cmd_help(self, chat_id: str, text: str) -> str:
        """Command /help"""
        return """🤖 NICOLE: Neural Intelligent Conversational Organism Language Engine

/newconvo - start new conversation

Just write me messages - I will learn and adapt!"""

    def _cmd_stats(self, chat_id: str, text: str) -> str:
        """Command /stats"""
        try:
            status = self.enhanced_nicole.get_system_status()

            stats_text = f"""📊 Nicole Statistics:

🤖 H2O Engine:
• Active transformers: {status['h2o_active_transformers']}
• Current session: {status['current_session']}
• Messages in session: {status['conversation_count']}

🧠 Learning:
• Learned patterns: {status['learning_stats'].get('learned_patterns', 0)}
• Architecture preferences: {status['learning_stats'].get('architecture_preferences', 0)}

💾 Memory:
• Total memories: {status['memory_stats'].get('total_memories', 0)}
• Associations: {status['memory_stats'].get('total_associations', 0)}

🔍 RAG:
• Queries: {status['rag_stats'].get('total_queries', 0)}
• Chaos factor: {status['rag_stats'].get('chaos_factor', 0):.3f}

🎯 Current transformer: {status['current_transformer'] or 'None'}"""

            return stats_text

        except Exception as e:
            return f"Error getting statistics: {e}"
            
    def _cmd_reset(self, chat_id: str, text: str) -> str:
        """Command /newconvo"""
        try:
            if self.enhanced_nicole.core.session_id:
                self.enhanced_nicole.core.end_conversation()
                
            return "⚡ New conversation started. Memory preserved."
            
        except Exception as e:
            return f"Error: {e}"

    def process_message(self, chat_id: str, message_text: str) -> str:
        """Processes user message"""
        # Log incoming message
        user_message = self.bot.simulate_user_message(chat_id, message_text)
        
        # Process message
        response = self.handle_message(chat_id, message_text)
        
        # Send response
        self.bot.send_message(chat_id, response)
        
        return response

def test_telegram_interface():
    """Testing telegram interface"""
    print("=== NICOLE TELEGRAM INTERFACE TEST ===")
    
    # Create interface
    tg_interface = NicoleTelegramInterface()
    tg_interface.start_bot()
    
    # Simulate user
    test_chat_id = "test_user_123"
    
    # Test commands
    print("\\n--- Command Test ---")
    commands_to_test = [
        "/start",
        "/help",
        "/newconvo"
    ]
    
    for cmd in commands_to_test:
        print(f"\\n> {cmd}")
        response = tg_interface.process_message(test_chat_id, cmd)
        time.sleep(0.2)
        
    # Test regular conversation
    print("\\n--- Conversation Test ---")
    conversation = [
        "Hello Nicole! My name is Tester",
        "I study neural networks",
        "Tell me about yourself",
        "How do you work without weights?",
        "This is very interesting!",
        "Show me your evolution"
    ]
    
    for msg in conversation:
        print(f"\\n> {msg}")
        response = tg_interface.process_message(test_chat_id, msg)
        time.sleep(0.3)
        
    # Final statistics
    print("\\n--- Final Statistics ---")
    final_stats = tg_interface.process_message(test_chat_id, "/stats")
    
    print("\\n--- Evolution Test ---")
    evolution_result = tg_interface.process_message(test_chat_id, "/evolve")
    
    print("\\n--- Memory After Conversation ---")
    memory_result = tg_interface.process_message(test_chat_id, "/memory")
    
    print("\\n=== TELEGRAM TEST COMPLETED ===")

class InteractiveNicole:
    """Interactive mode for console"""

    def __init__(self):
        self.tg_interface = NicoleTelegramInterface()
        self.chat_id = "console_user"

    def start_interactive(self):
        """Starts interactive mode"""
        print("🤖 Nicole Interactive Mode")
        print("Enter 'quit' to exit")
        print("-" * 40)
        
        self.tg_interface.start_bot()
        
        # Welcome
        welcome = self.tg_interface.process_message(self.chat_id, "/start")
        
        while True:
            try:
                user_input = input("\\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                    
                if not user_input:
                    continue
                    
                # Process message
                response = self.tg_interface.process_message(self.chat_id, user_input)
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def run_production_bot():
    """Starts production bot with real Telegram API"""
    token = os.getenv('TELEGRAM_TOKEN')
    if not token:
        print("❌ TELEGRAM_TOKEN not found in environment variables!")
        print("Create .env file or set environment variable")
        return

    if not TELEGRAM_AVAILABLE:
        print("❌ python-telegram-bot not installed!")
        print("Install: pip install python-telegram-bot")
        return

    print("🚀 Starting Nicole Production Telegram Bot...")
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
        print("Commands:")
        print("  python3 nicole_telegram.py test - testing")
        print("  python3 nicole_telegram.py interactive - interactive mode") 
        print("  python3 nicole_telegram.py bot - production bot")
