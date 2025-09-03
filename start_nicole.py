#!/usr/bin/env python3
"""
Nicole Startup Script - Главный запускальщик системы
Автоматически проверяет зависимости и запускает нужный режим.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Проверяет зависимости"""
    print("🔍 Проверяем зависимости...")
    
    # Проверяем python-telegram-bot
    try:
        import telegram
        print("✅ python-telegram-bot установлен")
        telegram_ok = True
    except ImportError:
        print("❌ python-telegram-bot НЕ установлен")
        telegram_ok = False
    
    # Проверяем переменные окружения
    token = os.getenv('TELEGRAM_TOKEN')
    if token:
        print(f"✅ TELEGRAM_TOKEN найден: {token[:10]}...")
        token_ok = True
    else:
        print("❌ TELEGRAM_TOKEN не найден")
        token_ok = False
        
    return telegram_ok, token_ok

def install_dependencies():
    """Устанавливает зависимости"""
    print("📦 Устанавливаем зависимости...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Зависимости установлены")
        return True
    except subprocess.CalledProcessError:
        print("❌ Ошибка установки зависимостей")
        return False

def main():
    print("🧠 === NICOLE STARTUP SYSTEM ===")
    print("Neural Organism Intelligence Conversational Language Engine")
    print("Посвящается Лео 💙")
    print()
    
    # Проверяем режим запуска
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    
    if mode == "local":
        print("🏠 Локальный режим - запускаем интерактивную консоль")
        os.system("python3 nicole_telegram.py interactive")
        
    elif mode == "test":
        print("🧪 Тестовый режим - проверяем все модули")
        print("\n--- Тест H2O ---")
        os.system("python3 h2o.py test")
        print("\n--- Тест Nicole Core ---")  
        os.system("python3 nicole.py test")
        print("\n--- Тест Telegram Interface ---")
        os.system("python3 nicole_telegram.py test")
        
    elif mode == "bot":
        print("🤖 Продакшен режим - запускаем Telegram бота")
        telegram_ok, token_ok = check_dependencies()
        
        if not telegram_ok:
            print("Устанавливаем зависимости...")
            if not install_dependencies():
                return
                
        if not token_ok:
            print("\n❌ Нужно настроить TELEGRAM_TOKEN!")
            print("1. Создайте бота у @BotFather")
            print("2. Скопируйте env.example в .env")
            print("3. Впишите токен в .env файл")
            print("4. Или установите переменную: export TELEGRAM_TOKEN=your_token")
            return
            
        os.system("python3 nicole_telegram.py bot")
        
    else:
        print("🎯 Доступные режимы:")
        print("  python3 start_nicole.py local - интерактивная консоль")
        print("  python3 start_nicole.py test - тестирование всех модулей") 
        print("  python3 start_nicole.py bot - Telegram бот")
        print("\n💡 Для первого запуска рекомендую: python3 start_nicole.py test")

if __name__ == "__main__":
    main()
