#!/usr/bin/env python3
"""
Простая загрузка переменных окружения из .env файла
Без внешних зависимостей типа python-dotenv
"""

import os

def load_env(env_file=".env"):
    """Загружает переменные из .env файла"""
    if not os.path.exists(env_file):
        print(f"⚠️  {env_file} не найден")
        return False
        
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Пропускаем комментарии и пустые строки
                if not line or line.startswith('#'):
                    continue
                    
                # Парсим KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Убираем кавычки если есть
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                        
                    os.environ[key] = value
                    print(f"✅ {key} = {value[:10]}..." if len(value) > 10 else f"✅ {key} = {value}")
                    
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки {env_file}: {e}")
        return False

if __name__ == "__main__":
    load_env()
