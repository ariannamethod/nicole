#!/usr/bin/env python3
"""
Nicole Startup Script - Main system launcher
Automatically checks dependencies and starts the appropriate mode.
"""

import os
import sys
import subprocess

def check_dependencies():
    try:
        import telegram
        telegram_ok = True
    except ImportError:
        telegram_ok = False
    
    token = os.getenv('TELEGRAM_TOKEN')
    token_ok = bool(token)
        
    return telegram_ok, token_ok

def install_dependencies():
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🧠 NICOLE")
    print("Neural Intelligent Conversational Organism Language Engine")
    print()
    
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    
    if mode == "local":
        os.system("python3 nicole_telegram.py interactive")
    elif mode == "test":
        os.system("python3 h2o.py test")
        os.system("python3 nicole.py test")
        os.system("python3 nicole_telegram.py test")
    elif mode == "bot":
        telegram_ok, token_ok = check_dependencies()
        if not telegram_ok and not install_dependencies():
            return
        if not token_ok:
            print("❌ TELEGRAM_TOKEN not found. Check README.")
            return
        os.system("python3 nicole_telegram.py bot")
    else:
        print("Usage:")
        print("  python3 start_nicole.py local")
        print("  python3 start_nicole.py test") 
        print("  python3 start_nicole.py bot")

if __name__ == "__main__":
    main()
