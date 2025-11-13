#!/usr/bin/env python3
"""
Simple loading of environment variables from .env file
No external dependencies like python-dotenv
"""

import os

def load_env(env_file=".env"):
    """Load variables from .env file"""
    if not os.path.exists(env_file):
        print(f"⚠️  {env_file} not found")
        return False
        
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                        
                    os.environ[key] = value
                    print(f"✅ {key} = {value[:10]}..." if len(value) > 10 else f"✅ {key} = {value}")
                    
        return True

    except Exception as e:
        print(f"❌ Error loading {env_file}: {e}")
        return False

if __name__ == "__main__":
    load_env()
