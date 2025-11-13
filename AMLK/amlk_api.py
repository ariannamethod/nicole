#!/usr/bin/env python3
"""
AMLK API - Programmatic Interface
Non-interactive AMLK system calls for Nicole integration

Unlike letsgo.py (interactive terminal), this provides clean API
for programmatic access to AMLK system functions.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional


class AMLKSystemAPI:
    """
    Programmatic AMLK system interface

    Provides system functions without interactive terminal overhead.
    Nicole calls this instead of spawning letsgo.py subprocess.
    """

    def __init__(self):
        self.platform = platform.system()
        self.is_linux = self.platform == "Linux"

    def execute_command(self, command: str, timeout: int = 5) -> Optional[str]:
        """
        Execute system command and return output

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds

        Returns:
            Command output as string, or None on error
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Return stdout if successful, stderr if failed
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                # Return stderr for debugging
                return f"Error: {result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            return f"Error: Command timeout ({timeout}s)"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'platform': self.platform,
            'python_version': sys.version,
            'cwd': os.getcwd(),
        }

        # Linux-specific info
        if self.is_linux:
            info['kernel'] = self.execute_command('uname -r')
            info['memory_total'] = self.execute_command("free -h | grep Mem | awk '{print $2}'")
            info['memory_used'] = self.execute_command("free -h | grep Mem | awk '{print $3}'")
            info['cpu_count'] = os.cpu_count()
            info['load_avg'] = os.getloadavg() if hasattr(os, 'getloadavg') else None

        return info

    def file_operations(self, action: str, path: str, content: str = None) -> Optional[str]:
        """
        File operations through AMLK

        Actions:
            - read: Read file content
            - write: Write content to file
            - list: List directory
            - exists: Check if path exists
        """
        try:
            path_obj = Path(path)

            if action == 'read':
                if path_obj.exists() and path_obj.is_file():
                    return path_obj.read_text()
                else:
                    return f"Error: File not found: {path}"

            elif action == 'write':
                if content is not None:
                    path_obj.write_text(content)
                    return f"Written {len(content)} bytes to {path}"
                else:
                    return "Error: No content provided"

            elif action == 'list':
                if path_obj.exists() and path_obj.is_dir():
                    items = [str(p.relative_to(path_obj)) for p in path_obj.iterdir()]
                    return '\n'.join(sorted(items))
                else:
                    return f"Error: Directory not found: {path}"

            elif action == 'exists':
                return str(path_obj.exists())

            else:
                return f"Error: Unknown action: {action}"

        except Exception as e:
            return f"Error: {str(e)}"

    def process_operations(self, action: str) -> Optional[str]:
        """
        Process and memory operations

        Actions:
            - list: List processes
            - memory: Memory usage
            - cpu: CPU usage
        """
        if not self.is_linux:
            return "Error: Process ops only supported on Linux"

        if action == 'list':
            return self.execute_command("ps aux | head -20")

        elif action == 'memory':
            return self.execute_command("free -h")

        elif action == 'cpu':
            return self.execute_command("top -bn1 | head -20")

        else:
            return f"Error: Unknown action: {action}"

    def network_operations(self, action: str) -> Optional[str]:
        """
        Network operations

        Actions:
            - status: Network status
            - interfaces: Network interfaces
            - ping: Ping test
        """
        if not self.is_linux:
            return "Error: Network ops only supported on Linux"

        if action == 'status':
            return self.execute_command("netstat -an | head -20")

        elif action == 'interfaces':
            return self.execute_command("ip addr show")

        elif action == 'ping':
            return self.execute_command("ping -c 3 8.8.8.8")

        else:
            return f"Error: Unknown action: {action}"


# Global API instance
_api = None

def get_amlk_api() -> AMLKSystemAPI:
    """Get global AMLK API instance"""
    global _api
    if _api is None:
        _api = AMLKSystemAPI()
    return _api


if __name__ == "__main__":
    # Test API
    print("=== AMLK API TEST ===")

    api = get_amlk_api()

    print("\n--- System Info ---")
    info = api.get_system_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    print("\n--- File Operations ---")
    print("PWD:", api.execute_command("pwd"))
    print("LS:", api.file_operations("list", "."))

    print("\n--- Process Operations ---")
    print("Memory:", api.process_operations("memory"))

    print("\n✅ API test complete")
