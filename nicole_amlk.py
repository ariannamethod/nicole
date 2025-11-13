"""
Nicole AMLK Integration Module
Integration of Nicole with Arianna Method Linux Kernel via letsgo.py

Nicole lives INSIDE AMLK as an operating system.
AMLK provides system functions through the letsgo.py terminal.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import threading
import queue
import subprocess

# Add AMLK to path for imports
AMLK_PATH = Path(__file__).parent / "AMLK"
sys.path.insert(0, str(AMLK_PATH))

class NicoleAMLKBridge:
    """
    Bridge between Nicole and AMLK operating system
    Nicole operates INSIDE AMLK, using its system calls
    """
    
    def __init__(self):
        self.amlk_process = None
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_running = False
        self.log_file = "amlk_system.log"
        
    def start_amlk_os(self):
        """Launch AMLK operating system"""
        try:
            # Launch letsgo.py as a system process
            self.amlk_process = subprocess.Popen(
                [sys.executable, str(AMLK_PATH / "letsgo.py")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self.is_running = True

            # Start output monitoring in a separate thread
            self._start_output_monitor()
            
            return True
        except Exception as e:
            # Log error instead of user output
            self._log_error(f"AMLK startup failed: {e}")
            return False
    
    def _start_output_monitor(self):
        """AMLK output monitoring in separate thread"""
        def monitor():
            while self.is_running and self.amlk_process:
                try:
                    line = self.amlk_process.stdout.readline()
                    if line:
                        self.response_queue.put(line.strip())
                except:
                    break
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def execute_system_command(self, command: str) -> Optional[str]:
        """
        Execute system command through AMLK
        Nicole uses this for system operations
        """
        if not self.is_running or not self.amlk_process:
            return None

        try:
            # Send command to AMLK
            self.amlk_process.stdin.write(f"{command}\n")
            self.amlk_process.stdin.flush()

            # Wait for response (with timeout)
            try:
                response = self.response_queue.get(timeout=5.0)
                return response
            except queue.Empty:
                return None

        except Exception as e:
            print(f"Error executing command in AMLK: {e}")
            return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get AMLK system information"""
        info = {}

        # Basic information via system commands
        commands = {
            'pwd': 'pwd',
            'ls': 'ls -la',
            'memory': 'free -h' if os.name != 'nt' else 'dir',
            'processes': 'ps aux' if os.name != 'nt' else 'tasklist'
        }
        
        for key, cmd in commands.items():
            result = self.execute_system_command(cmd)
            if result:
                info[key] = result
                
        return info
    
    def nicole_system_call(self, operation: str, **kwargs) -> Any:
        """
        Nicole system calls through AMLK
        This is the main interface for Nicole to use the OS
        """
        if operation == "file_ops":
            # File operations
            action = kwargs.get('action')
            path = kwargs.get('path')
            
            if action == 'read':
                return self.execute_system_command(f"cat {path}")
            elif action == 'write':
                content = kwargs.get('content', '')
                # Use echo for writing (safe for simple content)
                return self.execute_system_command(f'echo "{content}" > {path}')
            elif action == 'list':
                return self.execute_system_command(f"ls -la {path}")

        elif operation == "process_ops":
            # Processes and memory
            action = kwargs.get('action')
            
            if action == 'list':
                return self.execute_system_command("ps aux")
            elif action == 'memory':
                return self.execute_system_command("free -h")

        elif operation == "network_ops":
            # Network operations
            action = kwargs.get('action')
            
            if action == 'status':
                return self.execute_system_command("netstat -an")
                
        return None

    def shutdown_amlk(self):
        """Graceful AMLK shutdown"""
        if self.amlk_process:
            try:
                self.amlk_process.stdin.write("exit\n")
                self.amlk_process.stdin.flush()
                self.amlk_process.wait(timeout=5)
            except:
                self.amlk_process.terminate()
            finally:
                self.is_running = False

    def _log_info(self, message: str):
        """Logging for system, not for user"""
        with open(self.log_file, "a") as f:
            f.write(f"[AMLK:INFO] {message}\n")

    def _log_error(self, message: str):
        """Error logging for system"""
        with open(self.log_file, "a") as f:
            f.write(f"[AMLK:ERROR] {message}\n")

# Global bridge instance
_amlk_bridge = None

def get_amlk_bridge() -> NicoleAMLKBridge:
    """Get global AMLK bridge instance"""
    global _amlk_bridge
    if _amlk_bridge is None:
        _amlk_bridge = NicoleAMLKBridge()
    return _amlk_bridge

def start_nicole_in_amlk():
    """
    Launch Nicole inside AMLK operating system
    This is the main integration function
    """
    bridge = get_amlk_bridge()

    if bridge.start_amlk_os():
        # Get system information for internal use
        sys_info = bridge.get_system_info()
        bridge._log_info(f"AMLK OS active, sys_params: {len(sys_info)}")
        
        return bridge
    else:
        return None

if __name__ == "__main__":
    # Test integration launch
    bridge = start_nicole_in_amlk()

    if bridge:
        # Test system calls
        print("\n🔧 System operations test:")

        # Test file operations
        result = bridge.nicole_system_call("file_ops", action="list", path=".")
        print(f"File list: {result}")

        # Test processes
        result = bridge.nicole_system_call("process_ops", action="memory")
        print(f"System memory: {result}")

        # Shutdown
        bridge.shutdown_amlk()
        print("🏁 AMLK shutdown complete")
