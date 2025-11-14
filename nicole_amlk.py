"""
Nicole AMLK Integration Module
Integration of Nicole with Arianna Method Linux Kernel via amlk_api.py

Nicole uses AMLK system functions through clean programmatic API.
No subprocess spawning, direct system calls.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add AMLK to path for imports
AMLK_PATH = Path(__file__).parent / "AMLK"
sys.path.insert(0, str(AMLK_PATH))

# Import AMLK API
try:
    from amlk_api import get_amlk_api, AMLKSystemAPI
    AMLK_API_AVAILABLE = True
except ImportError:
    AMLK_API_AVAILABLE = False
    print("[Nicole:AMLK] Warning: amlk_api not available")

class NicoleAMLKBridge:
    """
    Bridge between Nicole and AMLK operating system
    Nicole operates WITH AMLK, using its system calls
    """

    def __init__(self):
        self.amlk_api = None
        self.is_running = False
        self.log_file = "amlk_system.log"

    def start_amlk_os(self):
        """Initialize AMLK API connection"""
        if not AMLK_API_AVAILABLE:
            self._log_error("AMLK API not available")
            return False

        try:
            # Get AMLK API instance (no subprocess needed!)
            self.amlk_api = get_amlk_api()
            self.is_running = True

            # Log system info
            sys_info = self.amlk_api.get_system_info()
            self._log_info(f"AMLK API initialized: {sys_info.get('platform')}")

            return True
        except Exception as e:
            self._log_error(f"AMLK API init failed: {e}")
            return False

    def execute_system_command(self, command: str) -> Optional[str]:
        """
        Execute system command through AMLK API
        Nicole uses this for system operations
        """
        if not self.is_running or not self.amlk_api:
            return None

        try:
            # Direct API call (no pipes, no subprocess!)
            return self.amlk_api.execute_command(command)
        except Exception as e:
            self._log_error(f"Command execution error: {e}")
            return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get AMLK system information"""
        if not self.is_running or not self.amlk_api:
            return {}

        try:
            # Use API method directly
            return self.amlk_api.get_system_info()
        except Exception as e:
            self._log_error(f"System info error: {e}")
            return {}
    
    def nicole_system_call(self, operation: str, **kwargs) -> Any:
        """
        Nicole system calls through AMLK API
        This is the main interface for Nicole to use the OS
        """
        if not self.is_running or not self.amlk_api:
            return None

        try:
            if operation == "file_ops":
                # Use API file operations
                action = kwargs.get('action')
                path = kwargs.get('path')
                content = kwargs.get('content')
                return self.amlk_api.file_operations(action, path, content)

            elif operation == "process_ops":
                # Use API process operations
                action = kwargs.get('action')
                return self.amlk_api.process_operations(action)

            elif operation == "network_ops":
                # Use API network operations
                action = kwargs.get('action')
                return self.amlk_api.network_operations(action)

            else:
                return None

        except Exception as e:
            self._log_error(f"System call error: {e}")
            return None

    def shutdown_amlk(self):
        """Graceful AMLK shutdown"""
        if self.is_running:
            self._log_info("AMLK API shutdown")
            self.is_running = False
            self.amlk_api = None

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
