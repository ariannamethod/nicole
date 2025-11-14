"""
Nicole AMLK Integration Module
Clean integration with AMLK System API

Nicole uses AMLK for system operations (files, processes, info).
This module provides clean Python API without subprocess overhead.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add AMLK to path for imports
AMLK_PATH = Path(__file__).parent / "AMLK"
sys.path.insert(0, str(AMLK_PATH))

# Import AMLK API
try:
    from amlk_api import AMLKSystemAPI, get_amlk_api
    AMLK_AVAILABLE = True
except ImportError as e:
    print(f"[AMLK] Warning: AMLK API not available: {e}")
    AMLK_AVAILABLE = False
    AMLKSystemAPI = None
    get_amlk_api = None


class NicoleAMLKBridge:
    """
    Bridge between Nicole and AMLK operating system

    Provides clean Python API for system operations.
    No subprocess overhead - direct Python calls.
    """

    def __init__(self):
        self.api = get_amlk_api() if AMLK_AVAILABLE else None
        self.is_available = AMLK_AVAILABLE

        if not self.is_available:
            print("[AMLK] Running without AMLK integration")

    # ==================== SYSTEM INFO ====================

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get complete system status

        Returns:
            Dict with CPU, memory, disk, network info
        """
        if not self.api:
            return {'error': 'AMLK not available'}

        return self.api.get_status()

    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU load information"""
        if not self.api:
            return {'error': 'AMLK not available'}

        return self.api.get_cpu_load()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        if not self.api:
            return {'error': 'AMLK not available'}

        return self.api.get_memory_info()

    def get_disk_usage(self, path: str = "/") -> Dict[str, Any]:
        """Get disk usage information"""
        if not self.api:
            return {'error': 'AMLK not available'}

        return self.api.get_disk_usage(path)

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        if not self.api:
            return {'error': 'AMLK not available'}

        return self.api.get_network_info()

    # ==================== EXECUTION ====================

    def execute_command(self, command: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute shell command

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds

        Returns:
            Dict with stdout, stderr, success status
        """
        if not self.api:
            return {'success': False, 'error': 'AMLK not available'}

        return self.api.execute_shell(command, timeout)

    def execute_python(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute Python code

        Args:
            code: Python code to execute
            timeout: Execution timeout

        Returns:
            Dict with output and status
        """
        if not self.api:
            return {'success': False, 'error': 'AMLK not available'}

        return self.api.execute_python(code, timeout)

    # ==================== FILE OPERATIONS ====================

    def read_file(self, path: str) -> Optional[str]:
        """
        Read file contents

        Args:
            path: File path to read

        Returns:
            File contents or None on error
        """
        if not self.api:
            return None

        result = self.api.read_file(path)

        if result.get('success'):
            return result.get('content')
        return None

    def write_file(self, path: str, content: str) -> bool:
        """
        Write content to file

        Args:
            path: File path to write
            content: Content to write

        Returns:
            True if successful, False otherwise
        """
        if not self.api:
            return False

        result = self.api.write_file(path, content)
        return result.get('success', False)

    def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        if not self.api:
            return False

        return self.api.file_exists(path)

    def list_directory(self, path: str = ".") -> List[Dict[str, Any]]:
        """
        List directory contents

        Args:
            path: Directory path

        Returns:
            List of file/directory info dicts
        """
        if not self.api:
            return []

        result = self.api.list_directory(path)

        if result.get('success'):
            return result.get('items', [])
        return []

    # ==================== PROCESS MANAGEMENT ====================

    def list_processes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List running processes

        Args:
            limit: Maximum number of processes

        Returns:
            List of process info dicts
        """
        if not self.api:
            return []

        return self.api.list_processes(limit)

    def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """
        Get process information

        Args:
            pid: Process ID

        Returns:
            Process info dict or None
        """
        if not self.api:
            return None

        result = self.api.get_process_info(pid)

        if result.get('success'):
            return result
        return None

    def kill_process(self, pid: int) -> bool:
        """
        Kill process by PID

        Args:
            pid: Process ID to kill

        Returns:
            True if successful
        """
        if not self.api:
            return False

        result = self.api.kill_process(pid)
        return result.get('success', False)

    # ==================== LEGACY COMPATIBILITY ====================

    def start_amlk_os(self) -> bool:
        """
        Legacy method for compatibility

        New API doesn't need startup - always ready.
        """
        return self.is_available

    def shutdown(self):
        """
        Shutdown AMLK bridge

        New API doesn't need cleanup - no subprocess.
        """
        pass


# Global singleton instance
_amlk_bridge = None

def get_amlk_bridge() -> NicoleAMLKBridge:
    """Get global AMLK bridge instance"""
    global _amlk_bridge
    if _amlk_bridge is None:
        _amlk_bridge = NicoleAMLKBridge()
    return _amlk_bridge


# Convenience functions for direct import
def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    return get_amlk_bridge().get_system_status()

def execute_command(command: str, timeout: int = 10) -> Dict[str, Any]:
    """Execute shell command"""
    return get_amlk_bridge().execute_command(command, timeout)

def read_file(path: str) -> Optional[str]:
    """Read file contents"""
    return get_amlk_bridge().read_file(path)

def write_file(path: str, content: str) -> bool:
    """Write file contents"""
    return get_amlk_bridge().write_file(path, content)


if __name__ == "__main__":
    # Test AMLK bridge
    bridge = NicoleAMLKBridge()

    print("=== Nicole AMLK Bridge Test ===\n")

    if not bridge.is_available:
        print("❌ AMLK not available")
        sys.exit(1)

    # Test system info
    print("System Status:")
    status = bridge.get_system_status()
    if 'error' not in status:
        print(f"  Platform: {status['platform']}")
        print(f"  Hostname: {status['hostname']}")
        print(f"  CPU: {status['cpu']['usage_percent']}%")
        print(f"  Memory: {status['memory']['percent']}%")
    print()

    # Test command execution
    print("Command Execution:")
    result = bridge.execute_command("echo 'Hello from Nicole AMLK Bridge'")
    print(f"  Success: {result['success']}")
    print(f"  Output: {result.get('stdout', 'N/A')}")
    print()

    # Test file operations
    print("File Operations:")
    test_file = "/tmp/nicole_amlk_test.txt"
    write_ok = bridge.write_file(test_file, "Nicole AMLK integration test")
    print(f"  Write: {write_ok}")

    content = bridge.read_file(test_file)
    print(f"  Read: {content is not None}")
    print(f"  Content: {content}")
    print()

    print("✅ Nicole AMLK Bridge working!")
