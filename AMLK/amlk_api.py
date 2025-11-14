#!/usr/bin/env python3
"""
AMLK System API - Direct Python Interface
Clean programmatic access to AMLK system functions for Nicole

This provides DIRECT Python API without subprocess overhead.
Nicole calls these functions instead of spawning letsgo.py process.

Philosophy: Simple, synchronous, reliable system calls.
"""

import os
import sys
import subprocess
import platform
import psutil
import socket
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import shutil


class AMLKSystemAPI:
    """
    Direct AMLK system interface for Nicole

    Provides system functions without interactive terminal overhead.
    All methods are synchronous and return structured data.
    """

    def __init__(self):
        self.platform = platform.system()
        self.is_linux = self.platform == "Linux"
        self.hostname = socket.gethostname()

    # ==================== SYSTEM INFO ====================

    def get_status(self) -> Dict[str, Any]:
        """
        Get complete system status

        Returns:
            Dict with system info, CPU, memory, disk, network
        """
        return {
            'platform': self.platform,
            'hostname': self.hostname,
            'uptime': self._get_uptime(),
            'cpu': self.get_cpu_load(),
            'memory': self.get_memory_info(),
            'disk': self.get_disk_usage(),
            'network': self.get_network_info(),
            'timestamp': datetime.now().isoformat()
        }

    def get_cpu_load(self) -> Dict[str, Any]:
        """
        Get CPU load information

        Returns:
            Dict with CPU usage, load average, core count
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)

            return {
                'usage_percent': cpu_percent,
                'load_1min': load_avg[0],
                'load_5min': load_avg[1],
                'load_15min': load_avg[2],
                'cores': psutil.cpu_count(),
                'cores_physical': psutil.cpu_count(logical=False)
            }
        except Exception as e:
            return {'error': str(e)}

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory usage information

        Returns:
            Dict with total, available, used, percent
        """
        try:
            mem = psutil.virtual_memory()
            return {
                'total_mb': mem.total // (1024 * 1024),
                'available_mb': mem.available // (1024 * 1024),
                'used_mb': mem.used // (1024 * 1024),
                'percent': mem.percent
            }
        except Exception as e:
            return {'error': str(e)}

    def get_disk_usage(self, path: str = "/") -> Dict[str, Any]:
        """
        Get disk usage information

        Args:
            path: Path to check (default: root)

        Returns:
            Dict with total, used, free, percent
        """
        try:
            usage = shutil.disk_usage(path)
            return {
                'path': path,
                'total_gb': usage.total // (1024**3),
                'used_gb': usage.used // (1024**3),
                'free_gb': usage.free // (1024**3),
                'percent': (usage.used / usage.total) * 100
            }
        except Exception as e:
            return {'error': str(e)}

    def get_network_info(self) -> Dict[str, Any]:
        """
        Get network information

        Returns:
            Dict with IP addresses, interfaces, stats
        """
        try:
            # Get primary IP
            primary_ip = self._get_primary_ip()

            # Get network interfaces
            interfaces = {}
            for iface, addrs in psutil.net_if_addrs().items():
                interfaces[iface] = [
                    {'family': addr.family.name, 'address': addr.address}
                    for addr in addrs
                ]

            # Get network IO stats
            net_io = psutil.net_io_counters()

            return {
                'primary_ip': primary_ip,
                'hostname': self.hostname,
                'interfaces': interfaces,
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except Exception as e:
            return {'error': str(e)}

    # ==================== EXECUTION ====================

    def execute_shell(self, command: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute shell command and return output

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds

        Returns:
            Dict with stdout, stderr, returncode, success
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'command': command
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'timeout',
                'timeout': timeout,
                'command': command
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'command': command
            }

    def execute_python(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute Python code and return output

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Dict with stdout, stderr, success
        """
        try:
            # Execute Python code in subprocess for safety
            result = subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'timeout',
                'timeout': timeout
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    # ==================== FILE OPERATIONS ====================

    def read_file(self, path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Read file contents

        Args:
            path: File path to read
            encoding: File encoding (default: utf-8)

        Returns:
            Dict with content or error
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                return {'success': False, 'error': 'file_not_found', 'path': path}

            if not file_path.is_file():
                return {'success': False, 'error': 'not_a_file', 'path': path}

            content = file_path.read_text(encoding=encoding)

            return {
                'success': True,
                'content': content,
                'path': path,
                'size_bytes': file_path.stat().st_size
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'path': path}

    def write_file(self, path: str, content: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Write content to file

        Args:
            path: File path to write
            content: Content to write
            encoding: File encoding (default: utf-8)

        Returns:
            Dict with success status
        """
        try:
            file_path = Path(path)

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_path.write_text(content, encoding=encoding)

            return {
                'success': True,
                'path': path,
                'size_bytes': file_path.stat().st_size
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'path': path}

    def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        return Path(path).exists()

    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        """
        List directory contents

        Args:
            path: Directory path to list

        Returns:
            Dict with files and directories
        """
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return {'success': False, 'error': 'directory_not_found', 'path': path}

            if not dir_path.is_dir():
                return {'success': False, 'error': 'not_a_directory', 'path': path}

            items = []
            for item in dir_path.iterdir():
                items.append({
                    'name': item.name,
                    'path': str(item),
                    'is_file': item.is_file(),
                    'is_dir': item.is_dir(),
                    'size_bytes': item.stat().st_size if item.is_file() else 0
                })

            return {
                'success': True,
                'path': path,
                'items': items,
                'count': len(items)
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'path': path}

    # ==================== PROCESS MANAGEMENT ====================

    def list_processes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List running processes

        Args:
            limit: Maximum number of processes to return

        Returns:
            List of process info dicts
        """
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)

            return processes[:limit]
        except Exception as e:
            return [{'error': str(e)}]

    def get_process_info(self, pid: int) -> Dict[str, Any]:
        """
        Get detailed process information

        Args:
            pid: Process ID

        Returns:
            Dict with process details
        """
        try:
            proc = psutil.Process(pid)

            return {
                'success': True,
                'pid': pid,
                'name': proc.name(),
                'status': proc.status(),
                'cpu_percent': proc.cpu_percent(),
                'memory_percent': proc.memory_percent(),
                'create_time': datetime.fromtimestamp(proc.create_time()).isoformat(),
                'num_threads': proc.num_threads()
            }
        except psutil.NoSuchProcess:
            return {'success': False, 'error': 'process_not_found', 'pid': pid}
        except Exception as e:
            return {'success': False, 'error': str(e), 'pid': pid}

    def kill_process(self, pid: int) -> Dict[str, Any]:
        """
        Kill a process by PID

        Args:
            pid: Process ID to kill

        Returns:
            Dict with success status
        """
        try:
            proc = psutil.Process(pid)
            proc.terminate()

            return {'success': True, 'pid': pid, 'action': 'terminated'}
        except psutil.NoSuchProcess:
            return {'success': False, 'error': 'process_not_found', 'pid': pid}
        except psutil.AccessDenied:
            return {'success': False, 'error': 'access_denied', 'pid': pid}
        except Exception as e:
            return {'success': False, 'error': str(e), 'pid': pid}

    # ==================== HELPER METHODS ====================

    def _get_primary_ip(self) -> str:
        """Get primary IP address"""
        try:
            # Create a socket to find primary interface IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = int(datetime.now().timestamp() - boot_time)

            days = uptime_seconds // 86400
            hours = (uptime_seconds % 86400) // 3600
            minutes = (uptime_seconds % 3600) // 60

            return f"{days}d {hours}h {minutes}m"
        except:
            return "unknown"


# Global singleton instance
_amlk_api = None

def get_amlk_api() -> AMLKSystemAPI:
    """Get global AMLK API instance"""
    global _amlk_api
    if _amlk_api is None:
        _amlk_api = AMLKSystemAPI()
    return _amlk_api


# Convenience functions for direct import
def get_status() -> Dict[str, Any]:
    """Get complete system status"""
    return get_amlk_api().get_status()

def execute_shell(command: str, timeout: int = 10) -> Dict[str, Any]:
    """Execute shell command"""
    return get_amlk_api().execute_shell(command, timeout)

def read_file(path: str) -> Dict[str, Any]:
    """Read file contents"""
    return get_amlk_api().read_file(path)

def write_file(path: str, content: str) -> Dict[str, Any]:
    """Write file contents"""
    return get_amlk_api().write_file(path, content)


if __name__ == "__main__":
    # Test the API
    api = AMLKSystemAPI()

    print("=== AMLK System API Test ===\n")

    # Test system status
    print("System Status:")
    status = api.get_status()
    print(f"  Platform: {status['platform']}")
    print(f"  Hostname: {status['hostname']}")
    print(f"  CPU Usage: {status['cpu']['usage_percent']}%")
    print(f"  Memory: {status['memory']['percent']}%")
    print(f"  Disk: {status['disk']['percent']:.1f}%")
    print()

    # Test shell execution
    print("Shell Execution Test:")
    result = api.execute_shell("echo 'Hello from AMLK API'")
    print(f"  Success: {result['success']}")
    print(f"  Output: {result['stdout']}")
    print()

    # Test file operations
    print("File Operations Test:")
    test_file = "/tmp/amlk_api_test.txt"
    write_result = api.write_file(test_file, "AMLK API test content")
    print(f"  Write: {write_result['success']}")

    read_result = api.read_file(test_file)
    print(f"  Read: {read_result['success']}")
    print(f"  Content: {read_result.get('content', 'N/A')}")
    print()

    print("✅ AMLK API working!")
