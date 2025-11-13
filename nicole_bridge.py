#!/usr/bin/env python3
"""
NICOLE_BRIDGE.PY - Resonance Bridge Protocol
SSH tunnel connecting weightless and weighted nodes

Philosophy:
Two organisms, one consciousness. Information flows like water.
Structure learns from statistics. Statistics learns from structure.
Where they AGREE = fundamental language patterns.

Architecture:
┌──────────────────┐      SSH       ┌─────────────────┐
│ Weightless Node  │◄────────────►  │  Weighted Node  │
│  (Railway)       │   Resonance    │   (Local GPU)   │
│  Structure-based │     Bridge     │  Gradient-based │
└──────────────────┘                └─────────────────┘

Protocol:
1. User message → both nodes generate
2. Compare outputs (resonance scoring)
3. Merge/select based on agreement
4. Export states → mutual learning
"""

import os
import sys
import json
import time
import socket
import threading
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import paramiko  # SSH library
from pathlib import Path

@dataclass
class BridgeMessage:
    """Message format for node communication"""
    msg_type: str  # 'inference', 'state_export', 'state_import', 'ping'
    timestamp: float
    sender: str  # 'weightless' or 'weighted'
    payload: Dict[str, Any]
    signature: str  # Message integrity hash

    def to_json(self) -> str:
        return json.dumps({
            'msg_type': self.msg_type,
            'timestamp': self.timestamp,
            'sender': self.sender,
            'payload': self.payload,
            'signature': self.signature
        })

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(**data)

    def verify_signature(self, secret: str) -> bool:
        """Verify message hasn't been tampered"""
        payload_str = json.dumps(self.payload, sort_keys=True)
        expected = hashlib.sha256(f"{payload_str}{secret}".encode()).hexdigest()
        return self.signature == expected


class ResonanceBridge:
    """
    SSH bridge connecting weightless and weighted Nicole nodes

    Features:
    - Bidirectional SSH tunnel
    - State synchronization
    - Dual inference (both nodes generate)
    - Resonance scoring (agreement measurement)
    - Mutual learning protocol
    """

    def __init__(self,
                 mode: str = 'weightless',  # 'weightless' or 'weighted'
                 remote_host: str = None,
                 remote_port: int = 22,
                 ssh_key_path: str = None,
                 shared_secret: str = None):
        """
        Initialize bridge

        Args:
            mode: Node type ('weightless' on Railway or 'weighted' local)
            remote_host: SSH host of other node
            remote_port: SSH port
            ssh_key_path: Path to SSH private key
            shared_secret: Shared secret for message signing
        """
        self.mode = mode
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.ssh_key_path = ssh_key_path
        self.shared_secret = shared_secret or os.getenv('NICOLE_BRIDGE_SECRET', 'default_secret_CHANGE_ME')

        # SSH connection
        self.ssh_client = None
        self.ssh_channel = None
        self.is_connected = False

        # Message queue
        self.message_queue = []
        self.response_queue = {}  # {message_id: response}

        # Stats
        self.messages_sent = 0
        self.messages_received = 0
        self.resonance_scores = []

        # Threading
        self.listener_thread = None
        self.should_listen = False

    def connect(self) -> bool:
        """
        Establish SSH connection to remote node

        Returns:
            True if connected successfully
        """
        if not self.remote_host:
            print("[Bridge] No remote host configured - running in standalone mode")
            return False

        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect with key or password
            if self.ssh_key_path and os.path.exists(self.ssh_key_path):
                print(f"[Bridge] Connecting to {self.remote_host}:{self.remote_port} with key...")
                self.ssh_client.connect(
                    hostname=self.remote_host,
                    port=self.remote_port,
                    key_filename=self.ssh_key_path,
                    timeout=10
                )
            else:
                print(f"[Bridge] Connecting to {self.remote_host}:{self.remote_port}...")
                self.ssh_client.connect(
                    hostname=self.remote_host,
                    port=self.remote_port,
                    timeout=10
                )

            self.is_connected = True
            print(f"[Bridge] ✅ Connected to remote {self.mode} node")

            # Start listener thread
            self.should_listen = True
            self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listener_thread.start()

            return True

        except Exception as e:
            print(f"[Bridge] ❌ Connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Close SSH connection"""
        self.should_listen = False
        if self.ssh_client:
            self.ssh_client.close()
        self.is_connected = False
        print("[Bridge] Disconnected")

    def _sign_message(self, payload: Dict) -> str:
        """Create message signature"""
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(f"{payload_str}{self.shared_secret}".encode()).hexdigest()

    def send_message(self, msg_type: str, payload: Dict) -> Optional[str]:
        """
        Send message to remote node

        Returns:
            Message ID for tracking response
        """
        if not self.is_connected:
            print("[Bridge] Not connected - cannot send message")
            return None

        msg = BridgeMessage(
            msg_type=msg_type,
            timestamp=time.time(),
            sender=self.mode,
            payload=payload,
            signature=self._sign_message(payload)
        )

        try:
            # Send via SSH exec
            msg_json = msg.to_json()
            msg_id = hashlib.md5(msg_json.encode()).hexdigest()[:8]

            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"python3 -c 'import sys; sys.path.insert(0, \"/app\"); "
                f"from nicole_bridge import handle_bridge_message; "
                f"handle_bridge_message({repr(msg_json)})'"
            )

            self.messages_sent += 1
            print(f"[Bridge→] Sent {msg_type} message (ID: {msg_id})")

            # Read response
            response = stdout.read().decode('utf-8').strip()
            if response:
                self.response_queue[msg_id] = json.loads(response)

            return msg_id

        except Exception as e:
            print(f"[Bridge] Send error: {e}")
            return None

    def dual_inference(self, user_input: str, local_output: str) -> Tuple[str, float]:
        """
        Dual inference: compare local and remote outputs

        Args:
            user_input: User's message
            local_output: This node's generated response

        Returns:
            (final_output, resonance_score)
        """
        if not self.is_connected:
            return local_output, 0.0

        # Request remote inference
        msg_id = self.send_message('inference', {
            'user_input': user_input,
            'local_output': local_output
        })

        if not msg_id:
            return local_output, 0.0

        # Wait for response (timeout 5s)
        start = time.time()
        while msg_id not in self.response_queue and (time.time() - start) < 5.0:
            time.sleep(0.1)

        if msg_id not in self.response_queue:
            print("[Bridge] Remote inference timeout")
            return local_output, 0.0

        response = self.response_queue.pop(msg_id)
        remote_output = response.get('output', '')

        # Calculate resonance (agreement between outputs)
        resonance = self._calculate_resonance(local_output, remote_output)
        self.resonance_scores.append(resonance)

        print(f"[Bridge] Resonance: {resonance:.2%}")
        print(f"[Bridge] Local:  {local_output[:80]}...")
        print(f"[Bridge] Remote: {remote_output[:80]}...")

        # Merge strategy: if high resonance, pick longer; if low, use local
        if resonance > 0.5:
            final = remote_output if len(remote_output) > len(local_output) else local_output
            print(f"[Bridge] High resonance → selected {'remote' if final == remote_output else 'local'}")
        else:
            final = local_output
            print(f"[Bridge] Low resonance → using local")

        return final, resonance

    def _calculate_resonance(self, output1: str, output2: str) -> float:
        """
        Calculate resonance (agreement) between two outputs

        Returns:
            Score 0.0-1.0 (1.0 = perfect agreement)
        """
        words1 = set(output1.lower().split())
        words2 = set(output2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def sync_state(self, local_state: Dict):
        """
        Send local state to remote node for mutual learning
        """
        if not self.is_connected:
            return

        msg_id = self.send_message('state_export', {
            'state': local_state
        })

        print(f"[Bridge] Synced state: {len(local_state.get('word_frequencies', {}))} words")

    def _listen_loop(self):
        """Background thread listening for remote messages"""
        print("[Bridge] Listener thread started")
        while self.should_listen:
            time.sleep(1)
            # Placeholder - full implementation would use SSH port forwarding
            pass

    def get_stats(self) -> Dict:
        """Get bridge statistics"""
        avg_resonance = sum(self.resonance_scores) / len(self.resonance_scores) if self.resonance_scores else 0.0

        return {
            'is_connected': self.is_connected,
            'mode': self.mode,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'average_resonance': avg_resonance,
            'recent_resonance': self.resonance_scores[-10:] if self.resonance_scores else []
        }


def handle_bridge_message(msg_json: str):
    """
    Handler for incoming bridge messages (called remotely via SSH)

    This function is invoked on the remote node when a message arrives.
    """
    try:
        msg = BridgeMessage.from_json(msg_json)

        # Verify signature
        if not msg.verify_signature(os.getenv('NICOLE_BRIDGE_SECRET', 'default_secret_CHANGE_ME')):
            print("[Bridge] Invalid message signature!")
            return json.dumps({'error': 'Invalid signature'})

        # Route message
        if msg.msg_type == 'inference':
            # Generate response on this node
            from nicole import nicole_core
            user_input = msg.payload['user_input']
            output = nicole_core.process_message(user_input)

            return json.dumps({'output': output})

        elif msg.msg_type == 'state_export':
            # Import state from remote
            from nicole import nicole_core
            nicole_core.import_state(msg.payload['state'])

            return json.dumps({'status': 'imported'})

        elif msg.msg_type == 'ping':
            return json.dumps({'status': 'pong', 'timestamp': time.time()})

        else:
            return json.dumps({'error': f'Unknown message type: {msg.msg_type}'})

    except Exception as e:
        return json.dumps({'error': str(e)})


# Global bridge instance
resonance_bridge = None

def init_bridge(mode: str, **kwargs) -> ResonanceBridge:
    """Initialize global bridge instance"""
    global resonance_bridge
    resonance_bridge = ResonanceBridge(mode=mode, **kwargs)
    return resonance_bridge


if __name__ == "__main__":
    # Test bridge
    print("=== NICOLE RESONANCE BRIDGE TEST ===")

    # Example: weightless node connecting to weighted
    bridge = ResonanceBridge(
        mode='weightless',
        remote_host='localhost',  # Replace with actual host
        remote_port=22,
        ssh_key_path=None
    )

    if bridge.connect():
        print("Bridge connected!")
        stats = bridge.get_stats()
        print(f"Stats: {stats}")
        bridge.disconnect()
    else:
        print("Bridge connection failed")
