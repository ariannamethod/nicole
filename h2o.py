#!/usr/bin/env python3
"""
H2O - Hydrogen Oxide
Minimal Python compiler and runtime environment tailored for transformers.
Lean and focused, only what Nicole needs.
"""

import ast
import sys
import os
import types
import sqlite3
from typing import Any, Dict, List, Optional, Union
import threading
import time

class H2ORuntime:
    """Minimal runtime environment for Nicole transformers"""
    
    def __init__(self):
        self.globals_dict = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'delattr': delattr,
                'vars': vars,
                'dir': dir,
                'id': id,
                'hash': hash,
                'repr': repr,
                'ord': ord,
                'chr': chr,
                'hex': hex,
                'oct': oct,
                'bin': bin,
                'any': any,
                'all': all,
                'sorted': sorted,
                'reversed': reversed,
                'slice': slice,
                'divmod': divmod,
                'pow': pow,
                'eval': eval,
                'exec': exec,
                'compile': compile,
                '__import__': __import__,
                '__build_class__': __build_class__,
                'super': super,
                'property': property,
                'staticmethod': staticmethod,
                'classmethod': classmethod,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'AttributeError': AttributeError,
            },
            '__name__': '__h2o__',
            'math': self._create_math_module(),
            'random': self._create_random_module(),
            'time': self._create_time_module(),
            'requests': self._create_requests_module(),
            'sqlite3': self._create_sqlite_module(),
        }
        self.locals_dict = {}
        self.execution_stack = []
        self.transformer_context = {}
        
    def _create_math_module(self):
        """Minimal math module for transformers"""
        import math
        return types.SimpleNamespace(
            sin=math.sin, cos=math.cos, tan=math.tan,
            exp=math.exp, log=math.log, sqrt=math.sqrt,
            pi=math.pi, e=math.e,
            tanh=math.tanh, sinh=math.sinh, cosh=math.cosh,
            atan2=math.atan2, pow=math.pow, fabs=math.fabs,
            floor=math.floor, ceil=math.ceil
        )
        
    def _create_random_module(self):
        """Minimal random module"""
        import random
        return types.SimpleNamespace(
            random=random.random,
            randint=random.randint,
            choice=random.choice,
            shuffle=random.shuffle,
            seed=random.seed,
            uniform=random.uniform,
            gauss=random.gauss,
            normalvariate=random.normalvariate
        )
        
    def _create_time_module(self):
        """Minimal time module"""
        return types.SimpleNamespace(
            time=time.time,
            sleep=time.sleep,
            perf_counter=time.perf_counter
        )
    
    def _create_requests_module(self):
        """Minimal requests module for objectivity"""
        try:
            import requests
            return types.SimpleNamespace(
                get=requests.get,
                post=requests.post,
                put=requests.put,
                delete=requests.delete,
                Session=requests.Session,
                RequestException=requests.RequestException
            )
        except ImportError:
            # Mock if requests not installed
            def mock_get(*args, **kwargs):
                class MockResponse:
                    status_code = 200
                    def json(self): return {}
                    def text(self): return ""
                return MockResponse()
            
            return types.SimpleNamespace(
                get=mock_get,
                post=mock_get,
                RequestException=Exception
            )
    
    def _create_sqlite_module(self):
        """Minimal sqlite3 module"""
        import sqlite3
        return types.SimpleNamespace(
            connect=sqlite3.connect,
            Row=sqlite3.Row,
            Error=sqlite3.Error
        )

class H2OCompiler:
    """Python code compiler for Nicole transformers"""
    
    def __init__(self, runtime: H2ORuntime):
        self.runtime = runtime
        self.compiled_cache = {}
        
    def compile_transformer_script(self, code: str, script_id: str) -> types.CodeType:
        """Compiles transformer script"""
        if script_id in self.compiled_cache:
            return self.compiled_cache[script_id]

        try:
            # Parse AST
            tree = ast.parse(code)

            # Optimize for transformers
            tree = self._optimize_for_transformers(tree)

            # Compile to bytecode
            compiled = compile(tree, f"<transformer_{script_id}>", 'exec')

            self.compiled_cache[script_id] = compiled
            return compiled

        except Exception as e:
            raise H2OCompilationError(f"Transformer {script_id} compilation error: {e}")
    
    def _optimize_for_transformers(self, tree: ast.AST) -> ast.AST:
        """Optimizations specific to Nicole transformers"""

        class TransformerOptimizer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Add decorators for metric tracking
                if node.name.startswith('attention_') or node.name.startswith('feed_forward_'):
                    # Wrap transformer functions in metrics
                    pass
                return self.generic_visit(node)

            def visit_For(self, node):
                # Optimize loops for matrix operations
                return self.generic_visit(node)

            def visit_ListComp(self, node):
                # Optimize list comprehensions for vector operations
                return self.generic_visit(node)
        
        optimizer = TransformerOptimizer()
        return optimizer.visit(tree)

class H2OExecutor:
    """Transformer script executor"""
    
    def __init__(self, runtime: H2ORuntime, compiler: H2OCompiler):
        self.runtime = runtime
        self.compiler = compiler
        self.active_transformers = {}
        self.execution_lock = threading.Lock()
        
    def execute_transformer(self, code: str, transformer_id: str, context: Dict[str, Any] = None) -> Any:
        """Executes transformer script in isolated environment"""

        with self.execution_lock:
            try:
                # Compile code
                compiled_code = self.compiler.compile_transformer_script(code, transformer_id)

                # Create isolated environment
                execution_globals = self.runtime.globals_dict.copy()
                execution_locals = {}

                # Add transformer context
                if context:
                    execution_globals.update(context)

                # Add special functions for transformers
                execution_globals.update({
                    'transformer_id': transformer_id,
                    'h2o_log': lambda msg: self._log_transformer_action(transformer_id, msg),
                    'h2o_metric': lambda name, value: self._record_metric(transformer_id, name, value),
                    'h2o_reshape': self._reshape_transformer,
                    'h2o_evolve': self._evolve_transformer,
                    # CRITICAL: add globals() function for objectivity scripts
                    'globals': lambda: execution_globals,
                    'locals': lambda: execution_locals,
                })

                # Execute code (IMPORTANT: only globals, so variables end up there)
                result = exec(compiled_code, execution_globals, execution_globals)

                # Save transformer state
                self.active_transformers[transformer_id] = {
                    'globals': execution_globals,
                    'locals': execution_locals,  # Now also globals
                    'last_execution': time.time(),
                    'code': code
                }

                # Log what was saved
                user_vars = {k: v for k, v in execution_globals.items()
                           if not k.startswith('__') and k not in ['transformer_id', 'h2o_log', 'h2o_metric', 'h2o_reshape', 'h2o_evolve', 'globals', 'locals', 'math', 'random', 'time', 'requests', 'sqlite3']}
                if user_vars:
                    self._log_transformer_action(transformer_id, f"Saved variables: {list(user_vars.keys())}")

                return result

            except Exception as e:
                self._log_transformer_error(transformer_id, str(e))
                raise H2OExecutionError(f"Transformer {transformer_id} execution error: {e}")
    
    def _log_transformer_action(self, transformer_id: str, message: str):
        """Logs transformer actions"""
        print(f"[H2O:{transformer_id}] {message}")

    def _record_metric(self, transformer_id: str, metric_name: str, value: float):
        """Records transformer metrics"""
        # Nicole integration for metric recording will go here
        pass

    def _reshape_transformer(self, transformer_id: str, new_architecture: Dict):
        """Changes transformer architecture on the fly"""
        # Architecture change logic will go here
        pass

    def _evolve_transformer(self, transformer_id: str, evolution_params: Dict):
        """Evolves transformer"""
        # Evolution logic will go here
        pass

    def _log_transformer_error(self, transformer_id: str, error: str):
        """Logs transformer errors"""
        print(f"[H2O:ERROR:{transformer_id}] {error}")

    def kill_transformer(self, transformer_id: str):
        """Kills transformer and frees resources"""
        if transformer_id in self.active_transformers:
            del self.active_transformers[transformer_id]
            self._log_transformer_action(transformer_id, "TERMINATED")

    def list_active_transformers(self) -> List[str]:
        """Returns list of active transformers"""
        return list(self.active_transformers.keys())

class H2OEngine:
    """Main H2O engine"""

    def __init__(self):
        self.runtime = H2ORuntime()
        self.compiler = H2OCompiler(self.runtime)
        self.executor = H2OExecutor(self.runtime, self.compiler)
        self.session_id = None

    def start_session(self, session_id: str):
        """Starts new session"""
        self.session_id = session_id
        print(f"[H2O] Session {session_id} started")

    def run_transformer_script(self, code: str, transformer_id: str = None, context: Dict = None) -> Any:
        """Runs transformer script"""
        if not transformer_id:
            transformer_id = f"transformer_{int(time.time() * 1000000)}"

        return self.executor.execute_transformer(code, transformer_id, context)

    def kill_all_transformers(self):
        """Kills all active transformers"""
        for transformer_id in self.executor.list_active_transformers():
            self.executor.kill_transformer(transformer_id)

    def end_session(self):
        """Ends session"""
        if self.session_id:
            self.kill_all_transformers()
            print(f"[H2O] Session {self.session_id} ended")
            self.session_id = None

# H2O Exceptions
class H2OError(Exception):
    """Base H2O exception"""
    pass

class H2OCompilationError(H2OError):
    """Compilation error"""
    pass

class H2OExecutionError(H2OError):
    """Execution error"""
    pass

# Global H2O engine instance
h2o_engine = H2OEngine()

def run_script(code: str, transformer_id: str = None, context: Dict = None) -> Any:
    """Convenient function for running scripts"""
    return h2o_engine.run_transformer_script(code, transformer_id, context)

def test_h2o():
    """Testing H2O engine"""
    print("=== H2O ENGINE TEST ===")

    # Test 1: Simple script
    test_code1 = """
def simple_transformer():
    h2o_log("Simple transformer started")
    return "Hello from transformer!"

result = simple_transformer()
h2o_log(f"Result: {result}")
"""

    h2o_engine.start_session("test_session")

    try:
        run_script(test_code1, "test_transformer_1")
        print("✓ Test 1 passed")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")

    # Test 2: Mathematical operations
    test_code2 = """
import math

def math_transformer():
    h2o_log("Math transformer")
    values = [math.sin(i) for i in range(10)]
    h2o_metric("values_sum", sum(values))
    return values

result = math_transformer()
h2o_log(f"Math result: {len(result)} values")
"""

    try:
        run_script(test_code2, "test_transformer_2")
        print("✓ Test 2 passed")
    except Exception as e:
        print("✗ Test 2 failed:", e)

    # Show active transformers
    active = h2o_engine.executor.list_active_transformers()
    print(f"Active transformers: {active}")

    h2o_engine.end_session()
    print("=== H2O TEST COMPLETED ===")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_h2o()
    else:
        print("H2O Engine ready")
        print("For testing run: python h2o.py test")
