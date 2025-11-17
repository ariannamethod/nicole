"""
Nicole Bootstrap Engine
Weightless sentence planning using JSON skeleton

This module will be integrated into Nicole's runtime AFTER bootstrap training.
For now, these are stubs.
"""

from .loader import load_skeleton, get_ngrams, get_shapes, get_clusters, get_style, get_banned, get_metadata
from .planner import choose_structure, filter_banned

__all__ = [
    'load_skeleton',
    'get_ngrams',
    'get_shapes',
    'get_clusters',
    'get_style',
    'get_banned',
    'get_metadata',
    'choose_structure',
    'filter_banned'
]
