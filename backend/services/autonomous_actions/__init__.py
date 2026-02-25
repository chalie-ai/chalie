"""
Autonomous Actions — Extensible action selection for spontaneous thoughts.

The drift engine produces thoughts on a regular cycle. This package adds a
decision layer: "What should I do with this thought?"

Rev 1: COMMUNICATE or NOTHING
Rev 2: REFLECT — internal enrichment via association linking
Future: USE_SKILL, PLAN, LEARN
"""

from .base import AutonomousAction, ActionResult, ThoughtContext
from .nothing_action import NothingAction
from .communicate_action import CommunicateAction
from .reflect_action import ReflectAction
from .seed_thread_action import SeedThreadAction
from .suggest_action import SuggestAction
from .nurture_action import NurtureAction
from .decision_router import ActionDecisionRouter
from .engagement_tracker import EngagementTracker

__all__ = [
    'AutonomousAction', 'ActionResult', 'ThoughtContext',
    'NothingAction', 'CommunicateAction', 'ReflectAction',
    'SeedThreadAction', 'SuggestAction', 'NurtureAction',
    'ActionDecisionRouter', 'EngagementTracker',
]
