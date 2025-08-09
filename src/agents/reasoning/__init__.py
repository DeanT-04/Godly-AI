"""
Multi-Modal Reasoning Cores

This package implements specialized reasoning cores for different cognitive domains,
each using liquid state machines with modality-specific adaptations.

The cross-modal synchronization system enables coordination between reasoning cores
through spike synchronization, competitive resource allocation, and cross-modal
information integration.
"""

from .base_reasoning_core import (
    BaseReasoningCore,
    ReasoningCoreParams,
    ReasoningCoreState,
    ModalityType
)

from .visual_reasoning_core import (
    VisualReasoningCore,
    create_visual_reasoning_core
)

from .audio_reasoning_core import (
    AudioReasoningCore,
    create_audio_reasoning_core
)

from .text_reasoning_core import (
    TextReasoningCore,
    create_text_reasoning_core
)

from .motor_reasoning_core import (
    MotorReasoningCore,
    create_motor_reasoning_core
)

from .cross_modal_sync import (
    CrossModalSynchronizer,
    SyncParams,
    SyncMode,
    create_cross_modal_synchronizer
)

__all__ = [
    # Base classes
    "BaseReasoningCore",
    "ReasoningCoreParams", 
    "ReasoningCoreState",
    "ModalityType",
    
    # Specialized cores
    "VisualReasoningCore",
    "AudioReasoningCore", 
    "TextReasoningCore",
    "MotorReasoningCore",
    
    # Cross-modal synchronization
    "CrossModalSynchronizer",
    "SyncParams",
    "SyncMode",
    
    # Factory functions
    "create_visual_reasoning_core",
    "create_audio_reasoning_core",
    "create_text_reasoning_core",
    "create_motor_reasoning_core",
    "create_cross_modal_synchronizer"
]