from .__version__ import __version__

# Boot self-test: dependency validation
try:
    from causalnerve.memory.replay_engine import StructuralReplayEngine
except ImportError:
    raise ImportError("causalnerve-observe requires causalnerve.memory.replay_engine. Please upgrade causalnerve.")

try:
    from causalnerve.memory import StructuralMemoryBank
    if not hasattr(StructuralMemoryBank, "retrieve_similar"):
        raise AttributeError("StructuralMemoryBank is missing retrieve_similar. Please upgrade causalnerve.")
except ImportError:
    raise ImportError("causalnerve-observe requires causalnerve.memory.StructuralMemoryBank. Please upgrade causalnerve.")

from .dashboard import observe

__all__ = ["observe", "__version__"]
