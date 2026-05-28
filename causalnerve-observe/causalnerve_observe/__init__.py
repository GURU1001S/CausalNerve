from .__version__ import __version__

def observe(nerve_instance, port=7860):
    """
    Launch the interactive CausalNerve Observatory.
    """
    from .dashboard import CausalRuntimeObservatory
    obs = CausalRuntimeObservatory(nerve_instance)
    obs.launch(port=port)

__all__ = ["observe", "__version__"]
