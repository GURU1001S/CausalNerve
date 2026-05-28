from .base import CausalDataset

class EEGDataset(CausalDataset):
    """Stub for EEG dataset adapter."""
    @property
    def citation(self) -> str:
        return "@article{eeg_stub, title={EEG dataset stub}, year={2026}}"
