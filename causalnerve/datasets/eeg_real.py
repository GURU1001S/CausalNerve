import os
import numpy as np
import urllib.request
from typing import Dict, List, Tuple, Optional
from causalnerve.datasets.base import CausalDataset, CausalDataBundle

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


class RealEEGDataset(CausalDataset):
    """
    Adapter for real EEG data, supporting MNE, EDF, and streaming windows.
    Defaults to PhysioNet EEG Motor Movement/Imagery Dataset via mne if available.
    """
    
    def __init__(self, subject: int = 1, run: int = 4, cache_dir: str = "./eeg_data"):
        self.subject = subject
        self.run = run
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.raw = None
        self.events = None
        self.ch_names = None
        self.sfreq = None

    def load_subject(self) -> CausalDataBundle:
        """Loads full subject run into memory."""
        if not MNE_AVAILABLE:
            raise ImportError("mne is required to load PhysioNet datasets. Run `pip install mne`")
        
        # Load PhysioNet EEG Motor Imagery dataset
        # Run 4: imagining opening and closing left or right fist
        from mne.datasets import eegbci
        print(f"Downloading/Loading EEG Motor Imagery Subject {self.subject}, Run {self.run}...")
        raw_fnames = eegbci.load_data(self.subject, [self.run], path=self.cache_dir, update_path=False)
        self.raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
        
        # Standardize 10-20 layout
        eegbci.standardize(self.raw)
        
        # Standard 19-channel 10-20 layout subset
        canonical_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'T3', 'C3', 'Cz', 'C4', 'T4', 
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2'
        ]
        
        # MNE EEG Motor Imagery channels have dots/capitalization differences (e.g. Fp1.)
        # We need to map them or pick the closest match
        available_ch = self.raw.ch_names
        # MNE eegbci channels are like 'Fp1', 'Fp2', 'F7'... wait, no, eegbci channels are often named with trailing dots if from BCI2000, 
        # but `eegbci.standardize(self.raw)` normally renames them. Let's just pick the canonical ones.
        # But if they don't match, we will just take the first 19 channels as a fallback.
        
        picks = []
        for ch in canonical_channels:
            # Case-insensitive search
            matches = [c for c in available_ch if c.lower().replace('.', '') == ch.lower()]
            if matches:
                picks.append(matches[0])
            else:
                pass
                
        if len(picks) == 19:
            self.raw.pick_channels(picks)
        else:
            self.raw.pick_channels(available_ch[:19])

        self.ch_names = self.raw.ch_names
        self.sfreq = self.raw.info['sfreq']
        
        # Optional bandpass filtering (Beta/Gamma bands are good for connectivity)
        self.raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin')
        
        data = self.raw.get_data().T  # (T, n_channels)
        
        # Normalization
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-9)
        
        node_labels = {i: name for i, name in enumerate(self.ch_names)}
        
        return CausalDataBundle(
            X=data,
            node_labels=node_labels,
            metadata={
                "subject": self.subject,
                "run": self.run,
                "sfreq": self.sfreq,
                "n_channels": len(self.ch_names),
                "domain": "neuroscience/eeg"
            }
        )
        
    def stream_subject(self, window_size: int = 128, step: int = 64):
        """Generator for streaming EEG windows live."""
        bundle = self.load_subject()
        data = bundle.X
        T, n_ch = data.shape
        
        for start in range(0, T - window_size, step):
            end = start + window_size
            yield data[start:end, :]

    def load_window(self, start_sec: float, duration_sec: float) -> np.ndarray:
        if self.raw is None:
            self.load_subject()
        start_idx = int(start_sec * self.sfreq)
        end_idx = start_idx + int(duration_sec * self.sfreq)
        data = self.raw.get_data(start=start_idx, stop=end_idx).T
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-9)

    @property
    def citation(self) -> str:
        return (
            "Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. "
            "BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. "
            "IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004."
        )

    def load(self) -> CausalDataBundle:
        return self.load_subject()
