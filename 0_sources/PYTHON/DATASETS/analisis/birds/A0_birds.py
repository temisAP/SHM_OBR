from .A1_Representation import Representation
from .A2_STFT import STFT
from .A3_Spectrogram import Spectrogram
from .A4_Mel_spectrogram import Mel_spectrogram
from .A5_Zero_crossing_rate import Zero_crossing_rate
from .A6_Harmonics_and_perceptual import Harmonics_and_perceptual
from .A7_Spectral_centroid import Spectral_centroid
from .A8_Chroma_frequencies import Chroma_frequencies
from .A9_BPM import BPM
from .A10_Spectral_rolloff import Spectral_rolloff

def birds(self,sample_files):

    sample_signals = dict.fromkeys(sample_files)
    for file in sample_files:
        sample_signals[file] = self.obrfiles[file]

    #Representation(sample_signals)
    #Spectrogram(sample_signals)
    Mel_spectrogram(sample_signals)
    Zero_crossing_rate(self.slices)
    Harmonics_and_perceptual(sample_signals)
    Spectral_centroid(sample_signals)
    Chroma_frequencies(sample_signals)
    BPM(self.slices)
    Spectral_rolloff(sample_signals)