from .A1_Representation import Representation
from .A2_Spectral_Shift import Spectral_Shift
from .A3_Spectrogram import Spectrogram
from .A5_Zero_crossing_rate import Zero_crossing_rate
from .A6_Spectral_features import Spectral_features
from .A7_Spectral_centroid import Spectral_centroid
from .A8_Chroma_frequencies import Chroma_frequencies
from .A9_BPM import BPM
from .A10_Spectral_rolloff import Spectral_rolloff

def birds(self,sample_files):

    sample_signals = dict.fromkeys(sample_files)
    for file in sample_files:
        sample_signals[file] = self.obrfiles[file]

    spectrograms = True
    spectral_features = True


    #Representation(sample_signals)
    #Spectral_Shift(sample_signals)

    if spectrograms:

        magnitude = 'module'
        #Spectrogram(sample_signals,magnitude=magnitude,stft='librosa',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='scipy',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='custom_stft',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='custom_stft',type='mel')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='custom_stft',type='chroma')

        magnitude = 'phase'
        #Spectrogram(sample_signals,magnitude=magnitude,stft='librosa',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='scipy',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='custom_stft',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='librosa',type='mel')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='librosa',type='chroma')

    if spectral_features:
        magnitude = 'Module'
        #Spectral_features(sample_signals,feature = 'Harmonics',magnitude=magnitude)
        #Spectral_features(sample_signals,feature = 'Perceptual', magnitude=magnitude)
        Spectral_features(sample_signals,feature = 'Spectral centroid',magnitude=magnitude)

    Zero_crossing_rate(self,sample_size = 50)
    BPM(self,sample_size=50)
    #Spectral_rolloff(sample_signals,type = 'Module-Phase')
    #Spectral_rolloff(sample_signals,type = 'Real-Imaginary')

    # To be created

    #chroma_cqt
    #chroma_cens
    #mfcc
    #rms
    #spectral_bandwidth
    #spectral_contrast
    #spectral_flatness
    #poly_features
    #toneetz

    #tempogram
    #fourier_tempogram
