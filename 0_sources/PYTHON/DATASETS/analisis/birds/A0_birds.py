from .A1_Representation import Representation
from .A2_Spectral_Shift import Spectral_Shift
from .A3_Spectrogram import Spectrogram
from .A5_Zero_crossing_rate import Zero_crossing_rate
from .A6_Features import Features
from .A9_BPM import BPM

def birds(self,sample_files):

    sample_signals = dict.fromkeys(sample_files)
    for file in sample_files:
        sample_signals[file] = self.obrfiles[file]

    spectrograms = True
    features = True


    #Representation(sample_signals)
    Spectral_Shift(sample_signals,magnitude='S_0')

    exit()

    if spectrograms:

        magnitude = 'module'
        #Spectrogram(sample_signals,magnitude=magnitude,stft='librosa',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='scipy',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='custom_stft',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='custom_stft',type='mel')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='custom_stft',type='chroma')
        Spectrogram(sample_signals,magnitude=magnitude,type='z-chrip')

        magnitude = 'phase'
        #Spectrogram(sample_signals,magnitude=magnitude,stft='librosa',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='scipy',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='custom_stft',type='normal')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='librosa',type='mel')
        #Spectrogram(sample_signals,magnitude=magnitude,stft='librosa',type='chroma')
        Spectrogram(sample_signals,magnitude=magnitude,type='z-chrip')

    if features:
        magnitude = 'Module'
        #Features(sample_signals,feature = 'Harmonics',magnitude=magnitude)
        #Features(sample_signals,feature = 'Percusive', magnitude=magnitude)
        Features(sample_signals,feature = 'Spectral centroid',magnitude=magnitude)
        Features(sample_signals,feature = 'Spectral rolloff',magnitude=magnitude)
        Features(sample_signals,feature = 'Spectral bandwidth', magnitude=magnitude)
        Features(sample_signals,feature = 'Spectral contrast',magnitude=magnitude)
        Features(sample_signals,feature = 'Spectral flatness',magnitude=magnitude)
        #Features(sample_signals,feature = 'Polyfeatures',magnitude=magnitude)
        Features(sample_signals,feature = 'RMS',magnitude=magnitude)
        #Features(sample_signals,feature = 'Tempogram',magnitude=magnitude)

    exit()

    Zero_crossing_rate(self,sample_size = 50)
    BPM(self,sample_size=50)


    # To be created

    #chroma_cqt
    #chroma_cens
    #mfcc
    #rms
    #toneetz

    #tempogram
    #fourier_tempogram
