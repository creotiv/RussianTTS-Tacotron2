
import numpy as np
import soundfile as sf
from audiomentations import Compose, TimeStretch, PitchShift
from scipy.io.wavfile import read
from librosa.effects import trim
import sys

augment = Compose([
    TimeStretch(min_rate=1.5, max_rate=1.5, p=1.0),
    PitchShift(min_semitones=4, max_semitones=4, p=1.0),
])


sampling_rate, audio = read(sys.argv[1])

augmented_samples = augment(samples=audio.astype(np.float32), sample_rate=sampling_rate)
sf.write('test.wav', trim(augmented_samples)[0], sampling_rate)