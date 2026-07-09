"""
Deepfake Voice Detector
Uses audio signal processing features + ML to detect AI-generated/deepfake speech
Features: MFCC, spectral, prosodic, temporal, voice quality measures
"""
import numpy as np
import io
import math
import struct
import wave

try:
    import librosa
    import librosa.feature
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_audio(audio_bytes: bytes) -> tuple:
    """Load audio from bytes, return (samples, sample_rate)"""
    
    if HAS_SF:
        try:
            audio_io = io.BytesIO(audio_bytes)
            samples, sr = sf.read(audio_io)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)  # Stereo to mono
            return samples.astype(float), sr
        except Exception:
            pass
    
    if HAS_LIBROSA:
        try:
            audio_io = io.BytesIO(audio_bytes)
            samples, sr = librosa.load(audio_io, sr=None, mono=True)
            return samples.astype(float), sr
        except Exception:
            pass
    
    # Fallback: try to parse as WAV
    try:
        audio_io = io.BytesIO(audio_bytes)
        with wave.open(audio_io, 'rb') as wav:
            sr = wav.getframerate()
            n_frames = wav.getnframes()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            raw = wav.readframes(n_frames)
            
            if sample_width == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768.0
            elif sample_width == 4:
                samples = np.frombuffer(raw, dtype=np.int32).astype(float) / 2147483648.0
            else:
                samples = np.frombuffer(raw, dtype=np.uint8).astype(float) / 128.0 - 1.0
            
            if n_channels > 1:
                samples = samples[::n_channels]
            
            return samples, sr
    except Exception:
        pass
    
    return None, None


def extract_audio_features(audio_bytes: bytes) -> np.ndarray:
    """Extract 50+ audio features for deepfake detection"""
    features = []
    
    samples, sr = load_audio(audio_bytes)
    
    if samples is None or len(samples) == 0:
        return np.zeros(50)
    
    # Ensure minimum length
    if len(samples) < 1000:
        samples = np.pad(samples, (0, 1000 - len(samples)))
    
    # Normalize
    if np.max(np.abs(samples)) > 0:
        samples = samples / np.max(np.abs(samples))
    
    if HAS_LIBROSA:
        try:
            y = samples.astype(np.float32)
            sr_use = sr if sr else 22050
            
            # --- MFCCs (Mel-frequency cepstral coefficients) ---
            # Key for voice authentication
            mfccs = librosa.feature.mfcc(y=y, sr=sr_use, n_mfcc=13)
            features.extend([float(np.mean(mfccs[i])) for i in range(13)])
            features.extend([float(np.std(mfccs[i])) for i in range(5)])  # 18 features
            
            # --- Spectral features ---
            # Deepfakes often have unnatural spectral characteristics
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr_use)[0]
            features.append(float(np.mean(spectral_centroids)))
            features.append(float(np.std(spectral_centroids)))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_use)[0]
            features.append(float(np.mean(spectral_rolloff)))
            features.append(float(np.std(spectral_rolloff)))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr_use)[0]
            features.append(float(np.mean(spectral_bandwidth)))
            features.append(float(np.std(spectral_bandwidth)))
            
            zero_crossing = librosa.feature.zero_crossing_rate(y)[0]
            features.append(float(np.mean(zero_crossing)))
            features.append(float(np.std(zero_crossing)))
            
            # 26 features so far
            
            # --- Chroma features ---
            chroma = librosa.feature.chroma_stft(y=y, sr=sr_use)
            features.append(float(np.mean(chroma)))
            features.append(float(np.std(chroma)))
            
            # --- RMS Energy ---
            rms = librosa.feature.rms(y=y)[0]
            features.append(float(np.mean(rms)))
            features.append(float(np.std(rms)))
            features.append(float(np.max(rms)))
            
            # --- Pitch/F0 features ---
            # Deepfakes often have unnatural pitch trajectories
            f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                          fmax=librosa.note_to_hz('C7'))
            f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            
            if len(f0_valid) > 0:
                features.append(float(np.mean(f0_valid)))
                features.append(float(np.std(f0_valid)))
                # Pitch variation ratio (deepfakes tend to be too smooth or too variable)
                features.append(float(np.std(f0_valid) / (np.mean(f0_valid) + 1e-6)))
                # Number of voiced frames ratio
                voiced_ratio = float(np.sum(voiced)) / max(len(voiced), 1) if voiced is not None else 0
                features.append(voiced_ratio)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # 35 features
            
            # --- Mel spectrogram statistics ---
            mel = librosa.feature.melspectrogram(y=y, sr=sr_use, n_mels=32)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            features.append(float(np.mean(mel_db)))
            features.append(float(np.std(mel_db)))
            features.append(float(np.min(mel_db)))
            
            # --- Temporal features ---
            # Deepfakes may have unnatural pauses or transitions
            # Silence ratio
            silence_threshold = 0.01
            silence_ratio = float(np.sum(np.abs(y) < silence_threshold) / len(y))
            features.append(silence_ratio)
            
            # RMS envelope smoothness (deepfakes often too smooth)
            rms_diff = np.diff(rms)
            features.append(float(np.mean(np.abs(rms_diff))))
            features.append(float(np.std(rms_diff)))
            
            # 41 features
            
            # --- Voice quality features ---
            # Harmonic-to-Noise Ratio (HNR) - important for naturalness
            harmonics, _ = librosa.effects.hpss(y)
            noise = y - harmonics
            harmonic_power = np.mean(harmonics ** 2)
            noise_power = np.mean(noise ** 2)
            hnr = 10 * np.log10(harmonic_power / (noise_power + 1e-10))
            features.append(float(hnr))
            
            # Spectral flatness
            spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features.append(float(np.mean(spec_flatness)))
            features.append(float(np.std(spec_flatness)))
            
            # Tonnetz (tonal centroid features)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr_use)
            features.append(float(np.mean(tonnetz)))
            features.append(float(np.std(tonnetz)))
            
            # 46 features
            
            # --- Codec artifacts (TTS/deepfakes often have compression artifacts) ---
            # High frequency energy ratio
            if sr_use > 8000:
                fft = np.abs(np.fft.rfft(y))
                freqs = np.fft.rfftfreq(len(y), 1.0/sr_use)
                hf_mask = freqs > 4000
                total_energy = np.sum(fft ** 2) + 1e-10
                hf_energy = np.sum(fft[hf_mask] ** 2) / total_energy
                features.append(float(hf_energy))
            else:
                features.append(0.0)
            
            # Low frequency energy
            if sr_use > 100:
                lf_mask = freqs < 300
                lf_energy = np.sum(fft[lf_mask] ** 2) / total_energy
                features.append(float(lf_energy))
            else:
                features.append(0.0)
            
            # Spectral kurtosis (deepfakes often have different kurtosis)
            if len(fft) > 0:
                fft_norm = fft / (np.sum(fft) + 1e-10)
                centroid = np.sum(freqs * fft_norm)
                variance = np.sum((freqs - centroid)**2 * fft_norm)
                kurtosis = np.sum((freqs - centroid)**4 * fft_norm) / (variance**2 + 1e-10)
                features.append(float(min(kurtosis, 1000)))
            else:
                features.append(0.0)
            
            # 49 features
            
            # Duration
            features.append(float(len(y) / sr_use))
            
            # 50 features
            
        except Exception as e:
            # If librosa processing fails, use basic features
            features = _extract_basic_features(samples, sr or 22050)
    else:
        features = _extract_basic_features(samples, sr or 22050)
    
    # Ensure exactly 50 features
    while len(features) < 50:
        features.append(0.0)
    
    return np.array(features[:50], dtype=float)


def _extract_basic_features(samples: np.ndarray, sr: int) -> list:
    """Basic audio features without librosa"""
    features = []
    
    # Statistical features
    features.append(float(np.mean(samples)))
    features.append(float(np.std(samples)))
    features.append(float(np.max(np.abs(samples))))
    features.append(float(np.percentile(np.abs(samples), 75)))
    features.append(float(np.percentile(np.abs(samples), 25)))
    
    # Zero crossing rate
    zcr = np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples))
    features.append(float(zcr))
    
    # RMS energy
    rms = np.sqrt(np.mean(samples**2))
    features.append(float(rms))
    
    # FFT-based features
    if len(samples) > 0:
        fft = np.abs(np.fft.rfft(samples[:min(len(samples), 4096)]))
        freqs = np.fft.rfftfreq(min(len(samples), 4096), 1.0/sr)
        
        # Spectral centroid
        total_power = np.sum(fft) + 1e-10
        centroid = np.sum(freqs * fft) / total_power
        features.append(float(centroid))
        
        # Spectral rolloff
        cumsum = np.cumsum(fft)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        features.append(float(freqs[min(rolloff_idx, len(freqs)-1)]))
        
        # Spectral spread
        spread = np.sqrt(np.sum((freqs - centroid)**2 * fft) / total_power)
        features.append(float(spread))
        
        # Band energies
        for low, high in [(0, 500), (500, 2000), (2000, 4000), (4000, sr//2)]:
            mask = (freqs >= low) & (freqs < high)
            features.append(float(np.sum(fft[mask]**2) / total_power if np.any(mask) else 0))
    
    # Silence ratio
    silence = np.sum(np.abs(samples) < 0.01) / len(samples)
    features.append(float(silence))
    
    # Duration
    features.append(float(len(samples) / sr))
    
    return features


def _generate_synthetic_audio_features(is_deepfake: bool, seed: int) -> np.ndarray:
    """Generate synthetic audio feature vectors for training"""
    rng = np.random.RandomState(seed)
    
    if not is_deepfake:
        # Real human voice characteristics
        f = [
            # MFCCs - natural voice patterns
            rng.normal(-200, 20),   # MFCC 1
            rng.normal(50, 15),     # MFCC 2
            rng.normal(-10, 10),    # MFCC 3
            rng.normal(5, 8),       # MFCC 4
            rng.normal(-5, 7),      # MFCC 5
            rng.normal(3, 6),       # MFCC 6
            rng.normal(-3, 5),      # MFCC 7
            rng.normal(2, 5),       # MFCC 8
            rng.normal(-2, 4),      # MFCC 9
            rng.normal(1, 4),       # MFCC 10
            rng.normal(-1, 3),      # MFCC 11
            rng.normal(0.5, 3),     # MFCC 12
            rng.normal(-0.5, 2),    # MFCC 13
            rng.normal(30, 8),      # MFCC std 1
            rng.normal(20, 5),      # MFCC std 2
            rng.normal(15, 4),      # MFCC std 3
            rng.normal(12, 3),      # MFCC std 4
            rng.normal(10, 3),      # MFCC std 5
            rng.normal(2500, 300),  # Spectral centroid mean
            rng.normal(400, 100),   # Spectral centroid std - more variation
            rng.normal(4000, 500),  # Spectral rolloff
            rng.normal(600, 100),   # Spectral rolloff std
            rng.normal(1500, 200),  # Spectral bandwidth
            rng.normal(250, 50),    # Spectral bandwidth std
            rng.normal(0.08, 0.02), # ZCR mean
            rng.normal(0.03, 0.01), # ZCR std
            rng.normal(0.5, 0.05),  # Chroma mean
            rng.normal(0.2, 0.05),  # Chroma std
            rng.normal(0.05, 0.02), # RMS mean
            rng.normal(0.02, 0.01), # RMS std
            rng.normal(0.15, 0.05), # RMS max
            rng.normal(200, 30),    # F0 mean (natural pitch variation)
            rng.normal(40, 15),     # F0 std (higher = more natural)
            rng.normal(0.2, 0.05),  # Pitch variation ratio
            rng.normal(0.6, 0.1),   # Voiced ratio
            rng.normal(-20, 5),     # Mel mean
            rng.normal(15, 3),      # Mel std
            rng.normal(-60, 10),    # Mel min
            rng.normal(0.1, 0.05),  # Silence ratio (natural pauses)
            rng.normal(0.005, 0.002), # RMS diff mean
            rng.normal(0.004, 0.002), # RMS diff std
            rng.normal(15, 5),      # HNR - high for clean speech
            rng.normal(0.05, 0.02), # Spectral flatness - lower = more tonal
            rng.normal(0.03, 0.01),
            rng.normal(0, 0.1),     # Tonnetz mean
            rng.normal(0.15, 0.05),
            rng.normal(0.2, 0.08),  # HF energy
            rng.normal(0.15, 0.05), # LF energy
            rng.normal(50, 20),     # Kurtosis
            rng.normal(3, 1),       # Duration
        ]
    else:
        # Deepfake/TTS voice characteristics
        # Key differences: too smooth pitch, unusual spectral patterns,
        # artifacts in high frequency, unnatural pauses
        f = [
            # MFCCs - TTS tends to have more regular patterns
            rng.normal(-195, 8),    # MFCC 1 - less variation
            rng.normal(48, 6),      # MFCC 2
            rng.normal(-8, 4),      # MFCC 3
            rng.normal(4, 3),       # MFCC 4
            rng.normal(-4, 3),      # MFCC 5
            rng.normal(2.5, 2),     # MFCC 6
            rng.normal(-2.5, 2),    # MFCC 7
            rng.normal(1.5, 2),     # MFCC 8
            rng.normal(-1.5, 1.5),  # MFCC 9
            rng.normal(0.8, 1.5),   # MFCC 10
            rng.normal(-0.8, 1),    # MFCC 11
            rng.normal(0.4, 1),     # MFCC 12
            rng.normal(-0.4, 0.8),  # MFCC 13
            rng.normal(15, 3),      # MFCC std 1 - LOWER std (too smooth)
            rng.normal(10, 2),      # MFCC std 2
            rng.normal(8, 2),       # MFCC std 3
            rng.normal(6, 1.5),     # MFCC std 4
            rng.normal(5, 1.5),     # MFCC std 5
            rng.normal(2800, 150),  # Spectral centroid - less natural variation
            rng.normal(200, 50),    # Spectral centroid std - LESS variation
            rng.normal(4200, 300),  # Spectral rolloff
            rng.normal(300, 60),    # Spectral rolloff std
            rng.normal(1400, 100),  # Spectral bandwidth
            rng.normal(150, 30),    # Spectral bandwidth std - less
            rng.normal(0.09, 0.01), # ZCR - slightly higher (codec artifacts)
            rng.normal(0.015, 0.005),# ZCR std - LESS variation
            rng.normal(0.5, 0.02),  # Chroma
            rng.normal(0.1, 0.02),
            rng.normal(0.04, 0.01),
            rng.normal(0.01, 0.005),
            rng.normal(0.12, 0.03),
            rng.normal(205, 15),    # F0 - LESS natural variation
            rng.normal(15, 5),      # F0 std - LOWER (too smooth pitch)
            rng.normal(0.07, 0.02), # Pitch variation ratio - lower
            rng.normal(0.8, 0.05),  # Voiced ratio - often higher (no natural breath)
            rng.normal(-18, 3),
            rng.normal(12, 2),
            rng.normal(-55, 5),
            rng.normal(0.03, 0.01), # Silence ratio - fewer natural pauses
            rng.normal(0.002, 0.001), # RMS diff - smoother
            rng.normal(0.001, 0.0005),
            rng.normal(8, 3),       # HNR - lower (codec noise)
            rng.normal(0.12, 0.03), # Spectral flatness - higher (more noise-like)
            rng.normal(0.08, 0.02),
            rng.normal(0.02, 0.05),
            rng.normal(0.08, 0.02),
            rng.normal(0.35, 0.1),  # HF energy - often different
            rng.normal(0.1, 0.03),  # LF energy
            rng.normal(120, 30),    # Kurtosis - often different
            rng.normal(3, 0.5),
        ]
    
    return np.array(f[:50], dtype=float)


class DeepfakeVoiceDetector:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=5,
            subsample=0.8, random_state=42
        )
        self.scaler = StandardScaler()
        self._train()
    
    def _train(self):
        X = []
        y = []
        
        # 100 each
        for i in range(100):
            X.append(_generate_synthetic_audio_features(False, seed=i))
            y.append(0)
        for i in range(100):
            X.append(_generate_synthetic_audio_features(True, seed=i+2000))
            y.append(1)
        
        X = np.array(X)
        y = np.array(y)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, audio_bytes: bytes) -> dict:
        features = extract_audio_features(audio_bytes)
        
        if np.all(features == 0):
            return {
                "error": "Could not process audio file",
                "is_deepfake": False,
                "confidence": 0,
                "message": "Please upload a valid audio file (WAV, MP3, FLAC, OGG)"
            }
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prob = self.model.predict_proba(features_scaled)[0]
        pred = int(np.argmax(prob))
        
        # Analyze specific indicators
        indicators = []
        
        # Check F0 std (index 32) - too low = too smooth
        if features[32] < 20 and features[31] > 0:
            indicators.append("Unusually smooth pitch trajectory (natural voice varies more)")
        
        # Check voiced ratio (index 34)
        if features[34] > 0.85:
            indicators.append("Abnormally high voiced segment ratio (no natural breath sounds)")
        
        # Check silence ratio (index 38)
        if features[38] < 0.02:
            indicators.append("Insufficient natural pauses between words")
        
        # Check HNR (index 41)
        if features[41] < 5:
            indicators.append("Low harmonic-to-noise ratio (possible codec artifacts)")
        
        # Check MFCC std (indices 13-17)
        mfcc_std_avg = np.mean(features[13:18])
        if mfcc_std_avg < 12:
            indicators.append("Low MFCC variance — over-regularized speech patterns")
        
        # Check spectral centroid std (index 19)
        if features[19] < 250:
            indicators.append("Limited spectral variation — may indicate synthetic generation")
        
        return {
            "is_deepfake": pred == 1,
            "confidence": round(float(max(prob)) * 100, 1),
            "deepfake_probability": round(float(prob[1]) * 100, 1),
            "indicators": indicators[:4],
            "audio_analysis": {
                "pitch_stability": round(float(features[32]), 2) if features[32] > 0 else None,
                "harmonic_noise_ratio": round(float(features[41]), 2),
                "voiced_ratio": round(float(features[34]) * 100, 1),
                "silence_ratio": round(float(features[38]) * 100, 1),
                "mfcc_variance": round(float(np.mean(features[13:18])), 2),
                "duration_seconds": round(float(features[49]), 2)
            },
            "librosa_available": HAS_LIBROSA
        }


_voice_detector = None

def get_voice_detector():
    global _voice_detector
    if _voice_detector is None:
        _voice_detector = DeepfakeVoiceDetector()
    return _voice_detector