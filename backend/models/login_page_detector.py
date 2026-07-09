"""
Screenshot-based Login Page Detector
Uses computer vision features + ML to detect fake/phishing login pages
"""
import numpy as np
from PIL import Image
import io
import re
import math

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


BRAND_COLORS = {
    'facebook': [(59, 89, 152), (24, 119, 242)],
    'google': [(66, 133, 244), (234, 67, 53), (251, 188, 4), (52, 168, 83)],
    'microsoft': [(0, 120, 212), (0, 153, 0), (255, 185, 0), (210, 20, 0)],
    'apple': [(0, 122, 255), (51, 51, 51)],
    'paypal': [(0, 112, 186), (0, 36, 72)],
    'amazon': [(255, 153, 0), (35, 47, 62)],
    'twitter': [(29, 161, 242)],
    'linkedin': [(0, 119, 181)],
}

BRAND_KEYWORDS = [
    'facebook', 'google', 'microsoft', 'apple', 'paypal', 'amazon',
    'twitter', 'instagram', 'linkedin', 'netflix', 'bank', 'chase',
    'wellsfargo', 'citibank', 'signin', 'login', 'password', 'username',
    'email', 'account', 'verify', 'secure'
]


def extract_image_features(image_bytes: bytes) -> np.ndarray:
    """Extract visual features from screenshot"""
    features = []
    
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize for consistent processing
        img_resized = img.resize((800, 600))
        img_array = np.array(img_resized)
        
        # --- Color features ---
        # 1-3. Mean RGB
        features.extend([
            float(np.mean(img_array[:, :, 0])),
            float(np.mean(img_array[:, :, 1])),
            float(np.mean(img_array[:, :, 2]))
        ])
        # 4-6. Std RGB
        features.extend([
            float(np.std(img_array[:, :, 0])),
            float(np.std(img_array[:, :, 1])),
            float(np.std(img_array[:, :, 2]))
        ])
        # 7. Brightness
        gray = np.mean(img_array, axis=2)
        features.append(float(np.mean(gray)))
        # 8. High contrast (phishing pages often high contrast)
        features.append(float(np.std(gray)))
        # 9. Dominant white regions (login forms often white)
        white_ratio = np.sum(gray > 240) / gray.size
        features.append(float(white_ratio))
        # 10. Dark regions
        dark_ratio = np.sum(gray < 50) / gray.size
        features.append(float(dark_ratio))
        
        # --- Structural features using edge detection ---
        # Convert to grayscale numpy
        gray_uint8 = gray.astype(np.uint8)
        
        # 11. Vertical edges (form fields have vertical edges)
        v_diff = np.diff(gray_uint8, axis=1)
        features.append(float(np.mean(np.abs(v_diff))))
        # 12. Horizontal edges
        h_diff = np.diff(gray_uint8, axis=0)
        features.append(float(np.mean(np.abs(h_diff))))
        # 13. Edge density
        edge_threshold = 30
        edges = (np.abs(v_diff) > edge_threshold).sum() + (np.abs(h_diff) > edge_threshold).sum()
        features.append(float(edges / gray.size))
        
        # --- Rectangle/form detection heuristics ---
        # 14. Horizontal line count (input fields = rectangles)
        row_edges = np.sum(np.abs(h_diff) > 20, axis=1)
        features.append(float(np.sum(row_edges > img_array.shape[1] * 0.3)))
        # 15. Symmetry (login pages often centered/symmetric)
        left_half = gray[:, :gray.shape[1]//2]
        right_half = np.fliplr(gray[:, gray.shape[1]//2:])
        min_w = min(left_half.shape[1], right_half.shape[1])
        symmetry = 1 - np.mean(np.abs(left_half[:, :min_w].astype(float) - 
                                       right_half[:, :min_w].astype(float))) / 255
        features.append(float(symmetry))
        
        # --- Color uniqueness (brand color matching) ---
        # 16. Blue-heavy (many legit sites use blue, but phishing mimics)
        blue_dom = (img_array[:,:,2] > img_array[:,:,0]) & (img_array[:,:,2] > img_array[:,:,1])
        features.append(float(np.mean(blue_dom)))
        # 17. Color variance (phishing pages often have fewer colors)
        unique_colors = len(np.unique(img_array.reshape(-1, 3)[:100], axis=0))
        features.append(float(unique_colors))
        # 18. Saturation (fake pages sometimes over-saturated)
        img_hsv = _rgb_to_hsv(img_array)
        features.append(float(np.mean(img_hsv[:,:,1])))
        
        # --- Layout analysis ---
        # 19. Central focus (login forms centered)
        center_region = gray[200:400, 250:550]
        peripheral = np.concatenate([gray[:100, :].flatten(), gray[-100:, :].flatten()])
        center_brightness = np.mean(center_region)
        peripheral_brightness = np.mean(peripheral)
        features.append(float(abs(center_brightness - peripheral_brightness)))
        # 20. Button-like region detection (dark rectangle)
        dark_regions = (gray_uint8 < 100).astype(float)
        # Horizontal dark bands = buttons
        horiz_dark = np.mean(dark_regions, axis=1)
        features.append(float(np.sum(horiz_dark > 0.3)))
        
        # 21-25. Histogram features
        for i in range(3):
            hist, _ = np.histogram(img_array[:,:,i], bins=8, range=(0, 256))
            hist_norm = hist / hist.sum()
            features.append(float(np.max(hist_norm)))  # Dominant bin
        features.append(float(np.sum(gray_uint8 > 200) / gray_uint8.size))  # Light region
        features.append(float(np.sum((gray_uint8 > 50) & (gray_uint8 < 200)) / gray_uint8.size))  # Mid-tone
        
    except Exception as e:
        features = [0.0] * 26
    
    # Ensure exactly 26 features
    while len(features) < 26:
        features.append(0.0)
    
    return np.array(features[:26], dtype=float)


def _rgb_to_hsv(rgb_array):
    """Convert RGB to HSV"""
    rgb = rgb_array.astype(float) / 255.0
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc
    
    v = maxc
    s = np.where(maxc != 0, diff / maxc, 0)
    
    h = np.zeros_like(r)
    mask = diff != 0
    
    r_max = (maxc == r) & mask
    g_max = (maxc == g) & mask
    b_max = (maxc == b) & mask
    
    h[r_max] = (60 * ((g[r_max] - b[r_max]) / diff[r_max]) % 360)
    h[g_max] = (60 * ((b[g_max] - r[g_max]) / diff[g_max]) + 120)
    h[b_max] = (60 * ((r[b_max] - g[b_max]) / diff[b_max]) + 240)
    
    return np.stack([h/360, s, v], axis=2)


def _generate_synthetic_features(is_phishing: bool, seed: int) -> np.ndarray:
    """Generate realistic synthetic image features for training"""
    rng = np.random.RandomState(seed)
    
    if not is_phishing:
        # Legit login pages: well-designed, consistent branding
        features = [
            200 + rng.normal(0, 20),   # mean R - white/light bg
            200 + rng.normal(0, 20),   # mean G
            220 + rng.normal(0, 20),   # mean B (slightly blue)
            40 + rng.normal(0, 10),    # std R
            40 + rng.normal(0, 10),    # std G  
            50 + rng.normal(0, 10),    # std B
            210 + rng.normal(0, 15),   # brightness
            45 + rng.normal(0, 10),    # contrast
            0.6 + rng.normal(0, 0.1),  # white ratio
            0.05 + rng.normal(0, 0.02),# dark ratio
            8 + rng.normal(0, 2),      # v_edges
            7 + rng.normal(0, 2),      # h_edges
            0.02 + rng.normal(0, 0.005),# edge density
            30 + rng.normal(0, 5),     # horiz lines
            0.8 + rng.normal(0, 0.05), # symmetry
            0.4 + rng.normal(0, 0.1),  # blue dominance
            80 + rng.normal(0, 10),    # unique colors
            0.3 + rng.normal(0, 0.05), # saturation
            20 + rng.normal(0, 5),     # center focus diff
            5 + rng.normal(0, 2),      # button regions
            0.8 + rng.normal(0, 0.1),  # dom R
            0.7 + rng.normal(0, 0.1),  # dom G
            0.9 + rng.normal(0, 0.1),  # dom B
            0.65 + rng.normal(0, 0.05),# light region
            0.3 + rng.normal(0, 0.05), # mid-tone
            0.0,                        # padding
        ]
    else:
        # Phishing pages: often hastily made, off-brand colors, inconsistent
        features = [
            180 + rng.normal(0, 40),   # mean R - more variable
            170 + rng.normal(0, 40),   # mean G
            190 + rng.normal(0, 40),   # mean B
            70 + rng.normal(0, 20),    # std R - more variance
            65 + rng.normal(0, 20),    # std G
            75 + rng.normal(0, 20),    # std B
            180 + rng.normal(0, 30),   # brightness - more variable
            70 + rng.normal(0, 20),    # contrast - higher
            0.4 + rng.normal(0, 0.15), # white ratio - lower
            0.15 + rng.normal(0, 0.05),# dark ratio - higher
            15 + rng.normal(0, 5),     # v_edges - more
            14 + rng.normal(0, 5),     # h_edges
            0.04 + rng.normal(0, 0.01),# edge density - higher
            15 + rng.normal(0, 8),     # horiz lines - fewer
            0.5 + rng.normal(0, 0.15), # symmetry - lower
            0.5 + rng.normal(0, 0.15), # blue dominance
            40 + rng.normal(0, 15),    # unique colors - fewer
            0.5 + rng.normal(0, 0.1),  # saturation - higher
            40 + rng.normal(0, 15),    # center focus
            15 + rng.normal(0, 5),     # button regions
            0.6 + rng.normal(0, 0.15), # dom R
            0.6 + rng.normal(0, 0.15), # dom G
            0.7 + rng.normal(0, 0.15), # dom B
            0.45 + rng.normal(0, 0.1), # light region
            0.45 + rng.normal(0, 0.1), # mid-tone
            0.0,                        # padding
        ]
    
    return np.clip(np.array(features[:26], dtype=float), 0, None)


class LoginPageDetector:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self._train()
    
    def _train(self):
        X = []
        y = []
        
        # 50 legit, 50 phishing synthetic samples
        for i in range(50):
            X.append(_generate_synthetic_features(False, seed=i))
            y.append(0)
        for i in range(50):
            X.append(_generate_synthetic_features(True, seed=i+1000))
            y.append(1)
        
        X = np.array(X)
        y = np.array(y)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, image_bytes: bytes) -> dict:
        features = extract_image_features(image_bytes)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prob = self.model.predict_proba(features_scaled)[0]
        pred = int(np.argmax(prob))
        
        # Analyze specific visual indicators
        indicators = []
        
        # White ratio check
        if features[8] < 0.3:
            indicators.append("Low white space — unusual for legitimate login pages")
        
        # Symmetry check
        if features[14] < 0.5:
            indicators.append("Asymmetric layout detected — poorly designed page")
        
        # Edge density
        if features[12] > 0.035:
            indicators.append("Unusual visual complexity")
        
        # Color variance
        if features[16] < 50:
            indicators.append("Limited color palette — may indicate template reuse")
        
        # High saturation
        if features[17] > 0.45:
            indicators.append("Over-saturated colors — inconsistent with brand guidelines")
        
        # Contrast check
        if features[7] > 65:
            indicators.append("High contrast — possible fake/hastily-created page")
        
        # Feature importances
        feature_names = ['color_profile', 'contrast', 'white_space', 'edge_density', 
                        'symmetry', 'color_variety']
        importances = self.model.feature_importances_
        
        return {
            "is_phishing": pred == 1,
            "confidence": round(float(max(prob)) * 100, 1),
            "phishing_probability": round(float(prob[1]) * 100, 1),
            "visual_indicators": indicators[:4],
            "visual_analysis": {
                "brightness": round(float(features[6]), 1),
                "symmetry_score": round(float(features[14]) * 100, 1),
                "white_space_ratio": round(float(features[8]) * 100, 1),
                "edge_density": round(float(features[12]) * 1000, 2),
                "color_variety": int(features[16])
            }
        }


_login_detector = None

def get_login_detector():
    global _login_detector
    if _login_detector is None:
        _login_detector = LoginPageDetector()
    return _login_detector