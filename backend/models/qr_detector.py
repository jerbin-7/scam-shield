"""
QR Code Phishing Detector
Decodes QR codes from images and analyzes embedded URLs/data for phishing
"""
import math
import re
from urllib.parse import urlparse

import cv2
import numpy as np
import tldextract
from sklearn.ensemble import GradientBoostingClassifier

SUSPICIOUS_QR_PATTERNS = [
    r'bit\.ly', r'tinyurl\.com', r't\.co', r'goo\.gl',
    r'\.tk/', r'\.ml/', r'\.ga/', r'\.cf/', r'\.gq/',
    r'login', r'signin', r'verify', r'account', r'secure',
    r'paypal', r'amazon', r'apple', r'google', r'microsoft',
    r'bank', r'wallet', r'crypto', r'bitcoin', r'free',
    r'prize', r'winner', r'claim', r'reward',
]

QR_SAFE_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
    'instagram.com', 'linkedin.com', 'github.com', 'wikipedia.org',
    'amazon.com', 'apple.com', 'microsoft.com', 'paypal.com',
    'netflix.com', 'spotify.com', 'zoom.us', 'dropbox.com',
}


def extract_qr_url_features(url: str) -> np.ndarray:
    """Feature extraction specifically for QR-decoded URLs."""
    features: list[float] = []

    if not url:
        return np.zeros(20)

    url_lower = url.lower()

    try:
        url_with_scheme = url if url.startswith(('http://', 'https://')) else 'http://' + url

        extracted = tldextract.extract(url_with_scheme)
        domain = extracted.domain or ''
        subdomain = extracted.subdomain or ''
        suffix = extracted.suffix or ''

        # 1. URL length
        features.append(len(url))
        # 2. Uses HTTPS
        features.append(1 if url.startswith('https://') else 0)
        # 3. Suspicious pattern count
        features.append(sum(1 for p in SUSPICIOUS_QR_PATTERNS if re.search(p, url_lower)))
        # 4. Is URL shortener
        shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'qr.io', 'qrco.de']
        features.append(1 if any(s in url_lower for s in shorteners) else 0)
        # 5. Suspicious TLD
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click']
        features.append(1 if f'.{suffix}' in suspicious_tlds else 0)
        # 6. Has IP address
        features.append(1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0)
        # 7. Number of dots
        features.append(url.count('.'))
        # 8. Number of hyphens
        features.append(url.count('-'))
        # 9. Subdomain depth
        features.append(len(subdomain.split('.')) if subdomain else 0)
        # 10. Brand in subdomain but not domain (impersonation)
        brands = ['paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook', 'netflix', 'bank']
        features.append(
            1 if any(b in subdomain.lower() for b in brands)
            and not any(b in domain.lower() for b in brands)
            else 0
        )
        # 11. Domain entropy
        entropy = 0.0
        if domain:
            freq: dict[str, int] = {}
            for c in domain:
                freq[c] = freq.get(c, 0) + 1
            for count in freq.values():
                p = count / len(domain)
                entropy -= p * math.log2(p)
        features.append(entropy)
        # 12. URL has encoded characters (% encoding)
        features.append(1 if '%' in url else 0)
        # 13. Number of path segments
        parsed = urlparse(url_with_scheme)
        features.append(parsed.path.count('/'))
        # 14. Has query parameters
        features.append(1 if '?' in url else 0)
        # 15. Query length
        features.append(len(parsed.query))
        # 16. Punycode
        features.append(1 if 'xn--' in url_lower else 0)
        # 17. Is known safe domain
        features.append(1 if f'{domain}.{suffix}' in QR_SAFE_DOMAINS else 0)
        # 18. Digits in domain
        features.append(sum(c.isdigit() for c in domain))
        # 19. Has @ symbol
        features.append(1 if '@' in url else 0)
        # 20. URL depth (capped at 10)
        features.append(min(parsed.path.count('/'), 10))

    except Exception:
        features = [0.0] * 20

    while len(features) < 20:
        features.append(0.0)

    return np.array(features[:20], dtype=float)


class QRPhishingDetector:
    def __init__(self) -> None:
        self.model = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42
        )
        self._train()

    def _train(self) -> None:
        legit_qr_urls = [
            "https://google.com/maps/place/eiffel+tower",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://github.com/anthropics/anthropic-sdk",
            "https://linkedin.com/in/johndoe",
            "https://amazon.com/dp/B09XYZ123",
            "https://docs.google.com/document/d/1abc",
            "https://zoom.us/j/1234567890",
            "https://spotify.com/playlist/abc123",
            "https://apple.com/shop/buy-iphone",
            "https://microsoft.com/en-us/microsoft-365",
            "https://en.wikipedia.org/wiki/Python",
            "https://stackoverflow.com/questions/1234",
            "https://twitter.com/anthropic",
            "https://facebook.com/events/123456",
            "https://instagram.com/p/ABC123",
            "https://dropbox.com/s/abc123/file.pdf",
            "https://drive.google.com/file/d/abc",
            "https://calendar.google.com/event?id=xyz",
            "https://forms.gle/abc123xyz",
            "https://meet.google.com/abc-def-ghi",
        ]

        phishing_qr_urls = [
            "http://paypal-verify.tk/qr/login",
            "https://amazon-prize.ml/qr/claim/free",
            "http://192.168.1.1/phishing/login",
            "http://bit.ly/2Xyz1Ab",
            "https://apple-id-verify.ga/qr/update",
            "http://microsoft-support.cf/qr/fix",
            "https://free-bitcoin-qr.xyz/wallet/claim",
            "http://bank-verify-qr.ml/account/update",
            "https://login.paypal.com.phish-site.tk/qr",
            "http://amazon.customer-qr-verify.ga/login",
            "https://netflix-billing-qr.cf/update-payment",
            "http://secure-qr-bank.xyz/verify/now",
            "https://facebook-qr-verify.tk/account",
            "http://google-signin-qr.ml/account/confirm",
            "https://crypto-wallet-qr.ga/claim/reward",
            "http://instagram-verify-qr.cf/login",
            "https://win-iphone-qr.xyz/claim/prize",
            "http://qr-code-bank-login.tk/secure",
            "https://linkedin-verify-qr.ml/profile/confirm",
            "http://tax-refund-qr.ga/claim/irs/refund",
        ]

        all_urls = legit_qr_urls + phishing_qr_urls
        X = np.array([extract_qr_url_features(u) for u in all_urls])
        y = np.array([0] * len(legit_qr_urls) + [1] * len(phishing_qr_urls))
        self.model.fit(X, y)

    def decode_qr(self, image_bytes: bytes) -> list[dict[str, str]]:
        """Decode QR code(s) from image bytes using OpenCV."""
        decoded_data: list[dict[str, str]] = []

        try:
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return [{"type": "ERROR", "data": "Unable to read image"}]

            detector = cv2.QRCodeDetector()

            # Attempt to decode multiple QR codes first
            success, decoded_info, *_ = detector.detectAndDecodeMulti(image)
            if success and decoded_info:
                for text in decoded_info:
                    if text and text.strip():
                        decoded_data.append({"type": "QRCODE", "data": text})

            # Fall back to single QR detection if nothing found
            if not decoded_data:
                text, *_ = detector.detectAndDecode(image)
                if text and text.strip():
                    decoded_data.append({"type": "QRCODE", "data": text})

            if not decoded_data:
                decoded_data.append({"type": "ERROR", "data": "No QR code detected"})

        except Exception as exc:
            decoded_data.append({"type": "ERROR", "data": str(exc)})

        return decoded_data

    def predict(self, image_bytes: bytes) -> dict:
        decoded = self.decode_qr(image_bytes)

        if not decoded:
            return {
                "qr_found": False,
                "message": "No QR code detected in image",
                "is_phishing": False,
                "confidence": 0,
            }

        results = []
        for item in decoded:
            url = item.get('data', '')

            if item.get('type') == 'ERROR':
                results.append({
                    "qr_content": url,
                    "content_type": "error",
                    "is_phishing": False,
                    "confidence": 0,
                    "phishing_probability": 0,
                    "risk_factors": [],
                })
                continue

            content_type = "url" if url.startswith(('http', 'www', 'ftp')) else "text"

            if content_type == "url":
                features = extract_qr_url_features(url)
                prob = self.model.predict_proba(features.reshape(1, -1))[0]
                pred = int(np.argmax(prob))

                risk_factors: list[str] = []
                url_lower = url.lower()

                if not url.startswith('https://'):
                    risk_factors.append("No HTTPS encryption")
                if any(s in url_lower for s in ['bit.ly', 'tinyurl', 'goo.gl']):
                    risk_factors.append("URL shortener hides real destination")
                if any(t in url_lower for t in ['.tk', '.ml', '.ga', '.cf']):
                    risk_factors.append("High-risk free domain TLD")
                if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
                    risk_factors.append("IP address instead of domain name")

                brands = ['paypal', 'amazon', 'apple', 'google', 'microsoft']
                ext = tldextract.extract(url if url.startswith('http') else 'http://' + url)
                if (any(b in ext.subdomain.lower() for b in brands)
                        and not any(b in ext.domain.lower() for b in brands)):
                    risk_factors.append("Brand name used in subdomain (impersonation)")

                results.append({
                    "qr_content": url,
                    "content_type": content_type,
                    "is_phishing": pred == 1,
                    "confidence": round(float(max(prob)) * 100, 1),
                    "phishing_probability": round(float(prob[1]) * 100, 1),
                    "risk_factors": risk_factors[:4],
                })
            else:
                suspicious_text = any(
                    kw in url.lower()
                    for kw in ['password', 'login', 'click here', 'verify', 'credit card']
                )
                results.append({
                    "qr_content": url[:200],
                    "content_type": content_type,
                    "is_phishing": suspicious_text,
                    "confidence": 60 if suspicious_text else 85,
                    "phishing_probability": 60 if suspicious_text else 10,
                    "risk_factors": ["Text QR contains suspicious keywords"] if suspicious_text else [],
                })

        overall_phishing = any(r.get('is_phishing', False) for r in results)
        max_prob = max((r.get('phishing_probability', 0) for r in results), default=0)

        return {
            "qr_found": True,
            "codes_found": len(decoded),
            "results": results,
            "is_phishing": overall_phishing,
            "phishing_probability": max_prob,
            "confidence": max((r.get('confidence', 0) for r in results), default=0),
        }


_qr_detector: QRPhishingDetector | None = None


def get_qr_detector() -> QRPhishingDetector:
    global _qr_detector
    if _qr_detector is None:
        _qr_detector = QRPhishingDetector()
    return _qr_detector