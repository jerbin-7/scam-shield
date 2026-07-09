"""
URL Phishing Detector - ML model using feature extraction + RandomForest
Features: lexical, host-based, content-based
"""
import re
import math
import tldextract
from urllib.parse import urlparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# Known legitimate TLDs and suspicious patterns
SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'update', 'secure', 'account', 'bank',
    'paypal', 'amazon', 'apple', 'google', 'microsoft', 'netflix', 'facebook',
    'instagram', 'twitter', 'confirm', 'password', 'credential', 'wallet',
    'crypto', 'urgent', 'alert', 'suspended', 'limited', 'click', 'free',
    'winner', 'prize', 'reward', 'bonus', 'lucky', 'gift'
]

LEGITIMATE_TLDS = {'.com', '.org', '.net', '.edu', '.gov', '.io', '.co'}

def calculate_entropy(text):
    """Shannon entropy of a string"""
    if not text:
        return 0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    entropy = 0
    for count in freq.values():
        prob = count / len(text)
        entropy -= prob * math.log2(prob)
    return entropy

def extract_url_features(url: str) -> np.ndarray:
    """Extract 30+ features from URL"""
    features = []
    
    # Ensure URL has scheme
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    try:
        parsed = urlparse(url)
        extracted = tldextract.extract(url)
        
        domain = extracted.domain or ''
        subdomain = extracted.subdomain or ''
        suffix = extracted.suffix or ''
        path = parsed.path or ''
        query = parsed.query or ''
        full_url = url
        
        # 1. URL length
        features.append(len(full_url))
        # 2. Domain length
        features.append(len(domain))
        # 3. Number of subdomains
        features.append(len(subdomain.split('.')) if subdomain else 0)
        # 4. Has IP address
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        features.append(1 if re.search(ip_pattern, full_url) else 0)
        # 5. Uses HTTPS
        features.append(1 if parsed.scheme == 'https' else 0)
        # 6. Number of dots in domain
        features.append(full_url.count('.'))
        # 7. Number of hyphens
        features.append(full_url.count('-'))
        # 8. Number of underscores
        features.append(full_url.count('_'))
        # 9. Number of special chars
        features.append(len(re.findall(r'[^a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]', full_url)))
        # 10. Number of digits in domain
        features.append(sum(c.isdigit() for c in domain))
        # 11. Suspicious keywords count
        url_lower = full_url.lower()
        features.append(sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower))
        # 12. Domain entropy
        features.append(calculate_entropy(domain))
        # 13. Path length
        features.append(len(path))
        # 14. Query length
        features.append(len(query))
        # 15. Number of parameters
        features.append(len(query.split('&')) if query else 0)
        # 16. Has @ symbol (credential stuffing)
        features.append(1 if '@' in full_url else 0)
        # 17. Has double slash in path
        features.append(1 if '//' in path else 0)
        # 18. URL shortener
        shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'is.gd', 'cli.gs']
        features.append(1 if any(s in full_url for s in shorteners) else 0)
        # 19. Suspicious TLD
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click', '.link']
        features.append(1 if f'.{suffix}' in suspicious_tlds else 0)
        # 20. Brand impersonation (brand name in subdomain but not domain)
        brands = ['paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook', 'netflix', 'bank']
        features.append(1 if any(b in subdomain.lower() for b in brands) and 
                        not any(b in domain.lower() for b in brands) else 0)
        # 21. Number of redirects-style symbols
        features.append(full_url.count('//') - 1)
        # 22. Hex encoding
        features.append(1 if '%' in full_url else 0)
        # 23. Domain age-like heuristic (newly registered pattern: many numbers)
        features.append(sum(c.isdigit() for c in domain) / max(len(domain), 1))
        # 24. Punycode / IDN homograph
        features.append(1 if 'xn--' in full_url.lower() else 0)
        # 25. Too many subdomains (>3)
        features.append(1 if len(subdomain.split('.')) > 3 else 0)
        # 26. Looks like legit site but typosquatted
        typos = {'paypa1', 'g00gle', 'amaz0n', 'micros0ft', 'faceb00k', 'app1e'}
        features.append(1 if any(t in url_lower for t in typos) else 0)
        # 27. Long subdomain
        features.append(len(subdomain))
        # 28. Path depth
        features.append(path.count('/'))
        # 29. Has port number
        features.append(1 if parsed.port else 0)
        # 30. Fragment presence
        features.append(1 if parsed.fragment else 0)
        
    except Exception:
        features = [0] * 30
    
    return np.array(features, dtype=float)


class URLPhishingDetector:
    def __init__(self):
        self.model = self._build_model()
        self._train_with_synthetic_data()
    
    def _build_model(self):
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def _train_with_synthetic_data(self):
        """Train on synthetic labeled data representing phishing patterns"""
        legitimate_urls = [
            "https://www.google.com/search?q=test",
            "https://github.com/user/repo",
            "https://amazon.com/product/123",
            "https://facebook.com/login",
            "https://microsoft.com/en-us/",
            "https://apple.com/iphone",
            "https://netflix.com/browse",
            "https://linkedin.com/feed",
            "https://twitter.com/home",
            "https://youtube.com/watch?v=abc",
            "https://wikipedia.org/wiki/Python",
            "https://stackoverflow.com/questions/123",
            "https://reddit.com/r/python",
            "https://dropbox.com/home",
            "https://slack.com/workspace",
            "https://zoom.us/meeting",
            "https://paypal.com/myaccount",
            "https://ebay.com/item/123456",
            "https://walmart.com/grocery",
            "https://target.com/store",
            "https://bestbuy.com/site/electronics",
            "https://chase.com/personal/banking",
            "https://wellsfargo.com/banking",
            "https://bankofamerica.com",
            "https://citibank.com/us/",
            "https://aws.amazon.com/console",
            "https://azure.microsoft.com",
            "https://cloud.google.com",
            "https://heroku.com/apps",
            "https://digitalocean.com/projects",
            "https://shopify.com/admin",
            "https://stripe.com/dashboard",
            "https://twilio.com/console",
            "https://sendgrid.com/marketing",
            "https://mailchimp.com/campaigns",
        ]
        
        phishing_urls = [
            "http://paypal-security-alert.tk/login/verify",
            "http://192.168.1.1/amazon/signin",
            "http://secure-apple-id.ml/account/update",
            "http://google-verify-login.ga/signin",
            "http://microsoft-account-suspended.cf/update",
            "http://amazon-prize-winner.xyz/claim",
            "http://netflix-account-verify.top/billing",
            "http://facebook-security.click/password/reset",
            "http://paypa1.com-verify.tk/login",
            "http://amaz0n-deals.ml/free/gift",
            "http://login.paypal.com.phish.net/verify",
            "http://account-verify.secure-bank.tk/update",
            "http://bit.ly/win-free-iphone",
            "http://tinyurl.com/amazon-gift-card",
            "http://secure.login.apple-id.com.phishing.ga",
            "http://update-your-info.microsoft-help.ml/urgent",
            "http://bank-alert-suspended.xyz/verify/now",
            "http://credential-update.login-secure.tk",
            "http://win-iphone-free.lucky-draw.ga/claim",
            "http://urgent-verify-account.paypal-alerts.cf",
            "http://xn--pypal-4ve.com/signin",
            "http://1nstagram-login.ml/password",
            "http://tw1tter-verify.tk/account",
            "http://amazon.customer-service-update.ga",
            "http://secure-login.apple-id-verify.ml/",
            "http://2562819204.com/verify/account",
            "http://login-secure-verify.amazon-alerts.top/",
            "http://your-account-suspended.netlflix.cf/",
            "http://free-bitcoin-wallet.crypto-prize.xyz/",
            "http://signin.google-accounts.verify-now.ml/",
            "http://paypal.login.verify.account.suspended.tk/",
            "http://microsoftt-support-alert.click/fix",
            "http://bankofamerica-security-update.ga/login",
            "http://chase-bank-alert-verify.xyz/account",
            "http://wells-fargo-suspended-alert.ml/verify",
        ]
        
        X = np.array([extract_url_features(u) for u in legitimate_urls + phishing_urls])
        y = np.array([0]*len(legitimate_urls) + [1]*len(phishing_urls))
        
        self.model.fit(X, y)
    
    def predict(self, url: str) -> dict:
        features = extract_url_features(url)
        features_2d = features.reshape(1, -1)
        
        prob = self.model.predict_proba(features_2d)[0]
        pred = self.model.predict(features_2d)[0]
        
        # Analyze specific risk factors
        risk_factors = []
        url_lower = url.lower()
        
        if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url):
            risk_factors.append("IP address used instead of domain name")
        if 'http://' in url and 'https://' not in url:
            risk_factors.append("No HTTPS encryption")
        if any(s in url_lower for s in ['bit.ly', 'tinyurl', 't.co', 'goo.gl']):
            risk_factors.append("URL shortener detected")
        for kw in SUSPICIOUS_KEYWORDS:
            if kw in url_lower:
                risk_factors.append(f"Suspicious keyword: '{kw}'")
                break
        if 'xn--' in url_lower:
            risk_factors.append("Punycode/homograph attack detected")
        if '@' in url:
            risk_factors.append("@ symbol in URL (credential bypass)")
        if any(t in url_lower for t in ['.tk', '.ml', '.ga', '.cf', '.gq']):
            risk_factors.append("High-risk free TLD detected")
        if url.count('.') > 5:
            risk_factors.append("Excessive subdomains")
        
        extracted = tldextract.extract(url if url.startswith('http') else 'http://' + url)
        brands = ['paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook', 'netflix']
        if any(b in extracted.subdomain.lower() for b in brands) and \
           not any(b in extracted.domain.lower() for b in brands):
            risk_factors.append("Brand name in subdomain (impersonation)")
        
        confidence = float(prob[1]) if pred == 1 else float(prob[0])
        
        return {
            "is_phishing": bool(pred == 1),
            "confidence": round(confidence * 100, 1),
            "phishing_probability": round(float(prob[1]) * 100, 1),
            "risk_factors": risk_factors[:5],
            "features": {
                "url_length": int(features[0]),
                "uses_https": bool(features[4] == 1),
                "suspicious_keywords": int(features[10]),
                "entropy": round(float(features[11]), 2),
                "has_ip": bool(features[3] == 1)
            }
        }


# Singleton
_url_detector = None

def get_url_detector():
    global _url_detector
    if _url_detector is None:
        _url_detector = URLPhishingDetector()
    return _url_detector