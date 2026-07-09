"""
Email Phishing Detector - NLP + Feature extraction
Uses TF-IDF + gradient boosting on header and body features
"""
import re
import math
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

PHISHING_SUBJECT_PATTERNS = [
    r'urgent', r'immediate action', r'account (suspended|blocked|locked|limited)',
    r'verify your', r'confirm your', r'update your', r'unusual (activity|sign-in)',
    r'security alert', r'password (expired|reset|changed)', r'click here',
    r'act now', r'you (have won|are selected)', r'free (gift|prize|reward)',
    r'congratulations', r'claim your', r'final notice', r'overdue',
    r'invoice attached', r'tax refund', r'irs notice', r'delivery failed',
    r'package (held|waiting|pending)', r'your order', r'transaction (declined|failed)',
]

PHISHING_BODY_PATTERNS = [
    r'click (here|below|this link|the link)',
    r'verify your (account|identity|email|information)',
    r'enter your (password|credentials|details|information)',
    r'your account (will be|has been) (closed|suspended|terminated)',
    r'within \d+ (hours|days)',
    r'bank (account|details|information)',
    r'social security (number|no)',
    r'credit card (number|details|information)',
    r'login (credentials|details|information)',
    r'dear (customer|user|member|account holder)',
    r'unfortunately.*account',
    r'we have noticed',
    r'suspicious (activity|login|access)',
    r'your (information|account|password) (has been|will be)',
]

LEGIT_SENDER_DOMAINS = {
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
    'company.com', 'business.org', 'university.edu'
}


def extract_email_features(email_data: dict) -> np.ndarray:
    """Extract numerical features from email"""
    features = []
    
    subject = email_data.get('subject', '')
    body = email_data.get('body', '')
    sender = email_data.get('sender', '')
    headers = email_data.get('headers', '')
    
    subject_lower = subject.lower()
    body_lower = body.lower()
    
    # 1. Subject length
    features.append(len(subject))
    # 2. Subject uppercase ratio
    upper_ratio = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
    features.append(upper_ratio)
    # 3. Phishing patterns in subject
    features.append(sum(1 for p in PHISHING_SUBJECT_PATTERNS if re.search(p, subject_lower)))
    # 4. Exclamation marks in subject
    features.append(subject.count('!'))
    # 5. Question marks in subject  
    features.append(subject.count('?'))
    # 6. Body length
    features.append(len(body))
    # 7. HTML in body
    features.append(1 if '<html' in body_lower or '<a href' in body_lower else 0)
    # 8. URLs in body
    url_count = len(re.findall(r'https?://\S+', body))
    features.append(url_count)
    # 9. Mismatched URLs (display text vs actual URL)
    href_pattern = re.findall(r'href=["\']([^"\']+)["\']', body_lower)
    features.append(len(href_pattern))
    # 10. Phishing patterns in body
    features.append(sum(1 for p in PHISHING_BODY_PATTERNS if re.search(p, body_lower)))
    # 11. Urgency words
    urgency = ['urgent', 'immediately', 'asap', 'right now', 'within 24', 'expire', 'deadline']
    features.append(sum(1 for u in urgency if u in body_lower))
    # 12. Personal info requests
    personal = ['password', 'ssn', 'social security', 'credit card', 'bank account', 'pin']
    features.append(sum(1 for p in personal if p in body_lower))
    # 13. Generic greeting (red flag)
    generic_greetings = ['dear customer', 'dear user', 'dear member', 'dear account']
    features.append(1 if any(g in body_lower for g in generic_greetings) else 0)
    # 14. Sender has suspicious domain
    sender_lower = sender.lower()
    suspicious_sender_patterns = ['.tk', '.ml', '.ga', 'noreply@', 'no-reply@']
    features.append(sum(1 for p in suspicious_sender_patterns if p in sender_lower))
    # 15. Sender impersonates brand
    brands = ['paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook', 'netflix', 'bank']
    sender_domain = sender_lower.split('@')[-1] if '@' in sender_lower else sender_lower
    brand_in_sender = any(b in sender_lower for b in brands)
    brand_in_domain = any(b in sender_domain for b in brands)
    features.append(1 if brand_in_sender and not brand_in_domain else 0)
    # 16. SPF/DKIM fail indicators
    features.append(1 if 'fail' in headers.lower() or 'spf=fail' in headers.lower() else 0)
    # 17. Number of external links
    external = len(re.findall(r'https?://(?!(?:www\.)?(?:gmail|google|microsoft|apple))', body))
    features.append(external)
    # 18. Body entropy (high randomness = generated text)
    features.append(_entropy(body[:500]) if body else 0)
    # 19. Spelling errors approximation (simple heuristic)
    common_mistakes = ['recieve', 'occured', 'untill', 'acces', 'verfy', 'suspened']
    features.append(sum(1 for m in common_mistakes if m in body_lower))
    # 20. Reply-to different from sender
    features.append(1 if 'reply-to' in headers.lower() else 0)
    
    return np.array(features, dtype=float)


def _entropy(text):
    if not text:
        return 0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    e = 0
    for count in freq.values():
        p = count / len(text)
        e -= p * math.log2(p)
    return e


class EmailPhishingDetector:
    def __init__(self):
        self.feature_model = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42
        )
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
        self._trained = False
        self._train()
    
    def _train(self):
        legit_emails = [
            {"subject": "Meeting tomorrow at 3pm", "body": "Hi John, just a reminder about our meeting tomorrow. Please bring the quarterly report. Best regards, Sarah", "sender": "sarah@company.com", "headers": "spf=pass"},
            {"subject": "Your order has shipped", "body": "Your Amazon order #123-456 has been shipped and is expected to arrive Friday. Track your package at amazon.com/orders", "sender": "shipment@amazon.com", "headers": "spf=pass dkim=pass"},
            {"subject": "Invoice #INV-2024-001", "body": "Please find attached invoice #INV-2024-001 for services rendered in March. Payment due within 30 days. Thank you for your business.", "sender": "billing@legitcompany.com", "headers": "spf=pass"},
            {"subject": "Team lunch this Friday", "body": "Hey everyone! We're having a team lunch this Friday at noon. Please RSVP by Thursday. The restaurant is Olive Garden on Main St.", "sender": "manager@office.com", "headers": "spf=pass"},
            {"subject": "New blog post: Python tips", "body": "Check out our latest blog post about Python best practices. This week we cover decorators, context managers, and async programming.", "sender": "newsletter@pythonblog.org", "headers": "spf=pass"},
            {"subject": "Your subscription renewal", "body": "Your Netflix subscription will renew on March 15th for $15.99. No action needed. To manage your subscription visit netflix.com/account.", "sender": "info@netflix.com", "headers": "spf=pass dkim=pass"},
            {"subject": "Interview scheduled for Monday", "body": "Dear candidate, we are pleased to inform you that your interview has been scheduled for Monday at 10am. Please bring your resume and portfolio.", "sender": "hr@techcorp.com", "headers": "spf=pass"},
            {"subject": "Project update - Q4 goals", "body": "Hi team, attached is the Q4 project update. We've achieved 80% of our goals. Let's discuss in our next standup meeting.", "sender": "pm@startup.io", "headers": "spf=pass"},
            {"subject": "Password reset confirmation", "body": "Your password has been successfully reset. If you did not request this, please contact support. For security, this link expires in 1 hour.", "sender": "security@github.com", "headers": "spf=pass dkim=pass"},
            {"subject": "Weekly digest - Tech news", "body": "This week in tech: OpenAI releases new model, Apple announces earnings, Google updates search algorithm. Read more on our website.", "sender": "digest@technews.com", "headers": "spf=pass"},
        ]
        
        phishing_emails = [
            {"subject": "URGENT: Your account has been suspended!", "body": "Dear Customer, We have detected unusual activity on your account. Click here immediately to verify your identity: http://paypal-verify.tk/login. Failure to verify within 24 hours will result in permanent suspension.", "sender": "security@paypa1.com", "headers": "spf=fail"},
            {"subject": "Congratulations! You've won $1,000,000!", "body": "Dear Lucky Winner! You have been selected as our grand prize winner. To claim your reward, please provide your bank account details and social security number.", "sender": "lottery@prize-winner.ml", "headers": "spf=fail"},
            {"subject": "Your Apple ID has been locked", "body": "Dear Apple User, Your Apple ID has been locked due to suspicious activity. Verify your account immediately at: http://apple-id-verify.ga/unlock. Enter your password and credit card to restore access.", "sender": "no-reply@appie.com", "headers": "spf=fail"},
            {"subject": "IRS Tax Refund Notice - Action Required", "body": "You have a pending tax refund of $3,847. To receive your refund, click here and enter your social security number and bank account details. This offer expires in 48 hours.", "sender": "refund@irs-gov.tk", "headers": "spf=fail"},
            {"subject": "Your Amazon Package - Delivery Failed", "body": "Dear Customer, we attempted to deliver your package but failed. To reschedule, click here and verify your address and credit card information: http://amazon-delivery.ml/reschedule", "sender": "delivery@amazom.com", "headers": "spf=fail"},
            {"subject": "Microsoft Account: Unusual sign-in activity", "body": "Dear user, we noticed unusual login activity on your Microsoft account. Someone from Russia tried to access your account. Click here to secure it now: http://microsoft-alert.cf/secure", "sender": "security@micros0ft.com", "headers": "spf=fail"},
            {"subject": "FINAL NOTICE: Overdue Invoice", "body": "This is your final notice regarding an overdue invoice. Legal action will be taken within 24 hours if payment is not received. Click here to view the invoice and pay now.", "sender": "collections@debt-notice.xyz", "headers": "spf=fail"},
            {"subject": "Your Netflix payment failed - Update now!", "body": "Dear valued customer, your recent Netflix payment has failed. To avoid service interruption, please update your payment information immediately: http://netflix-billing.tk/update-payment", "sender": "billing@netflx.com", "headers": "spf=fail"},
            {"subject": "Bank Account Security Alert!!!", "body": "URGENT!!! Your bank account has been compromised!!! Login credentials have been stolen!!! Click HERE NOW to change your password and verify identity!!! ACT IMMEDIATELY!!!", "sender": "alert@bank-security-verify.ga", "headers": "spf=fail"},
            {"subject": "You are selected for our exclusive rewards program", "body": "Congratulations dear member! You have been exclusively selected for our rewards program. Claim your free iPhone 15 by clicking here and entering your credit card for shipping verification.", "sender": "rewards@free-gifts-4u.ml", "headers": "spf=fail"},
        ]
        
        all_emails = legit_emails + phishing_emails
        y = [0] * len(legit_emails) + [1] * len(phishing_emails)
        
        # Extract numerical features
        X_features = np.array([extract_email_features(e) for e in all_emails])
        
        # TF-IDF on combined text
        texts = [f"{e['subject']} {e['body']}" for e in all_emails]
        X_tfidf = self.tfidf.fit_transform(texts)
        
        # Combine features
        X_combined = hstack([csr_matrix(X_features), X_tfidf])
        
        self.feature_model.fit(X_features, y)
        self._trained = True
    
    def predict(self, email_data: dict) -> dict:
        features = extract_email_features(email_data)
        prob = self.feature_model.predict_proba(features.reshape(1, -1))[0]
        pred = int(np.argmax(prob))
        
        # Identify specific red flags
        red_flags = []
        subject_lower = email_data.get('subject', '').lower()
        body_lower = email_data.get('body', '').lower()
        sender = email_data.get('sender', '').lower()
        
        for pattern in PHISHING_SUBJECT_PATTERNS[:5]:
            if re.search(pattern, subject_lower):
                red_flags.append(f"Suspicious subject pattern: '{pattern}'")
                break
        
        if sum(1 for p in PHISHING_BODY_PATTERNS if re.search(p, body_lower)) > 2:
            red_flags.append("Multiple phishing patterns in email body")
        
        if any(b in sender for b in ['paypal', 'amazon', 'apple', 'google', 'microsoft']):
            domain = sender.split('@')[-1] if '@' in sender else sender
            if not any(b + '.com' in domain or b + '.org' in domain 
                      for b in ['paypal', 'amazon', 'apple', 'google', 'microsoft']):
                red_flags.append("Sender impersonates trusted brand")
        
        urgency_count = sum(1 for u in ['urgent', 'immediately', 'within 24', 'expire'] 
                           if u in body_lower)
        if urgency_count >= 2:
            red_flags.append(f"High urgency language detected ({urgency_count} instances)")
        
        personal = sum(1 for p in ['password', 'credit card', 'bank account', 'social security', 'ssn'] 
                      if p in body_lower)
        if personal > 0:
            red_flags.append(f"Requests sensitive personal information ({personal} type(s))")
        
        if '<a href' in body_lower or re.findall(r'https?://\S+', email_data.get('body', '')):
            body_urls = re.findall(r'https?://(\S+)', email_data.get('body', ''))
            if any(t in str(body_urls) for t in ['.tk', '.ml', '.ga', '.cf', '.xyz']):
                red_flags.append("Links to suspicious/free TLD domains")
        
        return {
            "is_phishing": pred == 1,
            "confidence": round(float(max(prob)) * 100, 1),
            "phishing_probability": round(float(prob[1]) * 100, 1),
            "red_flags": red_flags[:5],
            "analysis": {
                "subject_risk": int(features[2]),
                "body_risk_patterns": int(features[9]),
                "urgency_score": int(features[10]),
                "personal_info_requests": int(features[11]),
                "sender_suspicious": bool(features[13] > 0)
            }
        }


_email_detector = None

def get_email_detector():
    global _email_detector
    if _email_detector is None:
        _email_detector = EmailPhishingDetector()
    return _email_detector