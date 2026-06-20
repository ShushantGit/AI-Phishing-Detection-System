import joblib
import os
import re
import math
import socket
import requests
import pandas as pd
from urllib.parse import urlparse
from colorama import Fore, Style, init
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print
import pyfiglet

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_auc_score
)


try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

init(autoreset=True)
console = Console()

API_KEY      = "Your key"
DATASET_PATH = "Database.csv"
MODEL_PATH   = "phishing_model.pkl"


def banner():
    ascii_banner = pyfiglet.figlet_format("Website Checker")
    print(f"[bold cyan]{ascii_banner}[/bold cyan]")


def fix_url(url):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


# ─────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION — 40+ lexical / statistical features
# ─────────────────────────────────────────────────────────────────────
def extract_features(url):
    url = str(url).strip()
    url = fix_url(url)

    try:
        parsed    = urlparse(url)
        domain    = parsed.netloc.lower().replace("www.", "")
        path      = parsed.path.lower()
        query     = parsed.query.lower()
    except ValueError:
        domain = path = query = ""

    def shannon_entropy(s):
        if not s:
            return 0.0
        freq = {c: s.count(c) / len(s) for c in set(s)}
        return -sum(p * math.log2(p) for p in freq.values())

    HIGH_RISK_KEYWORDS = ["login","verify","secure","update","bank","account",
                          "password","confirm","reward","signin","webscr",
                          "billing","customer","support","paypal","ebay",
                          "amazon","apple","microsoft"]
    MED_RISK_KEYWORDS  = ["free","gift","bonus","offer","promo","click",
                          "deal","winner","prize","cash"]
    FAKE_BRANDS        = ["g00gle","amaz0n","paypa1","faceb00k","micr0soft",
                          "app1e","netfl1x","inst4gram","twitt3r","g0ogle",
                          "arnazon","micosoft","paypai","linkedln"]
    SUSPICIOUS_TLDS    = [".xyz",".tk",".ru",".top",".ml",".ga",".cf",".gq",
                          ".pw",".club",".work",".date",".download",".racing",
                          ".review",".stream"]

    fake_brand_count = sum(1 for b in FAKE_BRANDS if b in url.lower())

    return {
        "url_length":           len(url),
        "domain_length":        len(domain),
        "path_length":          len(path),
        "has_at_symbol":        int("@" in url),
        "has_hyphen":           int("-" in domain),
        "has_ip_address":       int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", domain))),
        "count_dots":           domain.count("."),
        "count_slashes":        url.count("/"),
        "digit_count":          sum(c.isdigit() for c in url),
        "special_char_count":   sum(not c.isalnum() for c in url),
        "count_hyphens":        url.count("-"),
        "count_underscores":    url.count("_"),
        "count_question_marks": url.count("?"),
        "count_equal_signs":    url.count("="),
        "count_ampersands":     url.count("&"),
        "count_percent_signs":  url.count("%"),
        "uses_https":           int(url.startswith("https://")),
        "domain_entropy":       round(shannon_entropy(domain), 4),
        "domain_digit_ratio":   round(sum(c.isdigit() for c in domain) / len(domain) if domain else 0, 4),
        "consecutive_digits":   max((len(m.group()) for m in re.finditer(r"\d+", domain)), default=0),
        "subdomain_count":      max(domain.count(".") - 1, 0),
        "keyword_high_count":   sum(1 for w in HIGH_RISK_KEYWORDS if w in url.lower()),
        "keyword_med_count":    sum(1 for w in MED_RISK_KEYWORDS  if w in url.lower()),
        "has_login_word":       int("login"    in url.lower()),
        "has_verify_word":      int("verify"   in url.lower()),
        "has_secure_word":      int("secure"   in url.lower()),
        "has_update_word":      int("update"   in url.lower()),
        "has_bank_word":        int("bank"     in url.lower()),
        "has_free_word":        int("free"     in url.lower()),
        "has_account_word":     int("account"  in url.lower()),
        "has_password_word":    int("password" in url.lower()),
        "suspicious_tld":       int(any(domain.endswith(t) for t in SUSPICIOUS_TLDS)),
        "fake_brand":           int(fake_brand_count > 0),
        "fake_brand_count":     fake_brand_count,
        "has_hex_encoding":     int(bool(re.search(r"%[0-9a-fA-F]{2}", url))),
        "has_unicode_escape":   int("\\u" in url or "%u" in url),
        "has_double_slash":     int("//" in path),
        "has_port_in_url":      int(bool(re.search(r":\d{2,5}(/|$)", domain + path))),
    }


def prepare_dataset():
    df = pd.read_csv(DATASET_PATH)
    if "url" not in df.columns or "label" not in df.columns:
        console.print(Panel("Dataset must contain 'url' and 'label' columns.",
                            title="[bold red]DATASET ERROR[/bold red]", border_style="red"))
        return None, None, None

    feature_rows = [extract_features(str(u)) for u in df["url"]]
    X = pd.DataFrame(feature_rows)
    y = df["label"].map({"legitimate":0,"safe":0,"phishing":1,"malicious":1,"fraud":1})

    if y.isnull().any():
        console.print(Panel("Label column must only contain: legitimate, safe, phishing, malicious, fraud",
                            title="[bold red]LABEL ERROR[/bold red]", border_style="red"))
        return None, None, None

    return X, y, df["url"].values


# ─────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────
def show_confusion_matrix(tn, fp, fn, tp):
    table = Table(title="[bold cyan]Confusion Matrix[/bold cyan]",
                  show_header=True, header_style="bold magenta", border_style="cyan")
    table.add_column("Actual \\ Predicted", style="bold white",  width=22)
    table.add_column("Legitimate (0)",       style="bold green",  width=20, justify="center")
    table.add_column("Phishing (1)",          style="bold red",    width=20, justify="center")
    table.add_row("Legitimate (0)",
                  f"[bold green]TN = {tn:,}[/bold green]\n✔ Correct Negative",
                  f"[bold red]FP = {fp:,}[/bold red]\n✗ False Positive")
    table.add_row("Phishing (1)",
                  f"[bold yellow]FN = {fn:,}[/bold yellow]\n✗ False Negative",
                  f"[bold green]TP = {tp:,}[/bold green]\n✔ Correct Positive")
    console.print(table)

    console.print(Panel(
        f"""
[bold cyan]Confusion Matrix — Threat Intelligence Breakdown:[/bold cyan]

  [bold green]True Positive  (TP) = {tp:,}[/bold green]
    → Phishing URL correctly identified as malicious  ✔

  [bold green]True Negative  (TN) = {tn:,}[/bold green]
    → Legitimate URL correctly classified as benign  ✔

  [bold red]False Positive (FP) = {fp:,}[/bold red]
    → Legitimate URL incorrectly flagged as phishing  ✗  (False Alarm)

  [bold yellow]False Negative (FN) = {fn:,}[/bold yellow]
    → Phishing URL missed — classified as legitimate  ✗  (Critical Miss!)
        """,
        title="[bold cyan]Classification Report[/bold cyan]", border_style="cyan"))


# ─────────────────────────────────────────────────────────────────────
# COMPONENT-WISE EVALUATION (professor requirement)
# ─────────────────────────────────────────────────────────────────────
def evaluate_components_separately(model, feature_columns, X_test, y_test, urls_test):
    console.print(Panel(
        "[bold cyan]Evaluating ML model, rule-based method, and hybrid system separately...[/bold cyan]",
        title="[bold cyan]COMPONENT-WISE EVALUATION[/bold cyan]",
        border_style="cyan"
    ))

    # Component 1: ML Model alone
    y_pred_ml = model.predict(X_test)
    y_prob_ml = model.predict_proba(X_test)[:, 1]

    acc_ml = accuracy_score(y_test, y_pred_ml)
    prec_ml = precision_score(y_test, y_pred_ml, zero_division=0)
    rec_ml = recall_score(y_test, y_pred_ml, zero_division=0)
    f1_ml = f1_score(y_test, y_pred_ml, zero_division=0)

    cm_ml = confusion_matrix(y_test, y_pred_ml)
    tn_ml, fp_ml, fn_ml, tp_ml = cm_ml.ravel()

    console.print(Panel(
        f"""
Component 1: ML Voting Ensemble Model

Accuracy  : {acc_ml:.4f} ({acc_ml*100:.2f}%)
Precision : {prec_ml:.4f} ({prec_ml*100:.2f}%)
Recall    : {rec_ml:.4f} ({rec_ml*100:.2f}%)
F1 Score  : {f1_ml:.4f} ({f1_ml*100:.2f}%)

Confusion Matrix:
TN = {tn_ml:,}
FP = {fp_ml:,}
FN = {fn_ml:,}
TP = {tp_ml:,}
        """,
        title="[bold green]ML MODEL PERFORMANCE[/bold green]",
        border_style="green"
    ))

    # Component 2: Rule-Based method alone
    rb_preds = pd.Series(
        [1 if calculate_rule_based_risk(str(u)) >= 5 else 0 for u in urls_test],
        index=y_test.index
    )

    acc_rb = accuracy_score(y_test, rb_preds)
    prec_rb = precision_score(y_test, rb_preds, zero_division=0)
    rec_rb = recall_score(y_test, rb_preds, zero_division=0)
    f1_rb = f1_score(y_test, rb_preds, zero_division=0)

    cm_rb = confusion_matrix(y_test, rb_preds)
    tn_rb, fp_rb, fn_rb, tp_rb = cm_rb.ravel()

    console.print(Panel(
        f"""
Component 2: Rule-Based Heuristic Method

Accuracy  : {acc_rb:.4f} ({acc_rb*100:.2f}%)
Precision : {prec_rb:.4f} ({prec_rb*100:.2f}%)
Recall    : {rec_rb:.4f} ({rec_rb*100:.2f}%)
F1 Score  : {f1_rb:.4f} ({f1_rb*100:.2f}%)

Confusion Matrix:
TN = {tn_rb:,}
FP = {fp_rb:,}
FN = {fn_rb:,}
TP = {tp_rb:,}
        """,
        title="[bold yellow]RULE-BASED PERFORMANCE[/bold yellow]",
        border_style="yellow"
    ))

    # Component 3: Hybrid system
    hybrid_preds = pd.Series(
        [
            1 if (
                (p >= 0.85 and calculate_rule_based_risk(str(u)) >= 3)
                or (calculate_rule_based_risk(str(u)) >= 5 and p >= 0.45)
            ) else 0
            for p, u in zip(y_prob_ml, urls_test)
        ],
        index=y_test.index
    )

    acc_h = accuracy_score(y_test, hybrid_preds)
    prec_h = precision_score(y_test, hybrid_preds, zero_division=0)
    rec_h = recall_score(y_test, hybrid_preds, zero_division=0)
    f1_h = f1_score(y_test, hybrid_preds, zero_division=0)

    cm_h = confusion_matrix(y_test, hybrid_preds)
    tn_h, fp_h, fn_h, tp_h = cm_h.ravel()

    console.print(Panel(
        f"""
Component 3: Combined Hybrid System

Accuracy  : {acc_h:.4f} ({acc_h*100:.2f}%)
Precision : {prec_h:.4f} ({prec_h*100:.2f}%)
Recall    : {rec_h:.4f} ({rec_h*100:.2f}%)
F1 Score  : {f1_h:.4f} ({f1_h*100:.2f}%)

Confusion Matrix:
TN = {tn_h:,}
FP = {fp_h:,}
FN = {fn_h:,}
TP = {tp_h:,}
        """,
        title="[bold cyan]HYBRID SYSTEM PERFORMANCE[/bold cyan]",
        border_style="cyan"
    ))

    # Component 4: API Layer
    console.print(Panel(
        """
Component 4: External Threat Intelligence APIs

Google Safe Browsing API:
Used to check whether a URL is reported as phishing, malware, social engineering, or unwanted software.

VirusTotal API:
Used to check whether multiple security vendors have reported a URL as malicious or suspicious.

Note:
The APIs were evaluated qualitatively because live API results depend on database coverage, request limits, and whether a URL has already been reported.
        """,
        title="[bold magenta]API LAYER EVALUATION[/bold magenta]",
        border_style="magenta"
    ))



def build_fast_model():
    rf = RandomForestClassifier(
        n_estimators=200,     
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1              
    )

    if LGBM_AVAILABLE:
        lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=63,
            max_depth=-1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model = VotingClassifier(
            estimators=[("rf", rf), ("lgbm", lgbm)],
            voting="soft",
            n_jobs=-1
        )
        model_name = "Voting Ensemble (Random Forest + LightGBM)"
    else:
    
        model = rf
        model_name = "Random Forest (LightGBM not installed — run: pip install lightgbm)"

    return model, model_name


def train_random_forest_model():
    if os.path.exists(MODEL_PATH):
        console.print(Panel(
            "[bold green]Loading saved model from disk...[/bold green]",
            title="[bold cyan]MODEL LOAD[/bold cyan]",
            border_style="cyan"
        ))
        saved_data = joblib.load(MODEL_PATH)
        return saved_data["model"], saved_data["features"]

    X, y, all_urls = prepare_dataset()
    if X is None:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    urls_test = pd.Series(all_urls[X_test.index], index=y_test.index)

    model, model_name = build_fast_model()

    console.print(Panel(
        f"[bold cyan]Training {model_name}...[/bold cyan]\n"
        "[bold green]Expected time: 10–40 seconds[/bold green]",
        title="[bold cyan]MODEL TRAINING[/bold cyan]",
        border_style="cyan"
    ))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    console.print(Panel(
        f"""
[bold green]{model_name} — Trained Successfully[/bold green]

── Holdout Test Performance ──

Accuracy  : {accuracy:.4f}  →  {accuracy*100:.2f}%
Precision : {precision:.4f}  →  {precision*100:.2f}%
Recall    : {recall:.4f}  →  {recall*100:.2f}%
F1 Score  : {f1:.4f}  →  {f1*100:.2f}%

Test samples : {len(y_test):,}
Correct      : {tp+tn:,}
Wrong        : {fp+fn:,}

Confusion Matrix:
TN = {tn:,}
FP = {fp:,}
FN = {fn:,}
TP = {tp:,}
        """,
        title="[bold cyan]MODEL PERFORMANCE[/bold cyan]",
        border_style="cyan"
    ))

    evaluate_components_separately(
        model,
        X.columns.tolist(),
        X_test,
        y_test,
        urls_test
    )

    joblib.dump(
        {"model": model, "features": X.columns.tolist()},
        MODEL_PATH
    )

    console.print(Panel(
        "[bold green]Model saved to disk. Future runs will load instantly.[/bold green]",
        title="[bold cyan]MODEL SAVED[/bold cyan]",
        border_style="cyan"
    ))

    return model, X.columns.tolist()


def is_website_available(domain):
    try:
        socket.gethostbyname(domain)
        return True
    except Exception:
        return False


def check_google_safe_browsing(url):

    if not API_KEY:
        return False
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={API_KEY}"
    payload = {
        "client": {"clientId": "ai-phishing-detector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE","SOCIAL_ENGINEERING","UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=10)
        return r.status_code == 200 and "matches" in r.json()
    except Exception:
        return False


VT_API_KEY = "your key"


def check_virustotal(url):
    headers = {"x-apikey": VT_API_KEY}
    try:
        r = requests.post("https://www.virustotal.com/api/v3/urls",
                          headers=headers, data={"url": url})
        if r.status_code != 200:
            return False, 0
        analysis_id = r.json()["data"]["id"]
        r2 = requests.get(f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
                          headers=headers)
        stats = r2.json()["data"]["attributes"]["stats"]
        total = stats.get("malicious", 0) + stats.get("suspicious", 0)
        return total > 0, total
    except Exception:
        return False, 0


def calculate_rule_based_risk(url):
    risk_score = 0
    TRUSTED = ["google.com","youtube.com","facebook.com","amazon.com",
                "microsoft.com","apple.com","github.com","linkedin.com",
                "twitter.com","instagram.com","reddit.com","wikipedia.org"]
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    if any(domain.endswith(t) for t in TRUSTED):
        return 0

    FAKE_BRANDS     = ["g00gle","amaz0n","paypa1","faceb00k","micr0soft",
                       "app1e","netfl1x","inst4gram","twitt3r","g0ogle",
                       "arnazon","micosoft","paypai","linkedln"]
    SUSPICIOUS_WORDS = ["login","verify","secure","update","bank","free","gift",
                        "bonus","password","account","confirm","reward","signin",
                        "webscr","billing","support","paypal","ebay"]
    SUSPICIOUS_TLDS  = [".xyz",".ru",".tk",".top",".ml",".ga",".cf",".gq",
                        ".pw",".club",".work",".date",".download",".racing",
                        ".review",".stream"]

    for b in FAKE_BRANDS:
        if b in url.lower():     risk_score += 3
    for t in SUSPICIOUS_TLDS:
        if t in domain:          risk_score += 2
    for w in SUSPICIOUS_WORDS:
        if w in url.lower():     risk_score += 1
    if "@" in url:               risk_score += 2
    if "-" in domain:            risk_score += 1
    if len(url) > 75:            risk_score += 1
    if len(url) > 100:           risk_score += 1
    if url.startswith("http://"): risk_score += 2
    if re.search(r"\d+\.\d+\.\d+\.\d+", domain): risk_score += 3
    if domain.count(".") > 3:    risk_score += 2

    return min(risk_score, 10)


# ─────────────────────────────────────────────────────────────────────
# OUTPUT PANELS — identical design to original
# ─────────────────────────────────────────────────────────────────────
def show_safe(url, ml_probability, risk, google_status, vt_status):
    console.print(Panel(f"""
[bold green]✔ WEBSITE IS LEGITIMATE[/bold green]

Website: {url}

Google Safe Browsing  : {google_status}
VirusTotal Scan       : {vt_status}

ML Phishing Probability : {ml_probability:.2f}
Rule-Based Risk Score   : {risk}/10

[bold cyan]Result: Safe / Legitimate Website[/bold cyan]
""", title="[bold green]SAFE WEBSITE[/bold green]", border_style="green"))


def show_danger(url, ml_probability, risk, reason, google_status, vt_status):
    console.print(Panel(f"""
[bold red]⚠ DANGEROUS WEBSITE DETECTED ⚠[/bold red]

Website: {url}

Google Safe Browsing  : {google_status}
VirusTotal Scan       : {vt_status}

Reason: {reason}

ML Phishing Probability : {ml_probability:.2f}
Rule-Based Risk Score   : {risk}/10

[bold yellow]Possible phishing, scam, malware, or harmful website.[/bold yellow]
[bold red]Do not enter passwords, card details, or personal information.[/bold red]
""", title="[bold red]DANGER[/bold red]", border_style="red"))


def show_safe_unavailable(url, ml_probability, risk):
    console.print(Panel(f"""
[bold cyan]🛡 TARGET OFFLINE / DOMAIN UNREACHABLE[/bold cyan]

Website: {url}

ML Phishing Probability : {ml_probability:.2f}
Rule-Based Risk Score   : {risk}/10

[bold green]Verdict: No strong hostile indicators detected in URL structure.[/bold green]

[bold cyan]Cyber Analyst Note:[/bold cyan]
Target is offline or DNS is not resolving, but based on lexical URL analysis,
this does not appear to be an active phishing pattern.
""", title="[bold green]SAFE / LOW RISK[/bold green]", border_style="green"))


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    banner()

    model, feature_columns = train_random_forest_model()
    if model is None:
        return

    url    = input(Fore.CYAN + "Enter Website URL: " + Style.RESET_ALL)
    url    = fix_url(url)
    parsed = urlparse(url)
    domain = parsed.netloc

    print("\n[bold blue]Checking website...[/bold blue]\n")

    website_available = is_website_available(domain)

    google_flagged = False
    vt_flagged     = False
    vt_score       = 0
    google_status  = "[green]PASSED[/green]"
    vt_status      = "[green]PASSED[/green]"

    if website_available:
        google_flagged       = check_google_safe_browsing(url)
        vt_flagged, vt_score = check_virustotal(url)

    if google_flagged:
        google_status = "[bold red]FAILED[/bold red]"
    if vt_flagged:
        vt_status = f"[bold red]FAILED ({vt_score} engines flagged)[/bold red]"

    raw_features = extract_features(url)
    url_features = pd.DataFrame([raw_features])
    for col in feature_columns:
        if col not in url_features.columns:
            url_features[col] = 0
    url_features = url_features[feature_columns]

    probability = model.predict_proba(url_features)[0][1]
    rule_risk   = calculate_rule_based_risk(url)

    if google_flagged:
        show_danger(url, probability, 10,
                    "Flagged by Google Safe Browsing API",
                    google_status, vt_status)
    elif vt_flagged:
        show_danger(url, probability, rule_risk,
                    f"VirusTotal detected malicious activity ({vt_score} security vendors flagged this URL)",
                    google_status, vt_status)
    elif not website_available and (probability >= 0.70 or rule_risk >= 5):
        show_danger(url, probability, rule_risk,
                    "Offline domain with suspicious phishing indicators",
                    google_status, vt_status)
    elif probability >= 0.85 and rule_risk >= 3:
        show_danger(url, probability, rule_risk,
                    "Detected as phishing by ML Voting Ensemble",
                    google_status, vt_status)
    elif rule_risk >= 5 and probability >= 0.45:
        show_danger(url, probability, rule_risk,
                    "High rule-based phishing risk score",
                    google_status, vt_status)
    elif not website_available:
        show_safe_unavailable(url, probability, rule_risk)
    else:
        show_safe(url, probability, rule_risk, google_status, vt_status)


if __name__ == "__main__":
    
    main()
