"""
load_data.py
------------
Loads the Zenodo IT Support Ticket dataset.

Dataset files needed (download from https://zenodo.org/records/7648117):
    X_train.csv  — training ticket texts
    X_test.csv   — test ticket texts
    y_train.csv  — training labels
    y_test.csv   — test labels

Place all four files inside the data/ folder.

This script:
  1. Loads and merges the four CSVs
  2. Engineers a priority column from keyword rules
  3. Saves a single clean combined CSV: data/tickets_clean.csv
"""

import re
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")


# ── Priority engineering rules ────────────────────────────────────────────────
# Since the dataset has no priority column, we assign it using keyword signals.
# This is a common real-world technique called "rule-based labelling".

HIGH_KEYWORDS = [
    r"\b(urgent|asap|immediately|critical|emergency|down|outage)\b",
    r"\b(crash|crashed|crashing|broken|not working|stopped working)\b",
    r"\b(production|server|database|network|infrastructure)\b",
    r"\b(security|breach|hack|compromised|virus|malware|ransomware)\b",
    r"\b(data loss|lost data|cannot access|locked out|blocked)\b",
    r"!{2,}",
]

LOW_KEYWORDS = [
    r"\b(question|wondering|how (do|can|to)|where is|what is)\b",
    r"\b(request|suggestion|feature|improvement|enhancement)\b",
    r"\b(information|documentation|guide|tutorial|manual)\b",
    r"\b(when will|roadmap|future|plan|schedule)\b",
]


def assign_priority(text: str) -> str:
    """Assign High / Medium / Low priority based on keyword rules."""
    text = str(text).lower()
    high_hits = sum(1 for p in HIGH_KEYWORDS if re.search(p, text))
    low_hits  = sum(1 for p in LOW_KEYWORDS  if re.search(p, text))

    if high_hits >= 2:
        return "High"
    elif high_hits == 1 and low_hits == 0:
        return "High"
    elif low_hits >= 2:
        return "Low"
    elif low_hits == 1 and high_hits == 0:
        return "Low"
    else:
        return "Medium"


def load_zenodo_dataset() -> pd.DataFrame:
    """
    Load the four Zenodo CSVs, merge them, and engineer priority.
    Returns a unified DataFrame with columns: text, category, priority, split
    """
    x_train_path = os.path.join(DATA_DIR, "X_train.csv")
    x_test_path  = os.path.join(DATA_DIR, "X_test.csv")
    y_train_path = os.path.join(DATA_DIR, "y_train.csv")
    y_test_path  = os.path.join(DATA_DIR, "y_test.csv")

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"\nERROR: Missing file: {p}"
                f"\n\nPlease download the dataset from:"
                f"\nhttps://zenodo.org/records/7648117"
                f"\n\nAnd place X_train.csv, X_test.csv, y_train.csv, y_test.csv"
                f"\ninside the Data/ folder."
            )

    # Load
    X_train = pd.read_csv(x_train_path)
    X_test  = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test  = pd.read_csv(y_test_path)

    # Find text column (first string column)
    text_columns = X_train.select_dtypes(include=["object", "string"]).columns
    if len(text_columns) == 0:
        raise ValueError("X_train.csv does not contain a text column.")
    text_col = text_columns[0]
    label_col = y_train.columns[-1]

    # Build unified dataframe
    df_train = pd.DataFrame({
        "text"    : X_train[text_col].values,
        "category": y_train[label_col].values,
        "split"   : "train",
    })
    df_test = pd.DataFrame({
        "text"    : X_test[text_col].values,
        "category": y_test[label_col].values,
        "split"   : "test",
    })

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Preserve the dataset's official category names.
    category_names = {
        "Support General": "Support General",
        "Support general": "Support General",
        "Fileservice": "Fileservice",
        "Software": "Software",
        "O365": "O365",
        "Active Directory": "Active Directory",
        "Computer-Services": "Computer-Services",
        "EOL": "Eol",
        "Eol": "Eol",
    }
    df["category"] = (
        df["category"].astype(str).str.strip()
        .replace(category_names)
    )
    df = df.dropna(subset=["text", "category"])
    df = df[df["text"].str.strip() != ""]

    # Engineer priority
    df["priority"] = df["text"].apply(assign_priority)

    print(f"Dataset loaded: {len(df)} tickets")
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\nPriority distribution (engineered):")
    print(df["priority"].value_counts().to_string())

    return df


if __name__ == "__main__":
    df = load_zenodo_dataset()
    out = os.path.join(DATA_DIR, "tickets_clean.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
