"""
preprocessor.py
---------------
Text cleaning for the Zenodo IT support ticket dataset.
Handles multilingual text (English, German, Portuguese, Spanish).
Handles PII tags like [NAME], [EMAIL], [PHONE] already in the dataset.
"""

import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import hstack, csr_matrix


# ── Stopwords (English + common Portuguese/German/Spanish function words) ─────

STOPWORDS = {
    # English
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","please","would","could","get","got","also","want",
    "need","use","using","used","know","try","tried","via","however","since",
    "every","one","two","three","new","like","make","made","still","even",
    "see","seems","us","may","might","many","much","back","hi","hello","dear",
    "regards","thank","thanks","sincerely","attached","attachment","kindly",
    # Portuguese
    "eu","tu","ele","ela","nos","vos","eles","elas","um","uma","o","a","os",
    "as","de","da","do","em","na","no","por","para","com","que","se","mais",
    "mas","ao","dos","das","pelo","pela","como","foi","ser","ter","uma","isso",
    # German
    "ich","du","er","sie","es","wir","ihr","den","dem","der","die","das",
    "ein","eine","und","oder","aber","nicht","mit","auf","bei","von","zu",
    "ist","sind","hat","haben","war","wurde","werden","kann","muss",
    # Spanish
    "yo","tu","el","nosotros","vosotros","ellos","un","una","los","las",
    "del","al","por","para","con","que","pero","mas","como","fue","ser",
    "tener","estar","muy","bien","este","esta","estos","estas",
}

PII_TAGS = {"[name]","[email]","[phone]","[address]","[url]","[date]",
            "[number]","[id]","[ip]","[organization]","[location]"}


def clean_text(text: str) -> str:
    """
    Clean and normalise multilingual support ticket text.
    Handles PII placeholder tags from the Zenodo dataset.
    """
    text = str(text).lower()

    # Remove PII tags (already anonymised in dataset)
    for tag in PII_TAGS:
        text = text.replace(tag, " ")

    # Remove remaining bracket tags
    text = re.sub(r"\[.*?\]", " ", text)

    # Remove URLs and emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove standalone numbers, but preserve technical identifiers such as O365.
    text = re.sub(r"(?<![a-z0-9])\d+(?![a-z0-9])", " ", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Tokenise, remove stopwords, keep tokens >= 2 chars
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]

    return " ".join(tokens)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text and meta features to DataFrame."""
    df = df.copy()
    df["clean_text"]   = df["text"].apply(clean_text)

    # Meta features
    df["char_len"]     = df["text"].str.len()
    df["word_count"]   = df["text"].str.split().str.len()
    df["exclaim"]      = df["text"].str.count(r"!")
    df["question"]     = df["text"].str.count(r"\?")

    urgency_pat = (r"\b(?:urgent|asap|immediately|critical|emergency|"
                   r"broken|crash|production|hack|compromised|down|outage)\b")
    df["urgency_flag"] = (df["text"].str.lower()
                                    .str.contains(urgency_pat, regex=True)
                                    .astype(int))
    return df


META_COLS = ["char_len", "word_count", "exclaim", "question", "urgency_flag"]


def get_tfidf():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.92,
    )
