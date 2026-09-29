import re
from typing import Dict, Any, List, Set, Tuple, Optional
import fitz  # PyMuPDF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# Highlight color map (RGB normalized 0.0 - 1.0)
COLOR_PALETTE = {
    "Yellow": (1.0, 1.0, 0.2),
    "Green": (0.3, 0.9, 0.3),
    "Cyan": (0.3, 0.8, 1.0),
    "Magenta": (1.0, 0.4, 0.8),
    "Orange": (1.0, 0.7, 0.2),
}


def get_default_stopwords() -> Set[str]:
    """Retrieve NLTK English stopwords."""
    try:
        return set(stopwords.words("english"))
    except Exception:
        return {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "is", "are", "it"}


def extract_keywords_nlp(
    text: str,
    top_n: int = 10,
    custom_stopwords: Optional[Set[str]] = None,
    ngram_range: Tuple[int, int] = (1, 2)
) -> List[str]:
    """
    Extracts high-relevance key phrases using TF-IDF n-grams with stop-word filtering.
    """
    if not text.strip():
        return []

    stop_words_set = get_default_stopwords()
    if custom_stopwords:
        stop_words_set.update(custom_stopwords)

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words=list(stop_words_set),
            max_features=top_n * 3,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        # Sort features by TF-IDF score
        ranked_features = [
            feature_names[i]
            for i in scores.argsort()[::-1]
            if len(feature_names[i].strip()) > 2
        ]
        return ranked_features[:top_n]
    except Exception:
        # Fallback to simple frequency if TF-IDF cannot run
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        filtered = [w for w in words if w not in stop_words_set]
        counts = {}
        for w in filtered:
            counts[w] = counts.get(w, 0) + 1
        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:top_n]]


def highlight_keywords_in_pdf(
    doc: fitz.Document,
    keywords: List[str],
    color_name: str = "Yellow",
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Searches for keywords across all PDF pages and adds highlight annotations.
    Returns summary statistics of annotations added.
    """
    color = COLOR_PALETTE.get(color_name, COLOR_PALETTE["Yellow"])
    summary: Dict[str, Any] = {
        "total_highlights": 0,
        "keyword_counts": {k: 0 for k in keywords},
        "pages_affected": set()
    }

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        for kw in keywords:
            kw_clean = kw.strip()
            if not kw_clean:
                continue

            flags = 0 if case_sensitive else fitz.TEXT_DEHYPHENATE
            # Search for quads or rectangles
            rects = page.search_for(kw_clean, flags=flags)
            if rects:
                summary["keyword_counts"][kw] = summary["keyword_counts"].get(kw, 0) + len(rects)
                summary["total_highlights"] += len(rects)
                summary["pages_affected"].add(page_num)

                for r in rects:
                    annot = page.add_highlight_annot(r)
                    annot.set_colors(stroke=color)
                    annot.update()

    summary["pages_affected"] = sorted(list(summary["pages_affected"]))
    return summary
