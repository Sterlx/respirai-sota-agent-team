from pathlib import Path
import re
from collections import Counter
from math import log


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SKILL_REPOS = {
    "ai_research": PROJECT_ROOT / "external" / "AI-research-SKILLs",
    "scientific": PROJECT_ROOT / "external" / "scientific-agent-skills",
}

# Words that don't add signal for skill matching
STOPWORDS: set[str] = {
    "the", "and", "for", "that", "this", "with", "from", "your", "will",
    "have", "been", "are", "was", "not", "but", "you", "all", "can",
    "has", "had", "its", "each", "how", "use", "used", "using", "into",
    "more", "than", "some", "these", "those", "when", "what", "which",
    "very", "just", "also", "about", "after", "before", "between",
    "other", "only", "most", "make", "made", "making", "take", "does",
    "they", "them", "their", "here", "there", "well",
}


def tokenize(text: str) -> list[str]:
    """Split text into cleaned lowercase tokens, removing stopwords and short words."""
    # Remove markdown syntax, URLs, code blocks
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](url) → text
    text = re.sub(r"[#*>`|~\-_=\[\]{}():;.!?,/\\\"]+", " ", text)

    words = text.lower().split()
    return [
        w for w in words
        if len(w) >= 3 and w not in STOPWORDS and not w.isdigit()
    ]


def extract_headings(text: str) -> list[str]:
    """Extract heading text from markdown — headings carry more signal."""
    headings = re.findall(r"^#+\s+(.+)$", text, re.MULTILINE)
    return headings


def score_text(
    query: str,
    text: str,
    file_path: str = "",
    global_idf: dict[str, float] | None = None,
) -> float:
    """
    Improved scoring with:
    - TF-IDF style weighting (rare words matter more)
    - Phrase matching (bigrams)
    - Heading boost (words in markdown headings get 2× weight)
    - Path relevance bonus (filename words match query)
    """
    query_tokens = tokenize(query)

    # Build query bigrams for phrase matching
    query_bigrams = set()
    for i in range(len(query_tokens) - 1):
        query_bigrams.add(f"{query_tokens[i]} {query_tokens[i+1]}")

    text_tokens = tokenize(text)
    text_lower = text.lower()
    headings = extract_headings(text)
    heading_tokens = tokenize(" ".join(headings))

    # TF in the document
    text_counter = Counter(text_tokens)

    score = 0.0

    for token in query_tokens:
        if token not in text_counter:
            continue

        tf = text_counter[token]

        # IDF: rare words get higher weight (default idf=1.0 if no global_idf)
        idf = global_idf.get(token, 1.0) if global_idf else 1.0

        base = tf * idf

        # Heading boost: words that appear in markdown headings are more relevant
        if token in heading_tokens:
            base *= 2.0

        score += base

    # Bigram (phrase) bonus — bigrams are strong relevance signals
    for bigram in query_bigrams:
        count = len(re.findall(re.escape(bigram), text_lower))
        if count > 0:
            score += count * 3.0  # Phrases get 3× weight

    # Path relevance bonus: words in the skill file's path that match the query
    if file_path:
        path_tokens = set(tokenize(file_path))
        path_match = sum(1 for t in query_tokens if t in path_tokens)
        score += path_match * 1.5

    return score


def compute_idf(all_texts: list[str]) -> dict[str, float]:
    """Compute inverse document frequency across all skill files."""
    n_docs = len(all_texts)
    if n_docs == 0:
        return {}

    doc_freq: Counter[str] = Counter()
    for text in all_texts:
        unique_tokens = set(tokenize(text))
        doc_freq.update(unique_tokens)

    return {
        token: log((n_docs + 1) / (count + 1)) + 1.0
        for token, count in doc_freq.items()
    }


def read_markdown_files(repo_path: Path, max_files: int = 80) -> list[tuple[str, str]]:
    """
    Read markdown files from an external skill repository.

    Returns:
        A list of (relative_file_path, text) pairs.
    """
    if not repo_path.exists():
        raise FileNotFoundError(
            f"Skill repository not found: {repo_path}\n"
            "Make sure you cloned the external skill repos into external/."
        )

    markdown_files = list(repo_path.rglob("*.md"))[:max_files]
    results = []

    for file_path in markdown_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                results.append((str(file_path.relative_to(PROJECT_ROOT)), text))
        except Exception as exc:
            print(f"Warning: could not read {file_path}: {exc}")

    return results


def find_relevant_skills(
    query: str,
    max_results: int = 5,
    excerpt_chars: int = 3500,
) -> str:
    """
    Search external skill repositories and return relevant excerpts.

    Uses TF-IDF weighted scoring with bigram phrase matching
    and heading-aware relevance boosting.
    """
    # --- Collect all skill files ---
    all_files: list[tuple[str, str, str]] = []  # (repo_name, rel_path, text)

    for repo_name, repo_path in SKILL_REPOS.items():
        markdown_files = read_markdown_files(repo_path)
        for relative_path, text in markdown_files:
            all_files.append((repo_name, relative_path, text))

    if not all_files:
        return "No external skill files found. Clone the repos into external/."

    # --- Compute global IDF across all skill files ---
    global_idf = compute_idf([text for _, _, text in all_files])

    # --- Score each file ---
    scored_results = []

    for repo_name, relative_path, text in all_files:
        score = score_text(
            query=query,
            text=text,
            file_path=relative_path,
            global_idf=global_idf,
        )

        if score > 0:
            scored_results.append({
                "score": round(score, 2),
                "repo": repo_name,
                "path": relative_path,
                "text": text,
            })

    # --- Sort and select top results ---
    scored_results.sort(key=lambda item: item["score"], reverse=True)
    selected = scored_results[:max_results]

    if not selected:
        return "No relevant external skills found."

    # --- Format output ---
    parts = []

    for item in selected:
        excerpt = item["text"][:excerpt_chars]

        parts.append(
            f"""
SKILL_SOURCE_REPO: {item["repo"]}
SKILL_SOURCE_FILE: {item["path"]}
MATCH_SCORE: {item["score"]}

SKILL_EXCERPT:
{excerpt}
"""
        )

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    test_query = """
    RespirAI ICBHI2017 lung sound classification official 60/40 split
    dataset preprocessing training evaluation metrics clinical safety
    """
    result = find_relevant_skills(test_query)
    print(result)
    print(f"\n---\nFound {result.count('SKILL_SOURCE_REPO:')} skills.")