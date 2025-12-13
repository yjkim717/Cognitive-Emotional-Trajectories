import os
import re
from typing import Dict
from config import get_llm_path

# -------------------------------------------------------------------------
# MODEL TAGS
# -------------------------------------------------------------------------
MODEL_TAGS = {
    "DEEPSEEK": "DS",
    "GEMMA_4B": "G4B",
    "GEMMA_12B": "G12B",
    "LLAMA_MAVRICK": "LMK",
}


# -------------------------------------------------------------------------
# BASIC FILE OPERATIONS
# -------------------------------------------------------------------------
def read_text(path: str) -> str:
    """Read UTF-8 text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str):
    """Write text to file, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# -------------------------------------------------------------------------
# METADATA PARSING (supports Academic / Blogs / News / custom structures)
# -------------------------------------------------------------------------
def parse_metadata_from_path(human_path: str) -> Dict[str, str]:
    """
    Parse genre, subfield, author_id (batch), year, and index from a Human dataset path.
    
    Unified format for all genres (micro dataset):
        {GENRE}_{SUBFIELD}_{AUTHOR_ID}_{YEAR}_{ITEM_INDEX}.txt
    
    Examples:
      - Blogs_LIFESTYLE_01_2020_01.txt
      - News_10years_01_2012_01.txt
      - Academic_CHEMISTRY_03_2022_01.txt
    """
    parts = os.path.normpath(human_path).split(os.sep)

    try:
        # Support "Human", "cleaned_human", and "human" directories
        if "Human" in parts:
            idx = parts.index("Human")
        elif "cleaned_human" in parts:
            idx = parts.index("cleaned_human")
        elif "human" in parts:
            idx = parts.index("human")
        else:
            raise ValueError(f"No Human/cleaned_human/human directory found in path: {human_path}")
            
        genre = parts[idx + 1]           # e.g. Academic / Blogs / News
        filename = parts[-1]             # Always the filename
    except (ValueError, IndexError):
        raise ValueError(f"Unexpected path structure: {human_path}")

    # --- Unified format: {GENRE}_{SUBFIELD}_{AUTHOR_ID}_{YEAR}_{ITEM_INDEX}.txt
    # This matches all three genres: Blogs, News, Academic
    unified_pattern = r"^([A-Za-z]+)_([A-Za-z0-9]+)_(\d+)_(\d{4})_(\d+)\.txt$"
    m_unified = re.match(unified_pattern, filename)
    
    if m_unified:
        genre_from_name = m_unified.group(1)        # e.g., Blogs, News, Academic
        subfield_from_name = m_unified.group(2)     # e.g., LIFESTYLE, 10years, CHEMISTRY
        author_id = m_unified.group(3)              # e.g., 01, 03, 14
        year = m_unified.group(4)                   # e.g., 2020, 2012, 2022
        item_index = m_unified.group(5)             # e.g., 01, 02, 03
        
        # Use genre from filename if it matches the directory, otherwise use directory genre
        if genre_from_name.lower() == genre.lower():
            final_genre = genre_from_name
        else:
            final_genre = genre
        
        return {
            "genre": final_genre.lower(),          # Normalize to lowercase
            "subfield": subfield_from_name.lower(),  # Normalize to lowercase
            "batch": author_id,                    # Author ID (also called batch)
            "year": str(year),
            "index": str(item_index),
        }
    
    # --- Fallback: Try to extract genre from directory if filename doesn't match unified format
    # This handles legacy formats or files that haven't been renamed yet
    year_match = re.search(r"\d{4}", human_path)
    year = year_match.group(0) if year_match else "0000"
    
    # Try to extract subfield from directory structure (for Academic files in subdirectories)
    subfield = None
    try:
        if idx + 2 < len(parts):
            subfield = parts[idx + 2]  # e.g., bio, chemistry, cs
    except (ValueError, IndexError):
        pass
    
    return {
        "genre": genre.lower(),
        "subfield": subfield.lower() if subfield else "unknown",
        "batch": "01",  # Default batch/author_id
        "year": str(year),
        "index": "01",  # Default index
    }


def parse_macro_metadata_from_path(human_path: str) -> Dict[str, str]:
    """
    Parse metadata from macro_dataset human paths.
    
    Expected filename format for macro dataset:
        {GENRE}_{FIELD}_{YEAR}_{INDEX}.txt
    
    Examples:
      - NEWS_WORLD_2024_200.txt
      - Academic_BIOLOGY_2020_001.txt
    
    Returns a metadata dict compatible with BaseGenerator:
      - genre: lowercased genre name ("news", "academic", "blogs", ...)
      - subfield: lowercased field name
      - batch: fixed to "01" (no author/batch concept for macro, kept for compatibility)
      - year: 4-digit year string
      - index: index string (e.g., "200", "001")
    """
    parts = os.path.normpath(human_path).split(os.sep)

    try:
        # Find the "human" directory for macro dataset
        if "human" not in parts:
            raise ValueError(f"No human directory found in macro path: {human_path}")
        
        idx = parts.index("human")
        genre_dir = parts[idx + 1]      # e.g., news, academic
        filename = parts[-1]
    except (ValueError, IndexError):
        raise ValueError(f"Unexpected macro path structure: {human_path}")

    # Macro filename format: {GENRE}_{FIELD}_{YEAR}_{INDEX}.txt
    macro_pattern = r"^([A-Za-z]+)_([A-Za-z0-9]+)_(\d{4})_(\d+)\.txt$"
    m = re.match(macro_pattern, filename)

    if not m:
        raise ValueError(f"Filename does not match macro format {{GENRE}}_{{FIELD}}_{{YEAR}}_{{INDEX}}.txt: {filename}")

    genre_from_name = m.group(1)   # e.g., NEWS, Academic
    field = m.group(2)             # e.g., WORLD, BIOLOGY
    year = m.group(3)              # e.g., 2024
    index = m.group(4)             # e.g., 200, 001

    # Normalize genre: prioritize filename genre if it matches known genres,
    # otherwise fall back to directory name.
    if genre_from_name.lower() in {"news", "blogs", "academic"}:
        final_genre = genre_from_name.lower()
    else:
        final_genre = genre_dir.lower()

    return {
        "genre": final_genre,
        "subfield": field.lower(),
        "batch": "01",        # Macro dataset has no author id; keep field for compatibility
        "year": str(year),
        "index": str(index),
    }


def parse_macro_llm_metadata_from_path(llm_path: str) -> Dict[str, str]:
    """
    Parse metadata from macro_dataset LLM paths.

    Expected filename format:
        {GENRE}_{FIELD}_{YEAR}_{INDEX}_{MODEL}_{LEVEL}.txt

    Example:
        News_TECHNOLOGY_2023_021_G12B_LV1.txt
    """
    parts = os.path.normpath(llm_path).split(os.sep)

    try:
        if "llm" not in parts:
            raise ValueError(f"No llm directory found in macro path: {llm_path}")

        idx = parts.index("llm")
        genre_dir = parts[idx + 1]
        filename = parts[-1]
    except (ValueError, IndexError):
        raise ValueError(f"Unexpected macro LLM path structure: {llm_path}")

    macro_llm_pattern = r"^([A-Za-z]+)_([A-Za-z0-9]+)_(\d{4})_(\d+)_([A-Z0-9]+)_(LV[123])\.txt$"
    m = re.match(macro_llm_pattern, filename)

    if not m:
        raise ValueError(
            "Filename does not match macro LLM format "
            "{GENRE}_{FIELD}_{YEAR}_{INDEX}_{MODEL}_{LEVEL}.txt: "
            f"{filename}"
        )

    genre_from_name = m.group(1)
    field = m.group(2)
    year = m.group(3)
    index = m.group(4)
    model = m.group(5)
    level = m.group(6)

    if genre_from_name.lower() in {"news", "blogs", "academic"}:
        final_genre = genre_from_name.lower()
    else:
        final_genre = genre_dir.lower()

    return {
        "genre": final_genre,
        "subfield": field.lower(),
        "batch": "01",
        "year": str(year),
        "index": str(index),
        "model": model.upper(),
        "level": level.upper(),
    }


# -------------------------------------------------------------------------
# OUTPUT FILENAME BUILDER
# -------------------------------------------------------------------------
def build_llm_filename(meta: Dict[str, str], level: int | None = None, provider: str | None = None) -> str:
    """
    Build output filename for generated LLM files.
    Uses unified format for all genres: {GENRE}_{SUBFIELD}_{AUTHOR_ID}_{YEAR}_{ITEM_INDEX}_{MODEL}_{LEVEL}.txt
    
    Includes:
      - Model tag (DS / G4B / G12B / LMK)
      - Level suffix (_LV1, _LV2, _LV3)
    
    Args:
        meta: Metadata dictionary with genre, subfield, batch, year, index
        level: Generation level (1, 2, or 3)
        provider: LLM provider (DEEPSEEK, GEMMA_4B, GEMMA_12B, LLAMA_MAVRICK). 
                 If None, reads from environment variable LLM_PROVIDER.
    
    Examples:
      - Blogs_LIFESTYLE_01_2020_01_DS_LV1.txt
      - News_10years_01_2012_01_DS_LV1.txt
      - Academic_CHEMISTRY_03_2022_01_DS_LV1.txt
    """
    # Use provided provider or fall back to environment variable
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()
    else:
        provider = provider.upper()
    tag = MODEL_TAGS.get(provider, "DS")

    genre = meta["genre"]
    tag_with_level = f"{tag}_LV{level}" if level else tag

    # Unified format: {GENRE}_{SUBFIELD}_{AUTHOR_ID}_{YEAR}_{ITEM_INDEX}_{MODEL}_{LEVEL}.txt
    # Capitalize genre (first letter uppercase)
    genre_capitalized = genre.capitalize()  # e.g., blogs -> Blogs, news -> News, academic -> Academic
    
    # Handle subfield capitalization based on genre
    subfield = meta["subfield"]
    if genre.lower() == "blogs":
        # Blogs: SUBFIELD in uppercase (e.g., LIFESTYLE)
        subfield_formatted = subfield.upper()
    elif genre.lower() == "news":
        # News: subfield in lowercase (e.g., 10years)
        subfield_formatted = subfield.lower()
    elif genre.lower() == "academic":
        # Academic: SUBFIELD in uppercase (e.g., CHEMISTRY)
        subfield_formatted = subfield.upper()
    else:
        # Default: capitalize first letter
        subfield_formatted = subfield.capitalize()
    
    # Get author_id (batch)
    author_id = meta.get("batch", "01")  # Default to "01" if batch not present
    year = meta["year"]
    item_index = meta["index"]
    
    # Build unified filename
    return f"{genre_capitalized}_{subfield_formatted}_{author_id}_{year}_{item_index}_{tag_with_level}.txt"
