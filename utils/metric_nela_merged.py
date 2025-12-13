"""
Merged NELA feature extraction combining old and enhanced versions.

Merging strategy:
- Keep features from both old and enhanced versions
- Enhanced version features that replace old ones are kept alongside the old ones
- Removed features: punctuation_density, smog_index, sentence_density

Extracts:
- Emotional Layer (6 features):
  - polarity, subjectivity (TextBlob)
  - vader_compound, vader_pos, vader_neu, vader_neg (VADER)

- Stylistic Layer (9 features):
  From both versions:
  - word_diversity (Type-Token Ratio)
  - flesch_reading_ease (Readability)
  - gunning_fog (from old version, since smog_index is removed)
  
  From enhanced version:
  - average_word_length (length-independent)
  - verb_ratio
  
  From old version (added back):
  - num_words
  - avg_sentence_length
  - function_word_ratio
  - content_word_ratio
"""

import os
import re
from collections import Counter
from typing import Optional

import nltk
import pandas as pd
from textstat import flesch_reading_ease, gunning_fog
from textblob import TextBlob
from tqdm.auto import tqdm

from utils.file_utils import parse_macro_llm_metadata_from_path, parse_macro_metadata_from_path

# Download required NLTK data
_nltk_data_downloaded = False

def _ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    global _nltk_data_downloaded
    if _nltk_data_downloaded:
        return
    
    for pkg in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng", "vader_lexicon"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    
    _nltk_data_downloaded = True

_ensure_nltk_data()

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

from utils.parse_dataset_filename import parse_filename

# VADER sentiment analyzer
_vader_analyzer = None


def _is_macro_dataset(dataset_dir: str) -> bool:
    parts = os.path.abspath(dataset_dir).split(os.sep)
    return "macro_dataset" in parts


def _extract_metadata(file_name: str, file_path: str, is_llm: bool, macro_mode: bool, tag: str):
    if macro_mode:
        try:
            if is_llm:
                parsed = parse_macro_llm_metadata_from_path(file_path)
            else:
                parsed = parse_macro_metadata_from_path(file_path)
            return {
                "domain": parsed.get("genre"),
                "field": parsed.get("subfield"),
                "author_id": parsed.get("batch"),
                "year": parsed.get("year"),
                "item_index": parsed.get("index"),
                "model": parsed.get("model"),
                "level": parsed.get("level"),
            }
        except ValueError as exc:
            print(f"{tag} Warning: {exc}")
    return parse_filename(file_name, is_llm=is_llm)


def get_vader_analyzer():
    """Get or create VADER sentiment analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        from nltk.sentiment import SentimentIntensityAnalyzer
        _ensure_nltk_data()
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def compute_nela_features_merged(text: str) -> dict:
    """
    Compute merged NELA features from text.
    
    Combines features from old and enhanced versions.
    Removed: punctuation_density, smog_index, sentence_density
    
    Returns dict with 15 features:
    Emotional Layer (6):
    - polarity, subjectivity (TextBlob)
    - vader_compound, vader_pos, vader_neu, vader_neg (VADER)
    
    Stylistic Layer (9):
    - word_diversity, flesch_reading_ease, gunning_fog,
      average_word_length, num_words, avg_sentence_length,
      verb_ratio, function_word_ratio, content_word_ratio
    """
    if not text or len(text.strip()) == 0:
        return {
            # Emotional Layer
            'polarity': 0.0,
            'subjectivity': 0.0,
            'vader_compound': 0.0,
            'vader_pos': 0.0,
            'vader_neu': 0.0,
            'vader_neg': 0.0,
            # Stylistic Layer
            'word_diversity': 0.0,
            'flesch_reading_ease': 0.0,
            'gunning_fog': 0.0,
            'average_word_length': 0.0,
            'num_words': 0,
            'avg_sentence_length': 0.0,
            'verb_ratio': 0.0,
            'function_word_ratio': 0.0,
            'content_word_ratio': 0.0,
        }
    
    try:
        # Basic tokenization with fallback
        try:
            words = word_tokenize(text.lower())
        except LookupError:
            # Fallback if punkt_tab is not available
            words = text.lower().split()
        words = [w for w in words if w.isalnum()]  # Remove punctuation
        num_words = len(words)
        
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback: split by sentence-ending punctuation
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences) if sentences else 1
        
        # ============================================================
        # EMOTIONAL LAYER (6 features)
        # ============================================================
        
        # 1-2. TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # 3-6. VADER sentiment
        vader_analyzer = get_vader_analyzer()
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        vader_pos = vader_scores['pos']
        vader_neu = vader_scores['neu']
        vader_neg = vader_scores['neg']
        
        # ============================================================
        # STYLISTIC LAYER (4 features - removed punctuation_density, smog_index, sentence_density)
        # ============================================================
        
        # 1. word_diversity (Type-Token Ratio)
        unique_words = len(set(words))
        word_diversity = unique_words / num_words if num_words > 0 else 0.0
        
        # 2. flesch_reading_ease
        flesch_reading_ease_score = flesch_reading_ease(text)
        
        # 3. gunning_fog (from old version, since smog_index is removed)
        try:
            gunning_fog_score = gunning_fog(text)
        except Exception:
            gunning_fog_score = 0.0
        
        # 4. average_word_length (length-independent, from enhanced version)
        total_chars = len(text.replace(' ', ''))
        average_word_length = total_chars / num_words if num_words > 0 else 0.0
        
        # 6. num_words (from old version - added back)
        num_words_count = num_words
        
        # 7. avg_sentence_length (from old version - added back, since sentence_density is removed)
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0.0
        
        # 8. verb_ratio (from enhanced version, replaces function_word_ratio but we keep both)
        try:
            pos_tags = pos_tag(words)
            verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
            verb_ratio = verb_count / num_words if num_words > 0 else 0.0
        except LookupError:
            # If POS tagger is not available, try to download it and retry
            try:
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                pos_tags = pos_tag(words)
                verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
                verb_ratio = verb_count / num_words if num_words > 0 else 0.0
            except Exception:
                # If still fails, set verb_ratio to 0.0
                verb_ratio = 0.0
        
        # 9. function_word_ratio and content_word_ratio (from old version - added back)
        try:
            from nltk.corpus import stopwords
            stops = set(stopwords.words("english"))
            stopword_count = sum(1 for w in words if w.lower() in stops)
            function_word_ratio = stopword_count / num_words if num_words > 0 else 0.0
            content_word_ratio = 1 - function_word_ratio
        except LookupError:
            # If stopwords are not available
            try:
                nltk.download('stopwords', quiet=True)
                stops = set(stopwords.words("english"))
                stopword_count = sum(1 for w in words if w.lower() in stops)
                function_word_ratio = stopword_count / num_words if num_words > 0 else 0.0
                content_word_ratio = 1 - function_word_ratio
            except Exception:
                function_word_ratio = 0.0
                content_word_ratio = 0.0
        
        # ============================================================
        # Return features
        # ============================================================
        
        features = {
            # Emotional Layer (6)
            'polarity': polarity,
            'subjectivity': subjectivity,
            'vader_compound': vader_compound,
            'vader_pos': vader_pos,
            'vader_neu': vader_neu,
            'vader_neg': vader_neg,
            # Stylistic Layer (9) - merged from old and enhanced versions
            # Removed: punctuation_density, smog_index, sentence_density
            # Added back: num_words, avg_sentence_length, function_word_ratio, content_word_ratio
            'word_diversity': word_diversity,
            'flesch_reading_ease': flesch_reading_ease_score,
            'gunning_fog': gunning_fog_score,  # From old version (replaces smog_index)
            'average_word_length': average_word_length,  # From enhanced version (added back alongside num_words)
            'num_words': num_words_count,  # From old version (added back)
            'avg_sentence_length': avg_sentence_length,  # From old version (added back, since sentence_density is removed)
            'verb_ratio': verb_ratio,  # From enhanced version (added back alongside function_word_ratio)
            'function_word_ratio': function_word_ratio,  # From old version (added back)
            'content_word_ratio': content_word_ratio,  # From old version (added back)
        }
        
        return features
        
    except LookupError as e:
        # LookupError usually means missing NLTK data - try to download and retry
        import nltk
        try:
            # Try to download the missing resource
            if 'averaged_perceptron_tagger' in str(e) or 'tagger' in str(e):
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            # Retry the computation
            return compute_nela_features_merged(text)
        except Exception:
            # If retry fails, return zeros
            print(f"[NELA Merged LookupError] {e}")
            return {
                'polarity': 0.0, 'subjectivity': 0.0,
                'vader_compound': 0.0, 'vader_pos': 0.0, 'vader_neu': 0.0, 'vader_neg': 0.0,
                'word_diversity': 0.0, 'flesch_reading_ease': 0.0, 'average_word_length': 0.0,
                'verb_ratio': 0.0,
            }
    except Exception as e:
        # Print other exceptions for debugging
        print(f"[NELA Merged Error] {e}")
        import traceback
        traceback.print_exc()
        return {
            # Emotional Layer
            'polarity': 0.0,
            'subjectivity': 0.0,
            'vader_compound': 0.0,
            'vader_pos': 0.0,
            'vader_neu': 0.0,
            'vader_neg': 0.0,
            # Stylistic Layer
            'word_diversity': 0.0,
            'flesch_reading_ease': 0.0,
            'gunning_fog': 0.0,
            'average_word_length': 0.0,
            'num_words': 0,
            'avg_sentence_length': 0.0,
            'verb_ratio': 0.0,
            'function_word_ratio': 0.0,
            'content_word_ratio': 0.0,
        }


def extract_nela_features_merged(
    dataset_dir,
    label,
    save_path,
    domain: Optional[str] = None,
    model_name: Optional[str] = None,
    level: Optional[str] = None,
):
    """
    Extract merged NELA features from all .txt files in dataset_dir.
    
    Args:
        dataset_dir: Root directory containing .txt files (e.g., "dataset/human" or "dataset/llm")
        label: Label for the dataset ("human" or "llm")
        save_path: Path to save the output CSV
        domain: Optional domain filter (e.g., "academic")
        model_name: Optional model filter for LLM data (e.g., "G4B")
        level: Optional level filter for LLM data (e.g., "LV1")
    
    Returns:
        DataFrame with features and metadata
    """
    records = []
    total_files = 0
    is_llm = (label.lower() == "llm")

    domain_filter = domain.lower() if domain else None
    model_filter = model_name.upper() if model_name else None
    level_filter = level.upper() if level else None

    eligible_files = []
    macro_mode = _is_macro_dataset(dataset_dir)

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.lower().endswith(".txt"):
                continue
            file_path = os.path.join(root, file)
            meta = _extract_metadata(file, file_path, is_llm, macro_mode, "[NELA MERGED]")

            if domain_filter and (not meta or (meta.get("domain") or "").lower() != domain_filter):
                continue
            if is_llm:
                if model_filter and (not meta or (meta.get("model") or "").upper() != model_filter):
                    continue
                if level_filter and (not meta or (meta.get("level") or "").upper() != level_filter):
                    continue

            eligible_files.append((file, file_path, meta))

    total_files = len(eligible_files)
    print(
        f"[NELA Merged] Found {total_files} matching files in {dataset_dir} "
        f"(domain={domain or 'ANY'}, model={model_name or 'ANY'}, level={level or 'ANY'})"
    )

    if total_files == 0:
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Merged NELA features saved to {save_path} (0 files)")
        return df

    for file, file_path, meta in tqdm(eligible_files, desc="Extracting merged NELA features"):
        try:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
                    text = f.read()
            
            if len(text.strip()) == 0:
                continue
            
            # Extract merged NELA features
            feats = compute_nela_features_merged(text)
            
            # Build record with metadata
            record = {
                "filename": file,
                "path": file_path,
                "label": label
            }
            
            # Add parsed metadata if available
            if meta:
                record.update({
                    "domain": meta.get("domain"),
                    "field": meta.get("field"),
                    "author_id": meta.get("author_id"),
                    "year": meta.get("year"),
                    "item_index": meta.get("item_index"),
                    "model": meta.get("model"),
                    "level": meta.get("level")
                })
            else:
                # If parsing fails, still add the record but with None metadata
                record.update({
                    "domain": None,
                    "field": None,
                    "author_id": None,
                    "year": None,
                    "item_index": None,
                    "model": None,
                    "level": None
                })
            
            # Add merged NELA features
            record.update(feats)
            records.append(record)
        except Exception as e:
            print(f"[Error reading {file_path}]: {e}")
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    # Print summary
    processed_files = len(records)
    parsed_files = df["domain"].notna().sum()
    print(f"  Merged NELA features saved to {save_path}")
    print(f"    Processed: {processed_files} / {total_files} files")
    print(f"    Successfully parsed metadata: {parsed_files} files")
    if processed_files > parsed_files:
        print(f"    Warning: {processed_files - parsed_files} files could not be parsed")
    
    return df


def extract_nela_features_merged_by_model_level(dataset_dir, label, save_path, model, level):
    """
    Extract merged NELA features for a specific model and level only.
    
    Args:
        dataset_dir: Root directory containing .txt files
        label: Label for the dataset ("human" or "llm")
        save_path: Path to save the output CSV
        model: Model name (e.g., "LMK", "G12B", "G4B", "DS") - ignored for human
        level: Level name (e.g., "LV1", "LV2", "LV3") - ignored for human
    
    Returns:
        DataFrame with features and metadata
    """
    records = []
    total_files = 0
    is_llm = (label.lower() == "llm")
    
    macro_mode = _is_macro_dataset(dataset_dir)

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.lower().endswith(".txt"):
                continue
            
            # Filter by model and level for LLM files
            if is_llm:
                # Check if file matches the target model and level
                if f"_{model}_{level}.txt" not in file:
                    continue
            
            total_files += 1
            file_path = os.path.join(root, file)
            try:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
                        text = f.read()
                if len(text.strip()) == 0:
                    continue
                
                # Extract merged NELA features
                feats = compute_nela_features_merged(text)
                
                # Parse filename to extract metadata
                meta = _extract_metadata(file, file_path, is_llm, macro_mode, "[NELA MERGED]")
                
                # Build record with metadata
                record = {
                    "filename": file,
                    "path": file_path,
                    "label": label
                }
                
                # Add parsed metadata if available
                if meta:
                    record.update({
                        "domain": meta.get("domain"),
                        "field": meta.get("field"),
                        "author_id": meta.get("author_id"),
                        "year": meta.get("year"),
                        "item_index": meta.get("item_index"),
                        "model": meta.get("model"),
                        "level": meta.get("level")
                    })
                else:
                    # If parsing fails, still add the record but with None metadata
                    record.update({
                        "domain": None,
                        "field": None,
                        "author_id": None,
                        "year": None,
                        "item_index": None,
                        "model": None,
                        "level": None
                    })
                
                # Add merged NELA features
                record.update(feats)
                records.append(record)
            except Exception as e:
                print(f"[Error reading {file_path}]: {e}")
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    # Print summary
    processed_files = len(records)
    parsed_files = df["domain"].notna().sum()
    print(f"  Merged NELA features saved to {save_path}")
    print(f"    Processed: {processed_files} / {total_files} files")
    print(f"    Successfully parsed metadata: {parsed_files} files")
    if processed_files > parsed_files:
        print(f"    Warning: {processed_files - parsed_files} files could not be parsed")
    
    return df

