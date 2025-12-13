#!/usr/bin/env python3
"""
Parse unified dataset filename format to extract metadata.

Human format: <Domain>_<Field>_<AuthorID>_<Year>_<ItemIndex>.txt
LLM format:  <Domain>_<Field>_<AuthorID>_<Year>_<ItemIndex>_<Model>_<Level>.txt

Examples:
  - Academic_BIOLOGY_01_2020_01.txt
  - Academic_BIOLOGY_01_2020_01_G4B_LV1.txt
  - News_5years_02_2017_03.txt
  - News_5years_02_2017_03_DS_LV1.txt
"""
import re
from typing import Optional, Dict


def parse_human_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse human filename: <Domain>_<Field>_<AuthorID>_<Year>_<ItemIndex>.txt
    
    Returns dict with keys: domain, field, author_id, year, item_index
    """
    name = filename
    if name.endswith(".txt"):
        name = name[:-4]
    
    # Pattern: Domain_Field_AuthorID_Year_ItemIndex
    # Field can contain numbers (e.g., "5years", "11years")
    pattern = r'^([A-Za-z]+)_([A-Za-z0-9_]+)_(\d+)_(\d{4})_(\d+)\.txt?$'
    match = re.match(pattern, name + '.txt')
    
    if match:
        return {
            "domain": match.group(1).lower(),
            "field": match.group(2),
            "author_id": match.group(3),
            "year": match.group(4),
            "item_index": match.group(5),
            "model": None,
            "level": None
        }
    return None


def parse_llm_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse LLM filename: <Domain>_<Field>_<AuthorID>_<Year>_<ItemIndex>_<Model>_<Level>.txt
    
    Returns dict with keys: domain, field, author_id, year, item_index, model, level
    """
    name = filename
    if name.endswith(".txt"):
        name = name[:-4]
    
    # Pattern: Domain_Field_AuthorID_Year_ItemIndex_Model_Level
    # Field can contain numbers (e.g., "5years", "11years")
    pattern = r'^([A-Za-z]+)_([A-Za-z0-9_]+)_(\d+)_(\d{4})_(\d+)_([A-Z0-9]+)_(LV[123])\.txt?$'
    match = re.match(pattern, name + '.txt')
    
    if match:
        return {
            "domain": match.group(1).lower(),
            "field": match.group(2),
            "author_id": match.group(3),
            "year": match.group(4),
            "item_index": match.group(5),
            "model": match.group(6),
            "level": match.group(7)
        }
    return None


def parse_filename(filename: str, is_llm: bool = False) -> Optional[Dict[str, str]]:
    """
    Universal parser that detects format and parses accordingly.
    
    Args:
        filename: The filename to parse
        is_llm: If True, expects LLM format; if False, expects Human format.
                If None, auto-detects based on filename pattern.
    
    Returns:
        Dict with metadata or None if parsing fails
    """
    if is_llm:
        return parse_llm_filename(filename)
    else:
        result = parse_human_filename(filename)
        # If human parse fails, try LLM format (for backward compatibility)
        if result is None:
            result = parse_llm_filename(filename)
        return result
