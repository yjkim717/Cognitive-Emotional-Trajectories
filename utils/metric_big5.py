import os
from typing import Optional

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

from utils.file_utils import (
    parse_macro_llm_metadata_from_path,
    parse_macro_metadata_from_path,
)
from utils.parse_dataset_filename import parse_filename

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def _is_macro_dataset(dataset_dir: str) -> bool:
    parts = os.path.abspath(dataset_dir).split(os.sep)
    return "macro_dataset" in parts


def _extract_metadata(file_name: str, file_path: str, is_llm: bool, macro_mode: bool):
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
            print(f"[Big Five] Warning: {exc}")
    return parse_filename(file_name, is_llm=is_llm)


def load_big5_model():
    # Use pre-trained Big Five personality prediction model
    model_name = "Minej/bert-base-personality"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict_big5(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        # Get probabilities using softmax for classification
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    return dict(zip(TRAITS, probs))


def _filter_matches(meta: Optional[dict], domain: Optional[str], model_name: Optional[str], level: Optional[str], is_llm: bool) -> bool:
    if domain and (not meta or (meta.get("domain") or "").lower() != domain.lower()):
        return False
    if is_llm:
        if model_name and (not meta or (meta.get("model") or "").upper() != model_name.upper()):
            return False
        if level and (not meta or (meta.get("level") or "").upper() != level.upper()):
            return False
    return True


def extract_big5_features(
    dataset_dir,
    label,
    save_path,
    domain: Optional[str] = None,
    model_name: Optional[str] = None,
    level: Optional[str] = None,
):
    """
    Extract Big Five features with optional filtering by domain/model/level.
    """
    tokenizer, big5_model = load_big5_model()
    records = []
    is_llm = label.lower() == "llm"

    macro_mode = _is_macro_dataset(dataset_dir)
    eligible_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue
            file_path = os.path.join(root, file)
            meta = _extract_metadata(file, file_path, is_llm, macro_mode)
            if not _filter_matches(meta, domain, model_name, level, is_llm):
                continue
            eligible_files.append((file, file_path, meta))

    total_files = len(eligible_files)
    print(
        f"[Big Five] Found {total_files} matching files in {dataset_dir} "
        f"(domain={domain or 'ANY'}, model={model_name or 'ANY'}, level={level or 'ANY'})"
    )
    if total_files == 0:
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Big Five features saved to {save_path} (0 files)")
        return df

    for file, file_path, meta in tqdm(
        eligible_files,
        desc="[Big Five] Processing",
        total=total_files,
        unit="file",
        leave=False,
    ):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            scores = predict_big5(text, tokenizer, big5_model)
            record = {
                "filename": file,
                "path": file_path,
                "label": label,
                "domain": meta.get("domain") if meta else None,
                "field": meta.get("field") if meta else None,
                "author_id": meta.get("author_id") if meta else None,
                "year": meta.get("year") if meta else None,
                "item_index": meta.get("item_index") if meta else None,
                "model": meta.get("model") if meta else None,
                "level": meta.get("level") if meta else None,
            }
            record.update(scores)
            records.append(record)
        except Exception as e:
            print(f"[Big Five][Error] {file_path}: {e}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Big Five features saved to {save_path} ({len(df)} files)")
    return df
