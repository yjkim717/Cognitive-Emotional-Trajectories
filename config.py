# config.py
import os

# === LLM SETTINGS ===
LLM_PROVIDER = "deepseek"
LLM_MODEL = "deepseek-chat"      # 가장 저렴하고 빠른 버전
LLM_TEMPERATURE = 0.7
MAX_TOKENS = 1500

# === API SETTINGS ===
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# === PATH SETTINGS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
HUMAN_DIR = os.path.join(DATASET_DIR, "human")
LLM_DIR = os.path.join(DATASET_DIR, "llm")

GENRE_STRUCTURE = {
    "Academic": {
        "path": "Academic",
        "subfields": ["bio", "Chem", "CS", "med", "phy"]
    },
    "Blogs": {
        "path": "Blogs",
        "subfields": ["LIFESTYLE", "SOCIAL", "SPORT", "TECHNOLOGY"]
    },
    "News": {
        "path": "News",
        "subfields": [
            "3_years",
            "4_years",
            "5_years",
            "6_years",
            "7_years",
            "8_years",
            "9_years",
            "10_years",
            "11_years"
        ],
    },
}

def get_llm_path(genre, subfield, year):
    path = os.path.join(LLM_DIR, GENRE_STRUCTURE[genre]["path"], subfield, str(year))
    os.makedirs(path, exist_ok=True)
    return path
