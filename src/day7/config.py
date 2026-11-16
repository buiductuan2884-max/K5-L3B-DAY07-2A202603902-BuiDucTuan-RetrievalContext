import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DAY6_DIR = Path(os.environ.get("DAY6_DIR") or ROOT.parent / "K5-L3-Day06-DataIngestionEngineering" / "starter").resolve()
load_dotenv(DAY6_DIR / ".env")  # key stays in the Day 6 dir, never copied here

CHROMA_DIR = DAY6_DIR / "chroma_db"
COLLECTION = "strategy_B"
DAY6_QUESTIONS = DAY6_DIR.parent / "benchmark" / "questions.json"
DATA = ROOT / "data" / "day7"
CACHE = DATA / "cache"
CHUNKS_PATH = DATA / "chunks.json"
QUESTIONS_PATH = DATA / "questions.json"
NEW_QUESTIONS_PATH = DATA / "new_questions.json"
RESULTS = ROOT / "results"

ADV_PDF = Path(os.environ.get("ADV_PDF") or DAY6_DIR / "data" / "Economics2e.pdf")
ADV_SOURCE_ID = "openstax-econ2e-full"
ADV = ROOT / "advanced"
ADV_DATA = ADV / "data"
ADV_CHROMA_DIR = ADV / "chroma_db"
ADV_COLLECTION = "econ2e_full"
ADV_QUESTIONS_PATH = ADV_DATA / "questions.json"
VISION_MODEL = "gpt-4.1-mini"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
PRICES = {  # USD per 1M tokens
    EMBED_MODEL: {"input": 0.02},
    CHAT_MODEL: {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def openai_client():
    from openai import OpenAI
    return OpenAI()
