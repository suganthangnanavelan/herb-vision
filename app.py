# ============================================================
#  HerbVision – Flask Backend
#  Swin-B backbone · metadata.json TCM database
# ============================================================

import json, base64, io, time, unicodedata
from pathlib import Path

import numpy as np
import requests as req_lib
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

import torch
import torch.nn as nn
import timm
from torchvision import transforms

# ── Device ───────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[HerbVision] Using device: {DEVICE}")

# ── HuggingFace repo ─────────────────────────────────────────
HF_BASE_URL  = "https://huggingface.co/SuganthanGnanavelan/herbvision/resolve/main"
HF_CACHE_DIR = Path("hf_cache")          # cached next to app.py after first run

# ── Other paths ───────────────────────────────────────────────
METADATA_PATH = Path("metadata.json")
INDEX_HTML    = Path("index.html")

CONFIDENCE_THRESHOLD = 0.30


# ─────────────────────────────────────────────────────────────
#  Plain-requests downloader with local cache
# ─────────────────────────────────────────────────────────────
import ssl, urllib.request

# Headers that mimic a real browser — prevents CDN resets on some hosts
_DL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

def _stream_to_file(url: str, dest: Path, verify_ssl: bool = True) -> None:
    """Download *url* → *dest* with a progress bar."""
    session = req_lib.Session()
    session.headers.update(_DL_HEADERS)
    if not verify_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    with session.get(url, stream=True, timeout=180, verify=verify_ssl) as r:
        r.raise_for_status()
        total      = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(
                        f"\r  {pct:5.1f}%  "
                        f"({downloaded // 1_048_576} / {total // 1_048_576} MB)",
                        end="", flush=True,
                    )
    print()


def hf_download(filename: str) -> Path:
    """
    Download *filename* from the HuggingFace repo using plain requests.
    Files are cached in HF_CACHE_DIR so they are only fetched once.

    Strategy:
      1. Try with SSL verification enabled.
      2. If the SSL handshake is reset (common on Python 3.14 / Windows with
         HuggingFace's LFS CDN), retry without SSL verification.
      3. Both attempts retry up to MAX_RETRIES times with a short back-off.
    """
    MAX_RETRIES = 4
    HF_CACHE_DIR.mkdir(exist_ok=True)
    dest = HF_CACHE_DIR / filename

    if dest.exists():
        print(f"[HerbVision] Cache hit — {filename}")
        return dest

    url = f"{HF_BASE_URL}/{filename}"
    print(f"[HerbVision] Downloading {filename} …")

    tmp = dest.with_suffix(dest.suffix + ".tmp")

    for verify in (True, False):           # first try with SSL, then without
        label = "with SSL" if verify else "without SSL verification"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  Attempt {attempt}/{MAX_RETRIES} ({label}) …")
                _stream_to_file(url, tmp, verify_ssl=verify)
                tmp.rename(dest)
                print(f"[HerbVision] Saved → {dest}")
                return dest
            except Exception as exc:
                print(f"  [WARN] Attempt {attempt} failed: {exc}")
                if tmp.exists():
                    tmp.unlink()
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)   # 2 s, 4 s, 8 s …

    raise RuntimeError(
        f"Failed to download {filename} after all retries. "
        "Check your internet connection or download the file manually from "
        f"https://huggingface.co/SuganthanGnanavelan/herbvision and place it in hf_cache/"
    )


# ─────────────────────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────────────────────
class SwinClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224.ms_in22k",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.15,
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────
#  Herb database — multi-key lookup from metadata.json
# ─────────────────────────────────────────────────────────────
def build_herb_database(json_path: Path):
    """
    Returns (records_list, lookup_dict).

    class_names.json contains numeric IDs (e.g. "1", "42", "300").
    metadata.json has matching "id" fields ("001", "042", "300").

    Lookup is keyed by:
      • bare integer string  → "1", "42", "300"
      • zero-padded 3-digit  → "001", "042", "300"
    so both formats resolve correctly.
    """
    if not json_path.exists():
        print(f"[WARN] metadata.json not found at: {json_path.resolve()}")
        return [], {}

    with open(json_path, encoding="utf-8") as f:
        records: list = json.load(f)

    lookup: dict = {}

    for r in records:
        herb = {
            "zh":    r.get("zh",    "").strip(),
            "py":    r.get("py",    "").strip(),
            "sci":   r.get("sci",   "").strip(),
            "com":   r.get("com",   "").strip(),
            "fam":   r.get("fam",   "").strip(),
            "nat":   r.get("nat",   "Neutral").strip(),
            "taste": r.get("taste", "Various").strip(),
            "mer":   r.get("mer",   "Various").strip(),
            "parts": r.get("parts", "Various").strip(),
            "props": r.get("props", []),
            "desc":  r.get("desc",  "").strip(),
            "uses":  r.get("uses",  "").strip(),
        }

        raw_id = str(r.get("id", "")).strip()
        if not raw_id:
            continue

        # Register under both zero-padded ("001") and bare int ("1")
        lookup[raw_id] = herb                          # as stored, e.g. "001"
        if raw_id.isdigit():
            lookup[str(int(raw_id))] = herb            # bare int,  e.g. "1"

    print(f"[HerbVision] Herb DB loaded: {len(records)} species, {len(lookup)} ID keys.")
    return records, lookup


def parse_class_name(class_name: str):
    """
    class_names.json entries look like  "003.三七"  or  "42.SomeName".
    Returns (id_str, display_name):
      id_str       — numeric part only, e.g. "3"  (used to find the herb)
      display_name — everything after the first dot, e.g. "三七"
    If there is no dot, id_str == class_name and display_name == class_name.
    """
    s = class_name.strip()
    if "." in s:
        num_part, name_part = s.split(".", 1)
        return num_part.strip(), name_part.strip()
    return s, s


def get_herb(class_name: str) -> dict:
    """
    Parse 'NNN.HerbName' → look up herb by numeric ID in metadata.json.
    Always returns a full herb dict plus 'display_name' for the frontend.
    """
    id_str, display_name = parse_class_name(class_name)

    # Try exact key, bare int, zero-padded-3
    herb = None
    if id_str.isdigit():
        bare   = str(int(id_str))    # strip leading zeros: "003" → "3"
        padded = bare.zfill(3)       # "3" → "003"
        for key in (id_str, bare, padded):
            herb = HERB_LOOKUP.get(key)
            if herb:
                break
    else:
        herb = HERB_LOOKUP.get(id_str)

    if herb:
        return {**herb, "display_name": display_name}

    print(f"[WARN] No metadata match for class_name={class_name!r} (id={id_str!r})")
    return {
        "zh":           "",
        "py":           "",
        "sci":          "",
        "com":          display_name,   # at least show the label from class file
        "fam":          "—",
        "nat":          "—",
        "taste":        "—",
        "mer":          "—",
        "parts":        "—",
        "props":        [],
        "desc":         "No metadata available for this species.",
        "uses":         "No metadata available for this species.",
        "display_name": display_name,
    }


# ─────────────────────────────────────────────────────────────
#  Load artefacts (download from HuggingFace, cache locally)
# ─────────────────────────────────────────────────────────────
print("[HerbVision] Fetching class_names.json …")
class_names_path = hf_download("class_names.json")
with open(class_names_path, encoding="utf-8") as f:
    CLASS_NAMES: list = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)
print(f"[HerbVision] {NUM_CLASSES} classes. First 5: {CLASS_NAMES[:5]}")

_RECORDS, HERB_LOOKUP = build_herb_database(METADATA_PATH)

print("[HerbVision] Fetching model checkpoint …")
ckpt_path = hf_download("swin_best.pth")
model = SwinClassifier(num_classes=NUM_CLASSES, dropout=0.25).to(DEVICE)
ckpt  = torch.load(
    ckpt_path,
    map_location=DEVICE,
    weights_only=False,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"[HerbVision] Model ready — epoch {ckpt['epoch']}, "
      f"val acc {ckpt['val_acc']*100:.2f}%.")

# Preprocessing — identical to training val pipeline
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


def decode_image(img_b64: str) -> Image.Image:
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")


@app.route("/")
def index():
    return send_file(str(INDEX_HTML))


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    try:
        payload = request.get_json(force=True)
        img_b64 = payload.get("image", "")
        if not img_b64:
            return jsonify({"error": "No image provided"}), 400

        img    = decode_image(img_b64)
        tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(tensor)
            else:
                logits = model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top5_idx = probs.argsort()[::-1][:5]
        predictions = [
            {
                "herb":       get_herb(CLASS_NAMES[idx]),
                "confidence": float(probs[idx]),
            }
            for idx in top5_idx
        ]

        top_conf = predictions[0]["confidence"]
        unknown  = top_conf < CONFIDENCE_THRESHOLD
        elapsed  = round((time.time() - t0) * 1000)

        top_label = predictions[0]["herb"]["com"] or predictions[0]["herb"]["zh"]
        print(f"[predict] {top_label}  conf={top_conf:.3f}  {elapsed}ms")

        return jsonify({
            "unknown":     unknown,
            "predictions": predictions,
            "elapsed_ms":  elapsed,
        })

    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/debug/classes", methods=["GET"])
def debug_classes():
    """Shows first 20 class names + their resolved herb — useful for diagnosing mismatches."""
    sample = CLASS_NAMES[:20]
    resolved = [{"class": cn, "herb": get_herb(cn)["zh"] or get_herb(cn)["com"]} for cn in sample]
    return jsonify({"total_classes": NUM_CLASSES, "sample": resolved})


@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "num_classes": NUM_CLASSES,
        "device":      str(DEVICE),
        "herb_keys":   len(HERB_LOOKUP),
    })


if __name__ == "__main__":
    print("[HerbVision] Starting on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)