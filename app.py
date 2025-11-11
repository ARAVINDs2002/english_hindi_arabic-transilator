from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize.punkt import PunktSentenceTokenizer
import torch
import nltk
import os

# Download NLTK tokenizer data
nltk.download("punkt")

app = Flask(__name__)

# Load model from local path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

if not os.path.isdir(MODEL_DIR):
    raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Supported languages
LANG_CODE = {
    "en": "eng_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva"
}

# Use English tokenizer for all input
english_tokenizer = PunktSentenceTokenizer()

def translate_sentence(sentence, src_lang_code, tgt_lang_code):
    tokenizer.src_lang = src_lang_code
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code),
            max_length=512
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_paragraph(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text

    src_code = LANG_CODE.get(src_lang, "eng_Latn")
    tgt_code = LANG_CODE.get(tgt_lang)
    if not tgt_code:
        return "⚠️ Unsupported target language."

    translated = []
    for para in text.split("\n"):
        if not para.strip():
            translated.append("")
        else:
            sentences = english_tokenizer.tokenize(para)
            translated_sentences = [
                translate_sentence(s.strip(), src_code, tgt_code)
                for s in sentences if s.strip()
            ]
            translated.append(" ".join(translated_sentences))
    return "\n".join(translated)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    src_lang = "en"
    tgt_lang = "ar"
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "")
        src_lang = request.form.get("src_lang", "en")
        tgt_lang = request.form.get("tgt_lang", "ar")
        result = translate_paragraph(text, src_lang, tgt_lang)

    return render_template("index.html", result=result, text=text, src_lang=src_lang, tgt_lang=tgt_lang)

@app.route("/api/translate", methods=["POST"])
def api_translate():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        src_lang = data.get("src_lang", "").strip()
        tgt_lang = data.get("tgt_lang", "").strip()

        if not text or not tgt_lang or not src_lang:
            return jsonify({
                "status": False,
                "status_code": 400,
                "message": "Fields 'text', 'src_lang', and 'tgt_lang' are required.",
                "translated_text": None
            }), 400

        translated = translate_paragraph(text, src_lang, tgt_lang)

        return jsonify({
            "status": True,
            "status_code": 200,
            "message": "Translation successful.",
            "translated_text": translated
        }), 200

    except Exception as e:
        return jsonify({
            "status": False,
            "status_code": 500,
            "message": f"Internal server error: {str(e)}",
            "translated_text": None
        }), 500

@app.route("/api/translate_batch", methods=["POST"])
def api_translate_batch():
    try:
        data = request.get_json()
        items = data.get("text", [])
        src_lang_default = data.get("src_lang", "").strip()
        tgt_lang_default = data.get("tgt_lang", "").strip()

        if not isinstance(items, list) or not items:
            return jsonify({
                "status": False,
                "status_code": 400,
                "message": "messages must be a non-empty list.",
                "translated_texts": ""
            }), 400

        if not tgt_lang_default or not src_lang_default:
            return jsonify({
                "status": False,
                "status_code": 400,
                "message": "source and target languages are required.",
                "translated_texts": ""
            }), 400

        results = []
        for entry in items:
            if isinstance(entry, str):
                text = entry.strip()
                src_lang = src_lang_default
                tgt_lang = tgt_lang_default
            elif isinstance(entry, dict):
                text = entry.get("text", "").strip()
                src_lang = entry.get("src_lang", src_lang_default)
                tgt_lang = entry.get("tgt_lang", tgt_lang_default)
            else:
                results.append(None)
                continue

            if not text:
                results.append(None)
                continue

            translated = translate_paragraph(text, src_lang, tgt_lang)
            results.append({
                "original_text": text,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "translated_text": translated
            })

        return jsonify({
            "status": True,
            "status_code": 200,
            "message": "Batch translation successful.",
            "translated_texts": results
        }), 200

    except Exception as e:
        return jsonify({
            "status": False,
            "status_code": 500,
            "message": f"Internal server error: {str(e)}",
            "translated_texts": ""
        }), 500

if __name__ == "__main__":
    app.run(debug=True)