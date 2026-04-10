import json
import os
import tempfile
import time
from typing import Any

import requests
import ollama
import whisper
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# Initialize Ollama Client
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
llm_model = os.getenv("LLM_MODEL", "qwen2.5")

# Initialize Whisper Model (Multilingual V2T)
whisper_model_name = os.getenv("WHISPER_MODEL", "base")
# Loading the model globally to avoid reloading on every request
stt_model = whisper.load_model(whisper_model_name)

def get_model_id() -> str:
    return llm_model

def parse_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()
    
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(raw_text[start : end + 1])

def llm_json(system_prompt: str, user_prompt: str, model_id: str | None = None) -> dict[str, Any]:
    response = ollama.chat(
        model=model_id or get_model_id(),
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        format='json',
        options={'temperature': 0.2}
    )
    return parse_json_object(response['message']['content'])

def transcribe_audio_file(file_path: str, filename: str) -> str:
    # Whisper handles multiple formats (mp3, wav, webm, etc.) and languages (Hindi, Kannada, English) automatically.
    # No need to manually determine MIME types or send bytes; Whisper takes the file path.
    result = stt_model.transcribe(file_path)
    return result.get("text", "").strip()

def extract_symptoms(transcript: str) -> dict[str, Any]:
    system_prompt = """
You are a medical intake extraction assistant.
Extract only what the patient explicitly states or strongly implies.
Return compact JSON with:
- patient_summary: string
- symptoms: array of objects with name, duration, severity, body_part, notes
- risk_flags: array of strings
- missing_details: array of strings
Do not diagnose. Do not add facts not present in the transcript.
"""
    return llm_json(system_prompt, transcript)

def generate_candidate_conditions(symptom_data: dict[str, Any], transcript: str) -> dict[str, Any]:
    system_prompt = """
You create a preliminary differential diagnosis list from symptom descriptions.
Return JSON with:
- candidates: array of objects with name and why_considered
Rules:
- Keep to 3 to 6 common conditions.
- Use generic disease names, not long prose.
- Do not claim certainty.
- This is only a candidate list before evidence validation.
"""
    user_prompt = json.dumps(
        {
            "transcript": transcript,
            "symptom_extraction": symptom_data,
        },
        ensure_ascii=True,
    )
    return llm_json(system_prompt, user_prompt)

def pubmed_evidence(symptoms: list[dict[str, Any]], max_results: int = 5) -> list[dict[str, Any]]:
    symptom_terms = [item.get("name", "").strip() for item in symptoms if item.get("name")]
    if not symptom_terms:
        return []

    query = " AND ".join(f"({term})" for term in symptom_terms[:5]) + " AND (diagnosis OR differential diagnosis)"
    try:
        search_response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
                "term": query,
            },
            timeout=20,
        )
        search_response.raise_for_status()
        id_list = search_response.json().get("esearchresult", {}).get("idlist", [])
    except requests.RequestException:
        return []

    if not id_list:
        return []

    try:
        summary_response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                "db": "pubmed",
                "retmode": "json",
                "id": ",".join(id_list),
            },
            timeout=20,
        )
        summary_response.raise_for_status()
        result = summary_response.json().get("result", {})
    except requests.RequestException:
        return []

    evidence = []
    for pubmed_id in id_list:
        item = result.get(pubmed_id, {})
        if not item:
            continue
        evidence.append(
            {
                "pubmed_id": pubmed_id,
                "title": item.get("title"),
                "source": item.get("fulljournalname") or item.get("source"),
                "pubdate": item.get("pubdate"),
                "authors": item.get("authors", [])[:3],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
            }
        )
    return evidence

def search_condition_sources(condition_name: str, max_results: int = 3) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            "https://clinicaltables.nlm.nih.gov/api/conditions/v3/search",
            params={
                "terms": condition_name,
                "maxList": max_results,
                "df": "consumer_name,primary_name",
                "ef": "icd10cm_codes,info_link_data",
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return []

    codes = payload[1] if len(payload) > 1 else []
    extras = payload[2] if len(payload) > 2 and isinstance(payload[2], dict) else {}
    display_rows = payload[3] if len(payload) > 3 else []

    icd_lists = extras.get("icd10cm_codes", [])
    link_lists = extras.get("info_link_data", [])

    results = []
    for idx, code in enumerate(codes):
        display = display_rows[idx] if idx < len(display_rows) else []
        links = link_lists[idx] if idx < len(link_lists) else []
        icd_codes = icd_lists[idx] if idx < len(icd_lists) else []
        if isinstance(display, list):
            display_name = " | ".join(part for part in display if part)
        else:
            display_name = str(display)

        normalized_links = []
        if isinstance(links, list):
            for link_info in links:
                if isinstance(link_info, list) and len(link_info) >= 2:
                    normalized_links.append({"title": link_info[1], "url": link_info[0]})

        results.append(
            {
                "code": code,
                "display_name": display_name,
                "icd10_codes": icd_codes if isinstance(icd_codes, list) else [icd_codes],
                "links": normalized_links,
            }
        )
    return results

def medlineplus_for_icd(icd_code: str) -> dict[str, Any] | None:
    if not icd_code:
        return None

    try:
        response = requests.get(
            "https://connect.medlineplus.gov/application",
            params={
                "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.90",
                "mainSearchCriteria.v.c": icd_code,
                "mainSearchCriteria.v.dn": icd_code,
                "knowledgeResponseType": "application/json",
                "informationRecipient.languageCode.c": "en",
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return None
    entry = ((data.get("feed") or {}).get("entry") or [])
    if isinstance(entry, dict):
        entry = [entry]
    if not entry:
        return None

    first = entry[0]
    links = first.get("link", [])
    if isinstance(links, dict):
        links = [links]

    return {
        "title": ((first.get("title") or {}).get("_value") if isinstance(first.get("title"), dict) else first.get("title")),
        "summary": ((first.get("summary") or {}).get("_value") if isinstance(first.get("summary"), dict) else first.get("summary")),
        "links": [
            {"href": item.get("@href"), "title": item.get("@title")}
            for item in links
            if isinstance(item, dict) and item.get("@href")
        ],
    }

def build_grounded_predictions(
    transcript: str,
    symptom_data: dict[str, Any],
    candidate_data: dict[str, Any],
    pubmed_items: list[dict[str, Any]],
    validated_conditions: list[dict[str, Any]],
) -> dict[str, Any]:
    system_prompt = """
You are a clinical triage assistant.
Use only the provided patient transcript, extracted symptoms, PubMed evidence metadata, and validated condition records from official NCBI/NLM sources.
Return JSON with:
- predictions: array of objects with name, confidence_label, matching_symptoms, why_it_fits, why_it_may_not_fit, official_sources
- triage_advice: string
- emergency_warning: string
Rules:
- Do not claim a confirmed diagnosis.
- Prefer conditions supported by the validated condition list. If validation data is empty, you may use preliminary candidates only when the PubMed evidence still supports them.
- Keep predictions to at most 3.
- Mention uncertainty plainly.
- Encourage urgent care if severe red-flag symptoms are present.
"""
    user_prompt = json.dumps(
        {
            "transcript": transcript,
            "symptom_data": symptom_data,
            "candidate_data": candidate_data,
            "pubmed_items": pubmed_items,
            "validated_conditions": validated_conditions,
        },
        ensure_ascii=True,
    )
    return llm_json(system_prompt, user_prompt)

def analyze_text(transcript: str) -> dict[str, Any]:
    symptom_data = extract_symptoms(transcript)
    candidate_data = generate_candidate_conditions(symptom_data, transcript)
    pubmed_items = pubmed_evidence(symptom_data.get("symptoms", []))
    source_warnings = []

    if not pubmed_items:
        source_warnings.append("PubMed evidence was unavailable or no matching articles were found.")

    validated_conditions = []
    for candidate in candidate_data.get("candidates", []):
        name = candidate.get("name")
        if not name:
            continue
        matches = search_condition_sources(name)
        if not matches:
            continue
        top_match = matches[0]
        medlineplus = None
        icd_codes = top_match.get("icd10_codes") or []
        if icd_codes:
            medlineplus = medlineplus_for_icd(icd_codes[0])

        validated_conditions.append(
            {
                "candidate_name": name,
                "why_considered": candidate.get("why_considered"),
                "condition_match": top_match,
                "medlineplus": medlineplus,
            }
        )

    if not validated_conditions:
        source_warnings.append("NLM condition validation returned no direct matches, so the response may rely more heavily on the symptom transcript and PubMed evidence.")

    grounded = build_grounded_predictions(
        transcript=transcript,
        symptom_data=symptom_data,
        candidate_data=candidate_data,
        pubmed_items=pubmed_items,
        validated_conditions=validated_conditions,
    )

    return {
        "transcript": transcript,
        "symptoms": symptom_data,
        "candidate_conditions": candidate_data,
        "evidence": {
            "pubmed": pubmed_items,
            "validated_conditions": validated_conditions,
            "source_warnings": source_warnings,
            "official_sources": [
                "NCBI PubMed E-utilities",
                "NLM Clinical Tables Conditions API",
                "MedlinePlus Connect",
            ],
        },
        "analysis": grounded,
        "disclaimer": "This is not a medical diagnosis. Use it as informational triage support and consult a licensed clinician.",
    }

@app.route("/")
def index() -> Any:
    return send_from_directory("static", "index.html")

@app.get("/api/health")
def health() -> Any:
    return jsonify({"ok": True})

@app.post("/api/transcribe")
def transcribe() -> Any:
    audio = request.files.get("audio")
    if not audio or not audio.filename:
        return jsonify({"error": "Missing audio file."}), 400

    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            audio.save(temp_file.name)
            temp_path = temp_file.name

        transcript = transcribe_audio_file(temp_path, audio.filename)
        return jsonify({"transcript": transcript})
    except Exception as exc:
        return jsonify({"error": f"Transcription failed: {exc}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/api/analyze")
def analyze() -> Any:
    payload = request.get_json(silent=True) or {}
    transcript = (payload.get("text") or "").strip()
    if not transcript:
        return jsonify({"error": "Text is required for analysis."}), 400

    try:
        return jsonify(analyze_text(transcript))
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
