# Voice Symptom Checker

A simple Flask web app that:

- records live audio in the browser or uploads a saved audio file
- transcribes the audio with an LLM-compatible transcription API
- extracts symptoms with an LLM
- generates a disease shortlist grounded in official medical sources

## Sources used for medical grounding

The backend validates and supports the disease shortlist with:

- NCBI PubMed E-utilities
- NLM Clinical Tables Conditions API
- MedlinePlus Connect

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Open `http://127.0.0.1:5000`.

## Required environment variables

- `OPENAI_API_KEY`: required

## Optional environment variables

- `OPENAI_BASE_URL`: for OpenAI-compatible providers
- `LLM_MODEL`: defaults to `gpt-4.1-mini`
- `TRANSCRIPTION_MODEL`: defaults to `gpt-4o-mini-transcribe`
- `PORT`: defaults to `5000`

## API endpoints

- `POST /api/transcribe`
  - multipart form field: `audio`
- `POST /api/analyze`
  - JSON body: `{ "text": "..." }`
- `GET /api/health`

## Notes

- This is not a medical diagnosis tool.
- Browser live recording uses the MediaRecorder API and usually sends `webm` audio.
- The backend does not use a database.
