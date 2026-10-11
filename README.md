# SME Reception LLM - IGP Private Dev Repo

Offline voice assistant for SME appointment booking. Caller speaks, system books the appointment. No cloud, no GPU required at inference time  
UWE Bristol MSc Data Science (Iyenoma Goodnews Osabuohien, 26001003)

---

## What it does

A fully offline phone assistant for small businesses. Caller speaks → system books the appointment.

```
mic → Faster-Whisper STT → spaCy NER → fine-tuned LLM → JSON action → Piper TTS → spoken response
```

Handles: booking, cancellations, availability checks, name capture, calendar suggestions, profanity (3-strike), end-of-call detection. Runs on CPU, no cloud.

---

## Eval results

Deployed model: Qwen 2.5 0.5B, fine-tuned with QLoRA and quantised to Q4_K_M, running on CPU via llama.cpp. 100% action accuracy on the 60-record aligned test set, 814ms median latency on a laptop CPU with no GPU and no internet.

Fine-tuning is what makes the task work (GPU evaluation, 480-record synthetic set)
Condition Action Accuracy JSON valid Exact Match
Phi-3 mini (vanilla) 0.4% 12.5% 0.0%
Phi-3 mini (fine-tuned) 98.1% 100% 70.6%
Llama 3.2 3B (vanilla) 0.0% 0% 0.0%
Llama 3.2 3B (fine-tuned) 99.8% 100% 70.4%
Base models produce almost no valid JSON actions; fine-tuning takes both above 98%. Caveat: this GPU evaluation ran on the 480-record synthetic training set, so the fine-tuned rows are training-set accuracy. The held-out check is the 60-record aligned harness below, where the fine-tuned models still reach 98.3-100% action accuracy on data they never saw in training.

Model-size sweep (CPU, Q4_K_M, 60-record aligned test set)
Model Params Action JSON P50 (ms)
Phi-3 mini 3.8B 98.3% 100% 6112
Llama 3.2 3B 3B 100.0% 100% 3398
Qwen 2.5 1.5B 1.5B 98.3% 100% 1702
Llama 3.2 1B 1B 100.0% 100% 1347
Qwen 2.5 0.5B 0.5B 100.0% 100% 814
SmolLM2 360M 360M 93.3% 100% 691
Accuracy holds from 3.8B down to 0.5B and breaks at 360M, so 0.5B is the smallest usable model. Quantisation is nearly free down to 3-bit: Llama and Qwen hold their accuracy at 2-bit, Phi-3 collapses.

Real audio (133 clips, 14 speakers, four first languages, Faster-Whisper tiny, Q4_K_M) - re-evaluated 18 July 2026
Model Strict Scope-aware Mean WER
Phi-3 mini 3.8B 15.8% 48.9% 37.7%
Llama 3.2 3B 24.1% 63.9% 34.6%
Qwen 2.5 1.5B 24.1% 63.2% 37.0%
Llama 3.2 1B 21.8% 57.9% 37.2%
Qwen 2.5 0.5B 25.6% 63.9% 36.1%
SmolLM2 360M 24.8% 56.4% 42.9%
Supersedes the earlier 88-clip/10-speaker pilot below. Same finding, on a larger and more diverse sample (14 speakers, four first languages: English, Farsi, Bengali, Arabic): synthetic accuracy does not survive real speech, and the deployed model still copes best. Every model that scored at or near 100% on written scenarios falls to between 16% and 26% strict accuracy on real speech.

Deployed STT choice matters more than the LibriSpeech benchmark suggests. The LibriSpeech benchmark (tiny 17.3% vs small 8.5% WER) is measured on clean studio recordings; on the 133 real clips above, mean WER by model ranges from 34.6% to 42.9%, a considerably wider real-world gap than the clean benchmark implies.

Quantisation holds up on real speech, latency keeps dropping: Qwen 2.5 0.5B strict/scope-aware accuracy across quant levels on the 133-clip set, deployed Whisper tiny - Q4_K_M 25.6% / 63.9%, Q3_K_M 25.6% / 63.9%, Q2_K 30.1% / 66.9%. Accuracy holds within a few points across all three, with Q2_K scoring slightly higher, well within the noise of a 133-clip sample; transcription latency drops from about 2817ms (Whisper small) to 750ms (tiny) median, which is what keeps the total response near a second.

Earlier 88-clip/10-speaker pilot (Faster-Whisper small, superseded above): breakdown by difficulty for Qwen 2.5 0.5B was clean 25.9%, messy 26.1%, very short utterances 75.0%, non-native speakers 33.3%, the non-native and messy-phrasing cases were where accuracy was weakest. This difficulty breakdown has not yet been re-run on the full 133-clip set.

CPU vs on-premise GPU latency
Same deployed model and test set: 814ms on a laptop CPU, 137ms on an on-premise GPU, both fully offline. The GPU is a speed upgrade a business can add, not a cloud dependency.

Results in evaluation/cpu_results/ and evaluation/real_audio_results/.

---

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# text mode - safest, no audio hardware needed (mock model)
python demo.py --text --no-tts

# the real deployed model on CPU, one command (starts server, runs demo, stops server)
./start_demo.ps1 -Text        # or plain .\start_demo.ps1 to speak

# Streamlit UI
streamlit run app.py          # http://localhost:8501

# API server
uvicorn backend:app --port 5005
# Swagger:             http://localhost:5005/docs
# Metrics dashboard:   http://localhost:5005/metrics-dashboard
```

---

---

# Streamlit UI

The main demo interface. Imports the pipeline directly - no API server needed.

Features: chat bubbles, per-turn action badges (BOOK / CANCEL / AVAIL. / OOS / CLARIFY), latency counter, spaCy entity tags, Piper TTS inline audio playback, model selector, session reset.

Mock mode works on any machine. The fine-tuned models run on CPU via the llama.cpp server (start it first, see CPU inference below), no GPU needed.

# Conversation flow

Greeting: "Thank you for calling. Could I take your name please?"
Name capture: regex strips preambles ("it's / my name is"), spaCy PERSON fallback, spelled names joined (J-A-C-K R-E-A-C-H-E-R -> Jack Reacher)
Name confirmation: "Did I get that as Jack Reacher?" - handles yes / correction / booking intent
Normal turns: utterance -> spaCy NER -> LLM -> validated JSON action
Availability: suggests next slot ("Thursday 25th June at 9:00 AM. Does that work?")
Caller says yes: books slot, confirmation with ordinal date
Caller says no: next slot same day; "later date / another day" jumps to next calendar day
Specific date request ("the 26th"): NER + ordinal regex fallback, explains if nothing available
Cancellations: logged as a message for the reception team, never auto-cancelled (an automated line cannot verify caller identity or ownership of a booking)
Profanity: 3-strike, call ends on strike 3
"bye / that's all / no thanks": end_call detected, session closes
Calendar: Mon-Fri only. Past same-day slots and weekend slots are excluded.

# Testing

python test_pipeline.py # edge cases and conversation flow (mock)
python test_realistic.py # messy real-caller phrasing, invariants (mock)
python test_realistic.py --cpu # action intent against the live model (server running)
test_pipeline.py covers name capture, spelled names, booking intent loop guard, empty/noise/emoji input, profanity 3-strike, out-of-scope, date edge cases, slot exhaustion, ambiguous confirmation, session reuse, cancel-to-human, and end-call detection. test_realistic.py throws messy real-caller phrasing at the pipeline and asserts it never crashes, never goes silent, and never books after a decline; --cpu adds an action-intent check against the deployed Qwen model.

# CPU inference (no GPU)

Runs the fine-tuned models locally via GGUF quantisation and llama.cpp. No GPU, no internet.

# 1. Merge QLoRA adapter into base model (deployed model shown; others: llama3, phi3, ...)

python scripts/01_merge_adapter.py --model qwen0.5b

# 2. Convert to GGUF and quantise

python scripts/02_convert_gguf.py --model qwen0.5b --quant Q4_K_M

# 3. Start the llama.cpp server (port 8080)

python scripts/03_cpu_server.py --model qwen0.5b --quant Q4_K_M

# 4. Run the aligned evaluation (in a second terminal)

python scripts/06_aligned_eval.py --model qwen0.5b --quant Q4_K_M
06_aligned_eval.py runs the same 60-record test set and prompts as the GPU evaluation, so CPU and GPU numbers are directly comparable. 08_real_audio_eval.py scores the pipeline on real recordings.

Requires llama.cpp binaries in tools/llama_cpp/bin/. Download llama-bXXXX-bin-win-cpu-x64.zip from github.com/ggerganov/llama.cpp/releases and extract into tools/llama_cpp/bin/. The tools/ folder is gitignored.

# Provenance

Third-party code and recipes this project builds on, not authored by this team:

llama.cpp (ggerganov/llama.cpp, MIT licence). Not vendored: scripts/02_convert_gguf.py clones it at runtime, and scripts/03_cpu_server.py can instead fetch a prebuilt release binary. Used as an external CPU inference engine, invoked as a subprocess/HTTP server; none of its source is copied into this repo.
Piper TTS (rhasspy/piper, MIT licence). Not vendored: downloaded per the setup steps below. The voice model (en_US-lessac-medium / en_US-ryan-high) is a third-party asset from rhasspy/piper-voices on Hugging Face, not trained by this team.
QLoRA fine-tuning setup (scripts/train_qlora*.py, scripts/kaggle_train*.ipynb): the quantisation config, LoRA target modules, and SFTTrainer usage follow the standard HuggingFace PEFT + TRL QLoRA recipe (huggingface.co/docs/peft), adapted here for this project's data, models, and Kaggle T4 hardware constraints.

---

## File map

```
src/
  inference.py          pipeline: profanity -> name capture -> slot confirm -> LLM
  entity_extractor.py   spaCy NER: DATE, TIME, SERVICE, PERSON
  session_manager.py    in-memory session state per call
  sme_action_schema.py  Pydantic v2 discriminated union, 6 action types
  stt.py                Faster-Whisper microphone capture
  tts.py                Piper TTS subprocess wrapper (dual-flag retry for Windows)
  profanity.py          3-strike keyword filter + de-escalation messages
  calendar_store.py     JSON calendar: find_next_slot(), book_slot(), Mon-Fri filter
  metrics_logger.py     SQLite logger -> data/metrics.db
  call_log.py           consent-gated local call record (transcript + outcome)

scripts/
  01_merge_adapter.py   merge QLoRA adapter into base model
  02_convert_gguf.py    convert merged model to GGUF + quantise
  03_cpu_server.py      start llama.cpp server on port 8080
  06_aligned_eval.py    aligned CPU eval (same 60 records/prompts as GPU)
  08_real_audio_eval.py score the pipeline on real recordings
  gpu_latency_kaggle.py measure GPU latency on the same test set (Kaggle)

data/
  calendar.json         mock 3-week appointment schedule

evaluation/
  cpu_results/          aligned CPU eval JSON results and summary
  real_audio_results/   real-audio eval summary
  figures/              loss curves, latency plots, CPU-vs-GPU chart

checkpoints/            QLoRA adapters + merged models + GGUF files (gitignored)

app.py                  Streamlit UI
backend.py              FastAPI: /turn, /calls, /metrics, /metrics-dashboard
demo.py                 end-to-end demo (text or voice, mock or real model)
start_demo.ps1          one-command CPU demo (starts server, runs demo, stops server)
test_pipeline.py        edge case + flow tests (mock mode)
test_realistic.py       messy-phrasing robustness tests (mock, and --cpu)
```

---

---

# Demo video

## Recorded walkthrough of the system: https://drive.google.com/file/d/1dI0_J8AK8q2tKaEbaQFD2c3m4xcMDuNV/view?usp=drive_link

---

# Local voice demo (talk to your laptop)

Runs the full pipeline end-to-end: mic input, Faster-Whisper STT, LLM, Piper TTS out of your speakers. No phone needed.

# Set up Piper TTS first (see Piper TTS setup section below), then:

python demo.py
The system greets you, listens via your mic, processes through the pipeline, and speaks back. Say "bye" to end the call.

If mic input fails, check that your default recording device is set correctly in Windows sound settings (sounddevice uses the system default).

## For a demo recording: use OBS or Windows Game Bar (Win+G) to capture screen and mic audio simultaneously.

---

# Twilio phone call demo

Lets anyone call a real phone number and talk to the system. Twilio transcribes the caller's speech, sends it to your local server, and speaks back the response. You need the API server running and a public URL via cloudflared.

Step 1: Get a free trial phone number

Go to console.twilio.com, then Phone Numbers > Manage > Buy a number. Pick any UK or US number (trial credit covers it).

Step 2: Expose your local server

Download cloudflared (no account needed for quick tunnels) from github.com/cloudflare/cloudflared/releases, then:

# Terminal 1: start the API

uvicorn backend:app --port 5005

# Terminal 2: open the public tunnel

cloudflared tunnel --url http://localhost:5005

cloudflared prints a URL like https://abc-def-ghi.trycloudflare.com. Copy it.

Step 3: Set the Twilio webhook

In the Twilio console, open your phone number's configuration. Under "Voice and Fax", set "A call comes in" to:

Webhook: https://abc-def-ghi.trycloudflare.com/twilio/voice
HTTP POST
Save the config.

Step 4: Call the number

Call your Twilio number from any phone. The system answers, greets the caller, and handles the full booking conversation. Each call gets its own session (Twilio's CallSid is the session ID, so state persists across turns in the same call).

Twilio free trial can only call back your verified number. Buy credit (~$20) to remove that restriction.

To use the real fine-tuned model: start the llama.cpp server first (see CPU inference section), then point twilio_handler.py at the CPU model instead of mock mode.

---

---

# Piper TTS setup

Create a piper/ folder in the project root and put these two files in it:

Piper binary: download from github.com/rhasspy/piper/releases - pick the Windows or Linux build
Voice model: download en_US-ryan-high.onnx and its .json config from huggingface.co/rhasspy/piper-voices (this is the deployed voice; any Piper voice works, update the path in src/tts.py)

---

## Conversation flow

1. Turn 0: greeting + "Could I take your name please?"
2. Turn 1: name captured (regex strips "it's / I'm / my name is" preamble)
3. Turn 2: name confirmation ("Did I get that as X?") - yes/no/correction handled
4. Normal turns: utterance → spaCy NER → LLM → JSON action
5. `check_availability` → `calendar_store` → "Next slot is X, does that work?"
6. Caller says yes → `book_slot()` → confirmation with name
7. Caller says no → suggest next slot
8. Confusion escalation: 4-step retry with format hints, ends call on 4th failure
9. Profanity: 3-strike, ends call on strike 3
10. "bye / goodbye / thanks" → `end_call` → loop terminates

---

## Docker

```bash
docker build -t sme-backend .
docker run -p 5005:5005 sme-backend
```

No GPU needed. Image ~500MB.

---

## Do NOT re-run

- `pip install` / `spacy download` - already done
- Training - checkpoints already saved
- Evaluation - 480-sample results already in `evaluation/`
- PPTX generation - use `docs/SME_Viva_v3.pptx` (12 slides, current)

---

## Stack

Python 3.11, Faster-Whisper, spaCy en_core_web_sm, Phi-3 mini, Llama 3.2 3B, QLoRA/PEFT, BitsAndBytes 4-bit NF4, FastAPI, Pydantic v2, Piper TTS, SQLite, Docker

---

# Licence

## This project's own code is MIT licensed, see LICENSE. llama.cpp and Piper are separate MIT-licensed third-party tools this project calls at runtime, not part of this codebase, see Provenance above.
