from transformers import pipeline
import gradio as gr
import time
import re
from datetime import datetime
import os

speech_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=15,
    return_timestamps=True,
)

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
)


def seconds_to_timestamp(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"[{m:02d}:{sec:02d}]"


def build_timestamped_transcript(chunks: list) -> str:
    lines = []
    for chunk in chunks:
        ts = chunk.get("timestamp", (None, None))
        start = seconds_to_timestamp(ts[0]) if ts and ts[0] is not None else ""
        text = chunk.get("text", "").strip()
        lines.append(f"{start} {text}" if start else text)
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def llm_prompt(mode: str, text: str) -> str:
    prompts = {
        "Bullet Summary": (
            f"Summarize the following transcript into clear bullet-point key points.\n\n"
            f"Transcript:\n{text}\n\nKey Points:\n-"
        ),
        "Q&A": (
            f"Generate 3 insightful questions and answers based on the transcript below.\n\n"
            f"Transcript:\n{text}\n\nQ&A:\nQ1:"
        ),
        "Sentiment": (
            f"Analyse the overall sentiment (Positive / Negative / Neutral / Mixed) "
            f"of the transcript and explain briefly why.\n\n"
            f"Transcript:\n{text}\n\nSentiment Analysis:"
        ),
        "Action Items": (
            f"Extract a list of concrete action items or tasks mentioned in the transcript.\n\n"
            f"Transcript:\n{text}\n\nAction Items:\n-"
        ),
    }
    return prompts.get(mode, prompts["Bullet Summary"])


def process_audio(audio_file, analysis_mode, max_summary_tokens):
    if audio_file is None:
        return (
            "⚠️ No audio file provided. Please upload or record audio.",
            "", "", ""
        )

    start = time.time()

    try:
        result = speech_pipe(audio_file, batch_size=8)
        raw_text = result["text"].strip()
        chunks = result.get("chunks", [])
    except Exception as e:
        return f"❌ Transcription error: {e}", "", "", ""

    if not raw_text:
        return "⚠️ No speech detected in the audio.", "", "", ""

    timestamped = build_timestamped_transcript(chunks) if chunks else raw_text

    try:
        prompt = llm_prompt(analysis_mode, raw_text)
        prefix = "- " if analysis_mode in ("Bullet Summary", "Action Items") else ""
        llm_out = llm(prompt, max_length=int(max_summary_tokens))[0]["generated_text"]
        analysis = prefix + llm_out
    except Exception as e:
        analysis = f"❌ Analysis error: {e}"

    elapsed = round(time.time() - start, 1)
    wc = word_count(raw_text)
    stats = (
        f"✅ Done in {elapsed}s  |  "
        f"Words: {wc}  |  "
        f"Mode: {analysis_mode}  |  "
        f"Model: whisper-tiny.en + flan-t5-base  |  "
        f"Processed: {datetime.now().strftime('%H:%M:%S')}"
    )

    return timestamped, raw_text, analysis, stats


def export_results(transcript, analysis, mode):
    if not transcript and not analysis:
        return None
    filename = f"/tmp/transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    content = (
        f"AUDIO TRANSCRIPTION EXPORT\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*50}\n\n"
        f"TRANSCRIPT\n{'-'*30}\n{transcript}\n\n"
        f"{mode.upper()}\n{'-'*30}\n{analysis}\n"
    )
    with open(filename, "w") as f:
        f.write(content)
    return filename


css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --card:      #16161f;
    --border:    #252535;
    --accent:    #e8ff47;
    --accent2:   #47d4ff;
    --text:      #e8e8f0;
    --muted:     #5a5a72;
    --success:   #47ffb0;
    --radius:    14px;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 2rem 1.5rem !important;
}

/* ── Header ── */
#header-wrap {
    text-align: center;
    padding: 3rem 0 2.5rem;
    position: relative;
}
#header-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(232,255,71,.08) 0%, transparent 70%);
    pointer-events: none;
}
#app-title {
    font-family: var(--font-head) !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: var(--text) !important;
    margin: 0 !important;
    line-height: 1 !important;
}
#app-title span { color: var(--accent); }
#app-subtitle {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin-top: 0.75rem !important;
}

/* ── Panel cards ── */
.panel-card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    height: 100%;
}

/* ── Labels ── */
label span, .label-wrap span {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
}

/* ── Audio widget ── */
.audio-wrap, [data-testid="audio"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Radio buttons → pill toggle ── */
.radio-group fieldset { border: none !important; padding: 0 !important; }
.radio-group .wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
}
.radio-group label {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    padding: 0.35rem 1rem !important;
    cursor: pointer !important;
    transition: all .2s !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}
.radio-group label:has(input:checked) {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #0a0a0f !important;
    font-weight: 600 !important;
}
.radio-group input[type=radio] { display: none !important; }

/* ── Slider ── */
input[type=range] { accent-color: var(--accent) !important; }
.slider-container .output-number {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
}

/* ── Process button ── */
#run-btn {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.85rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: transform .15s, box-shadow .15s !important;
    box-shadow: 0 0 0 0 rgba(232,255,71,0) !important;
}
#run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 30px rgba(232,255,71,.25) !important;
}
#run-btn:active { transform: translateY(0) !important; }

/* ── Export button ── */
#export-btn {
    background: transparent !important;
    color: var(--accent2) !important;
    border: 1px solid var(--accent2) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 1.2rem !important;
    cursor: pointer !important;
    transition: background .2s !important;
    width: 100% !important;
    margin-top: 0.75rem !important;
}
#export-btn:hover { background: rgba(71,212,255,.08) !important; }

/* ── Stats bar ── */
#stats-box textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--success) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.73rem !important;
    padding: 0.6rem 0.9rem !important;
    resize: none !important;
}

/* ── Tabs ── */
.tabs { border: none !important; }
.tab-nav {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    margin-bottom: 1rem !important;
}
.tab-nav button {
    border-radius: 7px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.45rem 1rem !important;
    transition: all .2s !important;
}
.tab-nav button.selected {
    background: var(--card) !important;
    color: var(--text) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.4) !important;
}

/* ── Output textareas ── */
.output-box textarea, #timestamped-out textarea,
#raw-out textarea, #analysis-out textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.83rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
    resize: vertical !important;
}

/* ── File download ── */
.file-preview {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Divider ── */
.divider {
    height: 1px;
    background: var(--border);
    margin: 1.25rem 0;
}
"""

with gr.Blocks(
    css=css,
    title="Audio Intelligence",
    theme=gr.themes.Base(
        primary_hue="yellow",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("DM Mono"),
    ),
) as demo:

    with gr.Column(elem_id="header-wrap"):
        gr.Markdown(
            "<h1 id='app-title'>Audio <span>Intelligence</span></h1>",
        )
        gr.Markdown(
            "<p id='app-subtitle'>Speech → Transcript → AI Analysis</p>",
        )

    with gr.Row(equal_height=True):

        with gr.Column(scale=1, min_width=300, elem_classes=["panel-card"]):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio Input",
                elem_classes=["audio-wrap"],
            )
            gr.HTML("<div class='divider'></div>")
            analysis_mode = gr.Radio(
                choices=["Bullet Summary", "Q&A", "Sentiment", "Action Items"],
                value="Bullet Summary",
                label="Analysis Mode",
                elem_classes=["radio-group"],
            )
            gr.HTML("<div class='divider'></div>")
            max_tokens = gr.Slider(
                minimum=64, maximum=512, value=256, step=32,
                label="Max Output Tokens",
            )
            gr.HTML("<div class='divider'></div>")
            run_btn = gr.Button("▶  Process Audio", variant="primary", elem_id="run-btn")
            stats_box = gr.Textbox(
                label="",
                interactive=False,
                placeholder="Stats will appear here after processing...",
                lines=2,
                elem_id="stats-box",
            )

        with gr.Column(scale=2, elem_classes=["panel-card"]):
            with gr.Tabs():
                with gr.Tab("📝 Timestamped"):
                    timestamped_out = gr.Textbox(
                        label="Transcript with Timestamps",
                        lines=16,
                        interactive=False,
                        elem_id="timestamped-out",
                        elem_classes=["output-box"],
                        placeholder="Your timestamped transcript will appear here...",
                    )
                with gr.Tab("📄 Raw Text"):
                    raw_out = gr.Textbox(
                        label="Raw Transcript",
                        lines=16,
                        interactive=False,
                        elem_id="raw-out",
                        elem_classes=["output-box"],
                        placeholder="Plain transcript text will appear here...",
                    )
                with gr.Tab("🤖 AI Analysis"):
                    analysis_out = gr.Textbox(
                        label="Analysis Output",
                        lines=16,
                        interactive=False,
                        elem_id="analysis-out",
                        elem_classes=["output-box"],
                        placeholder="AI analysis results will appear here...",
                    )

            export_btn = gr.Button("⬇  Export to .txt", elem_id="export-btn")
            export_file = gr.File(label="Download", visible=True)

    run_btn.click(
        fn=process_audio,
        inputs=[audio_input, analysis_mode, max_tokens],
        outputs=[timestamped_out, raw_out, analysis_out, stats_box],
    )

    export_btn.click(
        fn=export_results,
        inputs=[timestamped_out, analysis_out, analysis_mode],
        outputs=[export_file],
    )

port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)