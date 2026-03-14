# 🎧 Audio Intelligence

### Speech → Transcript → AI Analysis

Audio Intelligence is a lightweight AI application that converts spoken audio into text and performs intelligent analysis on the transcript.
Upload or record audio, generate a transcript with timestamps, and use AI to summarize, extract insights, or analyze sentiment.

The app combines **speech recognition** and **language models** to create an interactive audio-analysis tool.

---

## ✨ Features

* 🎙 **Speech-to-Text Transcription**
  Converts audio into text using Whisper.

* ⏱ **Timestamped Transcript**
  Automatically generates timestamps for segments in the audio.

* 🤖 **AI Analysis Modes**

  * Bullet-point summaries
  * Question & Answer generation
  * Sentiment analysis
  * Action item extraction

* 📊 **Processing Statistics**

  * Word count
  * Processing time
  * Timestamp of analysis

* 💾 **Export Results**
  Download transcripts and analysis as a `.txt` file.

* 🎨 **Custom UI**
  Clean dark-themed interface built with Gradio + custom CSS.

---

## 🧠 Models Used

Speech Recognition

* Whisper Tiny

Language Model

* FLAN‑T5 Base

Interface Framework

* Gradio

---

## ⚙️ How It Works

```
Audio Input
     ↓
Whisper Speech Recognition
     ↓
Transcript + Timestamps
     ↓
FLAN-T5 AI Processing
     ↓
Analysis Output
```

The system performs three stages:

1️⃣ Speech recognition converts audio into text
2️⃣ The transcript is processed into structured segments
3️⃣ The LLM analyzes the transcript depending on the selected mode

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/audio-intelligence.git
cd audio-intelligence
```

Create a virtual environment:

```bash
python -m venv ai_env
ai_env\Scripts\activate
```

Install dependencies:

```bash
pip install transformers torch gradio
```

---

## ▶ Running the App

```bash
python app.py
```

The interface will start at:

```
http://127.0.0.1:7860
```

---

## 🎛 Analysis Modes

| Mode           | Description                                    |
| -------------- | ---------------------------------------------- |
| Bullet Summary | Generates key bullet points from transcript    |
| Q&A            | Creates questions and answers based on content |
| Sentiment      | Detects emotional tone of the speech           |
| Action Items   | Extracts tasks mentioned in the audio          |

---

## 📤 Export Feature

The export button creates a `.txt` file containing:

* Timestamped transcript
* AI analysis output
* Generation timestamp

---

## 📁 Project Structure

```
audio-intelligence/
│
├── app.py
├── README.md
└── requirements.txt
```

---

## 🚀 Future Improvements

* Real-time microphone streaming transcription
* Larger language models for improved analysis
* Speaker diarization (detect different speakers)
* Export to PDF / Markdown
* Cloud deployment

---

## 📜 License

Apache 2.0

---

## 🙌 Acknowledgements

* Hugging Face Transformers
* OpenAI Whisper
* Google FLAN-T5
* Gradio
