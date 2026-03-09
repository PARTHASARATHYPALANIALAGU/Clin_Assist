===============================================================================
CLINASSIST WER EVALUATION SAMPLES
===============================================================================

This directory contains sample scripts for Word Error Rate (WER) evaluation.

INSTRUCTIONS:
To run the evaluation, you must provide 5 audio recordings in WAV format that
correspond to the scripts below. Record yourself (or use text-to-speech) reading
each script naturally.

REQUIRED AUDIO FORMAT:
- Format: WAV
- Sample Rate: 16,000 Hz
- Bit Depth: 16-bit PCM
- Channels: Mono

FILE NAMING:
Place your recordings in this directory with these exact names:
  sample_1.wav
  sample_2.wav
  sample_3.wav
  sample_4.wav
  sample_5.wav

===============================================================================
SAMPLE SCRIPTS (Ground Truth Transcriptions)
===============================================================================

--- Sample 1: Critical Headache Case ---
FILE: sample_1.wav
SCRIPT: "I have a severe headache that started suddenly three hours ago. It's a nine out of ten and it's getting worse."

Expected Risk Level: CRITICAL (severity >= 9)


--- Sample 2: High-Risk Chest Pain ---
FILE: sample_2.wav
SCRIPT: "I've been experiencing chest pain for the past two days. The pain is about a seven and it seems to be worsening."

Expected Risk Level: CRITICAL (contains "chest pain" keyword) or HIGH (severity >= 7)


--- Sample 3: Low-Risk Cough ---
FILE: sample_3.wav
SCRIPT: "I have a mild cough that started gradually about five days ago. It's around a three out of ten and staying stable."

Expected Risk Level: LOW (low severity, stable progression)


--- Sample 4: Moderate Ankle Injury ---
FILE: sample_4.wav
SCRIPT: "My ankle hurts since yesterday when I twisted it. The pain is about a five and improving slightly. I also have some swelling and bruising."

Expected Risk Level: MODERATE (severity >= 4, plus associated symptoms)


--- Sample 5: Critical Breathing Difficulty ---
FILE: sample_5.wav
SCRIPT: "I've had difficulty breathing since this morning. It came on suddenly and it's an eight out of ten. I also feel dizzy and have chest tightness."

Expected Risk Level: CRITICAL (contains "difficulty breathing" keyword)

===============================================================================
HOW TO CREATE AUDIO FILES
===============================================================================

OPTION 1: Record Yourself
- Use any audio recording software (Audacity, Voice Recorder, etc.)
- Read each script naturally at normal speaking pace
- Export as WAV with the specifications above
- Use ffmpeg to convert if needed:
  ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 sample_1.wav

OPTION 2: Use Text-to-Speech
- Use online TTS tools (Google TTS, Amazon Polly, etc.)
- Generate speech from each script
- Download and convert to required WAV format

OPTION 3: Sample Generation Script (Python)
- Use gTTS or pyttsx3 library to generate samples programmatically

===============================================================================
RUNNING THE EVALUATION
===============================================================================

Once all 5 WAV files are in place:

1. Run: python evaluation.py
2. Review console output for individual WER scores
3. Check evaluation_report.txt for detailed results
4. Average WER should be < 0.10 (10%) for good accuracy

===============================================================================
EXPECTED OUTPUT
===============================================================================

The evaluation will compute:
- Word Error Rate for each sample
- Average WER across all samples
- Comparison of reference vs hypothesis transcripts
- Detailed report saved to evaluation_report.txt

Good WER benchmarks:
- Excellent: < 5% (0.05)
- Good: 5-10% (0.05-0.10)
- Acceptable: 10-20% (0.10-0.20)
- Poor: > 20% (0.20+)

===============================================================================
