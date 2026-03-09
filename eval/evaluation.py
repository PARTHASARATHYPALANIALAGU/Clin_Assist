"""
WER (Word Error Rate) evaluation module for ClinAssist
Computes transcription accuracy using Levenshtein distance
"""
import Levenshtein
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from config import EVALUATION_SAMPLES_DIR
from engine import stt


# Ground truth sample scripts for evaluation
EVALUATION_SAMPLES = [
    {
        "id": 1,
        "reference": "I have a severe headache that started suddenly three hours ago. It's a nine out of ten and it's getting worse.",
        "description": "Critical headache case"
    },
    {
        "id": 2,
        "reference": "I've been experiencing chest pain for the past two days. The pain is about a seven and it seems to be worsening.",
        "description": "High-risk chest pain"
    },
    {
        "id": 3,
        "reference": "I have a mild cough that started gradually about five days ago. It's around a three out of ten and staying stable.",
        "description": "Low-risk cough"
    },
    {
        "id": 4,
        "reference": "My ankle hurts since yesterday when I twisted it. The pain is about a five and improving slightly. I also have some swelling and bruising.",
        "description": "Moderate ankle injury"
    },
    {
        "id": 5,
        "reference": "I've had difficulty breathing since this morning. It came on suddenly and it's an eight out of ten. I also feel dizzy and have chest tightness.",
        "description": "Critical breathing difficulty"
    }
]


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate using Levenshtein distance
    
    WER = (S + D + I) / N
    where:
        S = substitutions
        D = deletions
        I = insertions
        N = number of words in reference
    
    Args:
        reference: Ground truth transcription
        hypothesis: STT system's transcription
    
    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    # Normalize: lowercase and split into words
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Calculate Levenshtein distance at word level
    distance = Levenshtein.distance(ref_words, hyp_words)
    
    # Number of words in reference
    num_words = len(ref_words)
    
    if num_words == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # WER = edit distance / reference length
    wer = distance / num_words
    
    return wer


def run_evaluation():
    """
    Run WER evaluation on all sample audio files
    Generates evaluation report with individual and average WER
    """
    print("=" * 60)
    print("ClinAssist WER Evaluation")
    print("=" * 60)
    print()
    
    results: List[Tuple[int, float, str, str]] = []
    
    for sample in EVALUATION_SAMPLES:
        sample_id = sample["id"]
        reference = sample["reference"]
        description = sample["description"]
        
        # Check if audio file exists
        audio_path = EVALUATION_SAMPLES_DIR / f"sample_{sample_id}.wav"
        
        if not audio_path.exists():
            print(f"Sample {sample_id}: SKIPPED - {audio_path.name} not found")
            print(f"  Expected: {reference}")
            print()
            continue
        
        # Load audio file
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # Transcribe using STT module (creates temporary session for logging)
        hypothesis = stt.transcribe_audio(audio_bytes, f"eval_{sample_id}")
        
        # Compute WER
        wer = compute_wer(reference, hypothesis)
        
        # Store result
        results.append((sample_id, wer, reference, hypothesis))
        
        # Print result
        print(f"Sample {sample_id}: {description}")
        print(f"  Reference:  {reference}")
        print(f"  Hypothesis: {hypothesis}")
        print(f"  WER: {wer:.4f} ({wer * 100:.2f}%)")
        print()
    
    # Calculate average WER
    if results:
        average_wer = sum(r[1] for r in results) / len(results)
        print("=" * 60)
        print(f"Average WER: {average_wer:.4f} ({average_wer * 100:.2f}%)")
        print("=" * 60)
        
        # Save report to file
        report_path = Path(__file__).parent / "evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("ClinAssist WER Evaluation Report\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            
            for sample_id, wer, reference, hypothesis in results:
                f.write(f"Sample {sample_id}:\n")
                f.write(f"  Reference:  {reference}\n")
                f.write(f"  Hypothesis: {hypothesis}\n")
                f.write(f"  WER: {wer:.4f} ({wer * 100:.2f}%)\n\n")
            
            f.write("=" * 60 + "\n")
            f.write(f"Average WER: {average_wer:.4f} ({average_wer * 100:.2f}%)\n")
            f.write("=" * 60 + "\n")
        
        print(f"\nReport saved to: {report_path}")
    else:
        print("No audio samples found for evaluation.")
        print(f"Place WAV files in: {EVALUATION_SAMPLES_DIR}")


if __name__ == "__main__":
    run_evaluation()
