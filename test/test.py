# Full Local Pipeline: WhisperX + SpeechBrain Diarization

import whisperx
import torch
import numpy as np
import librosa
from speechbrain.pretrained import EncoderClassifier, VAD
from sklearn.cluster import AgglomerativeClustering


def run_whisperx_transcription(audio_path, language="de", model_size="large", device="cuda"):
    model = whisperx.load_model(model_size, device)
    result = model.transcribe(audio_path, language=language, batch_size=16, vad_filter=True)
    
    # Word-level alignment
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
    return result_aligned  # result_aligned["word_segments"]


def run_speechbrain_diarization(audio_path, device="cuda", num_speakers=2):
    classifier = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    vad = VAD.from_hparams("speechbrain/vad-crdnn-libriparty", run_opts={"device": device})

    signal, sr = librosa.load(audio_path, sr=16000)
    frames = []
    timestamps = []
    window_size = 1.5
    hop = 0.75
    frame_len = int(sr * window_size)
    hop_len = int(sr * hop)

    for i in range(0, len(signal) - frame_len, hop_len):
        frame = signal[i:i+frame_len]
        if np.mean(frame**2) > 1e-4:
            emb = classifier.encode_batch(torch.tensor(frame).unsqueeze(0).to(device))
            frames.append(emb.squeeze().cpu().numpy())
            timestamps.append(i / sr)

    X = np.vstack(frames)
    cluster = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
    labels = cluster.fit_predict(X)

    diarization = []
    for i, (start, label) in enumerate(zip(timestamps, labels)):
        end = start + window_size
        diarization.append({"start": start, "end": end, "speaker": f"SPEAKER_{label:02d}"})
    return diarization


def merge_transcript_and_diarization(word_segments, diarization):
    merged = []
    for word in word_segments:
        word_start = word['start']
        word_end = word['end']
        text = word['word'].strip()
        speaker = "SPEAKER_UNKNOWN"

        for seg in diarization:
            if seg['start'] <= word_start <= seg['end']:
                speaker = seg['speaker']
                break

        merged.append({
            "start": word_start,
            "end": word_end,
            "speaker": speaker,
            "text": text
        })
    return merged


def print_diarized_transcript(merged):
    lines = []
    current_speaker = None
    buffer = []
    start_time, end_time = None, None

    for word in merged:
        if word['speaker'] != current_speaker:
            if buffer:
                line = f"[{start_time:.2f} - {end_time:.2f}] {current_speaker}: {' '.join(buffer)}"
                lines.append(line)
            current_speaker = word['speaker']
            buffer = [word['text']]
            start_time = word['start']
        else:
            buffer.append(word['text'])
        end_time = word['end']

    if buffer:
        lines.append(f"[{start_time:.2f} - {end_time:.2f}] {current_speaker}: {' '.join(buffer)}")

    print("\n".join(lines))


if __name__ == "__main__":
    audio_file = "20241121_Wassermann-FGR-2420.mp3"  # replace with your file path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔁 Running WhisperX transcription...")
    aligned = run_whisperx_transcription(audio_file, device=device)

    print("🔁 Running SpeechBrain diarization...")
    diarization = run_speechbrain_diarization(audio_file, device=device)

    print("🔁 Merging and formatting transcript...")
    merged = merge_transcript_and_diarization(aligned["word_segments"], diarization)
    print_diarized_transcript(merged)
