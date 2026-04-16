import os
import torch
import torchaudio
import subprocess
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation, Timeline
from pydub import AudioSegment
import numpy as np
import pandas as pd
import soundfile as sf

class AudioPipeline:
    """
    Advanced audio processing pipeline for vocal separation and speaker diarization.
    """
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def separate_music(self, file_path, output_dir="temp_output"):
        """
        Uses Demucs to separate vocals from music/noise.
        Returns path to vocals file and music file.
        Memory-efficient: Processes audio in chunks and streams output to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Separating {file_path} using Demucs (Streamed) on {self.device}")

        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import soundfile as sf
        import gc

        filename = os.path.splitext(os.path.basename(file_path))[0]
        result_dir = os.path.join(output_dir, "htdemucs", filename)
        os.makedirs(result_dir, exist_ok=True)

        vocals_path = os.path.join(result_dir, "vocals.wav")
        music_path = os.path.join(result_dir, "no_vocals.wav")

        # Load Pre-trained Model
        model = get_model('htdemucs')
        model.to(self.device)
        sr = model.samplerate

        # Calculate Chunk Size (2 minutes per chunk to manage RAM)
        chunk_seconds = 120
        chunk_samples = chunk_seconds * sr

        # Get total duration/frames from header
        info = sf.info(file_path)
        total_frames = info.frames
        original_sr = info.samplerate

        print(f"Total Frames: {total_frames}, Model Sample Rate: {sr}")

        # Iterate in chunks to prevent memory overflow
        reader = sf.SoundFile(file_path)

        # Create output writers (Stereo, 16-bit PCM)
        voc_writer = sf.SoundFile(vocals_path, mode='w', samplerate=sr, channels=2, subtype='PCM_16')
        mus_writer = sf.SoundFile(music_path, mode='w', samplerate=sr, channels=2, subtype='PCM_16')

        current_frame = 0

        try:
            while current_frame < total_frames:
                print(f"Processing chunk: {current_frame} / {total_frames} ({(current_frame/total_frames)*100:.1f}%)")

                read_frames = int(chunk_seconds * original_sr)
                audio_chunk = reader.read(frames=read_frames)

                if len(audio_chunk) == 0:
                    break

                # Transpose to (Channels, Frames)
                if len(audio_chunk.shape) == 1:
                    audio_chunk = audio_chunk.reshape(1, -1)
                else:
                    audio_chunk = audio_chunk.T

                # Convert to PyTorch tensor
                wav = torch.from_numpy(audio_chunk).float()

                # Resample if if source rate differs from model rate
                if original_sr != sr:
                     wav = torchaudio.functional.resample(wav, original_sr, sr)

                # Apply model with error handling for GPU out-of-memory
                try:
                    torch.cuda.empty_cache()
                    sources = apply_model(model, wav[None], device=self.device, shifts=0, split=True, overlap=0.25)[0]
                except RuntimeError as e:
                     if "out of memory" in str(e):
                         print("GPU OOM on chunk. Falling back to CPU.")
                         torch.cuda.empty_cache()
                         model.cpu()
                         sources = apply_model(model, wav[None], device="cpu", shifts=0, split=True, overlap=0.25)[0]
                         model.to(self.device) 
                     else:
                         raise e

                # Extract vocals and combine other sources into "music"
                if 'vocals' in model.sources:
                    vocals_idx = model.sources.index('vocals')
                    vocals = sources[vocals_idx]
                    other_indices = [i for i in range(len(sources)) if i != vocals_idx]
                    music = torch.sum(sources[other_indices], dim=0)
                else:
                    vocals = sources[0]
                    music = sources[1]

                # Write back to disk
                voc_np = vocals.cpu().numpy().T
                mus_np = music.cpu().numpy().T

                voc_writer.write(voc_np)
                mus_writer.write(mus_np)

                current_frame += audio_chunk.shape[1]

                # Manual garbage collection
                del sources, vocals, music, wav
                gc.collect()

        finally:
            reader.close()
            voc_writer.close()
            mus_writer.close()

        print("Vocal Separation Complete.")
        return vocals_path, music_path

    def diarize(self, vocal_track_path):
        """
        Uses pyannote.audio to identify speakers in the vocal track.
        """
        print("Loading Pyannote Diarization Pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            token=self.hf_token
        )
        pipeline.to(self.device)

        print(f"Diarizing {vocal_track_path}...")

        # Manually load via soundfile to avoid torchcodec issues on Windows
        data, samplerate = sf.read(vocal_track_path)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        else:
            data = data.T
        waveform = torch.from_numpy(data).float()

        input_data = {"waveform": waveform, "sample_rate": samplerate}

        # Run inference (Allowing up to 10 speakers to catch background artifacts)
        print("Running Speaker Diarization (min=2, max=10)...")
        diarization = pipeline(input_data, min_speakers=2, max_speakers=10)
        return diarization

    def process_segments(self, diarization):
        """
        Processes diarization results into clean segments (Speaker labels and Overlaps).
        """
        if hasattr(diarization, "speaker_diarization"):
             diarization = diarization.speaker_diarization
        elif hasattr(diarization, "annotation"):
             diarization = diarization.annotation

        # Identify Overlaps
        if hasattr(diarization, "get_overlap"):
            overlap_timeline = diarization.get_overlap()
        else:
             print("WARNING: get_overlap not found. Overlaps will not be explicitly labeled.")
             overlap_timeline = Timeline()

        labels = diarization.labels()
        speaker_timelines = {}
        for label in labels:
            speaker_timelines[label] = diarization.label_timeline(label)

        final_segments = []

        # Add Overlap segments first
        for segment in overlap_timeline:
            final_segments.append({
                'start': segment.start,
                'end': segment.end,
                'label': 'Overlap'
            })

        # Add Pure Speaker segments (Timeline MINUS OverlapTimeline)
        for label, timeline in speaker_timelines.items():
            if hasattr(timeline, "extrude"):
                 pure_timeline = timeline.extrude(overlap_timeline)
            else:
                 print(f"WARNING: Timeline extrude missing for {label}. Using full timeline.")
                 pure_timeline = timeline

            for segment in pure_timeline:
                final_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'label': label 
                })
                
        # Sort by start time for consistent playback
        final_segments.sort(key=lambda x: x['start'])

        # --- SMOOTHING / GAP FILLING ---
        smoothed_segments = []
        if not final_segments: return []

        current_seg = final_segments[0]
        MAX_GAP = 1.0 # Max gap in seconds to bridge for the same speaker

        for next_seg in final_segments[1:]:
            if (next_seg['label'] == current_seg['label'] and 
                (next_seg['start'] - current_seg['end']) <= MAX_GAP):

                # Merge segments
                current_seg['end'] = max(current_seg['end'], next_seg['end'])
            else:
                smoothed_segments.append(current_seg)
                current_seg = next_seg

        smoothed_segments.append(current_seg)
        print(f"Smoothing reduced segments from {len(final_segments)} to {len(smoothed_segments)}")

        return smoothed_segments

    def export_audio(self, segments, original_audio_path, output_dir="output", trim_start_sec=0.0):
        """
        Splits the audio into separate speaker tracks based on diarization segments.
        """
        print(f"Exporting to {output_dir} (Trim: {trim_start_sec}s)")
        os.makedirs(output_dir, exist_ok=True)

        try:
            original = AudioSegment.from_file(original_audio_path)
            trim_ms = int(trim_start_sec * 1000)
            duration_ms = max(0, len(original) - trim_ms)

            # Initialize silent base tracks
            track_A = AudioSegment.silent(duration=duration_ms)
            track_B = AudioSegment.silent(duration=duration_ms)
            track_Overlap = AudioSegment.silent(duration=duration_ms)

            # Identify top 2 speakers by duration
            speaker_durations = {}
            for s in segments:
                lbl = s['label']
                if lbl == 'Overlap': continue
                dur = s['end'] - s['start']
                speaker_durations[lbl] = speaker_durations.get(lbl, 0) + dur

            top_speakers = sorted(speaker_durations.keys(), key=lambda l: speaker_durations[l], reverse=True)
            label_map = {}
            if len(top_speakers) > 0: label_map[top_speakers[0]] = 'A'
            if len(top_speakers) > 1: label_map[top_speakers[1]] = 'B'

            print(f"Assigning Speakers: {label_map}")

            for seg in segments:
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                label = seg['label']

                if end_ms <= trim_ms: continue

                read_start = max(start_ms, trim_ms)
                read_end = end_ms
                write_pos = read_start - trim_ms
                chunk = original[read_start:read_end]

                if label == 'Overlap':
                    track_Overlap = track_Overlap.overlay(chunk, position=write_pos)
                elif label in label_map:
                    mapped = label_map[label]
                    if mapped == 'A':
                        track_A = track_A.overlay(chunk, position=write_pos)
                    elif mapped == 'B':
                        track_B = track_B.overlay(chunk, position=write_pos)

            # Save files
            track_A.export(os.path.join(output_dir, "track_speaker_A.wav"), format="wav")
            track_B.export(os.path.join(output_dir, "track_speaker_B.wav"), format="wav")
            track_Overlap.export(os.path.join(output_dir, "track_overlap.wav"), format="wav")

            print("Export complete.")
            return label_map

        except Exception as e:
            print(f"Export Error: {e}")
            raise e
