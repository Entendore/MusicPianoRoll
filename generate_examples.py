#!/usr/bin/env python3
"""
🎵 Piano Roll Player — Example File Generator

Run this script to generate a set of test MIDI and WAV files.
It creates a 'PianoRoll_Examples/' folder with various musical
examples, including an auto-paired MIDI+Audio set to test
the batch processing and pairing features of the player.

Requirements: pip install mido
"""

import os
import math
import struct
import wave
import random

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: 'mido' library is required.")
    print("Install it with: pip install mido")
    exit(1)


OUTPUT_DIR = "PianoRoll_Examples"
SAMPLE_RATE = 44100

# MIDI note numbers (Fixed definitions)
C3 = 48; D3 = 50; E3 = 52; F3 = 53; G3 = 55; A3 = 57; B3 = 59
C4 = 60; D4 = 62; E4 = 64; F4 = 65; G4 = 67; A4 = 69; B4 = 71
C5 = 72; C6 = 84

def ensure_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"📁 Output directory: ./{OUTPUT_DIR}/")

# ═══════════════════════════════════════════════════════════════
#  MIDI GENERATORS
# ═══════════════════════════════════════════════════════════════

def create_scale_midi():
    """Generates a simple C Major scale."""
    path = os.path.join(OUTPUT_DIR, "C_Major_Scale.mid")
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=500000, time=0))
    track.append(MetaMessage('track_name', name='C Major Scale', time=0))

    notes = [C4, D4, E4, F4, G4, A4, B4, C5, C5, B4, A4, G4, F4, E4, D4, C4]
    duration = 480  # ticks per note (quarter note at 120 BPM)

    for note in notes:
        track.append(Message('note_on', note=note, velocity=100, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=duration))

    mid.save(path)
    print(f"🎹 Created: {path}")
    return notes

def create_arpeggio_midi():
    """Generates a fast C Major arpeggio across 3 octaves."""
    path = os.path.join(OUTPUT_DIR, "Fast_Arpeggio.mid")
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=250000, time=0))  # Fast tempo
    track.append(MetaMessage('track_name', name='Fast Arpeggio', time=0))

    # C major arpeggio up and down
    notes = [C3, E3, G3, C4, E4, G4, C5, E4, G4, C4, E3, G3] * 4
    duration = 120  # 16th notes

    for note in notes:
        track.append(Message('note_on', note=note, velocity=random.randint(80, 127), time=0))
        track.append(Message('note_off', note=note, velocity=64, time=duration))

    mid.save(path)
    print(f"🎹 Created: {path}")

def create_chords_midi():
    """Generates a chord progression (C - Am - F - G) with a bassline."""
    path = os.path.join(OUTPUT_DIR, "Chord_Progression.mid")
    mid = MidiFile()
    
    # Track 1: Chords
    track1 = MidiTrack()
    mid.tracks.append(track1)
    track1.append(MetaMessage('set_tempo', tempo=500000, time=0))
    track1.append(MetaMessage('track_name', name='Chords', time=0))

    # Track 2: Bass
    track2 = MidiTrack()
    mid.tracks.append(track2)
    track2.append(MetaMessage('track_name', name='Bass', time=0))

    chords = [
        [C4, E4, G4],  # C
        [C4, E4, A4],  # Am
        [F3, A3, C4],  # F (using F3 to space out the voicing)
        [G3, B3, D4],  # G
    ]
    basses = [C3, A3, F3, G3]

    whole_note = 1920

    for i in range(4): # 4 measures
        for chord in chords:
            for note in chord:
                track1.append(Message('note_on', note=note, velocity=90, time=0))
            for note in chord:
                track1.append(Message('note_off', note=note, velocity=64, time=whole_note))
        
        bass = basses[i % 4]
        track2.append(Message('note_on', note=bass, velocity=120, time=0))
        track2.append(Message('note_off', note=bass, velocity=64, time=whole_note))

    mid.save(path)
    print(f"🎹 Created: {path}")

def create_chaos_midi():
    """Generates random fast notes to test particle systems and rendering."""
    path = os.path.join(OUTPUT_DIR, "Visual_Stress_Test.mid")
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=200000, time=0))
    track.append(MetaMessage('track_name', name='Chaos', time=0))

    for _ in range(200):
        note = random.randint(30, 100)
        vel = random.randint(40, 127)
        dur = random.choice([60, 120, 180])
        track.append(Message('note_on', note=note, velocity=vel, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=dur))

    mid.save(path)
    print(f"🎹 Created: {path}")

# ═══════════════════════════════════════════════════════════════
#  WAV (AUDIO) GENERATORS
# ═══════════════════════════════════════════════════════════════

def create_sine_wav(filename, frequencies_and_durations):
    """Helper to generate a 16-bit mono WAV file from a list of (freq, duration_secs)."""
    path = os.path.join(OUTPUT_DIR, filename)
    
    audio_data = []
    volume = 0.5
    
    for freq, dur in frequencies_and_durations:
        num_samples = int(SAMPLE_RATE * dur)
        for i in range(num_samples):
            t = float(i) / SAMPLE_RATE
            # Simple sine wave with an envelope to avoid clicking
            envelope = min(1.0, min(i, num_samples - i) / (SAMPLE_RATE * 0.01))
            value = volume * envelope * math.sin(2.0 * math.pi * freq * t)
            audio_data.append(value)

    with wave.open(path, 'w') as wav_file:
        n_channels = 1
        sampwidth = 2
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(SAMPLE_RATE)
        
        for value in audio_data:
            data = struct.pack('<h', int(value * 32767))
            wav_file.writeframesraw(data)
            
    print(f"🎵 Created: {path}")

def create_paired_audio():
    """Generates a WAV file that matches the C Major Scale MIDI to test auto-pairing."""
    # C4 to C5 and back (frequencies in Hz)
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,
             523.25, 493.88, 440.00, 392.00, 349.23, 329.63, 293.66, 261.63]
    note_duration = 0.5 # seconds
    
    create_sine_wav(
        "C_Major_Scale.wav", 
        [(f, note_duration) for f in freqs]
    )

def create_unpaired_audio():
    """Generates a standalone WAV file without a MIDI counterpart."""
    # A simple ambient drone chord (C3 + G3 + C4)
    dur = 8.0
    create_sine_wav(
        "Unpaired_Ambient_Drone.wav", 
        [(130.81, dur), (196.00, dur), (261.63, dur)]
    )


# ═══════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("🎵 Piano Roll Player - Example File Generator")
    print("=" * 50)
    
    ensure_dir()
    print("\nGenerating MIDI files...")
    create_scale_midi()
    create_arpeggio_midi()
    create_chords_midi()
    create_chaos_midi()
    
    print("\nGenerating WAV files...")
    create_paired_audio()
    create_unpaired_audio()
    
    print("\n" + "=" * 50)
    print("✅ Generation Complete!")
    print("=" * 50)
    print(f"\nTo test the player, open the '{OUTPUT_DIR}' folder:")
    print(f"  1. Click '📂 Scan Folder' in the playlist panel.")
    print(f"  2. Select the '{OUTPUT_DIR}' folder.")
    print(f"  3. Say 'Yes' to recursive scan.")
    print(f"  4. Say 'Yes' to Auto-Pair (This will link C_Major_Scale.mid + C_Major_Scale.wav).")
    print(f"\nTry the 'Visual_Stress_Test.mid' with the 🌈 Psychedelic theme to see particle effects!")