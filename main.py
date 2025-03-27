import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def create_streamer():
    '''
    This is what works so far:
    format: jack
        gx_head_amp:out_0
        ardour:Guitar/audio_out 1
        ardour:General MIDI Synth/audio_out 1
        Midi-Bridge:Oxygen 61 MKV M-Audio Oxygen 61 (playback)

    format: pulse
        alsa_input.usb-Focusrite_Scarlett_2i2_USB-00.HiFi__Mic1__source

    :return:
    '''

    streamer = torchaudio.io.StreamReader("zynaddsubfx:out_1", format="jack", option={"channels":"1"})
    print(streamer.get_src_stream_info(0))
    # prints: SourceAudioStream(media_type='audio', codec='pcm_f32le', codec_long_name='PCM 32-bit floating point little-endian', format='flt', bit_rate=1536000, num_frames=0, bits_per_sample=0, metadata={}, sample_rate=48000.0, num_channels=1)
    streamer.add_basic_audio_stream(-1)
    # streamer.add_basic_audio_stream(48000, sample_rate=48000, num_channels=1) # pactl list sources | grep "Sample Specification"
    # streamer.add_audio_stream(8000)
        # format="s32p" - If None, the output chunk has dtype corresponding to the precision of the source audio.
        # depending on sample rate value (i.e 16k or 48k). The size of chunk.size[0] will be different
        # print(streamObj.get_src_stream_info(0)) -> SourceAudioStream(media_type='audio', codec='pcm_f32le', codec_long_name='PCM 32-bit floating point little-endian', format='flt', bit_rate=3072000, num_frames=0, bits_per_sample=0, metadata={}, sample_rate=48000.0, num_channels=2)

    return streamer

def frequency_to_note(frequency):
    """
    Convert a frequency in Hz to a musical note name

    Args:
        frequency: Frequency in Hz

    Returns:
        String containing note name, octave, and cents deviation if applicable
    """
    if frequency <= 0:
        return "Silent"

    # A4 = 440Hz is our reference
    A4 = 440.0
    C0 = A4 * 2 ** (-4.75)  # C0 reference frequency

    # Calculate how many half steps away from C0
    half_steps = 12 * np.log2(frequency / C0)

    # Round to the nearest half step
    half_steps_rounded = round(half_steps)

    # Calculate the octave
    octave = int(half_steps_rounded // 12)

    # Calculate the note index (0=C, 1=C#, etc.)
    note_idx = int(half_steps_rounded % 12)

    # Note names
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Get the note name
    note_name = note_names[note_idx] + str(octave)

    # Calculate cents deviation
    cents = 100 * (half_steps - half_steps_rounded)
    if abs(cents) > 5:  # Only show if more than 5 cents off
        note_name += f" ({cents:+.0f}¢)"

    return note_name

def get_frequency_autocorrelation(waveform, sample_rate, min_freq=50, max_freq=2000):
    """
    Detect pitch using autocorrelation - simple and effective for single tones

    Args:
        waveform: Audio waveform (1D tensor)
        sample_rate: Sample rate in Hz
        min_freq: Minimum frequency to detect
        max_freq: Maximum frequency to detect

    Returns:
        Estimated fundamental frequency in Hz, or 0 if none detected
    """
    # Convert to numpy array
    wave_np = waveform.numpy()

    # Check if signal is too quiet
    if np.abs(wave_np).mean() < 0.01:
        return 0.0

    # Normalize waveform
    wave_np = wave_np / (np.abs(wave_np).max() + 1e-10)

    # Calculate autocorrelation
    corr = np.correlate(wave_np, wave_np, mode='full')
    corr = corr[len(corr) // 2:]  # Only use positive lags

    # Convert frequency range to lag samples
    min_lag = int(sample_rate / max_freq) if max_freq > 0 else 1
    max_lag = int(sample_rate / min_freq) if min_freq > 0 else len(corr) - 1

    # Ensure we stay within array bounds
    max_lag = min(max_lag, len(corr) - 1)

    if max_lag <= min_lag:
        return 0.0

    # Skip the first few lags to avoid the main peak
    start_lag = min(min_lag, 10)

    # Normalize correlation
    corr_norm = corr[start_lag:max_lag] / (corr[0] + 1e-10)

    # Find local maxima (peaks)
    peaks = []
    for i in range(1, len(corr_norm) - 1):
        if corr_norm[i] > corr_norm[i - 1] and corr_norm[i] > corr_norm[i + 1]:
            peaks.append((i + start_lag, corr_norm[i]))

    # If no peaks found, return 0
    if not peaks:
        return 0.0

    # Sort peaks by correlation value (highest first)
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Get lag of the highest peak
    best_lag = peaks[0][0]

    # Convert lag to frequency
    if best_lag > 0:
        frequency = sample_rate / best_lag
        return frequency
    else:
        return 0.0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    torch.set_printoptions(threshold=float('inf'))

    streamObj = create_streamer()
    sample_rate = streamObj.get_src_stream_info(0).sample_rate
    print(f"Sample rate: {sample_rate} Hz")
    freq_buffer = []
    buffer_size = 3

    while streamObj.stream():
        chunk = next(streamObj.stream())
        chunkTensor = chunk[0]
        waveform = chunkTensor[:,0]
        # print(waveform)

        freq = get_frequency_autocorrelation(waveform, sample_rate)

        # Add frequency to buffer for smoothing
        if freq > 0:
            freq_buffer.append(freq)
            if len(freq_buffer) > buffer_size:
                freq_buffer.pop(0)

            # Use median for smoothing
            if len(freq_buffer) > 1:
                smoothed_freq = np.median(freq_buffer)
                note_name = frequency_to_note(smoothed_freq)
                print(f"Frequency: {smoothed_freq:.1f} Hz | Note: {note_name}")
            else:
                note_name = frequency_to_note(freq)
                print(f"Frequency: {freq:.1f} Hz | Note: {note_name}")
