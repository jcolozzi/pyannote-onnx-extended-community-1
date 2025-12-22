# Pyannote ONNX Extended

A pure **ONNX Runtime** implementation of the Pyannote Speaker Diarization 3.1 (multi-speaker) pipeline.
This project removes the heavy PyTorch dependency for inference, making it lightweight, fast, and easy to deploy.
Based on the [pyannote-audio](https://github.com/pyannote/pyannote-audio) models and inspired by [pyannote-onnx](https://github.com/pengzhendong/pyannote-onnx).

## Key Features

*   **Pure ONNX Runtime**: No PyTorch required for inference.
*   **Robust Overlap Handling**: Implements "Average Stitching" to handle overlapping speech segments smoothly across sliding windows.
*   **Two-Stage Clustering**: Uses a specialized clustering approach where stable "long" segments defined the speakers, and "short" segments are assigned to the nearest speaker. This significantly improves stability for short utterances.
*   **Lightweight**: Minimal dependencies compared to the full PyTorch pipeline.


## Exporting Models (Optional)

If you'd like to export the PyTorch models to ONNX format by yourself, you can do so by running the following command:

```bash
pip install -r requirements.txt
```

> You will need a Hugging Face token with access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).


```bash
python export_onnx.py --use_auth_token YOUR_HF_TOKEN
```

This will create a `models_onnx` folder containing:
*   `segmentation.onnx`
*   `embedding.onnx`

## Installation

```bash
pip install .
```

## Usage

```python
from onnx_pyannote import ONNXSpeakerDiarization

# Initialize the pipeline
pipeline = ONNXSpeakerDiarization(
    model_name="speaker-diarization-3.1",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] # Use CUDA if available
)

# Process an audio file
audio_path = "path/to/your/audio.wav"
annotation = pipeline(audio_path)

# Print result
for turn, _, speaker in annotation.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker={speaker}")
```

## Credits

*   **Pyannote.audio**: Hervé Bredin - The original state-of-the-art speaker diarization framework.
*   **Pyannote ONNX**: Peng Zhendong - Original inspiration for ONNX porting (pyannote/segmentation-3.0).
