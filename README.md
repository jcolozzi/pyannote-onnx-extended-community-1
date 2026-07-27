# Pyannote ONNX Extended

A pure ONNX Runtime implementation of the pyannote speaker diarization 3.1 style pipeline.

This project removes the heavy PyTorch dependency for inference, making it lightweight, fast, and easy to deploy.

Based on pyannote-audio models and inspired by pyannote-onnx.

## Key Features

- Pure ONNX Runtime inference (no PyTorch required at runtime)
- Overlap-aware segmentation stitching
- Two-stage clustering for more stable short-utterance assignment
- Native exclusive diarization option
- Lightweight dependency footprint compared to full pyannote.audio

## Recent Changes

- Added native exclusive diarization behavior in ONNXSpeakerDiarization.
- Added test coverage for exclusive diarization behavior.
- Added GitHub Actions CI that runs pytest on push and pull request.

## Installation

```bash
pip install .
```

## Usage

```python
from onnx_pyannote import ONNXSpeakerDiarization

pipeline = ONNXSpeakerDiarization(
    model_name="speaker-diarization-3.1",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    return_exclusive=True,
)

annotation = pipeline("path/to/your/audio.wav")

for turn, _, speaker in annotation.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker={speaker}")
```

## Exclusive Diarization

By default, return_exclusive=True.

- return_exclusive=True returns a single active speaker per time instant.
- return_exclusive=False returns the regular overlap-preserving diarization.

If you already have an annotation, you can convert it directly:

```python
exclusive = ONNXSpeakerDiarization.build_exclusive_annotation(annotation)
```

## Community-1 Local ONNX Models

You can run with locally converted community-1 compatible artifacts by passing explicit paths:

```python
pipeline = ONNXSpeakerDiarization(
    model_name="speaker-diarization-3.1",
    segmentation_path="/path/to/segmentation-community.1.onnx",
    embedding_path="/path/to/wespeaker-voxceleb-resnet34-LM.onnx",
    providers=["CPUExecutionProvider"],
    return_exclusive=True,
)
```

If segmentation_path and embedding_path are not provided, defaults are fetched from Hugging Face.

## Exporting Models (Optional)

If you want to export models to ONNX yourself:

```bash
pip install -r requirements.txt
python export_onnx.py --use_auth_token YOUR_HF_TOKEN
```

This creates:

- models_onnx/segmentation.onnx
- models_onnx/embedding.onnx

## Testing

Run tests locally:

```bash
pip install pytest
pytest -q
```

Current tests include focused checks for exclusive diarization behavior under overlapping and non-overlapping segment conditions.

## CI

GitHub Actions workflow: .github/workflows/ci.yml

- Triggers on push and pull request
- Installs package in editable mode
- Runs pytest
