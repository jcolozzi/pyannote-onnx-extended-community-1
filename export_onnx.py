import torch
from pyannote.audio import Pipeline
import os


def export_onnx(use_auth_token=None):
    print("Loading pipeline...")
    # Load the pipeline
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=use_auth_token)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Please ensure you have accepted the user agreement on Hugging Face (pyannote/speaker-diarization-3.1) and provided your access token.")
        return

    # Create output directory
    os.makedirs("models_onnx", exist_ok=True)
    
    print("Exporting Segmentation Model...")
    # 1. Export Segmentation Model
    # The segmentation model takes a waveform chunk and outputs speaker probabilities
    try:
        segmentation_model = pipeline._segmentation.model
        segmentation_model.eval()
        
        # Dummy input for segmentation: (batch_size, channels, samples)
        # Pyannote segmentation usually works on 10s chunks at 16kHz
        # 10s * 16000 Hz = 160000 samples
        dummy_input_seg = torch.randn(1, 1, 160000) 
        
        torch.onnx.export(
            segmentation_model,
            dummy_input_seg,
            "models_onnx/segmentation.onnx",
            opset_version=12,
            input_names=["input_values"],
            output_names=["segmentation"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "num_channels", 2: "num_samples"},
                "segmentation": {0: "batch_size", 1: "num_frames"}
            }
        )
        print("Segmentation model exported to models_onnx/segmentation.onnx")
    except Exception as e:
        print(f"Failed to export segmentation model: {e}")

    print("Exporting Embedding Model...")
    # 2. Export Embedding Model
    # The pyannote 3.1 pipeline uses wespeaker-resnet34-LM.
    # We download the ONNX version from onnx-community.
    
    # Check if pipeline uses standard embedding
    if hasattr(pipeline, "_embedding") and pipeline._embedding is not None:
        try:
            embedding_model = pipeline._embedding.model_.resnet

            # Wrapper to keep only the first output
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    _, out = self.model(x)
                    return out  
            embedding_model = ModelWrapper(embedding_model)
            embedding_model.eval()

            dummy_input_emb = torch.randn(1, 1, 80)
            
            torch.onnx.export(
                embedding_model,
                dummy_input_emb,
                "models_onnx/embedding.onnx",
                opset_version=12,
                input_names=["input_features"],
                output_names=["embedding"],
                dynamic_axes={
                    "input_features": {0: "batch_size", 1: "time"},
                    "embedding": {0: "batch_size"}
                }
            )
            print("Embedding model exported to models_onnx/embedding.onnx")
        except Exception as e:
            print(f"Failed to export embedding model: {e}")

    else:
        print("Pipeline has no separate _embedding attribute or it is not available.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_auth_token", type=str, default=None)
    args = parser.parse_args()

    export_onnx(args.use_auth_token)
