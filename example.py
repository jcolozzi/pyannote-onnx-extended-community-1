from onnx_diarization import ONNXSpeakerDiarization
import time

def main():
    # Initialize the pipeline
    # Make sure you have run 'python export_onnx.py' first!
    pipeline = ONNXSpeakerDiarization(
        model_name="speaker-diarization-3.1",
        providers=['CPUExecutionProvider'] # Change to 'CUDAExecutionProvider' for GPU
    )

    audio_path = "example.wav" # Replace with your audio file
    
    # Create a dummy file if it doesn't exist for testing
    import os
    if not os.path.exists(audio_path):
        print(f"File {audio_path} not found. Please provide an audio file path.")
        return

    print(f"Processing {audio_path}...")
    start_time = time.time()
    
    # Run diarization
    annotation = pipeline(audio_path)
    
    end_time = time.time()
    print(f"Diarization completed in {end_time - start_time:.2f} seconds.")

    # Print results
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        print(f"[{turn.start:.2f} - {turn.end:.2f}] {speaker}")

if __name__ == "__main__":
    main()
