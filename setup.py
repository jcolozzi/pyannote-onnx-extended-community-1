from setuptools import setup, find_packages

setup(
    name="pyannote-onnx-extended",
    version="0.1.0",
    description="A pure ONNX Runtime implementation of Pyannote Speaker Diarization 3.1",
    author="User",
    py_modules=["onnx_pyannote"],
    install_requires=[
        "onnxruntime>=1.16.0",
        "numpy>=1.24.0",
        "av>=11",
        "scikit-learn",
        "pyannote.core",
        "scipy",
        "librosa",
    ],
    python_requires=">=3.8",
)
