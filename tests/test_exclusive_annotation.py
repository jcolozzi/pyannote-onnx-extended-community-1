from pyannote.core import Annotation, Segment

from onnx_pyannote import ONNXSpeakerDiarization


def _as_triplets(annotation: Annotation):
    triplets = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        triplets.append((round(float(segment.start), 3), round(float(segment.end), 3), speaker))
    return triplets


def test_exclusive_annotation_is_empty_for_empty_input():
    annotation = Annotation()

    exclusive = ONNXSpeakerDiarization.build_exclusive_annotation(annotation)

    assert _as_triplets(exclusive) == []


def test_exclusive_annotation_keeps_non_overlapping_segments():
    annotation = Annotation()
    annotation[Segment(0.0, 1.0)] = "SPEAKER_00"
    annotation[Segment(1.0, 2.0)] = "SPEAKER_01"

    exclusive = ONNXSpeakerDiarization.build_exclusive_annotation(annotation)

    assert _as_triplets(exclusive) == [
        (0.0, 1.0, "SPEAKER_00"),
        (1.0, 2.0, "SPEAKER_01"),
    ]


def test_exclusive_annotation_removes_overlap_with_single_active_speaker():
    annotation = Annotation()
    annotation[Segment(0.0, 2.0)] = "SPEAKER_00"
    annotation[Segment(0.5, 0.7)] = "SPEAKER_01"
    annotation[Segment(1.6, 1.9)] = "SPEAKER_01"

    exclusive = ONNXSpeakerDiarization.build_exclusive_annotation(annotation)

    assert _as_triplets(exclusive) == [
        (0.0, 0.5, "SPEAKER_00"),
        (0.5, 0.7, "SPEAKER_01"),
        (0.7, 1.6, "SPEAKER_00"),
        (1.6, 1.9, "SPEAKER_01"),
        (1.9, 2.0, "SPEAKER_00"),
    ]