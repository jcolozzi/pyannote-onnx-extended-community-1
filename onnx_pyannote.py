# The MIT License (MIT)
#
# Copyright (c) 2025- Samson Woof

from itertools import permutations

from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
import librosa
from pyannote.core import Segment, Annotation, Timeline
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

from utils.audio import decode_audio


class ONNXSpeakerDiarization:
    def __init__(self,
                 model_name="speaker-diarization-community-1",
                 segmentation_path=None,
                 embedding_path=None,
                 providers=['CPUExecutionProvider'],
                 onset=0.5,
                 offset=0.5,
                 min_duration_on=0.5,
                 min_duration_off=0.3,
                 return_exclusive=True):

        if model_name in {"speaker-diarization-community-1", "speaker-diarization-3.1"}:
            if segmentation_path is None:
                segmentation_path = hf_hub_download(
                    repo_id="onnx-community/pyannote-segmentation-3.0", 
                    filename="onnx/model.onnx"
                )
            if embedding_path is None:
                embedding_path = hf_hub_download(
                    repo_id="onnx-community/wespeaker-voxceleb-resnet34-LM", 
                    filename="onnx/model.onnx"
                )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.segmentation_session = ort.InferenceSession(
            segmentation_path, providers=providers)
        self.embedding_session = ort.InferenceSession(
            embedding_path, providers=providers)
        self.segmentation_input_name = self.segmentation_session.get_inputs()[0].name
        self.segmentation_output_name = self.segmentation_session.get_outputs()[0].name
        self.embedding_input_name = self.embedding_session.get_inputs()[0].name
        self.embedding_output_name = self.embedding_session.get_outputs()[0].name
        self.sample_rate = 16000
        self.duration = 10.0 # Segmentation model duration
        self.step = 0.5 * self.duration # 50% overlap
        self.onset = onset
        self.offset = offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.return_exclusive = return_exclusive

    @staticmethod
    def sample2frame(sample):
        return (sample - 721) // 270

    @staticmethod
    def reorder(overlap_prob, prob):
        perms = [np.array(perm).T for perm in permutations(prob.T)]
        perms = np.array(perms)
        sum_perms = np.sum(
            perms[:, :overlap_prob.shape[0], :] - overlap_prob, axis=1)
        diffs = np.sum(np.abs(sum_perms), axis=1)
        return perms[np.argmin(diffs)]

    @staticmethod
    def build_exclusive_annotation(annotation):
        timeline = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            start = float(segment.start)
            stop = float(segment.end)
            if stop <= start:
                continue
            timeline.append((start, stop, str(speaker)))

        if not timeline:
            return Annotation()

        boundaries = sorted({t for start, stop, _ in timeline for t in (start, stop)})
        if len(boundaries) < 2:
            return Annotation()

        speaker_order = []
        for _, _, speaker in timeline:
            if speaker not in speaker_order:
                speaker_order.append(speaker)

        exclusive = Annotation()
        for idx in range(len(boundaries) - 1):
            sub_start = boundaries[idx]
            sub_stop = boundaries[idx + 1]
            if sub_stop <= sub_start:
                continue

            active = {
                speaker
                for start, stop, speaker in timeline
                if start < sub_stop and stop > sub_start
            }
            if not active:
                continue

            if len(active) == 1:
                chosen_speaker = next(iter(active))
            else:
                overlap_center = (sub_start + sub_stop) / 2.0

                def score(candidate):
                    covered = sum(
                        min(sub_stop, stop) - max(sub_start, start)
                        for start, stop, speaker in timeline
                        if speaker == candidate and start < sub_stop and stop > sub_start
                    )
                    dist = min(
                        abs(overlap_center - ((start + stop) / 2.0))
                        for start, stop, speaker in timeline
                        if speaker == candidate
                    )
                    return (covered, -dist)

                sorted_active = sorted(
                    active,
                    key=lambda candidate: (
                        score(candidate)[0],
                        score(candidate)[1],
                        -speaker_order.index(candidate),
                    ),
                    reverse=True,
                )
                chosen_speaker = sorted_active[0]

            exclusive[Segment(sub_start, sub_stop)] = chosen_speaker

        return exclusive.support(collar=0.0)

    def __call__(self, audio, num_speakers=None, return_exclusive=None):
        """
        Args:
            audio (str or np.ndarray): Audio path or waveform.
            num_speakers (int, optional): Number of speakers. Defaults to None.
        """
        if isinstance(audio, str):
            audio_path = audio
            waveform = decode_audio(audio_path, self.sample_rate)
        else:
            waveform = audio

        # 1. Segmentation
        segments = self.run_segmentation(waveform)

        # 2. Embedding extraction
        embeddings, valid_segments = self.extract_embeddings(waveform, segments)

        if len(embeddings) == 0:
            return Annotation()

        # 3. Clustering
        labels = self.cluster_embeddings(embeddings, valid_segments, num_speakers=num_speakers)

        # 4. Build Annotation
        annotation = Annotation()
        for segment, label in zip(valid_segments, labels):
            annotation[segment] = f"SPEAKER_{label:02d}"

        annotation = annotation.support()  # Merge consecutive segments

        if return_exclusive is None:
            return_exclusive = self.return_exclusive

        if return_exclusive:
            return self.build_exclusive_annotation(annotation)

        return annotation

    def run_segmentation(self, waveform):
        # Sliding window segmentation
        step_samples = int(self.step * self.sample_rate)
        window_samples = int(self.duration * self.sample_rate)

        # Pad waveform
        pad_width = window_samples - (len(waveform) % window_samples)
        if pad_width < window_samples:
            waveform = np.pad(waveform, (0, pad_width), mode='constant')

        # Accumulate activations
        num_frames = self.sample2frame(window_samples)
        seconds_per_frame = self.duration / num_frames

        # Store all frame-level scores to merge them later (simple averaging for overlap)
        total_duration = len(waveform) / self.sample_rate
        total_frames = int(total_duration / seconds_per_frame) + 100 # buffer

        # We use a simple array to store max probability for speech at each frame
        global_scores = np.zeros((total_frames, 3))
        overlap_frames = self.sample2frame(window_samples - step_samples)

        for i_sample in range(0, len(waveform) - window_samples + 1, step_samples):
            chunk = waveform[i_sample:i_sample+window_samples]
            chunk = chunk[np.newaxis, np.newaxis, :]

            out = self.segmentation_session.run(
                [self.segmentation_output_name],
                {self.segmentation_input_name: chunk},
            )[0][0]
            out = np.exp(out) # (frames, speakers)

            # Get speech probability
            out[:, 1] += out[:, 4] + out[:, 5]
            out[:, 2] += out[:, 4] + out[:, 6]
            out[:, 3] += out[:, 5] + out[:, 6]
            speech_prob = out[:, 1:4]

            # Map to global frames
            start_frame_global = int((i_sample / self.sample_rate) / seconds_per_frame)
            end_frame_global = start_frame_global + len(speech_prob)

            # Overlap handling: take the average confidence
            if i_sample > 0:
                current_slice = global_scores[start_frame_global:end_frame_global]
                overlap_slice = current_slice[:overlap_frames]
                speech_prob = self.reorder(overlap_slice, speech_prob)

                overlap_end = start_frame_global + overlap_frames
                global_scores[start_frame_global:overlap_end] = (
                    overlap_slice + speech_prob[:overlap_frames]) / 2
                global_scores[overlap_end:end_frame_global] = speech_prob[overlap_frames:]
            else:
                global_scores[start_frame_global:end_frame_global] = speech_prob

        # Hysteresis Thresholding
        is_actives = [False] * 3
        list_segments = [[], [], []]
        start_ts = [0, 0, 0]
        end_ts = [0, 0, 0]

        # Iterate through global scores
        for f, scores in enumerate(global_scores):
            t = f * seconds_per_frame
            if t > total_duration:
                break

            for i_speaker in range(3):
                if not is_actives[i_speaker]:
                    if scores[i_speaker] > self.onset:
                        is_actives[i_speaker] = True
                        start_ts[i_speaker] = t
                else:
                    if scores[i_speaker] < self.offset:
                        is_actives[i_speaker] = False
                        end_ts[i_speaker] = t
                        if end_ts[i_speaker] - start_ts[i_speaker] >= self.min_duration_on:
                            list_segments[i_speaker].append(
                                Segment(start_ts[i_speaker], end_ts[i_speaker]))

        # Handle last segment
        for i_speaker in range(3):
            if is_actives[i_speaker]:
                end_ts[i_speaker] = total_duration
                if end_ts[i_speaker] - start_ts[i_speaker] >= self.min_duration_on:
                    list_segments[i_speaker].append(
                        Segment(start_ts[i_speaker], end_ts[i_speaker]))

        # Merge consecutive segments
        segments = []
        for i_speaker in range(3):
            segments.extend(
                list(Timeline(list_segments[i_speaker]).support(collar=self.min_duration_off)))

        return segments

    def extract_embeddings(self, waveform, segments):
        embeddings = []
        valid_segments = []

        for segment in segments:
            start_sample = int(segment.start * self.sample_rate)
            end_sample = int(segment.end * self.sample_rate)

            if end_sample > len(waveform):
                end_sample = len(waveform)

            chunk = waveform[start_sample:end_sample]

            if len(chunk) < 400: # Too short
                continue

            # Feature Extraction
            n_fft = 400
            hop_length = 160
            n_mels = 80

            melspec = librosa.feature.melspectrogram(
                y=chunk, sr=self.sample_rate, n_fft=n_fft,
                hop_length=hop_length, n_mels=n_mels,
                window='hamming', center=False
            )
            log_melspec = np.log(melspec + 1e-6)
            features = log_melspec.T
            features = features - np.mean(features, axis=0)
            features = features[np.newaxis, :, :]

            try:
                emb = self.embedding_session.run(
                    [self.embedding_output_name],
                    {self.embedding_input_name: features},
                )[0][0]

                # Normalize embedding (L2 norm) - Important for Cosine Distance
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm

                embeddings.append(emb)
                valid_segments.append(segment)
            except Exception as e:
                pass

        return np.array(embeddings), valid_segments

    def cluster_embeddings(self, embeddings, segments, num_speakers=None):
        if len(embeddings) == 0:
            return []
        elif len(embeddings) == 1:
            return [0]

        # Two-Stage Clustering
        # 1. Cluster long segments to find core speakers
        # 2. Assign short segments to the nearest core speaker
        duration = sum([s.end - s.start for s in segments])
        min_duration = np.clip(duration / 60, 2, 5)
        long_indices = [
            i for i, s in enumerate(segments) if (s.end - s.start) >= min_duration]
        short_indices = [
            i for i, s in enumerate(segments) if (s.end - s.start) < min_duration]

        # Fallback: if not enough long segments, cluster everything
        if len(long_indices) < 2 or (num_speakers and num_speakers <= 1):
            return [0] * len(embeddings)

        long_embeddings = embeddings[long_indices]

        if num_speakers:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers, metric='euclidean', linkage='ward')
        else:
            distance_threshold = max((350 - duration) / 350, 0.73)
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance_threshold,
                metric='euclidean', linkage='single')

        clustering.fit(long_embeddings)
        long_labels = clustering.labels_

        # Prepare final labels array
        labels = np.zeros(len(embeddings), dtype=int)
        labels[long_indices] = long_labels

        # Assign short segments
        if len(short_indices) > 0:
            # Calculate centroids of the found clusters
            unique_labels = np.unique(long_labels)
            centroids = []
            centroid_labels = []

            for label in unique_labels:
                # Get all embeddings belonging to this cluster
                cluster_emb = long_embeddings[long_labels == label]
                # Compute mean (centroid)
                centroid = np.mean(cluster_emb, axis=0)
                centroids.append(centroid)
                centroid_labels.append(label)

            centroids = np.array(centroids)

            # Predict labels for short embeddings based on nearest centroid
            short_embeddings = embeddings[short_indices]

            # Calculate distances: (n_short, n_centroids)
            dists = cdist(short_embeddings, centroids, metric='euclidean')
            nearest_centroid_indices = np.argmin(dists, axis=1)

            short_labels = [centroid_labels[i] for i in nearest_centroid_indices]
            labels[short_indices] = short_labels

        return labels
