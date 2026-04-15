# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n
# Lightweight build — removed torchcodec, SAMAudioJudgeProcessor, load_video

import json
import logging
import math
import os
from typing import Callable, List, Optional, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from torch.nn.utils.rnn import pad_sequence

from sam_audio.model.config import SAMAudioConfig

logger = logging.getLogger(__name__)

Anchor = Tuple[str, float, float]


def batch_audio(
    audios: list[str | torch.Tensor], audio_sampling_rate: int = 48_000
) -> Tuple[torch.Tensor, torch.Tensor]:
    wavs = []
    for audio in audios:
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)
            if sr != audio_sampling_rate:
                wav = torchaudio.functional.resample(wav, sr, audio_sampling_rate)
        else:
            wav = audio
        wavs.append(wav.mean(0))
    sizes = torch.tensor([wav.size(-1) for wav in wavs])
    return pad_sequence(wavs, batch_first=True).unsqueeze(1), sizes


class Batch:
    def __init__(
        self,
        audios: torch.Tensor,
        sizes: torch.Tensor,
        wav_sizes: torch.Tensor,
        descriptions: list[str],
        hop_length: int,
        audio_sampling_rate: int,
        anchors: Optional[list[list[Anchor]]] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
        masked_video: Optional[torch.Tensor] = None,
    ):
        self.audios = audios
        self.sizes = sizes
        self.wav_sizes = wav_sizes
        self.descriptions = descriptions
        self.audio_pad_mask = audio_pad_mask
        self.masked_video = masked_video
        self.hop_length = hop_length
        self.audio_sampling_rate = audio_sampling_rate
        self.process_anchors(anchors)
        assert self.audios.size(0) == len(self.descriptions)

    def _wav_to_feature_idx(self, wav_idx: int):
        return math.ceil(wav_idx / self.hop_length)

    def to(self, device: torch.device):
        self.audios = self.audios.to(device)
        self.anchor_ids = self.anchor_ids.to(device)
        self.anchor_alignment = self.anchor_alignment.to(device)
        self.sizes = self.sizes.to(device)
        self.wav_sizes = self.wav_sizes.to(device)
        if self.audio_pad_mask is not None:
            self.audio_pad_mask = self.audio_pad_mask.to(device)
        if self.masked_video is not None:
            self.masked_video = [v.to(device) for v in self.masked_video]
        return self

    def process_anchors(self, anchors: Optional[list[list[Anchor]]]):
        batch_size = len(self.audios)
        anchor_dict = {"<null>": 0, "+": 1, "-": 2, "<pad>": 3}
        if anchors is None:
            anchor_ids = torch.full(
                (batch_size, 2), anchor_dict["<null>"], dtype=torch.long
            )
            anchor_ids[:, 1] = anchor_dict["<pad>"]
            anchor_alignment = torch.full(
                (
                    batch_size,
                    self.audio_pad_mask.size(-1),
                ),
                0,
                dtype=torch.long,
            )
            anchor_alignment[~self.audio_pad_mask] = 1  # point to pad token
        else:
            anchor_alignment = torch.full(
                (
                    batch_size,
                    self.audio_pad_mask.size(-1),
                ),
                0,
                dtype=torch.long,
            )
            anchor_alignment[~self.audio_pad_mask] = 1  # point to pad token
            ids = []

            for i, anchor_list in enumerate(anchors):
                current = [anchor_dict["<null>"], anchor_dict["<pad>"]]
                for token, start_time, end_time in anchor_list:
                    start_idx = self._wav_to_feature_idx(
                        start_time * self.audio_sampling_rate
                    )
                    end_idx = self._wav_to_feature_idx(
                        end_time * self.audio_sampling_rate
                    )
                    anchor_alignment[i, start_idx:end_idx] = len(current)
                    current.append(anchor_dict[token])
                ids.append(torch.tensor(current))
            anchor_ids = pad_sequence(
                ids, batch_first=True, padding_value=anchor_dict["<pad>"]
            )
        self.anchor_ids = anchor_ids.to(self.audios.device)
        self.anchor_alignment = anchor_alignment.to(self.audios.device)
        self.anchors = anchors


def mask_from_sizes(sizes: torch.Tensor) -> torch.Tensor:
    return torch.arange(sizes.max()).expand(len(sizes), -1) < sizes.unsqueeze(1)


class Processor:
    config_cls: Callable

    def __init__(self, audio_hop_length: int, audio_sampling_rate: int):
        self.audio_hop_length = audio_hop_length
        self.audio_sampling_rate = audio_sampling_rate

    @classmethod
    def _get_config(cls, model_name_or_path: str):
        if os.path.exists(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
        else:
            config_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="config.json",
                revision=cls.revision,
            )
        with open(config_path) as fin:
            config = cls.config_cls(**json.load(fin))
        return config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "Processor":
        config = cls._get_config(model_name_or_path)
        return cls(
            audio_hop_length=config.audio_codec.hop_length,
            audio_sampling_rate=config.audio_codec.sample_rate,
        )

    def feature_to_wav_idx(self, feature_idx):
        return feature_idx * self.audio_hop_length

    def wav_to_feature_idx(self, wav_idx):
        if torch.is_tensor(wav_idx):
            ceil = torch.ceil
        else:
            ceil = math.ceil
        return ceil(wav_idx / self.audio_hop_length)


class SAMAudioProcessor(Processor):
    config_cls = SAMAudioConfig
    revision = None

    def __call__(
        self,
        descriptions: list[str],
        audios: list[str | torch.Tensor],
        anchors: Optional[list[list[Anchor]]] = None,
        masked_videos: Optional[list[str | torch.Tensor]] = None,
    ):
        """
        Processes input data for the model.

        Args:
            descriptions (list[str]): List of text descriptions corresponding to each audio sample.
            audios (list[str]): List of audio file paths or tensors.
            anchors (Optional[list[list[Anchor]]], optional): List of anchors for each sample.
            masked_videos: Not supported in lightweight build.
        """

        assert len(descriptions) == len(audios)
        assert anchors is None or len(descriptions) == len(anchors)

        audios, wav_sizes = batch_audio(audios, self.audio_sampling_rate)

        sizes = self.wav_to_feature_idx(wav_sizes)
        audio_pad_mask = mask_from_sizes(sizes)

        if masked_videos is not None:
            logger.warning(
                "Video input is not supported in sam_audio_lite. Ignoring masked_videos."
            )

        return Batch(
            audios=audios,
            sizes=sizes,
            descriptions=descriptions,
            audio_pad_mask=audio_pad_mask,
            anchors=anchors,
            masked_video=None,
            hop_length=self.audio_hop_length,
            audio_sampling_rate=self.audio_sampling_rate,
            wav_sizes=wav_sizes,
        )


__all__ = ["SAMAudioProcessor", "Batch"]
