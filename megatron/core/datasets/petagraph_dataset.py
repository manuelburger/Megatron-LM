# =============================================================================
# 
# Copyright (C) 2024, Manuel Burger
# 
# Petagraph Dataset
#
# =============================================================================
from torchdata.datapipes.iter import IterableWrapper, S3FileLoader, \
    FileOpener, Mapper, StreamReader

import torch
import random
import zstd
import numpy as np
from typing import Dict, Optional, Tuple

from pathlib import Path

from megatron.training import print_rank_0
from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.training.training import cyclic_iter


class PetaGraphStreamDataset(torch.utils.data.Dataset):

    def __init__(self, 
        url_list: list[str],
        from_cloud: bool = False,
        maxlen: int = 128,
        samples_per_epoch: int = 10000,
        create_attention_mask: bool = True,
        debug: bool = False
    ):

        self.samples_per_epoch = samples_per_epoch
        self.maxlen = maxlen
        self.create_attention_mask = create_attention_mask
        self.debug = debug

        print_rank_0("=====================================")
        print_rank_0(f"Creating PetaGraphStreamDataset with {samples_per_epoch} samples per epoch and maxlen {maxlen}")
        print_rank_0(f"Num. URLs: {len(url_list)}")
        print_rank_0(f"From Cloud: {from_cloud}")
        print_rank_0("=====================================")

        self.VOCAB = {
            "BOS": 0, "EOS": 1, "PAD": 2, "UNK": 3,
            "A": 4, "C": 5, "G": 6, "T": 7
        }
        self._pad_token_id = self.VOCAB["PAD"]
        self._eos_token_id = self.VOCAB["EOS"]

        if from_cloud:
            dp_s3_urls = IterableWrapper(url_list).list_files_by_s3()
            # In order to make sure data are shuffled and sharded in the
            # distributed environment, `shuffle`  and `sharding_filter`
            # are required. For detail, please check our tutorial in:
            # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
            sharded_s3_urls = dp_s3_urls.shuffle().sharding_filter()
            opened_files = S3FileLoader(sharded_s3_urls)

        else:

            files_names = IterableWrapper(url_list).shuffle().sharding_filter()
            opened_files = FileOpener(files_names, mode="rb")

        decoded_files = StreamReader(opened_files)
        decompressed_files = Mapper(decoded_files, self.decompression_func)
        sequences_batched = Mapper(decompressed_files, self.fasta_parsing_func)
        sequences_unbatched = sequences_batched.unbatch()

        sequences_crop = Mapper(sequences_unbatched, self.crop_maxlen)
        sequences_tokenized = Mapper(sequences_crop, self.tokenize_and_pad)

        if from_cloud:
            self.iterable_dataset = iter(sequences_tokenized)
        else:
            self.iterable_dataset = cyclic_iter(sequences_tokenized)
        print_rank_0(f"Sample: {next(self.iterable_dataset)}")

    def decompression_func(self, input_data):
        path, data = input_data
        if self.debug:
            print_rank_0(f"Decompressing {path}")
            
        return path, zstd.decompress(data)

    def fasta_parsing_func(self, input_data):
        path, data = input_data
        if self.debug:
            print_rank_0(f"Parsing {path}")

        sequences = []
        current_sequence = ""
        for line in data.decode().split("\n"):
            if line.startswith(">"):
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = ""
            else:
                current_sequence += line
        return sequences

    def crop_maxlen(self, input_sequence: str):
        maxlen_without_special_tokens = self.maxlen - 2
        if len(input_sequence) <= maxlen_without_special_tokens:
            return input_sequence
        else:
            # Crop the sequence to the maximum length
            # Get random starting point
            start = random.randint(0, len(input_sequence) - maxlen_without_special_tokens)
            return input_sequence[start:start + maxlen_without_special_tokens]

    def tokenize_and_pad(self, input_sequence: str):
        maxlen = self.maxlen

        # Tokenize the sequence
        tokenized_sequence = [0] # start with BOS token
        tokenized_sequence.extend([self.VOCAB.get(base, 3) for base in input_sequence]) # 3 is the UNK token
        tokenized_sequence.append(1) # end with EOS token

        # Pad the sequence
        if len(tokenized_sequence) < maxlen:
            tokenized_sequence.extend([2] * (maxlen - len(tokenized_sequence))) # 2 is the PAD token

        return np.array(tokenized_sequence, dtype=np.int16)
        
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        text = next(self.iterable_dataset)
        text = torch.from_numpy(text).long()

        # if self.config.add_extra_token_to_sequence:
        #     tokens = text[:-1].contiguous()
        #     labels = text[1:].contiguous()
        # else:

        tokens = text
        labels = torch.roll(text, shifts=-1, dims=0)
        labels[-1] = self._pad_token_id

        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                eod_token=self._eos_token_id,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=True,
                create_attention_mask=True,
            )

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }