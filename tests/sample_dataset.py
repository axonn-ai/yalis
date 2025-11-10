# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This module has been adapted from the original vLLM codebase:
#  https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_dataset.py

"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet

TODO: Implement CustomDataset to parse a JSON file and convert
its contents into SampleRequest instances, similar to the approach
used in ShareGPT.
"""
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and
        random seed. Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = (
            random_seed if random_seed is not None else self.DEFAULT_SEED
        )
        self.data = None

    def apply_multimodal_chat_transformation(self, prompt: str) -> list[dict]:
        """
        Transform a prompt into a chat format.
        This method is used for chat models that expect a specific
        conversation format.
        """
        content = [{"text": prompt, "type": "text"}]
        return [{"role": "user", "content": content}]

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        raise NotImplementedError(
            "load_data must be implemented in subclasses."
        )

    @abstractmethod
    def sample(
        self, tokenizer: PreTrainedTokenizerBase, num_requests: int
    ) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific
        logic for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(
        self, requests: list[SampleRequest], num_requests: int
    ) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(
                requests, k=num_requests - len(requests)
            )
            requests.extend(additional)
            print(
                f"Oversampled requests to reach {num_requests} total samples."
            )


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 1.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> list[SampleRequest]:
        vocab_size = tokenizer.vocab_size

        prefix_token_ids = (
            np.random.randint(0, vocab_size, size=prefix_len).tolist()
            if prefix_len > 0
            else []
        )

        input_low = int(input_len * range_ratio)
        output_low = int(output_len * range_ratio)

        input_lens = np.random.randint(
            input_low, input_len + 1, size=num_requests
        )
        output_lens = np.random.randint(
            output_low, output_len + 1, size=num_requests
        )
        offsets = np.random.randint(0, vocab_size, size=num_requests)

        requests = []
        for i in range(num_requests):
            inner_seq = (
                (offsets[i] + i + np.arange(input_lens[i])) % vocab_size
            ).tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(token_sequence)
            total_input_len = prefix_len + int(input_lens[i])
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                )
            )
        return requests


# -----------------------------------------------------------------------------
# Alpaca Dataset Implementation
# -----------------------------------------------------------------------------


class AlpacaDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_PREFIX_LEN = 10
    DEFAULT_INPUT_LEN = 16384
    DEFAULT_OUTPUT_LEN = 150

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path="yahma/alpaca-cleaned", **kwargs)
        self.dataset_split = "train"
        self.dataset_subset = None
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
        )
        data = self.data.shuffle(seed=self.random_seed)

        self.data = data.map(
            lambda example: {
                "user_prompt": f"{example['instruction']} {example['input']} \n\n"  # noqa: E501
            }
        )["user_prompt"]
        print(f"Length of data: {len(self.data[0])}")

    def sample(
        self,
        tokenizer,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        return_prompt_formatted: bool = False,
        **kwargs,
    ) -> list:
        # Calculate average token length for a poem line.
        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens) for tokens in tokenized_lines) / len(
            tokenized_lines
        )
        # print (f"Average length of tokenized lines: {avg_len}")

        # Find one minimum length line.
        index_min = np.argmin(
            [len(tokens) for tokens in tokenized_lines]
        ).item()
        min_len = len(tokenized_lines[index_min])
        padding_line = self.data[int(index_min)]
        # print (f"Minimum length line: {padding_line}")

        # Build the base prompt.
        base_prompt = "You are a helpful chatbot. Answer as many of the following questions as possible.\n"  # noqa: E501
        base_msg = [{"role": "system", "content": base_prompt}]
        base_fmt = tokenizer.apply_chat_template(
            base_msg, add_generation_prompt=True, tokenize=False
        )
        base_offset = len(tokenizer(base_fmt).input_ids)
        # print (f"Base offset: {base_offset}")
        if input_len <= base_offset:
            raise ValueError(
                f"'input_len' must be higher than the base prompt length "
                f"({base_offset})."
            )

        # Determine how many lines to use.
        num_input_lines = round((input_len - base_offset + avg_len) / avg_len)
        # print (f"Number of input lines: {num_input_lines}")

        samples = []
        for _ in range(num_requests):
            data_lines = random.choices(self.data, k=num_input_lines)
            user_prompt = f"{''.join(data_lines)}"
            user_msg = [{"role": "user", "content": user_prompt}]
            user_prompt_formatted = tokenizer.apply_chat_template(
                user_msg, add_generation_prompt=True, tokenize=False
            )
            user_prompt_len = len(tokenizer(user_prompt_formatted).input_ids)
            if user_prompt_len < input_len:
                # Add the minimum padding line to the prompt.
                num_reps = (input_len - user_prompt_len) // min_len
                padding_lines = [padding_line] * num_reps
                data_lines.extend(padding_lines)

            prompt = f"{''.join(data_lines)}"
            msg = [
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt_formatted = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            samples.append(
                SampleRequest(
                    prompt=(
                        prompt_formatted if return_prompt_formatted else prompt
                    ),
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                )
            )
        return samples


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset."""

    DEFAULT_PREFIX_LEN = 0
    DEFAULT_INPUT_LEN = 16384
    DEFAULT_OUTPUT_LEN = 250

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_path="Lambent/shakespeare_sonnets_diffused", **kwargs
        )
        self.dataset_split = "train"
        self.dataset_subset = None
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
        )
        self.data = self.data.shuffle(seed=self.random_seed)["Variation Text"]

    def sample(
        self,
        tokenizer,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        return_prompt_formatted: bool = False,
        **kwargs,
    ) -> list:
        # Calculate average token length for a poem line.
        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens) for tokens in tokenized_lines) / len(
            tokenized_lines
        )

        # Build the base prompt.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_msg = [{"role": "user", "content": base_prompt}]
        base_fmt = tokenizer.apply_chat_template(
            base_msg, add_generation_prompt=True, tokenize=False
        )
        base_offset = len(tokenizer(base_fmt).input_ids)
        if input_len <= base_offset:
            raise ValueError(
                f"'input_len' must be higher than the base prompt length "
                f"({base_offset})."
            )

        # Determine how many lines to use.
        num_input_lines = (
            round(((input_len - base_offset) + avg_len) / avg_len) + 100
        )

        samples = []
        for _ in range(num_requests):
            extra_lines = random.choices(self.data, k=num_input_lines)
            prompt = f"{base_prompt}{''.join(extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            samples.append(
                SampleRequest(
                    prompt=(
                        prompt_formatted if return_prompt_formatted else prompt
                    ),
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                )
            )
        return samples


if __name__ == "__main__":
    # Test AlpacaDataset
    dataset = AlpacaDataset(random_seed=42)
    print(dataset.data[0])
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    samples = dataset.sample(
        tokenizer, 8, input_len=16384, return_prompt_formatted=True
    )

    for sample in samples:
        print("Prompt:", len(sample.prompt))

        # Get number of words in the prompt
        num_words = len(sample.prompt.split())
        print("Number of words in the prompt:", num_words)

        print("Prompt Length:", sample.prompt_len)
