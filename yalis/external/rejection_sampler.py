# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# File modified from:
#  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rejection_sampler.py  # noqa: E501

from functools import cached_property
from typing import Tuple

import torch
import torch.jit


import torch
import torch.jit
import torch.nn as nn


class RejectionSampler(nn.Module):

    def __init__(self):
        super().__init__()

    @property
    def probs_dtype(self):
        return torch.float32

    @property
    def token_id_dtype(self):
        return torch.int64

    @cached_property
    def _smallest_positive_value(self) -> float:
        """Return the smallest positive value representable by the probs dtype.
        This value is used when constructing a distribution from which to
        sample recovered tokens in the first rejection case.

        See _get_recovered_probs for more details

        Note that this isn't actually the smallest positive value representable
        by float32, but the smallest positive normal value.
        See https://en.wikipedia.org/wiki/Subnormal_number for more info.
        """
        return torch.finfo(self.probs_dtype).tiny

    def _create_output(
        self,
        accepted: torch.Tensor,  # [batch_size, k]
        substitute_token_ids: torch.Tensor,  # [batch_size, k]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        bonus_token_ids: torch.Tensor,  # [batch_size]
    ) -> torch.Tensor:
        """Format output. Returns a matrix of token ids. When
        a token is rejected via sampling, all subsequent token ids are
        set to -1 for the sequence.

        Args:
            accepted: A boolean tensor indicating if the corresponding
            draft token in draft_token_ids should be accepted or not.
            substitute_token_ids: A tensor of token_ids that can be used
            as substitutes for the draft token ids if the proposed token
            is rejected.
            draft_token_ids: A tensor of token ids speculated by the
            draft model.
            bonus_token_ids: Token ids to use as the bonus token if
            all the draft tokens are accepted.
        Returns:
            A tensor containing the accepted token ids. The shape of the
            tensor is [batch_size, k + num_bonus_tokens]
        """
        batch_size, k = substitute_token_ids.shape
        bonus_token_ids = bonus_token_ids.squeeze()
        # Determine the index of the first False value for each row.
        limits = (accepted == 0).max(1).indices
        limits[~(accepted == 0).any(1)] = k

        # Create masks using the indices.
        indices = torch.arange(k, device=accepted.device).unsqueeze(0)
        accepted_mask = indices < limits.unsqueeze(1)
        after_false_mask = indices == limits.unsqueeze(1)

        # Create an extended output tensor
        output_with_bonus_tokens = -torch.ones(
            (batch_size, k + 1),
            # self._num_bonus_tokens),
            dtype=self.token_id_dtype,
            device=accepted.device,
        )
        output = output_with_bonus_tokens[:, :k]

        # Fill in the first k columns of the output tensor using masks and data
        # tensors.
        output[:, :k] = torch.where(
            accepted_mask, draft_token_ids, -torch.ones_like(draft_token_ids)
        )

        # Fill the last column.
        # We check output directly as accepted may have True values
        # inconsistent with causal acceptance.
        output_with_bonus_tokens[:, -1] = torch.where(
            output[:, -1] != -1, bonus_token_ids, -1
        )

        # Fill the recovered token ids.
        output.mul_(~after_false_mask).add_(
            substitute_token_ids.mul(after_false_mask)
        )

        return output_with_bonus_tokens

    def forward(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ):
        batch_size, k, _ = draft_probs.shape

        draft_tokens = draft_tokens.view(batch_size, k)

        # Assuming only one bonus token per batch
        bonus_tokens = bonus_tokens.view(batch_size, 1)

        accepted, recovered_token_ids = (
            self._batch_modified_rejection_sampling(
                target_probs[:, :-1],
                draft_probs,
                draft_tokens,
            )
        )

        output_token_ids = self._create_output(
            accepted,
            recovered_token_ids,
            draft_tokens,
            bonus_tokens,
        )

        return output_token_ids

    def _batch_modified_rejection_sampling(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform modified rejection sampling on each sequence.

        Returns:
            A tuple of two tensors:
            0: A bool tensor of which tokens in each sequence is accepted.
                shape = [batch_size, k]
            1: Token ids sampled from a recovered distribution, to be used
                when a token is rejected.
                shape = [batch_size, k]
        """

        batch_size, k, vocab_size = draft_probs.shape

        # shape [batch_size, k]
        accepted = self._get_accepted(
            target_probs, draft_probs, draft_token_ids
        )

        recovered_probs = self._get_recovered_probs(
            target_probs, draft_probs
        ).reshape(batch_size * k, vocab_size)

        # NOTE: the recovered_probs are overwritten by this method.
        recovered_token_ids = _multinomial(
            recovered_probs,
            num_samples=1,
            k=k,
        ).reshape(batch_size, k)

        return accepted, recovered_token_ids

    def _create_uniform_samples(
        self, batch_size: int, k: int, device: torch.device
    ) -> torch.Tensor:
        return torch.rand(batch_size, k + 1, device=device)

    def _get_accepted(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
    ) -> torch.Tensor:
        r"""Create bool matrix over the proposed draft tokens. If
        True, then a token can be accepted, else it should be
        rejected.

        Given :math:`q(\hat{x}_{n+1}|x_1, \dots, x_n)`, the probability of
        :math:`\hat{x}_{n+1}` given context :math:`x_1, \dots, x_n` according
        to the target model, and :math:`p(\hat{x}_{n+1}|x_1, \dots, x_n)`, the
        same conditional probability according to the draft model, the token
        is accepted with probability:

        .. math::
            \min\left(1, \frac{q(\hat{x}_{n+1}|x_1, \dots, x_n)}
                           {p(\hat{x}_{n+1}|x_1, \dots, x_n)}\right)

        This implementation does not apply causality. When using the output,
        if a token is rejected, subsequent tokens should not be used.

        Returns a bool tensor of shape [batch_size, k] specifying which tokens
        are accepted.
        """
        batch_size, k, _ = draft_probs.shape
        batch_indices = torch.arange(batch_size, device=target_probs.device)[
            :, None
        ]
        probs_indicies = torch.arange(k, device=target_probs.device)

        # shape [batch_size, k]
        selected_draft_probs = draft_probs[
            batch_indices, probs_indicies, draft_token_ids
        ]

        # shape [batch_size, k]
        selected_target_probs = target_probs[
            batch_indices, probs_indicies, draft_token_ids
        ]

        uniform_rand = self._create_uniform_samples(
            batch_size, k - 1, target_probs.device
        )

        capped_ratio = torch.minimum(
            selected_target_probs / selected_draft_probs,
            torch.full((1,), 1, device=target_probs.device),
        )
        accepted = uniform_rand < capped_ratio

        return accepted

    def _get_recovered_probs(
        self,
        target_probs: torch.Tensor,  # [k, vocab_size]
        draft_probs: torch.Tensor,  # [k, vocab_size]
    ) -> torch.Tensor:
        r"""Create a probability distribution for each proposed token which can
        be sampled if the proposed token is rejected.

        When this routine is applied sequentially, the true distribution of the
        target model is recovered (within hardware numerics).

        The probability distribution used in this rejection case is constructed
        as follows. Given :math:`q(x|x_1, \dots, x_n)`, the probability of
        :math:`x` given context :math:`x_1, \dots, x_n` according to the target
        model and :math:`p(x|x_1, \dots, x_n)`, the same conditional
        probability according to the draft model:

        .. math::
            x_{n+1} \sim (q(x|x_1, \dots, x_n) - p(x|x_1, \dots, x_n))_+

        where :math:`(f(x))_+` is defined as:

        .. math::
            (f(x))_+ = \frac{\max(0, f(x))}{\sum_x \max(0, f(x))}

        See https://github.com/vllm-project/vllm/pull/2336 for a visualization
        of the draft, target, and recovered probability distributions.

        Returns a tensor of shape [batch_size, k, vocab_size].

        Note: This batches operations on GPU and thus constructs the recovered
        distribution for all tokens, even if they are accepted. This causes
        division-by-zero errors, so we use self._smallest_positive_value to
        avoid that. This introduces some drift to the distribution.
        """
        _, k, _ = draft_probs.shape

        # shape [batch_size, k, vocab_size]
        difference = target_probs - draft_probs

        # TODO(cade): Can we use logprobs instead of probs, and avoid the
        # division-by-zero errors without introducing distribution drift?

        # shape [batch_size, k, vocab_size]
        f = torch.clamp(difference, min=self._smallest_positive_value)

        # shape [batch_size, k, vocab_size]
        recovered_probs = f / torch.sum(f, dim=-1).reshape(-1, k, 1)

        return recovered_probs


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead that skips the sync.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
@torch.compile(dynamic=True)
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    k: int,
) -> torch.Tensor:

    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        probs = (
            probs[:, None, :]
            .expand(probs.shape[0], num_samples, probs.shape[1])
            .contiguous()
            .view(-1, probs.shape[1])
        )
    q = torch.empty_like(probs)
    q.exponential_(1.0)

    return probs.div_(q).argmax(dim=1).view(-1, num_samples)
