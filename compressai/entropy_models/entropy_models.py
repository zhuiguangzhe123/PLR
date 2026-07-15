# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import math

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ops import quantize_ste
from torch import Tensor
import torch.special as sp

from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ops import LowerBound


class _EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders

        if method not in available_entropy_coders():
            methods = ", ".join(available_entropy_coders())
            raise ValueError(
                f'Unknown entropy coder "{method}"' f" (available: {methods})"
            )

        if method == "ans":
            from compressai import ans

            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == "rangecoder":
            import range_coder

            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self.name = method
        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)


def default_entropy_coder():
    from compressai import get_entropy_coder

    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf: Tensor, precision: int = 16) -> Tensor:
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


def _forward(self, *args: Any) -> Any:
    raise NotImplementedError()


class EntropyModel(nn.Module):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(
        self,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
    ):
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["entropy_coder"] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length

    # See: https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def _quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        warnings.warn("_quantize is deprecated. Use quantize instead.", stacklevel=2)
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(
        inputs: Tensor, means: Optional[Tensor] = None, dtype: torch.dtype = torch.float
    ) -> Tensor:
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs

    @classmethod
    def _dequantize(cls, inputs: Tensor, means: Optional[Tensor] = None) -> Tensor:
        warnings.warn("_dequantize. Use dequantize instead.", stacklevel=2)
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    def compress(self, inputs, indexes, means=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs, "symbols", means)

        if len(inputs.size()) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = self.dequantize(outputs, means, dtype)
        return outputs


class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md>`__
    for an introduction.
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.matrices.append(nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.biases.append(nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.factors.append(nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _get_medians(self) -> Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: bool = False, update_quantiles: bool = False) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        if update_quantiles:
            self._update_quantiles()

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)
        samples = samples[None, :] + pmf_start[:, None, None]

        pmf, lower, upper = self._likelihood(samples, stop_gradient=True)
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) -> Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self.matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = self.biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits = logits + bias

            if i < len(self.filters):
                factor = self.factors[i]
                if stop_gradient:
                    factor = factor.detach()
                logits = logits + torch.tanh(factor) * torch.tanh(logits)
        return logits

    def _likelihood(
        self, inputs: Tensor, stop_gradient: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        half = float(0.5)
        lower = self._logits_cumulative(inputs - half, stop_gradient=stop_gradient)
        upper = self._logits_cumulative(inputs + half, stop_gradient=stop_gradient)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        return likelihood, lower, upper

    def forward(
        self, x: Tensor, training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = torch.cat(
                (
                    torch.tensor([1, 0], dtype=torch.long, device=x.device),
                    torch.arange(2, x.ndim, dtype=torch.long, device=x.device),
                )
            )
            inv_perm = perm
        else:
            raise NotImplementedError()
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            # perm = (1, 2, 3, 0)
            # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize

        outputs = self.quantize(
            values, "noise" if training else "dequantize", self._get_medians()
        )

        if not torch.jit.is_scripting():
            likelihood, _, _ = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    @torch.no_grad()
    def _update_quantiles(self, search_radius=1e5, rtol=1e-4, atol=1e-3):
        """Fast quantile update via bisection search.

        Often faster and much more precise than minimizing aux loss.
        """
        device = self.quantiles.device
        shape = (self.channels, 1, 1)
        low = torch.full(shape, -search_radius, device=device)
        high = torch.full(shape, search_radius, device=device)

        def f(y, self=self):
            return self._logits_cumulative(y, stop_gradient=True)

        for i in range(len(self.target)):
            q_i = self._search_target(f, self.target[i], low, high, rtol, atol)
            self.quantiles[:, :, i] = q_i[:, :, 0]

    @staticmethod
    def _search_target(f, target, low, high, rtol=1e-4, atol=1e-3, strict=False):
        assert (low <= high).all()
        if strict:
            assert ((f(low) <= target) & (target <= f(high))).all()
        else:
            low = torch.where(target <= f(high), low, high)
            high = torch.where(f(low) <= target, high, low)
        while not torch.isclose(low, high, rtol=rtol, atol=atol).all():
            mid = (low + high) / 2
            f_mid = f(mid)
            low = torch.where(f_mid <= target, mid, low)
            high = torch.where(f_mid >= target, mid, high)
        return (low + high) / 2

    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)


class GaussianConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/api_docs/python/tfc/GaussianConditional.md>`__
    for more information.
    """

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales: Tensor) -> Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes






class GaussianConditional_ST(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/api_docs/python/tfc/GaussianConditional.md>`__
    for more information.
    """

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor) -> Tensor:
        const = float(-(2**-0.5))
        values = (half - torch.abs(inputs-means)) / scales
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(const * values)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    def update(self):
        half = float(0.5)
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples_scale = self.scale_table.unsqueeze(1).float()
        upper = self._standardized_cumulative(torch.arange(max_length, device=device).int(), pmf_center[:, None], samples_scale, half)
        lower = self._standardized_cumulative(torch.arange(max_length, device=device).int(), pmf_center[:, None], samples_scale, -half)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]
        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2



    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        half = float(0.5)

        scales = self.lower_bound_scale(scales)
        upper = self._standardized_cumulative(inputs, means, scales, half)
        lower = self._standardized_cumulative(inputs, means, scales, -half)
        likelihood = upper - lower

        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        if training:
            outputs = inputs
        else:
            outputs = inputs.int()
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        # reg = torch.tensor(0.).to(inputs.device)
        return outputs, likelihood#, reg

    def build_indexes(self, scales: Tensor) -> Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes
    
    # def compress(self, inputs, indexes, means=None):
    #     """
    #     Compress input tensors to char strings.

    #     Args:
    #         inputs (torch.Tensor): input tensors
    #         indexes (torch.IntTensor): tensors CDF indexes
    #         means (torch.Tensor, optional): optional tensor means
    #     """
    #     symbols = (inputs - torch.round(means)).int()

    #     if len(inputs.size()) < 2:
    #         raise ValueError(
    #             "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
    #         )

    #     if inputs.size() != indexes.size():
    #         raise ValueError("`inputs` and `indexes` should have the same size.")

    #     self._check_cdf_size()
    #     self._check_cdf_length()
    #     self._check_offsets_size()

    #     strings = []
    #     for i in range(symbols.size(0)):
    #         rv = self.entropy_coder.encode_with_indexes(
    #             symbols[i].reshape(-1).int().tolist(),
    #             indexes[i].reshape(-1).int().tolist(),
    #             self._quantized_cdf.tolist(),
    #             self._cdf_length.reshape(-1).int().tolist(),
    #             self._offset.reshape(-1).int().tolist(),
    #         )
    #         strings.append(rv)
    #     return strings
    @torch.no_grad()
    def _build_cdf(self, scales, means, nonzero, abs_max):
        scales = scales[:, nonzero]
        means = means[:, nonzero]
        
        num_samples = abs_max * 2 + 1
        TINY = 1e-10
        device = scales.device

        scales = scales.clamp_(0.11, 256)
        means += abs_max

        scales_ = scales.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        means_ = means.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        num_latents = scales_.size(0)

        samples = (
            torch.arange(num_samples).to(device).unsqueeze(0).expand(num_latents, -1)
        )

        pmf = torch.zeros_like(samples).float()
        pmf += (
            0.5
            * (
                1
                + torch.erf(
                    (samples + 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
            - 0.5
            * (
                1
                + torch.erf(
                    (samples - 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
        )

        cdf_limit = 2**self.entropy_coder_precision - 1
        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, 1, keepdim=True).expand(-1, num_samples)

        cdf = F.pad(
            torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, 1).int(),
            (1, 0),
            "constant",
            0,
        )
        pmf_quantized = torch.diff(cdf, dim=1)

        # We can't have zeros in PMF because rANS won't be able to encode it.
        # Try to fix this by "stealing" probability from some unlikely symbols.

        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)

        _, pmf_first_stealable_indices = torch.min(
            torch.where(
                pmf_quantized > pmf_zero_count.unsqueeze(-1).expand(-1, num_samples),
                pmf_quantized,
                torch.tensor(cdf_limit + 1).int().to(device),
            ),
            dim=1,
        )

        pmf_real_zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        pmf_quantized[pmf_real_zero_indices[0], pmf_real_zero_indices[1]] += 1

        pmf_real_steal_indices = torch.cat(
            (
                torch.arange(num_latents).to(device).unsqueeze(-1),
                pmf_first_stealable_indices.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[
            pmf_real_steal_indices[0], pmf_real_steal_indices[1]
        ] -= pmf_zero_count

        cdf = F.pad(torch.cumsum(pmf_quantized, 1).int(), (1, 0), "constant", 0)
        cdf = F.pad(cdf, (0, 1), "constant", cdf_limit + 1)

        return cdf

    def compress(self, y, scales, means):
        abs_max = (
            max(torch.abs(y.max()).int().item(), torch.abs(y.min()).int().item()) + 1
        )
        abs_max = 1 if abs_max < 1 else abs_max

        zero_bitmap = torch.where(
            torch.sum(torch.abs(y), (3, 2)).squeeze(0) == 0, 0, 1
        )
        print(zero_bitmap)
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        symbols = y[:, nonzero] + abs_max
        cdf = self._build_cdf(scales, means, nonzero, abs_max)
        num_latents = cdf.size(0)
        flatten_symbols = symbols.reshape(-1).int().tolist()
        assert len(flatten_symbols) == num_latents, "CDF and symbols size mismatch"
        rv = self.entropy_coder._encoder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(),
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )
        # return (rv, abs_max, zero_bitmap)
        return [rv]

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = (outputs + torch.round(means)).int()
        return outputs

class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 初始化随机正交矩阵（保证可逆）
        w_init = torch.qr(torch.randn(channels, channels))[0]
        self.weight = nn.Parameter(w_init)  # 可学习参数

    def forward(self, x, reverse=False):
        _, _, H, W = x.shape
        
        if not reverse:
            # 前向计算：y = Wx
            logdet = torch.slogdet(self.weight)[1] * H * W  # 对数行列式
            out = F.conv2d(x, self.weight.view(self.channels, self.channels, 1, 1))
            return out, logdet
        else:
            # 反向计算：x = W^{-1}y
            inv_weight = torch.inverse(self.weight)
            out = F.conv2d(x, inv_weight.view(self.channels, self.channels, 1, 1))
            return out, None




class GaussianStudentConditional_ST(nn.Module):
    r"""Polynomial laplace conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.
    """
        
    def __init__(
        self,
        scale_bound: float = 0.11,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.lower_bound_scale = LowerBound(scale_bound)
        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

    def _laplace_standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor):
        values = (half - torch.abs(inputs-means)) / scales
        exp = torch.exp(-torch.abs(values))
        return torch.where(values > 0, 2 - exp, exp) / 2
    
    def _gaussian_standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor) -> Tensor:
        const = float(-(2**-0.5))
        values = (half - torch.abs(inputs-means)) / scales
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(const * values)
    

    def _student_t_standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, df: Tensor, half: Tensor) -> Tensor:
        """
        Student-t 分布的 standardized cumulative 函数
        inputs: (Tensor) 待计算点
        means:  (Tensor) 均值
        scales: (Tensor) 尺度
        df:     (Tensor or float) 自由度 (ν)
        half:   (Tensor or float) 半宽度参数 (和 Gaussian/Laplace 版本接口保持一致)
        """
        # 标准化
        z = (half - torch.abs(inputs - means)) / scales
        df = F.softplus(df) + 2.1   # 保证 df > 2
        t2 = df / (df + z**2)
        # regularized incomplete beta
        ibeta = sp.betainc(df/2.0, 0.5, t2)

        # Student-t CDF
        cdf = 0.5 + 0.5 * torch.sign(z) * ibeta
        return cdf
        

    def _likelihood(
        self, inputs: Tensor, scales1: Tensor, means1: Tensor, scales2: Tensor, means2: Tensor, u: Tensor, weights: Tensor
    ) -> Tensor:
        half = float(0.5)
        weights = torch.sigmoid(weights)
        # print(inputs.shape)
        scales1 = self.lower_bound_scale(scales1)
        scales2 = self.lower_bound_scale(scales2)
        upper_gau = self._gaussian_standardized_cumulative(inputs, means1, scales1, half)
        lower_gau = self._gaussian_standardized_cumulative(inputs, means1, scales1, -half)
        upper_lap = self._student_t_standardized_cumulative(inputs, means2, scales2, u, half)
        lower_lap = self._student_t_standardized_cumulative(inputs, means2, scales2, u, -half)
        upper = upper_gau * weights + upper_lap * (1 - weights)
        lower = lower_gau * weights + lower_lap * (1 - weights)
        likelihood = upper - lower
        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales1: Tensor,
        means1: Tensor,
        scales2: Tensor,
        means2: Tensor,
        u: Tensor,
        weights: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        if training:
            outputs = inputs
        else:
            outputs = inputs.int()
        likelihood = self._likelihood(outputs, scales1, means1, scales2, means2, u, weights)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        reg = torch.tensor(0.).to(inputs.device)
        return outputs, likelihood, reg
    
    @torch.no_grad()
    def _build_cdf(self, scales, means, nonzero, abs_max):
        scales = scales[:, nonzero]
        means = means[:, nonzero]
        
        num_samples = abs_max * 2 + 1
        TINY = 1e-10
        device = scales.device

        scales = scales.clamp_(0.11, 256)
        means += abs_max

        scales_ = scales.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        means_ = means.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        num_latents = scales_.size(0)

        samples = (
            torch.arange(num_samples).to(device).unsqueeze(0).expand(num_latents, -1)
        )

        pmf = torch.zeros_like(samples).float()
        pmf += (
            0.5
            * (
                1
                + torch.erf(
                    (samples + 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
            - 0.5
            * (
                1
                + torch.erf(
                    (samples - 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
        )

        cdf_limit = 2**self.entropy_coder_precision - 1
        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, 1, keepdim=True).expand(-1, num_samples)

        cdf = F.pad(
            torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, 1).int(),
            (1, 0),
            "constant",
            0,
        )
        pmf_quantized = torch.diff(cdf, dim=1)

        # We can't have zeros in PMF because rANS won't be able to encode it.
        # Try to fix this by "stealing" probability from some unlikely symbols.

        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)

        _, pmf_first_stealable_indices = torch.min(
            torch.where(
                pmf_quantized > pmf_zero_count.unsqueeze(-1).expand(-1, num_samples),
                pmf_quantized,
                torch.tensor(cdf_limit + 1).int().to(device),
            ),
            dim=1,
        )

        pmf_real_zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        pmf_quantized[pmf_real_zero_indices[0], pmf_real_zero_indices[1]] += 1

        pmf_real_steal_indices = torch.cat(
            (
                torch.arange(num_latents).to(device).unsqueeze(-1),
                pmf_first_stealable_indices.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[
            pmf_real_steal_indices[0], pmf_real_steal_indices[1]
        ] -= pmf_zero_count

        cdf = F.pad(torch.cumsum(pmf_quantized, 1).int(), (1, 0), "constant", 0)
        cdf = F.pad(cdf, (0, 1), "constant", cdf_limit + 1)

        return cdf

    def compress(self, y, scales, means):
        abs_max = (
            max(torch.abs(y.max()).int().item(), torch.abs(y.min()).int().item()) + 1
        )
        abs_max = 1 if abs_max < 1 else abs_max

        zero_bitmap = torch.where(
            torch.sum(torch.abs(y), (3, 2)).squeeze(0) == 0, 0, 1
        )
        print(zero_bitmap)
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        symbols = y[:, nonzero] + abs_max
        cdf = self._build_cdf(scales, means, nonzero, abs_max)
        num_latents = cdf.size(0)
        flatten_symbols = symbols.reshape(-1).int().tolist()
        assert len(flatten_symbols) == num_latents, "CDF and symbols size mismatch"
        rv = self.entropy_coder._encoder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(),
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )
        # return (rv, abs_max, zero_bitmap)
        return [rv]

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = (outputs + torch.round(means)).int()
        return outputs


class GMMConditional_ST(nn.Module):
    r"""GMM conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.
    """
        
    def __init__(
        self,
        scale_bound: float = 0.11,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.lower_bound_scale = LowerBound(scale_bound)
        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

    def _laplace_standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor):
        values = (half - torch.abs(inputs-means)) / scales
        exp = torch.exp(-torch.abs(values))
        return torch.where(values > 0, 2 - exp, exp) / 2
    
    def _gaussian_standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor) -> Tensor:
        const = float(-(2**-0.5))
        values = (half - torch.abs(inputs-means)) / scales
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(const * values)

    def _likelihood(
        self, inputs: Tensor, scales1: Tensor, scales2: Tensor, scales3: Tensor, means1: Tensor, means2: Tensor, means3: Tensor, weights1: Tensor,
        weights2: Tensor, weights3: Tensor
    ) -> Tensor:
        half = float(0.5)
        weights = torch.cat((weights1, weights2, weights3), dim=-1)
        weights = torch.softmax(weights, dim=-1)
        weights1, weights2, weights3 = weights.chunk(3, dim=-1)
        # print(inputs.shape)
        scales1 = self.lower_bound_scale(scales1)
        scales2 = self.lower_bound_scale(scales2)
        scales3 = self.lower_bound_scale(scales3)
        # print('mean shape', means.shape, 'scales1 shape', scales1.shape, 'y shape', inputs.shape, 'weights shape', weights1.shape)
        upper_gau1 = self._gaussian_standardized_cumulative(inputs, means1, scales1, half)
        lower_gau1 = self._gaussian_standardized_cumulative(inputs, means1, scales1, -half)

        upper_gau2 = self._gaussian_standardized_cumulative(inputs, means2, scales2, half)
        lower_gau2 = self._gaussian_standardized_cumulative(inputs, means2, scales2, -half)

        upper_gau3 = self._gaussian_standardized_cumulative(inputs, means3, scales3, half)
        lower_gau3 = self._gaussian_standardized_cumulative(inputs, means3, scales3, -half)
        upper = upper_gau1 * weights1 + upper_gau2 * weights2 + upper_gau3 * weights3
        lower = lower_gau1 * weights1 + lower_gau2 * weights2 + lower_gau3 * weights3 
        likelihood = upper - lower
        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales1: Tensor,
        scales2: Tensor,
        scales3: Tensor,
        means1: Tensor,
        means2: Tensor,
        means3: Tensor,
        weights1: Tensor,
        weights2: Tensor,    
        weights3: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        if training:
            outputs = inputs
        else:
            outputs = inputs.int()
        likelihood = self._likelihood(outputs, scales1, scales2, scales3, means1, means2, means3, weights1, weights2, weights3)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood
    
    @torch.no_grad()
    def _build_cdf(self, scales, means, nonzero, abs_max):
        scales = scales[:, nonzero]
        means = means[:, nonzero]
        
        num_samples = abs_max * 2 + 1
        TINY = 1e-10
        device = scales.device

        scales = scales.clamp_(0.11, 256)
        means += abs_max

        scales_ = scales.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        means_ = means.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        num_latents = scales_.size(0)

        samples = (
            torch.arange(num_samples).to(device).unsqueeze(0).expand(num_latents, -1)
        )

        pmf = torch.zeros_like(samples).float()
        pmf += (
            0.5
            * (
                1
                + torch.erf(
                    (samples + 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
            - 0.5
            * (
                1
                + torch.erf(
                    (samples - 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
        )

        cdf_limit = 2**self.entropy_coder_precision - 1
        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, 1, keepdim=True).expand(-1, num_samples)

        cdf = F.pad(
            torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, 1).int(),
            (1, 0),
            "constant",
            0,
        )
        pmf_quantized = torch.diff(cdf, dim=1)

        # We can't have zeros in PMF because rANS won't be able to encode it.
        # Try to fix this by "stealing" probability from some unlikely symbols.

        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)

        _, pmf_first_stealable_indices = torch.min(
            torch.where(
                pmf_quantized > pmf_zero_count.unsqueeze(-1).expand(-1, num_samples),
                pmf_quantized,
                torch.tensor(cdf_limit + 1).int().to(device),
            ),
            dim=1,
        )

        pmf_real_zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        pmf_quantized[pmf_real_zero_indices[0], pmf_real_zero_indices[1]] += 1

        pmf_real_steal_indices = torch.cat(
            (
                torch.arange(num_latents).to(device).unsqueeze(-1),
                pmf_first_stealable_indices.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[
            pmf_real_steal_indices[0], pmf_real_steal_indices[1]
        ] -= pmf_zero_count

        cdf = F.pad(torch.cumsum(pmf_quantized, 1).int(), (1, 0), "constant", 0)
        cdf = F.pad(cdf, (0, 1), "constant", cdf_limit + 1)

        return cdf

    def compress(self, y, scales, means):
        abs_max = (
            max(torch.abs(y.max()).int().item(), torch.abs(y.min()).int().item()) + 1
        )
        abs_max = 1 if abs_max < 1 else abs_max

        zero_bitmap = torch.where(
            torch.sum(torch.abs(y), (3, 2)).squeeze(0) == 0, 0, 1
        )
        print(zero_bitmap)
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        symbols = y[:, nonzero] + abs_max
        cdf = self._build_cdf(scales, means, nonzero, abs_max)
        num_latents = cdf.size(0)
        flatten_symbols = symbols.reshape(-1).int().tolist()
        assert len(flatten_symbols) == num_latents, "CDF and symbols size mismatch"
        rv = self.entropy_coder._encoder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(),
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )
        # return (rv, abs_max, zero_bitmap)
        return [rv]

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = (outputs + torch.round(means)).int()
        return outputs




class PolynomialLaplaceConditional_ST2(nn.Module):
    r"""Polynomial laplace conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.
    """
        
    def __init__(
        self,
        scale_bound: float = 0.11,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.lower_bound_scale = LowerBound(scale_bound)
        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

    def _laplace_standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor):
        values = (half - torch.abs(inputs-means)) / scales
        exp = torch.exp(-torch.abs(values))
        return torch.where(values > 0, 2 - exp, exp) / 2
    
    def _gaussian_standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor) -> Tensor:
        const = float(-(2**-0.5))
        values = (half - torch.abs(inputs-means)) / scales
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(const * values)
    
    def _scale_variation_loss(self, scales: Tensor) -> Tensor:
        """
        鼓励尺度参数保持合理差异
        防止所有尺度收敛到相同值
        
        参数:
            scales: 尺度参数张量
        
        返回:
            尺度差异损失值
        """
        # 计算尺度对之间的比值
        scale_ratios = scales.unsqueeze(0) / scales.unsqueeze(1)
        
        # 我们希望尺度比值在合理范围内 (如0.5-2.0)
        target_min = 0.5
        target_max = 2.0
        loss = torch.where(
            scale_ratios < target_min,
            (scale_ratios - target_min)**2,
            torch.where(
                scale_ratios > target_max,
                (scale_ratios - target_max)**2,
                torch.zeros_like(scale_ratios)
            )
        )
        return torch.mean(loss)
    

    
    def _kl_divergence_loss(self, means1: Tensor, scales1: Tensor, means2: Tensor, scales2: Tensor) -> Tensor:
        """
        计算高斯分布和拉普拉斯分布之间的KL散度
        (注意: KL散度不对称，这里计算的是 KL(Gaussian || Laplace))
        
        Args:
            means1, scales1: 高斯分布的参数 (μ, σ)
            means2, scales2: 拉普拉斯分布的参数 (μ, b)
        
        Returns:
            KL散度损失值
        """
        # # 高斯分布的概率密度函数在拉普拉斯分布下的期望
        # # KL(p||q) = E_p[log(p(x)) - log(q(x))]
        
        # # 对于高斯分布N(μ1,σ1)和拉普拉斯分布L(μ2,b2):
        # # KL = log(b2) + σ1^2/(2b2^2) + |μ1-μ2|/b2 - 0.5*(1 + log(2πσ1^2))
        # scales1 = self.lower_bound_scale(scales1)
        # scales2 = self.lower_bound_scale(scales2)
        # b2 = scales2
        # sigma1 = scales1
        # mu_diff = torch.abs(means1 - means2)
        
        # term1 = torch.log(b2)
        # term2 = (sigma1**2) / (2 * b2**2)
        # term3 = mu_diff / b2
        # term4 = 0.5 * (1 + torch.log(2 * torch.tensor(torch.pi)) + 2 * torch.log(sigma1))
        
        # kl = term1 + term2 + term3 - term4
        """
        计算 KL( N(mu1, sigma1^2) || Laplace(mu2, b2) )
        所有输入可以是标量或张量，支持自动广播和梯度回传。
        """
        # 确保正数
        b2 = scales2
        sigma1 = scales1
        eps = 1e-9
        sigma1 = torch.clamp(sigma1, min=eps)
        b2 = torch.clamp(b2, min=eps)

        delta = means1 - means2
        abs_delta = torch.abs(delta)

        # 高斯的绝对偏差期望 E[|X - mu2|]
        # 使用误差函数 erf
        erf_arg = abs_delta / (sigma1 * math.sqrt(2.0))
        exp_term = torch.exp(-0.5 * (delta / sigma1)**2) * sigma1 * math.sqrt(2.0 / math.pi)
        erf_term = abs_delta * torch.erf(erf_arg)
        abs_dev = exp_term + erf_term  # E_p[|x - mu2|]

        # E_p[log p(x)] for Gaussian
        Ep_logp = -0.5 * torch.log(2 * math.pi * sigma1**2) - 0.5

        # E_p[log q(x)] for Laplace
        Ep_logq = -torch.log(2 * b2) - abs_dev / b2

        # KL
        kl = Ep_logp - Ep_logq

        return torch.mean(kl)
    
    def _shape_difference_loss(self, x: Tensor, means: Tensor, scales: Tensor) -> Tensor:
        """
        通过采样比较两种分布的形状差异
        
        参数:
            x: 输入数据
            means: 共享均值
            scales: 共享尺度
        
        返回:
            形状差异损失值
        """
       
        # 计算两种分布在数据点的PDF比值
        gauss_pdf = torch.exp(-0.5*((x-means)/scales)**2)/(scales*np.sqrt(2*np.pi))
        laplace_pdf = torch.exp(-torch.abs(x-means)/scales)/(2*scales)
        
        ratio = gauss_pdf / (laplace_pdf + 1e-10)
        
        # 我们希望比值在某些区域有差异 (如0.5-2.0)
        target_min = 0.5
        target_max = 2.0
        loss = torch.where(
            ratio < target_min,
            (ratio - target_min)**2,
            torch.where(
                ratio > target_max,
                (ratio - target_max)**2,
                torch.zeros_like(ratio)
            )
        )
        return torch.mean(loss)
    
    def _js_divergence_loss(self, means1: Tensor, scales1: Tensor, means2: Tensor, scales2: Tensor) -> Tensor:
        """
        计算高斯分布和拉普拉斯分布之间的JS散度
        
        Args:
            means1, scales1: 高斯分布的参数 (μ, σ)
            means2, scales2: 拉普拉斯分布的参数 (μ, b)
        
        Returns:
            JS散度损失值
        """
        # JS散度是对称的，计算方式为:
        # JS(p||q) = 0.5*(KL(p||m) + KL(q||m)), 其中 m = 0.5*(p+q)
        scales1 = self.lower_bound_scale(scales1)
        scales2 = self.lower_bound_scale(scales2)
        
        # 由于混合分布的解析解复杂，我们使用蒙特卡洛近似
        num_samples = 1000
        device = means1.device
        
        # 从高斯分布采样
        samples_p = means1 + scales1 * torch.randn(num_samples, device=device)
        
        # 从拉普拉斯分布采样
        uniforms = torch.rand(num_samples, device=device) - 0.5
        samples_q = means2 - scales2 * torch.sign(uniforms) * torch.log(1 - 2 * torch.abs(uniforms))
        
        # 合并样本
        samples_m = torch.cat([samples_p, samples_q])
        
        # 计算各分布在样本点的对数概率密度
        log_p = -0.5 * ((samples_m - means1) / scales1)**2 - torch.log(scales1) - 0.5 * torch.log(2 * torch.tensor(torch.pi))
        log_q = -torch.abs(samples_m - means2) / scales2 - torch.log(2 * scales2)
        log_m = torch.logsumexp(torch.stack([log_p, log_q]), dim=0) - torch.log(torch.tensor(2.0))
        
        # 计算KL散度
        kl_pm = torch.mean(log_p - log_m)
        kl_qm = torch.mean(log_q - log_m)
        
        js = 0.5 * (kl_pm + kl_qm)
        return js


    def _likelihood(
        self, inputs: Tensor, scales1: Tensor, means1: Tensor, scales2: Tensor, means2: Tensor, weights: Tensor
    ) -> Tensor:
        half = float(0.5)
        weights = torch.sigmoid(weights)
        # print(inputs.shape)
        scales1 = self.lower_bound_scale(scales1)
        scales2 = self.lower_bound_scale(scales2)
        upper_gau = self._gaussian_standardized_cumulative(inputs, means1, scales1, half)
        lower_gau = self._gaussian_standardized_cumulative(inputs, means1, scales1, -half)
        upper_lap = self._laplace_standardized_cumulative(inputs, means2, scales2, half)
        lower_lap = self._laplace_standardized_cumulative(inputs, means2, scales2, -half)
        upper = upper_gau * weights + upper_lap * (1 - weights)
        lower = lower_gau * weights + lower_lap * (1 - weights)
        likelihood = upper - lower
        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales1: Tensor,
        means1: Tensor,
        scales2: Tensor,
        means2: Tensor,
        weights: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        if training:
            outputs = inputs
        else:
            outputs = inputs.int()
        likelihood = self._likelihood(outputs, scales1, means1, scales2, means2, weights)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        reg = self._kl_divergence_loss(means1, scales1, means2, scales2)
        return outputs, likelihood, reg
    
    @torch.no_grad()
    def _build_cdf(self, scales, means, nonzero, abs_max):
        scales = scales[:, nonzero]
        means = means[:, nonzero]
        
        num_samples = abs_max * 2 + 1
        TINY = 1e-10
        device = scales.device

        scales = scales.clamp_(0.11, 256)
        means += abs_max

        scales_ = scales.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        means_ = means.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        num_latents = scales_.size(0)

        samples = (
            torch.arange(num_samples).to(device).unsqueeze(0).expand(num_latents, -1)
        )

        pmf = torch.zeros_like(samples).float()
        pmf += (
            0.5
            * (
                1
                + torch.erf(
                    (samples + 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
            - 0.5
            * (
                1
                + torch.erf(
                    (samples - 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
        )

        cdf_limit = 2**self.entropy_coder_precision - 1
        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, 1, keepdim=True).expand(-1, num_samples)

        cdf = F.pad(
            torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, 1).int(),
            (1, 0),
            "constant",
            0,
        )
        pmf_quantized = torch.diff(cdf, dim=1)

        # We can't have zeros in PMF because rANS won't be able to encode it.
        # Try to fix this by "stealing" probability from some unlikely symbols.

        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)

        _, pmf_first_stealable_indices = torch.min(
            torch.where(
                pmf_quantized > pmf_zero_count.unsqueeze(-1).expand(-1, num_samples),
                pmf_quantized,
                torch.tensor(cdf_limit + 1).int().to(device),
            ),
            dim=1,
        )

        pmf_real_zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        pmf_quantized[pmf_real_zero_indices[0], pmf_real_zero_indices[1]] += 1

        pmf_real_steal_indices = torch.cat(
            (
                torch.arange(num_latents).to(device).unsqueeze(-1),
                pmf_first_stealable_indices.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[
            pmf_real_steal_indices[0], pmf_real_steal_indices[1]
        ] -= pmf_zero_count

        cdf = F.pad(torch.cumsum(pmf_quantized, 1).int(), (1, 0), "constant", 0)
        cdf = F.pad(cdf, (0, 1), "constant", cdf_limit + 1)

        return cdf

    def compress(self, y, scales, means):
        abs_max = (
            max(torch.abs(y.max()).int().item(), torch.abs(y.min()).int().item()) + 1
        )
        abs_max = 1 if abs_max < 1 else abs_max

        zero_bitmap = torch.where(
            torch.sum(torch.abs(y), (3, 2)).squeeze(0) == 0, 0, 1
        )
        print(zero_bitmap)
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        symbols = y[:, nonzero] + abs_max
        cdf = self._build_cdf(scales, means, nonzero, abs_max)
        num_latents = cdf.size(0)
        flatten_symbols = symbols.reshape(-1).int().tolist()
        assert len(flatten_symbols) == num_latents, "CDF and symbols size mismatch"
        rv = self.entropy_coder._encoder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(),
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )
        # return (rv, abs_max, zero_bitmap)
        return [rv]

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = (outputs + torch.round(means)).int()
        return outputs



class PolynomialLaplaceConditional_ST(nn.Module):
    r"""Polynomial laplace conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.
    """
        
    def __init__(
        self,
        scale_bound: float = 0.11,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.lower_bound_scale = LowerBound(scale_bound)
        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

    def _laplace_standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor):
        values = (half - torch.abs(inputs-means)) / scales
        exp = torch.exp(-torch.abs(values))
        return torch.where(values > 0, 2 - exp, exp) / 2
    
    def _gaussian_standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor) -> Tensor:
        const = float(-(2**-0.5))
        values = (half - torch.abs(inputs-means)) / scales
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(const * values)
    
    def _distribution_overlap_loss(self, means: Tensor, scales: Tensor) -> Tensor:
        
        # # 计算所有分布对之间的距离
        # pairwise_dist = torch.cdist(means.unsqueeze(-1), means.unsqueeze(-1))
        
        # 计算平均尺度作为参考
        avg_scale = scales.mean()
        
        # 我们希望分布之间的距离与尺度成比例
        target_dist = 2.0 * avg_scale  # 2倍尺度作为理想距离
    
        return torch.mean((0 - target_dist)**2)
        # """
        # 鼓励尺度参数保持合理差异
        # 防止所有尺度收敛到相同值
        
        # 参数:
        #     scales: 尺度参数张量
        
        # 返回:
        #     尺度差异损失值
        # """
        # min_val = 0.2
        # max_val = 2.0
        # scales = torch.clamp(scales, min=0.11)
        # # 映射到 [0,1] 区间
        # normed = (scales - min_val) / (max_val - min_val)
        # # 惩罚在 [0,1] 之外的区域
        # sharpness=10.0
        # penalty = torch.sigmoid(-sharpness * normed) + torch.sigmoid(sharpness * (normed - 1))
        # return torch.mean(penalty)

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Tensor, weights: Tensor
    ) -> Tensor:
        half = float(0.5)
        weights = torch.sigmoid(weights)
        # print(inputs.shape)
        scales = self.lower_bound_scale(scales)
        upper_gau = self._gaussian_standardized_cumulative(inputs, means, scales, half)
        lower_gau = self._gaussian_standardized_cumulative(inputs, means, scales, -half)
        upper_lap = self._laplace_standardized_cumulative(inputs, means, scales, half)
        lower_lap = self._laplace_standardized_cumulative(inputs, means, scales, -half)
        upper = upper_gau * weights + upper_lap * (1 - weights)
        lower = lower_gau * weights + lower_lap * (1 - weights)
        likelihood = upper - lower
        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Tensor,
        weights: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        if training:
            outputs = inputs
        else:
            outputs = inputs.int()
        likelihood = self._likelihood(outputs, scales, means, weights)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        reg = self._distribution_overlap_loss(means, scales)
        return outputs, likelihood, reg
    
    @torch.no_grad()
    def _build_cdf(self, scales, means, nonzero, abs_max):
        scales = scales[:, nonzero]
        means = means[:, nonzero]
        
        num_samples = abs_max * 2 + 1
        TINY = 1e-10
        device = scales.device

        scales = scales.clamp_(0.11, 256)
        means += abs_max

        scales_ = scales.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        means_ = means.reshape(-1).unsqueeze(-1).expand(-1, num_samples)
        num_latents = scales_.size(0)

        samples = (
            torch.arange(num_samples).to(device).unsqueeze(0).expand(num_latents, -1)
        )

        pmf = torch.zeros_like(samples).float()
        pmf += (
            0.5
            * (
                1
                + torch.erf(
                    (samples + 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
            - 0.5
            * (
                1
                + torch.erf(
                    (samples - 0.5 - means_) / ((scales_ + TINY) * 2**0.5)
                )
            )
        )

        cdf_limit = 2**self.entropy_coder_precision - 1
        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, 1, keepdim=True).expand(-1, num_samples)

        cdf = F.pad(
            torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, 1).int(),
            (1, 0),
            "constant",
            0,
        )
        pmf_quantized = torch.diff(cdf, dim=1)

        # We can't have zeros in PMF because rANS won't be able to encode it.
        # Try to fix this by "stealing" probability from some unlikely symbols.

        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)

        _, pmf_first_stealable_indices = torch.min(
            torch.where(
                pmf_quantized > pmf_zero_count.unsqueeze(-1).expand(-1, num_samples),
                pmf_quantized,
                torch.tensor(cdf_limit + 1).int().to(device),
            ),
            dim=1,
        )

        pmf_real_zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        pmf_quantized[pmf_real_zero_indices[0], pmf_real_zero_indices[1]] += 1

        pmf_real_steal_indices = torch.cat(
            (
                torch.arange(num_latents).to(device).unsqueeze(-1),
                pmf_first_stealable_indices.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[
            pmf_real_steal_indices[0], pmf_real_steal_indices[1]
        ] -= pmf_zero_count

        cdf = F.pad(torch.cumsum(pmf_quantized, 1).int(), (1, 0), "constant", 0)
        cdf = F.pad(cdf, (0, 1), "constant", cdf_limit + 1)

        return cdf

    def compress(self, y, scales, means):
        abs_max = (
            max(torch.abs(y.max()).int().item(), torch.abs(y.min()).int().item()) + 1
        )
        abs_max = 1 if abs_max < 1 else abs_max

        zero_bitmap = torch.where(
            torch.sum(torch.abs(y), (3, 2)).squeeze(0) == 0, 0, 1
        )
        print(zero_bitmap)
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        symbols = y[:, nonzero] + abs_max
        cdf = self._build_cdf(scales, means, nonzero, abs_max)
        num_latents = cdf.size(0)
        flatten_symbols = symbols.reshape(-1).int().tolist()
        assert len(flatten_symbols) == num_latents, "CDF and symbols size mismatch"
        rv = self.entropy_coder._encoder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(),
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )
        # return (rv, abs_max, zero_bitmap)
        return [rv]

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = (outputs + torch.round(means)).int()
        return outputs



class LogisticConditional(GaussianConditional_ST):
  """Conditional logistic entropy model.

  This is a conditionally Logistic entropy model, analogous to
  `GaussianConditional`.
  """

  def _standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor):
    values = (half - torch.abs(inputs-means)) / scales
    return torch.sigmoid(values)

  def _standardized_quantile(self, quantile):
    return scipy.stats.logistic.ppf(quantile)


class LaplacianConditional(GaussianConditional_ST):
  """Conditional Laplacian entropy model.

  This is a conditionally Laplacian entropy model, analogous to
  `GaussianConditional`.
  """

  def _standardized_cumulative(self, inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor):
    values = (half - torch.abs(inputs-means)) / scales
    exp = torch.exp(-torch.abs(values))
    return torch.where(values > 0, 2 - exp, exp) / 2

  def _standardized_quantile(self, quantile):
    return scipy.stats.laplace.ppf(quantile)


# def standardized_cumulative(x):
#     """ 分段多项式近似（类似JPEG XL的方案） """
#     # 预计算常数
#     const1 = torch.tensor(1.0 / (2 * 2.685035))
#     const2 = torch.tensor(1.0 / (2 * 2.685035 + 0.5))
    
#     # 核心计算
#     abs_x = torch.abs(x)
#     mask = abs_x < 1.5
#     y = torch.where(
#         mask,
#         x * (0.5 + x * x * (const1 - const2 * abs_x)),
#         torch.sign(x) * (1 - torch.exp(-abs_x * 2.685035))
#     )
#     return 0.5 * (1 + y)



##################GMM##################

class GaussianMixtureConditional(GaussianConditional):
    def __init__(
        self,
        K=3,
        scale_table: Optional[Union[List, Tuple]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(scale_table, *args, **kwargs)

        self.K = K

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Tensor, weights: Tensor
    ) -> Tensor:
        likelihood = torch.zeros_like(inputs)
        M = inputs.size(1)

        for k in range(self.K):
            likelihood += (
                super()._likelihood(
                    inputs,
                    scales[:, M * k : M * (k + 1)],
                    means[:, M * k : M * (k + 1)],
                )
                * weights[:, M * k : M * (k + 1)]
            )

        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Tensor,
        weights: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs = self.quantize(
            inputs, "noise" if training else "dequantize", means=None
        )
        likelihood = self._likelihood(outputs, scales, means, weights)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    @torch.no_grad()
    def _build_cdf(self, scales, means, weights, abs_max):
        num_latents = scales.size(1)
        num_samples = abs_max * 2 + 1
        TINY = 1e-10
        device = scales.device

        scales = scales.clamp_(0.11, 256)
        means += abs_max

        scales_ = scales.unsqueeze(-1).expand(-1, -1, num_samples)
        means_ = means.unsqueeze(-1).expand(-1, -1, num_samples)
        weights_ = weights.unsqueeze(-1).expand(-1, -1, num_samples)

        samples = (
            torch.arange(num_samples).to(device).unsqueeze(0).expand(num_latents, -1)
        )

        pmf = torch.zeros_like(samples).float()
        for k in range(self.K):
            pmf += (
                0.5
                * (
                    1
                    + torch.erf(
                        (samples + 0.5 - means_[k]) / ((scales_[k] + TINY) * 2**0.5)
                    )
                )
                - 0.5
                * (
                    1
                    + torch.erf(
                        (samples - 0.5 - means_[k]) / ((scales_[k] + TINY) * 2**0.5)
                    )
                )
            ) * weights_[k]

        cdf_limit = 2**self.entropy_coder_precision - 1
        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, 1, keepdim=True).expand(-1, num_samples)

        cdf = F.pad(
            torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, 1).int(),
            (1, 0),
            "constant",
            0,
        )
        pmf_quantized = torch.diff(cdf, dim=1)

        # We can't have zeros in PMF because rANS won't be able to encode it.
        # Try to fix this by "stealing" probability from some unlikely symbols.

        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)

        _, pmf_first_stealable_indices = torch.min(
            torch.where(
                pmf_quantized > pmf_zero_count.unsqueeze(-1).expand(-1, num_samples),
                pmf_quantized,
                torch.tensor(cdf_limit + 1).int(),
            ),
            dim=1,
        )

        pmf_real_zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        pmf_quantized[pmf_real_zero_indices[0], pmf_real_zero_indices[1]] += 1

        pmf_real_steal_indices = torch.cat(
            (
                torch.arange(num_latents).to(device).unsqueeze(-1),
                pmf_first_stealable_indices.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[
            pmf_real_steal_indices[0], pmf_real_steal_indices[1]
        ] -= pmf_zero_count

        cdf = F.pad(torch.cumsum(pmf_quantized, 1).int(), (1, 0), "constant", 0)
        cdf = F.pad(cdf, (0, 1), "constant", cdf_limit + 1)

        return cdf

    def reshape_entropy_parameters(self, scales, means, weights, nonzero):
        reshape_size = (scales.size(0), self.K, scales.size(1) // self.K, -1)

        scales = (
            scales.reshape(*reshape_size)[:, :, nonzero]
            .permute(1, 0, 2, 3)
            .reshape(self.K, -1)
        )
        means = (
            means.reshape(*reshape_size)[:, :, nonzero]
            .permute(1, 0, 2, 3)
            .reshape(self.K, -1)
        )
        weights = (
            weights.reshape(*reshape_size)[:, :, nonzero]
            .permute(1, 0, 2, 3)
            .reshape(self.K, -1)
        )
        return scales, means, weights

    def compress(self, y, scales, means, weights):
        abs_max = (
            max(torch.abs(y.max()).int().item(), torch.abs(y.min()).int().item()) + 1
        )
        abs_max = 1 if abs_max < 1 else abs_max

        y_quantized = torch.round(y)
        zero_bitmap = torch.where(
            torch.sum(torch.abs(y_quantized), (3, 2)).squeeze(0) == 0, 0, 1
        )

        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        symbols = y_quantized[:, nonzero] + abs_max
        cdf = self._build_cdf(
            *self.reshape_entropy_parameters(scales, means, weights, nonzero), abs_max
        )

        num_latents = cdf.size(0)

        # rv = self.entropy_coder._encoder.encode_with_indexes(
        #     symbols.reshape(-1).int().tolist(),
        #     torch.arange(num_latents).int().tolist(),
        #     cdf.cpu().to(torch.int32),
        #     torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
        #     torch.tensor(0).repeat(num_latents).int().tolist(),
        # )
        rv = self.entropy_coder._encoder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(),
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )

        return (rv, abs_max, zero_bitmap), y_quantized

    def decompress(self, strings, abs_max, zero_bitmap, scales, means, weights):
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        cdf = self._build_cdf(
            *self.reshape_entropy_parameters(scales, means, weights, nonzero), abs_max
        )

        num_latents = cdf.size(0)

        # values = self.entropy_coder._decoder.decode_with_indexes(
        #     strings,
        #     torch.arange(num_latents).int().tolist(),
        #     cdf.cpu().to(torch.int32),
        #     torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
        #     torch.tensor(0).repeat(num_latents).int().tolist(),
        # )
        values = self.entropy_coder._decoder.decode_with_indexes(
            strings,
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )

        symbols = torch.tensor(values) - abs_max
        symbols = symbols.reshape(scales.size(0), -1, scales.size(2), scales.size(3))

        y_hat = torch.zeros(
            scales.size(0), zero_bitmap.size(0), scales.size(2), scales.size(3)
        )
        y_hat[:, nonzero] = symbols.float()

        return y_hat
