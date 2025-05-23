from contextlib import nullcontext
import copy
import os
import pickle
from tempfile import TemporaryDirectory

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.nn.modules import Linear8bitLt
from tests.helpers import (
    TRUE_FALSE,
    get_available_devices,
    id_formatter,
    torch_load_from_buffer,
    torch_save_to_buffer,
)


# contributed by Alex Borzunov, see:
# https://github.com/bigscience-workshop/petals/blob/main/tests/test_linear8bitlt.py
@pytest.mark.parametrize("device", get_available_devices())
def test_linear_no_igemmlt(device):
    linear = torch.nn.Linear(1024, 3072)
    x = torch.randn(3, 1024, dtype=torch.half)
    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
    )

    # TODO: Remove, this is no longer implemented
    linear_custom.state.force_no_igemmlt = True

    linear_custom.weight = bnb.nn.Int8Params(
        linear.weight.data.clone(),
        requires_grad=False,
        has_fp16_weights=False,
    ).to(linear.weight.dtype)
    linear_custom.bias = linear.bias
    linear_custom = linear_custom.to(device)
    linear = linear.half().to(device)

    x_ref = x.clone().to(device).requires_grad_(True)
    x_ours = x.clone().to(device).requires_grad_(True)
    fx_ref = linear(x_ref).float()
    grad_proj = torch.randn_like(fx_ref)
    (fx_ref * grad_proj).mean().backward()

    fx_ours = linear_custom(x_ours).float()
    (fx_ours * grad_proj).mean().backward()

    assert linear_custom.state.CB is not None
    assert not linear_custom.state.has_fp16_weights

    idx = torch.isclose(fx_ref, fx_ours, atol=0.02, rtol=1e-5)
    assert (idx == 0).sum().item() < fx_ref.numel() * 2.5e-4
    torch.testing.assert_close(fx_ref, fx_ours, atol=0.03, rtol=1e-5)
    torch.testing.assert_close(x_ref.grad, x_ours.grad, atol=0.01, rtol=1e-5)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("has_fp16_weights", TRUE_FALSE, ids=id_formatter("has_fp16_weights"))
@pytest.mark.parametrize("threshold", [0.0, 6.0], ids=id_formatter("threshold"))
@pytest.mark.parametrize("serialize_before_forward", TRUE_FALSE, ids=id_formatter("serialize_before_forward"))
@pytest.mark.parametrize("deserialize_before_cuda", TRUE_FALSE, ids=id_formatter("deserialize_before_cuda"))
@pytest.mark.parametrize("save_before_forward", TRUE_FALSE, ids=id_formatter("save_before_forward"))
@pytest.mark.parametrize("load_before_cuda", TRUE_FALSE, ids=id_formatter("load_before_cuda"))
def test_linear_serialization(
    device,
    has_fp16_weights,
    threshold,
    serialize_before_forward,
    deserialize_before_cuda,
    save_before_forward,
    load_before_cuda,
):
    if device != "cuda" and has_fp16_weights:
        pytest.skip("has_fp16_weights is only supported on CUDA and is deprecated")

    linear = torch.nn.Linear(32, 96)
    # TODO: Fallback for bad shapes
    x = torch.randn(4, 32, dtype=torch.half)
    # x = torch.randn(3, 32, dtype=torch.half)

    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=has_fp16_weights,
        threshold=threshold,
    )

    linear_custom.weight = bnb.nn.Int8Params(
        linear.weight.data.clone(),
        requires_grad=has_fp16_weights,
        has_fp16_weights=has_fp16_weights,
    )
    linear_custom.bias = linear.bias
    linear_custom = linear_custom.to(device)

    if serialize_before_forward:
        state_dict_8bit = linear_custom.state_dict()

    if save_before_forward:
        bytes_8bit = torch_save_to_buffer(linear_custom)

    x_first = x.clone().to(device).requires_grad_(True)
    fx_first = linear_custom(x_first).float()
    grad_proj = torch.randn_like(fx_first)
    (fx_first * grad_proj).mean().backward()

    if not serialize_before_forward:
        state_dict_8bit = linear_custom.state_dict()

    if not save_before_forward:
        bytes_8bit = torch_save_to_buffer(linear_custom)

    with TemporaryDirectory() as tmpdir:
        state_path_8bit = os.path.join(tmpdir, "state_8bit.pth")
        state_path = os.path.join(tmpdir, "state.pth")

        torch.save(linear.state_dict(), state_path)
        torch.save(state_dict_8bit, state_path_8bit)

        if not has_fp16_weights:
            assert os.path.getsize(state_path_8bit) < 0.5 * os.path.getsize(state_path)

        new_state_dict = torch.load(state_path_8bit, weights_only=False)

    new_linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=has_fp16_weights,
        threshold=threshold,
    )

    if deserialize_before_cuda:
        with nullcontext() if has_fp16_weights else pytest.raises(RuntimeError):
            new_linear_custom.load_state_dict(new_state_dict, strict=True)

    if load_before_cuda:
        new_linear_custom2 = torch_load_from_buffer(bytes_8bit)

    new_linear_custom = new_linear_custom.to(device)

    if not deserialize_before_cuda:
        new_linear_custom.load_state_dict(new_state_dict, strict=True)

    if not load_before_cuda:
        new_linear_custom2 = torch_load_from_buffer(bytes_8bit)

    x_second = x.clone().to(device).requires_grad_(True)
    fx_second = new_linear_custom(x_second).float()
    (fx_second * grad_proj).mean().backward()

    x_third = x.clone().to(device).requires_grad_(True)
    fx_third = new_linear_custom2(x_third).float()
    (fx_third * grad_proj).mean().backward()

    # if 8-bit weights were loaded before .cuda, state is incorrect anyway and RuntimeError was raised
    if has_fp16_weights or not deserialize_before_cuda:
        assert torch.allclose(fx_first, fx_second, atol=1e-5)
        assert torch.allclose(x_first.grad, x_second.grad, atol=1e-5)
    assert torch.allclose(fx_first, fx_third, atol=1e-5)
    assert torch.allclose(x_first.grad, x_third.grad, atol=1e-5)


@pytest.fixture
def linear8bit(requires_cuda):
    linear = torch.nn.Linear(32, 96)
    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
    )
    linear_custom.weight = bnb.nn.Int8Params(
        linear.weight.data.clone(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    linear_custom.bias = linear.bias
    linear_custom = linear_custom.cuda()
    return linear_custom


def test_linear8bit_copy_param(linear8bit):
    shallow_copy = copy.copy(linear8bit)
    assert linear8bit.weight is shallow_copy.weight
    assert linear8bit.bias is shallow_copy.bias
    assert linear8bit.weight.data.data_ptr() == shallow_copy.weight.data.data_ptr()


def test_linear8bit_deepcopy_param(linear8bit):
    deep_copy = copy.deepcopy(linear8bit)
    assert linear8bit.weight is not deep_copy.weight
    assert linear8bit.bias is not deep_copy.bias
    assert linear8bit.weight.data.data_ptr() != deep_copy.weight.data.data_ptr()
    assert torch.allclose(linear8bit.weight.data, deep_copy.weight.data)
    assert linear8bit.state == deep_copy.state

    # check for a bug where SCB and CB were not copied
    assert deep_copy.weight.SCB is not None
    assert (linear8bit.weight.SCB == deep_copy.weight.SCB).all()
    assert deep_copy.weight.CB is not None
    assert (linear8bit.weight.CB == deep_copy.weight.CB).all()


def test_linear8bit_serialization(linear8bit):
    serialized = pickle.dumps(linear8bit)
    deserialized = pickle.loads(serialized)
    assert linear8bit.weight.data.data_ptr() != deserialized.weight.data.data_ptr()
    assert torch.allclose(linear8bit.weight.data, deserialized.weight.data)
    assert linear8bit.bias.data.data_ptr() != deserialized.bias.data.data_ptr()
    assert torch.allclose(linear8bit.bias.data, deserialized.bias.data)
    assert linear8bit.state == deserialized.state

    # check for a bug where SCB and CB were not copied
    assert (linear8bit.weight.SCB == deserialized.weight.SCB).all()
    assert (linear8bit.weight.CB == deserialized.weight.CB).all()
