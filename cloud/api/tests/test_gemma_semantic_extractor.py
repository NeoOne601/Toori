from __future__ import annotations

import torch

from cloud.perception.gemma_semantic_extractor import _unpack_affine_row


def test_unpack_affine_row_restores_group_values():
    packed = torch.tensor([0x76543210, 0xFEDCBA98], dtype=torch.uint32)
    scales = torch.tensor([2.0 / 64.0], dtype=torch.bfloat16)
    biases = torch.tensor([-1.0], dtype=torch.bfloat16)

    restored = _unpack_affine_row(packed, scales, biases)

    assert restored.shape == (16,)
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.float32)
    expected = (expected * float(scales[0])) + float(biases[0])
    assert torch.allclose(torch.from_numpy(restored), expected, atol=1e-5)
