from typing import Optional

import torch


def get_coords_by_mask(mask: torch.Tensor, exclude_diagnal: bool = True, without_pairs: bool = True):
    """Get the coordinates from a 2D mask where True values occured.

    In tasks like image retrieval

    Example:
        >>> mask = torch.randn(3, 4) > 0.5
        >>> _ = get_coords_by_mask(mask)
    """
    assert len(mask.shape) == 2 and mask.shape[0] == mask.shape[1], f"Expected an N x N matrix. Got {mask.shape}."

    if without_pairs and exclude_diagnal:
        mask = torch.tril(mask, diagonal=-1)
    elif without_pairs:
        mask = torch.tril(mask, diagonal=1)
    elif exclude_diagnal:
        mask = mask.fill_diagonal_(fill_value=False, wrap=False)

    index_matrix_0 = torch.arange(0, mask.shape[1]).unsqueeze(0).expand(mask.shape[0], -1)
    index_matrix_1 = torch.arange(0, mask.shape[0]).unsqueeze(1).expand(-1, mask.shape[1])

    idx_0 = torch.masked_select(index_matrix_1, mask)
    idx_1 = torch.masked_select(index_matrix_0, mask)
    return torch.stack([idx_0, idx_1])


def iterate_cosine(
    features: torch.Tensor, max_block_size: Optional[int] = None, output_device: Optional[torch.device] = None
) -> torch.Tensor:
    """Compute cosine similarity across the input features iteratively.

    Compute the similarity matrix iteratively across the input feature vector.
    Input :math:`(B, n_feat)` and output :math:`(B, B)`.

    Args:
        features (torch.Tensor): input features shaped as :math:`(B, n_feat)`.
        max_block_size (int, optional): max number of samples for each block. Especially useful
            when a very large features obtained to avoid OOM. Default is None, which will compute
            over the entire feature vector.
        output_device (device, optional): device that stores the output tensor.

    Returns:
        torch.Tensor: output a tensor that shaped as :math:`(B, B)`.

    Example:
        >>> from torch.testing import assert_allclose
        >>> out = torch.randn(16, 128)
        >>> _ = assert_allclose(iterate_cosine(out, max_block_size=4), iterate_cosine(out))
    """
    assert len(features.shape) == 2, \
        f"features must be shaped as (B, n_feat). Got {features.shape}."

    def _normalize(A: torch.Tensor) -> torch.Tensor:
        lengths = (A ** 2).sum(axis=1, keepdims=True) ** .5
        return A / lengths

    def _cosine_similarity_cross(feat_1: torch.Tensor, feat_2: torch.Tensor) -> torch.Tensor:
        feat_1 = _normalize(feat_1)
        feat_2 = _normalize(feat_2)
        return torch.matmul(feat_1, feat_2.T)

    def _cosine_similarity_single(feat: torch.Tensor) -> torch.Tensor:
        feat = _normalize(feat)
        return torch.matmul(feat, feat.T)

    if max_block_size is None or max_block_size >= len(features):
        # If it can be computed in one goal.
        out = _cosine_similarity_single(features)
        return out

    if output_device is None:
        # If max_block_size is assigned, we prefer saving more GPU memories.
        output_device = torch.device("cpu")

    splits = len(features) // max_block_size

    output = torch.empty((len(features), len(features)), device=output_device, dtype=features.dtype)

    # Generate an NxN matrix. Optimized by:
    # 1. compute and assign diagnal only once
    # 2. symetric matrix M[i,j] == M[j,i]
    for i in range(splits):
        for j in range(i, splits):
            if i != j:
                seg_output = _cosine_similarity_cross(
                    features[max_block_size * i:max_block_size * (i + 1)],
                    features[max_block_size * j:max_block_size * (j + 1)]
                )
                output[
                    max_block_size * i:max_block_size * (i + 1),
                    max_block_size * j:max_block_size * (j + 1)
                ] = seg_output
                output[
                    max_block_size * j:max_block_size * (j + 1),
                    max_block_size * i:max_block_size * (i + 1)
                ] = seg_output.T
            else:  # i == j, the diagnal
                seg_output = _cosine_similarity_single(
                    features[max_block_size * j:max_block_size * (j + 1)]
                )
                output[
                    max_block_size * i:max_block_size * (i + 1),
                    max_block_size * j:max_block_size * (j + 1)
                ] = seg_output

    return output
