from typing import Callable, Union
import warnings

import torch
import torch.nn as nn

import kornia
from kornia.augmentation.base import _AugmentationBase, MixAugmentationBase
from kornia.augmentation import (
    CenterCrop,
    RandomCrop,
    RandomResizedCrop,
    CenterCrop3D,
    RandomCrop3D
)


class TTABase(nn.Sequential):
    r"""Test Time Augmentation (TTA) to increase of the testing set by different augmentations.

    TTA smoothed predictions by aggregating several transformed testing samples. It has three stages:
    transform, inference and aggregate results. Specifically, for non-classification tasks, it normally
    requires to "inverse the geometric transform" prior to the aggregation.

    Args:
        *args (_AugmentationBase): a list of augmentation module.
        aggregate_fn (Callable or str): the aggregation function to aggregate the TTA results.
            If Callable, it should accept :math:`(T, B, ...)` where T is how many times that TTA applied.
            If str, only "mean" is implemented. Default: "mean".

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> input = torch.randn(2, 3, 3, 4)
        >>> tta = TTABase(
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.augmentation.ColorJitter(p=1.0),
        ... )
        >>> output = tta.apply(input)
        >>> inversed = tta.detransform(output)
    """

    def __init__(self, *args: _AugmentationBase, aggregate_fn: Union[Callable, str] = 'mean') -> None:
        super(TTABase, self).__init__(*args)
        self.transforms: torch.Tensor = None
        if isinstance(aggregate_fn, str):
            if aggregate_fn == 'mean':
                self.aggregate_fn = self.average_aggregation
            else:
                raise NotImplementedError(aggregate_fn)
        else:
            self.aggregate_fn = aggregate_fn
        self._crop_augmentations = (RandomCrop, CenterCrop, RandomResizedCrop, RandomCrop3D, CenterCrop3D)
        for aug in args:
            aug.return_transform = True
            if isinstance(aug, MixAugmentationBase):
                raise NotImplementedError(f"`MixAugmentations` are not supported at this moment. Got {aug}.")
            if isinstance(aug, self._crop_augmentations):
                warnings.warn(f"{aug} is not recommended apart from classification tasks.")

    def forward(self, input: torch.Tensor):
        raise NotImplementedError

    def apply(self, input: torch.Tensor, tta_times: int = 3) -> torch.Tensor:
        """Apply TTA on input. (B, C, H, W) => (B * tta_times, C, H, W).
        """
        assert len(input.shape) == 4, f"Input must be (B, C, H, W). Got {input.shape}."
        input = torch.cat([input] * tta_times)
        input, self.transforms = super().forward(input)
        return input

    def detransform(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Undo the geometric transformations.
        """
        assert len(input.shape) == 4, f"Input must be (B, C, H, W). Got {input.shape}."
        assert input.size(0) == self.transforms.size(0), (
            "Input must have the same batch size as recorded transforms. "
            f"Got {input.size(0)} and {self.transforms.size(0)}.")
        return torch.cat([
            kornia.warp_affine(tensor[None], trans.inverse()[None, :2], dsize=tuple(tensor.shape[-2:]), **kwargs)
            for tensor, trans in zip(input, self.transforms)
        ])

    def average_aggregation(self, input: torch.Tensor) -> torch.Tensor:
        """Perform mean aggregation on the first dimension."""
        return torch.mean(input, dim=0)


class TTAClassificationWrapper(TTABase):
    """Test Time Augmentation (TTA) for classification tasks.

    TTA smoothed predictions by aggregating several transformed testing samples. It has three stages:
    transform, inference and aggregate results.

    Args:
        *args (_AugmentationBase): a list of augmentation module.
        aggregate_fn (Callable or str): the aggregation function to aggregate the TTA results.
            If Callable, it should accept :math:`(T, B, ...)` where T is how many times that TTA applied.
            If str, only "mean" is implemented. Default: "mean".

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> input = torch.randn(2, 1, 3, 4)
        >>> tta = TTAClassificationWrapper(
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ... )
        >>> model = lambda x: nn.AdaptiveAvgPool2d(1)(x).squeeze()
        >>> tta(model, input, tta_times=10)
        tensor([-0.0681,  0.2209])
    """

    def forward(self, model: Callable, input: torch.Tensor, tta_times: int = 16) -> torch.Tensor:
        """Perform TTA.

        Args:
            model (Callable): the model for inference.
            input (torch.Tensor): the input tensor for TTA that shaped as :math:`(B, C, H, W)`.
            tta_times (int): the times of applying random transformations for inferencing.
                Normally, the larger, the better.
        """
        input = self.apply(input, tta_times)
        output = model(input)
        output = output.view(tta_times, input.size(0) // tta_times, *output.shape[1:])
        return self.aggregate_fn(output)


class TTASegmentationWrapper(TTABase):
    """Test Time Augmentation (TTA) for segmentation tasks.

    TTA smoothed predictions by aggregating several transformed testing samples. It has three stages:
    transform, inference and aggregate results. Specifically, for segmentation tasks, it normally
    requires to "inverse the geometric transform" prior to the aggregation.

    Args:
        *args (_AugmentationBase): a list of augmentation module.
        aggregate_fn (Callable or str): the aggregation function to aggregate the TTA results.
            If Callable, it should accept :math:`(T, B, ...)` where T is how many times that TTA applied.
            If str, only "mean" is implemented. Default: "mean".

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> input = torch.randn(2, 3, 3, 4)
        >>> tta = TTASegmentationWrapper(
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.augmentation.ColorJitter(p=1.0),
        ... )
        >>> model = nn.Conv2d(3, 3, kernel_size=(2, 3))
        >>> tta(model, input, tta_times=3)
        tensor([[[[ 6.8983e-02, -1.2261e-01],
                  [ 1.8148e-01,  1.8101e-01]],
        <BLANKLINE>
                 [[ 7.6052e-02,  2.3324e-02],
                  [-4.1025e-02,  7.4649e-02]],
        <BLANKLINE>
                 [[-3.4549e-02,  3.4902e-02],
                  [-2.1821e-01, -1.8697e-01]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[ 9.3109e-03,  6.8537e-03],
                  [ 9.8155e-02,  3.5413e-05]],
        <BLANKLINE>
                 [[-4.5008e-03, -4.6529e-03],
                  [-4.3216e-02, -5.9058e-02]],
        <BLANKLINE>
                 [[-2.4113e-02, -2.5761e-02],
                  [-1.8114e-01, -8.6980e-02]]]], grad_fn=<MeanBackward1>)
    """

    def forward(self, model: Callable, input: torch.Tensor, tta_times: int = 16, **kwargs) -> torch.Tensor:
        """Perform TTA.

        Args:
            model (Callable): the model for inference.
            input (torch.Tensor): the input tensor for TTA that shaped as :math:`(B, C, H, W)`.
            tta_times (int): the times of applying random transformations for inferencing.
                Normally, the larger, the better.
        """
        input = self.apply(input, tta_times)
        output = model(input)
        output = self.detransform(output, **kwargs)
        output = output.view(tta_times, input.size(0) // tta_times, *output.shape[1:])
        return self.aggregate_fn(output)
