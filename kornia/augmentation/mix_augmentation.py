from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
import torch.nn as nn
from torch.nn.functional import pad

from kornia.augmentation.augmentation import AugmentationBase
from kornia.constants import Resample, BorderType
import kornia.augmentation.functional as F
import kornia.augmentation.random_generator as rg
from kornia.augmentation.utils import (
    _infer_batch_shape
)


class MixAugmentation(AugmentationBase):
    r"""MixAugmentation base class for customized augmentation implementations. For any augmentation,
    the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    In "apply_transform", both input and label tensors are required.

    Args:
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.

    """
    def __init__(self):
        super(MixAugmentation, self).__init__(return_transform=False)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,     # type: ignore
                        params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:   # type: ignore
        raise NotImplementedError

    def forward(
        self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # type: ignore
        label: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None,
        return_transform: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:  # type: ignore
        if return_transform is None:
            return_transform = self.return_transform
        if params is None:
            batch_shape = self.infer_batch_shape(input)
            self._params = self.generate_parameters(batch_shape)
        else:
            self._params = params

        if isinstance(input, tuple):
            output = self.apply_transform(input[0], label, self._params)
            transformation_matrix = self.compute_transformation(input[0], self._params)
            if return_transform:
                return output, input[1] @ transformation_matrix
            else:
                return output, input[1]

        output = self.apply_transform(input, label, self._params)
        if return_transform:
            transformation_matrix = self.compute_transformation(input, self._params)
            return output, transformation_matrix
        return output


class RandomMixUp(MixAugmentation):
    """
    Implemention for `mixup: BEYOND EMPIRICAL RISK MINIMIZATION <https://arxiv.org/pdf/1710.09412.pdf>`.
    The function returns (inputs, labels), in which the inputs is the tensor that contains the mixup images
    while the labels is a :math:`(B, 3)` tensor that contains (label_batch, label_permuted_batch, lambda) for
    each image. The implementation is on top of `https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py`.
    The loss and accuracy are computed as:
        ```
        def loss_mixup(y, logits):
            criterion = F.cross_entropy
            loss_a = criterion(logits, y[:, 0].long(), reduction='none')
            loss_b = criterion(logits, y[:, 1].long(), reduction='none')
            return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()

        def acc_mixup(y, logits):
            pred = torch.argmax(logits, dim=1).to(y.device)
            return (1 - y[:, 2]) * pred.eq(y[:, 0]).float() + y[:, 2] * pred.eq(y[:, 1]).float()
        ```

    Args:
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated

    Shape:
        - Input: :math:`(B, C, H, W)`, :math:`(B,)`
        - Output: :math:`(B, C, H, W)`, :math:`(B, 3)`

    Note:
        This implementation would randomly mixup images in a batch. Ideally, the larger batch size would be preferred.

    Examples:
        >>> rng = torch.manual_seed(1)
        >>> input = torch.rand(2, 1, 3, 3)
        >>> label = torch.tensor([0, 1])
        >>> mixup = RandomMixUp()
        >>> mixup(input, label)
        (tensor([[[[0.7576, 0.2793, 0.4031],
                  [0.7347, 0.0293, 0.7999],
                  [0.3971, 0.7544, 0.5695]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0.4388, 0.6387, 0.5247],
                  [0.6826, 0.3051, 0.4635],
                  [0.4550, 0.5725, 0.4980]]]]), tensor([[0.0000, 0.0000, 0.1980],
                [1.0000, 1.0000, 0.4162]]))
    """
    def __init__(self, p: float = 1.0, max_lambda: Optional[Union[torch.Tensor, float]] = None,
                 same_on_batch: bool = False) -> None:
        super(RandomMixUp, self).__init__()
        self.p = p
        if max_lambda is None:
            self.max_lambda = torch.tensor(1.)
        else:
            self.max_lambda = \
                cast(torch.Tensor, max_lambda) if isinstance(max_lambda, torch.Tensor) else torch.tensor(max_lambda)
        self.same_on_batch = same_on_batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, max_lambda={self.max_lambda}, same_on_batch={self.same_on_batch}"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_mixup_generator(batch_shape[0], self.p, self.max_lambda, same_on_batch=self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,  # type: ignore
                        params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        return F.apply_mixup(input, label, params)


class RandomCutMix(MixAugmentation):
    """
    Implemention for `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    <https://arxiv.org/pdf/1905.04899.pdf>`.
    The function returns (inputs, labels), in which the inputs is the tensor that contains the mixup images
    while the labels is a :math:`(num_mixes, B, 3)` tensor that contains (label_permuted_batch, lambda)
    for each cutmix. The implementation referred to `https://github.com/clovaai/CutMix-PyTorch`.

    The onehot label may be computed as :
        ```
        def onehot(size, target):
            vec = torch.zeros(size, dtype=torch.float32)
            vec[target] = 1.
            return vec

        def cutmix_label(labels, out_labels, size):
            lb_onehot = onehot(size, labels)
            for out_label in out_labels:
                label_permuted_batch, lam = out_label[:, 0], out_label[:, 1]
                label_permuted_onehot = onehot(size, label_permuted_batch)
                lb_onehot = lb_onehot * lam + label_permuted_onehot * (1. - lam)
            return lb_onehot
        ```

    Args:
        height (int): the width of the input image
        width (int): the width of the input image
        p (float): probability for performing cutmix. Default is 0.5.
        num_mix (int): cut mix times. Default is 1.
        beta (float or torch.Tensor, optional): hyperparameter for beta distribution. It controls the cut size.
            If None, it will be set to 1.

    Shape:
        - Input: :math:`(B, C, H, W)`, :math:`(B,)`
        - Output: :math:`(B, C, H, W)`, :math:`(num_mix, B, 2)`

    Note:
        This implementation would randomly cutmix images in a batch. Ideally, the larger batch size would be preferred.

    Examples:
        >>> rng = torch.manual_seed(42)
        >>> input = torch.rand(2, 1, 3, 3)
        >>> input[0] = torch.ones((1, 3, 3))
        >>> label = torch.tensor([0, 1])
        >>> cutmix = RandomCutMix(3, 3)
        >>> cutmix(input, label)
        (tensor([[[[1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0.1332, 0.9346, 0.5936],
                  [0.8694, 0.5677, 0.7411],
                  [0.4294, 0.8854, 0.5739]]]]), tensor([[[0.0000, 0.0000, 0.4444],
                 [1.0000, 1.0000, 1.0000]]]))
    """
    def __init__(self, height: int, width: int, p: float = 0.5, num_mix: int = 1,
                 beta: Optional[Union[torch.Tensor, float]] = None, same_on_batch: bool = False) -> None:
        super(RandomCutMix, self).__init__()
        self.height = height
        self.width = width
        self.p = p
        self.num_mix = num_mix
        if beta is None:
            self.beta = torch.tensor(1.)
        else:
            self.beta = cast(torch.Tensor, beta) if isinstance(beta, torch.Tensor) else torch.tensor(beta)
        self.same_on_batch = same_on_batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, num_mix={num_mix}, beta={self.beta}"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_cutmix_generator(batch_shape[0], width=self.width, height=self.height, p=self.p,
                                          num_mix=self.num_mix, beta=self.beta, same_on_batch=self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,  # type: ignore
                        params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        return F.apply_cutmix(input, label, params)
