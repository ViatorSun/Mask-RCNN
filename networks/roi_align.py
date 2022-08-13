#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time     : 2022.04
# @Author   : 绿色羽毛
# @Email    : lvseyumao@foxmail.com
# @Blog     : https://blog.csdn.net/ViatorSun
# @Paper    : 
# @arXiv    : 
# @version  : "1.0" 
# @Note     : 
# 
#




import torchvision
from torchvision.ops import roi_align
from torchvision.ops.boxes import box_area

from typing import Optional, List, Dict, Tuple, Union

from typing import List, Union

import torch
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
from torch.jit.annotations import BroadcastingList2
from torchvision.extension import _assert_has_ops




def _cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    # TODO add back the assert
    # assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = _cat([b for b in boxes], dim=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(torch.full_like(b[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            assert _tensor.size(1) == 4, \
                'The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]'
    elif isinstance(boxes, torch.Tensor):
        assert boxes.size(1) == 5, 'The boxes tensor shape is not correct as Tensor[K, 5]'
    else:
        assert False, 'boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]'
    return


def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:

    """
    Performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.

    Args:
        input (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each element
            contains ``C`` feature maps of dimensions ``H x W``.
            If the tensor is quantized, we expect a batch size of ``N == 1``.
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            If a single Tensor is passed, then the first column should
            contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
            If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
            in the batch.
        output_size (int or Tuple[int, int]): the size of the output (in bins or pixels) after the pooling
            is performed, as (height, width).
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ``ceil(roi_width / output_width)``, and likewise for height). Default: -1
        aligned (bool): If False, use the legacy implementation.
            If True, pixel shift the box coordinates it by -0.5 for a better alignment with the two
            neighboring pixel indices. This version is used in Detectron2

    Returns:
        Tensor[K, C, output_size[0], output_size[1]]: The pooled RoIs.
    """
    _assert_has_ops()
    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    return torchvision.ops.roi_align(input, rois, spatial_scale,
                                           output_size[0], output_size[1],
                                           sampling_ratio, aligned)


class RoIAlign(nn.Module):
    """
    See :func:`roi_align`.
    """
    def __init__(
        self,
        output_size: BroadcastingList2[int],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale='  + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ', aligned=' + str(self.aligned)
        tmpstr += ')'
        return tmpstr






# copying result_idx_in_level to a specific index in result[]
# is not supported by ONNX tracing yet.
# _onnx_merge_levels() is an implementation supported by ONNX
# that merges the levels to the right indices
@torch.jit.unused
def _onnx_merge_levels(levels: Tensor, unmerged_results: List[Tensor]) -> Tensor:
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros((levels.size(0), first_result.size(1),
                       first_result.size(2), first_result.size(3)),
                      dtype=dtype, device=device)
    for level in range(len(unmerged_results)):
        index = torch.where(levels == level)[0].view(-1, 1, 1, 1)
        index = index.expand(index.size(0),
                             unmerged_results[level].size(1),
                             unmerged_results[level].size(2),
                             unmerged_results[level].size(3))
        res = res.scatter(0, index, unmerged_results[level])
    return res


# TODO: (eellison) T54974082 https://github.com/pytorch/pytorch/issues/26744/pytorch/issues/26744
def initLevelMapper(
    k_min: int,
    k_max: int,
    canonical_scale: int = 224,
    canonical_level: int = 4,
    eps: float = 1e-6,
):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__( self, k_min: int, k_max: int,
                  canonical_scale: int = 224,
                  canonical_level: int = 4,
                  eps: float = 1e-6,  ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        Args:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics specified in eq. 1
    of the `Feature Pyramid Network paper <https://arxiv.org/abs/1612.03144>`_.
    They keyword-only parameters ``canonical_scale`` and ``canonical_level``
    correspond respectively to ``224`` and ``k0=4`` in eq. 1, and
    have the following meaning: ``canonical_level`` is the target level of the pyramid from
    which to pool a region of interest with ``w x h = canonical_scale x canonical_scale``.

    Args:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
        canonical_scale (int, optional): canonical_scale for LevelMapper
        canonical_level (int, optional): canonical_level for LevelMapper

    Examples::

        >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
        >>> i = OrderedDict()
        >>> i['feat1'] = torch.rand(1, 5, 64, 64)
        >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        >>> i['feat3'] = torch.rand(1, 5, 16, 16)
        >>> # create some random bounding boxes
        >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        >>> # original image size, before computing the feature maps
        >>> image_sizes = [(512, 512)]
        >>> output = m(i, [boxes], image_sizes)
        >>> print(output.shape)
        >>> torch.Size([6, 5, 3, 3])

    """

    __annotations__ = { 'scales': Optional[List[float]],
                        'map_levels': Optional[LevelMapper] }

    def __init__(self,
                featmap_names: List[str],
                output_size: Union[int, Tuple[int], List[int]],
                sampling_ratio: int,
                *,
                canonical_scale: int = 224,
                canonical_level: int = 4    ):

        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def convert_to_roi_format(self, boxes: List[Tensor]) -> Tensor:
        concat_boxes = torch.cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat([ torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device)
                          for i, b in enumerate(boxes)  ],
                        dim=0   )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def infer_scale(self, feature: Tensor, original_size: List[int]) -> float:
        # assumption: the scale is of the form 2 ** (-k), with k integer
        size = feature.shape[-2:]
        possible_scales: List[float] = []
        for s1, s2 in zip(size, original_size):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        return possible_scales[0]

    def setup_scales( self, features: List[Tensor], image_shapes: List[Tuple[int, int]]) -> None:
        assert len(image_shapes) != 0
        max_x = 0
        max_y = 0
        for shape in image_shapes:
            max_x = max(shape[0], max_x)
            max_y = max(shape[1], max_y)
        original_input_shape = (max_x, max_y)

        scales = [self.infer_scale(feat, original_input_shape) for feat in features]
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.scales = scales
        self.map_levels = initLevelMapper(  int(lvl_min),
                                            int(lvl_max),
                                            canonical_scale=self.canonical_scale,
                                            canonical_level=self.canonical_level )

    def forward(self,
                x: Dict[str, Tensor],
                boxes: List[Tensor],
                image_shapes: List[Tuple[int, int]]) -> Tensor:
        """
        Args:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        x_filtered = []
        for k, v in x.items():
            if k in self.featmap_names:
                x_filtered.append(v)
        num_levels = len(x_filtered)
        rois = self.convert_to_roi_format(boxes)
        if self.scales is None:
            self.setup_scales(x_filtered, image_shapes)

        scales = self.scales
        assert scales is not None

        if num_levels == 1:
            return roi_align(   x_filtered[0], rois,
                                output_size=self.output_size,
                                spatial_scale=scales[0],
                                sampling_ratio=self.sampling_ratio  )

        mapper = self.map_levels
        assert mapper is not None

        levels = mapper(boxes)

        num_rois = len(rois)
        num_channels = x_filtered[0].shape[1]

        dtype, device = x_filtered[0].dtype, x_filtered[0].device
        result = torch.zeros(
            (num_rois, num_channels,) + self.output_size,
            dtype=dtype,
            device=device,
        )

        tracing_results = []
        for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
            idx_in_level = torch.where(levels == level)[0]
            rois_per_level = rois[idx_in_level]

            result_idx_in_level = roi_align(
                per_level_feature, rois_per_level,
                output_size=self.output_size,
                spatial_scale=scale, sampling_ratio=self.sampling_ratio)

            if torchvision._is_tracing():
                tracing_results.append(result_idx_in_level.to(dtype))
            else:
                # result and result_idx_in_level's dtypes are based on dtypes of different
                # elements in x_filtered.  x_filtered contains tensors output by different
                # layers.  When autocast is active, it may choose different dtypes for
                # different layers' outputs.  Therefore, we defensively match result's dtype
                # before copying elements from result_idx_in_level in the following op.
                # We need to cast manually (can't rely on autocast to cast for us) because
                # the op acts on result in-place, and autocast only affects out-of-place ops.
                result[idx_in_level] = result_idx_in_level.to(result.dtype)

        if torchvision._is_tracing():
            result = _onnx_merge_levels(levels, tracing_results)

        return result

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
                f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})")
