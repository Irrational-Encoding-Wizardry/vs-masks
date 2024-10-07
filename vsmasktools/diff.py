from __future__ import annotations

from typing import Any, TypeAlias, overload

from vsexprtools import ExprOp, norm_expr
from vskernels import Bilinear, Catrom, Kernel, KernelT, NoScale
from vsrgtools import RemoveGrainMode, bilateral, gauss_blur, removegrain
from vstools import (
    ColorRange, CustomValueError, FuncExceptT, KwargsT, VSFunction, check_variable, depth,
    get_w, get_y, insert_clip, iterate, vs
)

from .edge import EdgeDetect, EdgeDetectT, ExLaplacian4
from .morpho import Morpho
from .types import XxpandMode

__all__ = [
    'diff_rescale',
    'diff_creditless',
    'diff_creditless_oped',
    'credit_mask',
    'based_diff_mask'
]


def diff_rescale(
    clip: vs.VideoNode, height: int, kernel: KernelT = Catrom,
    thr: float = 0.216, expand: int = 2, func: FuncExceptT | None = None
) -> vs.VideoNode:
    return based_diff_mask(clip, height, kernel, thr, expand=2 + expand, func=func)


def diff_creditless(
    credit_clip: vs.VideoNode, nc_clip: vs.VideoNode, thr: float = 0.01,
    start_frame: int = 0, expand: int = 2, *, prefilter: bool | int = False,
    edgemask: EdgeDetectT = ExLaplacian4, ep_clip: vs.VideoNode | None = None,
    func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    mask = based_diff_mask(
        credit_clip, nc_clip, thr=thr, prefilter=prefilter,
        postfilter=0,
        ampl=EdgeDetect.ensure_obj(edgemask, func), expand=2 + expand,
        func=func
    )

    if not ep_clip or ep_clip.num_frames == mask.num_frames:
        return mask

    assert mask.format

    return insert_clip(ep_clip.std.BlankClip(format=mask.format.id, keep=True), mask, start_frame)


def diff_creditless_oped(
    ep: vs.VideoNode, ncop: vs.VideoNode, nced: vs.VideoNode, thr: float = 0.1,
    opstart: int | None = None, opend: int | None = None,
    edstart: int | None = None, edend: int | None = None,
    func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = func or diff_creditless_oped

    op_mask = ed_mask = None

    kwargs |= KwargsT(expand=4, prefilter=False, func=func, ep_clip=ep) | kwargs

    if opstart is not None and opend is not None:
        op_mask = diff_creditless(ep[opstart:opend + 1], ncop[:opend - opstart + 1], thr, opstart, **kwargs)

    if edstart is not None and edend is not None:
        ed_mask = diff_creditless(ep[edstart:edend + 1], nced[:edend - edstart + 1], thr, edstart, **kwargs)

    if op_mask and ed_mask:
        return ExprOp.ADD.combine(op_mask, ed_mask)
    elif op_mask or ed_mask:
        return op_mask or ed_mask  # type: ignore

    raise CustomValueError(
        'You must specify one or both of ("opstart", "opend"), ("edstart", "edend")', func
    )


def credit_mask(
    clip: vs.VideoNode, ref: vs.VideoNode, thr: float,
    blur: float | None = 1.65, prefilter: bool | int = 5,
    expand: int = 8
) -> vs.VideoNode:
    if blur is not None:
        clip, ref = gauss_blur(clip, blur), gauss_blur(ref, blur)

    credit_mask = based_diff_mask(
        clip, ref,
        thr=thr,
        prefilter=prefilter, postfilter=0,
        ampl=ExLaplacian4, expand=4
    )

    credit_mask = Morpho.erosion(credit_mask, 6)
    credit_mask = iterate(credit_mask, lambda x: x.std.Minimum().std.Maximum(), 8)

    if expand:
        credit_mask = Morpho.dilation(credit_mask, expand)

    credit_mask = Morpho.inflate(credit_mask, 3)

    return credit_mask


Count: TypeAlias = int


@overload
def based_diff_mask(
    clip: vs.VideoNode, ref: vs.VideoNode,
    /,
    *,
    thr: float = 0.216,
    prefilter: int | KwargsT | bool | VSFunction = False,
    postfilter: int | tuple[Count, RemoveGrainMode] | list[tuple[Count, RemoveGrainMode]] | VSFunction = 2,
    ampl: str | type[EdgeDetect] | EdgeDetect = ...,
    expand: int = 4,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Make a difference mask between a clean source and a reference clip with additionnal pre and post processing

    :param clip:        Source clip
    :param ref:         Reference clip
    :param thr:         Threshold of the amplification expr, defaults to 0.216
    :param prefilter:   Filter applied before extracting the difference between clip and ref:
                        - int -> equivalent of number of taps used in the bilateral call applied to the clips
                        - True -> 5 taps
                        - KwargsT -> Arguments passed to the bilateral function
    :param postfilter:  Filter applied to the difference clip. Default is RemoveGrainMode.MINMAX_AROUND2 applied twice.
    :param ampl:        Amplification expression.
    :param expand:      Additional expand radius applied to the mask, defaults to 4
    :return:            Generated mask
    """


@overload
def based_diff_mask(
    clip: vs.VideoNode, height: int, kernel: KernelT = ...,
    /,
    thr: float = 0.216,
    prefilter: int | KwargsT | bool | VSFunction = False,
    postfilter: int | tuple[Count, RemoveGrainMode] | list[tuple[Count, RemoveGrainMode]] | VSFunction = 2,
    ampl: str | type[EdgeDetect] | EdgeDetect = ...,
    expand: int = 4,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Make a difference mask between a clean source and a rescaled clip with additionnal pre and post processing

    :param clip:        Source clip
    :param height:      Height to be descaled to
    :param kernel:      Kernel used for descaling and rescaling
    :param thr:         Threshold of the amplification expr, defaults to 0.216
    :param prefilter:   Filter applied before extracting the difference between clip and ref:
                        - int -> equivalent of number of taps used in the bilateral call applied to the clips
                        - True -> 5 taps
                        - KwargsT -> Arguments passed to the bilateral function
    :param postfilter:  Filter applied to the difference clip. Default is RemoveGrainMode.MINMAX_AROUND2 applied twice.
    :param ampl:        Amplification expression.
    :param expand:      Additional expand radius applied to the mask, defaults to 4
    :return:            Generated mask
    """


def based_diff_mask(
    clip: vs.VideoNode, ref_or_height: vs.VideoNode | int, kernel: KernelT = NoScale,
    /,
    thr: float = 0.216,
    prefilter: int | KwargsT | bool | VSFunction = False,
    postfilter: int | tuple[Count, RemoveGrainMode] | list[tuple[Count, RemoveGrainMode]] | VSFunction = 2,
    ampl: str | type[EdgeDetect] | EdgeDetect = 'x 2 4 pow * {thr} < 0 1 ?',
    expand: int = 4,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or based_diff_mask
    assert check_variable(clip, func)

    if isinstance(ref_or_height, vs.VideoNode):
        ref = ref_or_height
    else:
        clip = get_y(clip)

        kernel = Kernel.ensure_obj(kernel, func)

        ref = kernel.descale(clip, get_w(ref_or_height), ref_or_height)
        ref = kernel.scale(ref, clip.width, clip.height)

    assert clip.format
    assert ref.format

    if clip.format.num_planes != ref.format.num_planes:
        clip, ref = get_y(clip), get_y(ref)

    if prefilter:
        if callable(prefilter):
            clip, ref = prefilter(clip), prefilter(ref)
        else:
            if isinstance(prefilter, int):
                sigma = 5 if prefilter is True else prefilter
                kwargs = KwargsT(sigmaS=((sigma ** 2 - 1) / 12) ** 0.5, sigmaR=sigma / 10)
            else:
                kwargs = prefilter

            clip, ref = bilateral(clip, **kwargs), bilateral(ref, **kwargs)

    ref = depth(ref, clip)
    assert clip.format

    dst_fmt = clip.format.replace(subsampling_w=0, subsampling_h=0)
    diff_fmt = dst_fmt.replace(color_family=vs.GRAY)

    mask = ExprOp.mae(dst_fmt)((Bilinear.resample(c, dst_fmt) for c in [clip, ref]), format=diff_fmt, split_planes=True)
    mask = ColorRange.FULL.apply(mask)

    if postfilter:
        if isinstance(postfilter, int):
            mask = iterate(mask, removegrain, postfilter, RemoveGrainMode.MINMAX_AROUND2)
        elif isinstance(postfilter, tuple):
            mask = iterate(mask, removegrain, postfilter[0], postfilter[1])
        elif isinstance(postfilter, list):
            mask = mask
            for count, rgmode in postfilter:
                mask = iterate(mask, removegrain, count, rgmode)
        else:
            mask = postfilter(mask)

    if isinstance(ampl, str):
        mask = norm_expr(mask, ampl.format(thr=thr), func=func)
    else:
        mask = EdgeDetect.ensure_obj(ampl, func).edgemask(mask, lthr=thr, hthr=thr)

    if expand:
        mask = Morpho.expand(mask, expand, mode=XxpandMode.ELLIPSE)

    return mask
