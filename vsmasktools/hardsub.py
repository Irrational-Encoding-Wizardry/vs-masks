from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type

from vsexprtools import ExprOp, ExprToken, expr_func, norm_expr
from vskernels import Bilinear, Catrom, Point
from vsrgtools.util import mean_matrix
from vssource import IMWRI, Indexer
from vstools import (
    ColorRange, CustomOverflowError, FileNotExistsError, FilePathType, FrameRangeN, FrameRangesN,
    Matrix, VSFunction, check_variable, core, depth, fallback, get_lowest_value, get_neutral_value,
    get_neutral_values, get_peak_value, get_y, iterate, normalize_ranges, replace_ranges,
    scale_value, scale_delta, vs, vs_object
)

from .abstract import DeferredMask, GeneralMask
from .edge import SobelStd
from .morpho import Morpho
from .types import GenericMaskT, XxpandMode
from .utils import max_planes, normalize_mask

__all__ = [
    'CustomMaskFromFolder',
    'CustomMaskFromRanges',

    'HardsubMask',
    'HardsubSignFades',
    'HardsubSign',
    'HardsubLine',
    'HardsubLineFade',
    'HardsubASS',

    'bounded_dehardsub',
    'diff_hardsub_mask',

    'get_all_sign_masks'
]


class _base_cmaskcar(vs_object):
    clips: list[vs.VideoNode]

    @abstractmethod
    def frame_ranges(self, clip: vs.VideoNode) -> list[list[tuple[int, int]]]:
        ...

    def __vs_del__(self, core_id: int) -> None:
        self.clips.clear()


@dataclass
class CustomMaskFromClipsAndRanges(GeneralMask, _base_cmaskcar):
    processing: VSFunction = field(default=core.lazy.std.Binarize, kw_only=True)
    idx: Indexer | Type[Indexer] = field(default=IMWRI, kw_only=True)

    def get_mask(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        assert check_variable(clip, self.get_mask)

        mask = clip.std.BlankClip(
            format=clip.format.replace(color_family=vs.GRAY, subsampling_h=0, subsampling_w=0).id,
            keep=True, color=0
        )

        matrix = Matrix.from_video(clip)

        for maskclip, mask_ranges in zip(self.clips, self.frame_ranges(clip)):
            maskclip = Point.resample(
                maskclip.std.AssumeFPS(clip), mask, matrix,
                range_in=ColorRange.FULL, range=ColorRange.FULL
            )
            maskclip = self.processing(maskclip).std.Loop(mask.num_frames)

            mask = replace_ranges(mask, maskclip, mask_ranges, **kwargs)

        return mask


@dataclass
class CustomMaskFromFolder(CustomMaskFromClipsAndRanges):
    folder_path: FilePathType

    def __post_init__(self) -> None:
        if not (folder_path := Path(str(self.folder_path))).is_dir():
            raise FileNotExistsError('"folder_path" must be an existing path directory!', self.get_mask)

        self.files = list(folder_path.glob('*'))

        self.clips = [self.idx.source(file, bits=-1) for file in self.files]

    def frame_ranges(self, clip: vs.VideoNode) -> list[list[tuple[int, int]]]:
        return [
            [(other[-1] if other else end, end)]
            for (*other, end) in (map(int, name.stem.split('_')) for name in self.files)
        ]


@dataclass
class CustomMaskFromRanges(CustomMaskFromClipsAndRanges):
    ranges: dict[FilePathType, FrameRangeN | FrameRangesN]

    def __post_init__(self) -> None:
        self.clips = [self.idx.source(str(file), bits=-1) for file in self.ranges.keys()]

    def frame_ranges(self, clip: vs.VideoNode) -> list[list[tuple[int, int]]]:
        return [normalize_ranges(clip, ranges) for ranges in self.ranges.values()]


class HardsubMask(DeferredMask):
    bin_thr: float = 0.75

    def get_progressive_dehardsub(
        self, hardsub: vs.VideoNode, ref: vs.VideoNode, partials: list[vs.VideoNode]
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        """
        Dehardsub using multiple superior hardsubbed sources and one inferior non-subbed source.

        :param hardsub:  Hardsub master source (eg Wakanim RU dub).
        :param ref:      Non-subbed reference source (eg CR, Funi, Amazon).
        :param partials: Sources to use for partial dehardsubbing (eg Waka DE, FR, SC).

        :return:         Dehardsub stages and masks used for progressive dehardsub.
        """

        masks = [self.get_mask(hardsub, ref)]
        partials_dehardsubbed = [hardsub]
        dehardsub_masks = []
        partials = partials + [ref]

        assert masks[-1].format is not None

        thr = scale_value(self.bin_thr, 32, masks[-1])

        for p in partials:
            masks.append(
                ExprOp.SUB.combine(masks[-1], self.get_mask(p, ref))
            )
            dehardsub_masks.append(
                iterate(expr_func([masks[-1]], f"x {thr} < 0 x ?"), core.std.Maximum, 4).std.Inflate()
            )
            partials_dehardsubbed.append(
                partials_dehardsubbed[-1].std.MaskedMerge(p, dehardsub_masks[-1])
            )

            masks[-1] = masks[-1].std.MaskedMerge(masks[-1].std.Invert(), masks[-2])

        return partials_dehardsubbed, dehardsub_masks

    def apply_dehardsub(
        self, hardsub: vs.VideoNode, ref: vs.VideoNode, partials: list[vs.VideoNode] | None = None
    ) -> vs.VideoNode:
        if partials:
            partials_dehardsubbed, _ = self.get_progressive_dehardsub(hardsub, ref, partials)
            dehardsub = partials_dehardsubbed[-1]
        else:
            dehardsub = hardsub.std.MaskedMerge(ref, self.get_mask(hardsub, ref))

        return replace_ranges(hardsub, dehardsub, self.ranges)


class HardsubSignFades(HardsubMask):
    highpass: float
    expand: int
    edgemask: GenericMaskT
    expand_mode: XxpandMode

    def __init__(
        self, *args: Any, highpass: float = 0.0763, expand: int = 8, edgemask: GenericMaskT = SobelStd,
        expand_mode: XxpandMode = XxpandMode.RECTANGLE,
        **kwargs: Any
    ) -> None:
        self.highpass = highpass
        self.expand = expand
        self.edgemask = edgemask
        self.expand_mode = expand_mode

        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        clipedge, refedge = (
            normalize_mask(self.edgemask, x, **kwargs).std.Convolution(mean_matrix)
            for x in (clip, ref)
        )

        highpass = scale_delta(self.highpass, 32, clip)

        mask = norm_expr(
            [clipedge, refedge], f'x y - {highpass} < 0 {ExprToken.RangeMax} ?'
        ).std.Median()

        return max_planes(Morpho.inflate(Morpho.expand(mask, self.expand, mode=self.expand_mode), iterations=4))


class HardsubSign(HardsubMask):
    """
    Hardsub scenefiltering helper using `Zastin <https://github.com/kgrabs>`_'s hardsub mask.

    :param thr:             Binarization threshold, [0, 1] (Default: 0.06).
    :param minimum:         std.Minimum iterations (Default: 1).
    :param expand:          std.Maximum iterations (Default: 8).
    :param inflate:         std.Inflate iterations (Default: 7).
    :param expand_mode:     Specifies the XxpandMode used for mask growth (Default: XxpandMode.RECTANGLE).
    """

    thr: float
    minimum: int
    expand: int
    inflate: int
    expand_mode: XxpandMode

    def __init__(
        self, *args: Any, thr: float = 0.06, minimum: int = 1, expand: int = 8, inflate: int = 7,
        expand_mode: XxpandMode = XxpandMode.RECTANGLE,
        **kwargs: Any
    ) -> None:
        self.thr = thr
        self.minimum = minimum
        self.expand = expand
        self.inflate = inflate
        self.expand_mode = expand_mode
        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        hsmf = norm_expr([clip, ref], 'x y - abs')
        hsmf = Bilinear.resample(hsmf, clip.format.replace(subsampling_w=0, subsampling_h=0))  # type: ignore

        hsmf = ExprOp.MAX(hsmf, split_planes=True)

        hsmf = Morpho.binarize(hsmf, self.thr)
        hsmf = Morpho.minimum(hsmf, iterations=self.minimum)
        hsmf = Morpho.expand(hsmf, self.expand, mode=self.expand_mode)
        hsmf = Morpho.inflate(hsmf, iterations=self.inflate)

        return hsmf.std.Limiter()


class HardsubLine(HardsubMask):
    expand: int | None

    def __init__(self, *args: Any, expand: int | None = None, **kwargs: Any) -> None:
        self.expand = expand

        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        assert clip.format

        expand_n = fallback(self.expand, clip.width // 200)

        y_range = get_peak_value(clip) - get_lowest_value(clip)
        uv_range = get_peak_value(clip, chroma=True) - get_lowest_value(clip, chroma=True)

        uv_abs = f' {get_neutral_value(clip)} - abs '
        yexpr = f'x y - abs {y_range * 0.7} > 255 0 ?'
        uv_thr = uv_range * 0.8
        uvexpr = f'x {uv_abs} {uv_thr} < y {uv_abs} {uv_thr} < and 255 0 ?'

        upper = scale_value(0.8, 32, clip)
        lower = scale_value(0.2, 32, clip)
        mindiff = y_range * 0.1

        difexpr = f'x {upper} > x {lower} < or x y - abs {mindiff} > and 255 0 ?'

        right = core.resize.Point(clip, src_left=4)

        subedge = norm_expr(
            [clip, right], (yexpr, uvexpr), format=clip.format.replace(sample_type=vs.INTEGER, bits_per_sample=8)
        )

        subedge = ExprOp.MIN(Catrom.resample(subedge, vs.YUV444P8), split_planes=True)

        clip_y, ref_y = get_y(clip), depth(get_y(ref), clip)

        clips = [clip_y.std.Convolution(mean_matrix), ref_y.std.Convolution(mean_matrix)]
        diff = core.std.Expr(clips, difexpr, vs.GRAY8).std.Maximum().std.Maximum()

        mask: vs.VideoNode = core.misc.Hysteresis(subedge, diff)
        mask = iterate(mask, core.std.Maximum, expand_n)
        mask = mask.std.Inflate().std.Inflate().std.Convolution(mean_matrix)

        return depth(mask, clip, range_in=ColorRange.FULL, range_out=ColorRange.FULL)


class HardsubLineFade(HardsubLine):
    def __init__(self, *args: Any, refframe: float = 0.5, **kwargs: Any) -> None:
        if refframe < 0 or refframe > 1:
            raise CustomOverflowError('"refframe" must be between 0 and 1!', self.__class__)

        self.ref_float = refframe

        super().__init__(*args, refframes=None, **kwargs)

    def get_mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:  # type: ignore
        self.refframes = [
            r[0] + round((r[1] - r[0]) * self.ref_float)
            for r in normalize_ranges(ref, self.ranges)
        ]

        return super().get_mask(clip, ref)


class HardsubASS(HardsubMask):
    filename: str
    fontdir: str | None
    shift: int | None

    def __init__(
        self, filename: str, *args: Any, fontdir: str | None = None, shift: int | None = None, **kwargs: Any
    ) -> None:
        self.filename = filename
        self.fontdir = fontdir
        self.shift = shift
        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        ref = ref[0] * self.shift + ref if self.shift else ref
        mask = ref.sub.TextFile(self.filename, fontdir=self.fontdir, blend=False).std.PropToClip('_Alpha')
        mask = mask[self.shift:] if self.shift else mask
        mask = mask.std.Binarize(1)
        mask = iterate(mask, core.std.Maximum, 3)
        mask = iterate(mask, core.std.Inflate, 3)
        return mask.std.Limiter()


def bounded_dehardsub(
    hrdsb: vs.VideoNode, ref: vs.VideoNode, signs: list[HardsubMask], partials: list[vs.VideoNode] | None = None
) -> vs.VideoNode:
    for sign in signs:
        hrdsb = sign.apply_dehardsub(hrdsb, ref, partials)

    return hrdsb


def diff_hardsub_mask(a: vs.VideoNode, b: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
    assert check_variable(a, diff_hardsub_mask)
    assert check_variable(b, diff_hardsub_mask)

    return a.std.BlankClip(color=get_neutral_values(a), keep=True).std.MaskedMerge(
        a.std.MakeDiff(b), HardsubLine(**kwargs).get_mask(a, b)
    )


def get_all_sign_masks(hrdsb: vs.VideoNode, ref: vs.VideoNode, signs: list[HardsubMask]) -> vs.VideoNode:
    assert check_variable(hrdsb, get_all_sign_masks)
    assert check_variable(ref, get_all_sign_masks)

    mask = ref.std.BlankClip(
        format=ref.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id, keep=True
    )

    for sign in signs:
        mask = replace_ranges(mask, ExprOp.ADD.combine(mask, max_planes(sign.get_mask(hrdsb, ref))), sign.ranges)

    return mask.std.Limiter()
