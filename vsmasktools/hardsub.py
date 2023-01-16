from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vsexprtools import ExprOp
from vstools import (
    FileNotExistsError, FrameRangeN, FrameRangesN, VSFunction, check_variable, core, depth, fallback,
    get_neutral_values, get_y, insert_clip, iterate, normalize_ranges, replace_ranges, scale_thresh, split, vs,
    vs_object
)

from .abstract import DeferredMask, GeneralMask

__all__ = [
    'HardsubManual',

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


@dataclass
class HardsubManual(GeneralMask, vs_object):
    path: str | Path
    processing: VSFunction = core.lazy.std.Binarize  # type: ignore

    def __post_init__(self) -> None:
        if not (path := Path(self.path)).is_dir():
            raise FileNotExistsError('"path" must be an existing path directory!', self.get_mask)

        files = [file.stem for file in path.glob('*')]

        self.clips = [
            core.imwri.Read(file) for file in files
        ]

        self.ranges = [
            (other[-1] if other else end, end)
            for (*other, end) in (map(int, name.split('_')) for name in files)
        ]

    def get_mask(self, clip: vs.VideoNode, *args: Any) -> vs.VideoNode:  # type: ignore[override]
        assert check_variable(clip, self.get_mask)

        mask = clip.std.BlankClip(
            format=clip.format.replace(color_family=vs.GRAY, subsampling_h=0, subsampling_w=0).id,
            keep=True, color=0
        )

        for maskclip, (start_frame, end_frame) in zip(self.clips, self.ranges):
            maskclip = maskclip.std.AssumeFPS(clip).resize.Point(format=mask.format.id)  # type: ignore
            maskclip = self.processing(maskclip).std.Loop(end_frame - start_frame + 1)

            mask = insert_clip(mask, maskclip, start_frame)

        return mask

    def __vs_del__(self, core_id: int) -> None:
        super().__vs_del__(core_id)

        self.clips.clear()


class HardsubMask(DeferredMask):
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

        thresh = scale_thresh(0.75, masks[-1])

        for p in partials:
            masks.append(
                ExprOp.SUB.combine(masks[-1], self.get_mask(p, ref))
            )
            dehardsub_masks.append(
                iterate(core.akarin.Expr([masks[-1]], f"x {thresh} < 0 x ?"), core.std.Maximum, 4).std.Inflate()
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
    highpass: int
    expand: int

    def __init__(self, *args: Any, highpass: int = 5000, expand: int = 8, **kwargs: Any) -> None:
        self.highpass = highpass
        self.expand = expand

        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
        clip = core.fmtc.bitdepth(clip, bits=16).std.Convolution([1] * 9)
        ref = core.fmtc.bitdepth(ref, bits=16).std.Convolution([1] * 9)
        clipedge = get_y(clip).std.Sobel()
        refedge = get_y(ref).std.Sobel()
        mask = core.std.Expr([clipedge, refedge], f'x y - {self.highpass} < 0 65535 ?').std.Median()
        mask = iterate(mask, core.std.Maximum, self.expand)
        return iterate(mask, core.std.Inflate, 4)


class HardsubSign(HardsubMask):
    """
    Hardsub scenefiltering helper using `Zastin <https://github.com/kgrabs>`_'s hardsub mask.

    :param thresh:  Binarization threshold, [0, 1] (Default: 0.06).
    :param expand:  std.Maximum iterations (Default: 8).
    :param inflate: std.Inflate iterations (Default: 7).
    """

    thresh: float
    minimum: int
    expand: int
    inflate: int

    def __init__(
        self, *args: Any, thresh: float = 0.06, minimum: int = 1, expand: int = 8, inflate: int = 7, **kwargs: Any
    ) -> None:
        self.thresh = thresh
        self.minimum = minimum
        self.expand = expand
        self.inflate = inflate
        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
        assert clip.format
        hsmf = core.akarin.Expr([clip, ref], 'x y - abs')
        hsmf = hsmf.resize.Point(format=clip.format.replace(subsampling_w=0, subsampling_h=0).id)
        hsmf = core.akarin.Expr(split(hsmf), "x y z max max")
        hsmf = hsmf.std.Binarize(scale_thresh(self.thresh, hsmf))
        hsmf = iterate(hsmf, core.std.Minimum, self.minimum)
        hsmf = iterate(hsmf, core.std.Maximum, self.expand)
        hsmf = iterate(hsmf, core.std.Inflate, self.inflate)

        return hsmf.std.Limiter()


class HardsubLine(HardsubMask):
    expand: int | None

    def __init__(self, *args: Any, expand: int | None = None, **kwargs: Any) -> None:
        self.expand = expand

        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
        clp_f = clip.format
        assert clp_f
        bits = clp_f.bits_per_sample
        stype = clp_f.sample_type

        expand_n = fallback(self.expand, clip.width // 200)

        fmt_args = (clp_f.color_family, vs.INTEGER, 8, clp_f.subsampling_w, clp_f.subsampling_h)
        yuv_fmt = core.query_video_format(*fmt_args)

        y_range = 219 << (bits - 8) if stype == vs.INTEGER else 1
        uv_range = 224 << (bits - 8) if stype == vs.INTEGER else 1
        offset = 16 << (bits - 8) if stype == vs.INTEGER else 0

        uv_abs = ' abs ' if stype == vs.FLOAT else ' {} - abs '.format((1 << bits) // 2)
        yexpr = 'x y - abs {thr} > 255 0 ?'.format(thr=y_range * 0.7)
        uvexpr = 'x {uv_abs} {thr} < y {uv_abs} {thr} < and 255 0 ?'.format(uv_abs=uv_abs, thr=uv_range * 0.8)

        difexpr = 'x {upper} > x {lower} < or x y - abs {mindiff} > and 255 0 ?'.format(
            upper=y_range * 0.8 + offset, lower=y_range * 0.2 + offset, mindiff=y_range * 0.1
        )

        # right shift by 4 pixels.
        # fmtc uses at least 16 bit internally, so it's slower for 8 bit,
        # but its behaviour when shifting/replicating edge pixels makes it faster otherwise
        if bits < 16:
            right = core.resize.Point(clip, src_left=4)
        else:
            right = core.fmtc.resample(clip, sx=4, flt=False)
        subedge = core.std.Expr([clip, right], [yexpr, uvexpr], yuv_fmt.id)
        c444 = split(subedge.resize.Bicubic(format=vs.YUV444P8, filter_param_a=0, filter_param_b=0.5))
        subedge = core.std.Expr(c444, 'x y z min min')

        clip, ref = get_y(clip), get_y(ref)
        ref = ref if clip.format == ref.format else depth(ref, bits)

        clips = [clip.std.Convolution([1] * 9), ref.std.Convolution([1] * 9)]
        diff = core.std.Expr(clips, difexpr, vs.GRAY8).std.Maximum().std.Maximum()

        mask: vs.VideoNode = core.misc.Hysteresis(subedge, diff)  # type: ignore[assignment]
        mask = iterate(mask, core.std.Maximum, expand_n)
        mask = mask.std.Inflate().std.Inflate().std.Convolution([1] * 9)

        return depth(mask, bits, range_in=1, range_out=1)


class HardsubLineFade(HardsubLine):
    ref_float: float

    def __init__(
        self, ranges: FrameRangeN | FrameRangesN, *args: Any, refframe: float = 0.5, **kwargs: Any
    ) -> None:
        if refframe < 0 or refframe > 1:
            raise ValueError("HardsubLineFade: '`refframe` must be between 0 and 1!'")
        ranges = ranges if isinstance(ranges, list) else [ranges]
        self.ref_float = refframe
        super().__init__(ranges, *args, refframes=None, **kwargs)

    def get_mask(self, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:  # type: ignore[override]
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

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
        ref = ref[0] * self.shift + ref if self.shift else ref
        mask: vs.VideoNode = ref.sub.TextFile(  # type: ignore[attr-defined]
            self.filename, fontdir=self.fontdir, blend=False
        )[1]
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
        mask = replace_ranges(mask, ExprOp.ADD.combine(mask, sign.get_mask(hrdsb, ref)), sign.ranges)

    return mask.std.Limiter()
