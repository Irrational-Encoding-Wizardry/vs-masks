from __future__ import annotations

from itertools import zip_longest
from math import sqrt
from typing import Any, Literal, Sequence, cast

from vsexprtools import ExprList, ExprOp, TupleExprList, complexpr_available, norm_expr
from vsrgtools import BlurMatrix
from vstools import (
    ConvMode, CustomValueError, FuncExceptT, PlanesT, SpatialConvModeT, VSFunctionAllArgs,
    copy_signature, core, fallback, inject_self, iterate, scale_mask, scale_value, to_arr, vs
)

from .types import Coordinates, XxpandMode

__all__ = [
    'RadiusT',
    'Morpho',
    'grow_mask'
]

RadiusT = int | tuple[int, SpatialConvModeT]


def _morpho_method(
    self: Morpho,
    clip: vs.VideoNode,
    radius: RadiusT = 1,
    thr: float | None = None,
    iterations: int = 1,
    coords: Sequence[int] | None = None,
    multiply: float | None = None,
    planes: PlanesT = None,
    *,
    func: FuncExceptT | None = None,
    **kwargs: Any
) -> vs.VideoNode:
    raise NotImplementedError


def _xxpand_method(
    self: Morpho,
    clip: vs.VideoNode,
    sw: int, sh: int | None = None,
    mode: XxpandMode = XxpandMode.RECTANGLE,
    thr: float | None = None,
    planes: PlanesT = None,
    *,
    func: FuncExceptT | None = None,
    **kwargs: Any
) -> vs.VideoNode:
    raise NotImplementedError


class Morpho:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @classmethod
    def _morpho_xx_imum(
        cls,
        clip: vs.VideoNode,
        radius: tuple[int, ConvMode],
        thr: float | None,
        coords: Sequence[int] | None,
        multiply: float | None,
        clamp: bool,
        *,
        op: Literal[ExprOp.MIN, ExprOp.MAX],
        func: FuncExceptT
    ) -> TupleExprList:
        if coords:
            _, expr = cls._get_matrix_from_coords(coords, func)
        else:
            expr = ExprOp.matrix('x', *radius, [(0, 0)])

        for e in expr:
            e.extend([op] * e.mlength)

            if thr is not None:
                e.append("x", scale_value(thr, 32, clip))
                limit = (ExprOp.SUB, ExprOp.MAX) if op == ExprOp.MIN else (ExprOp.ADD, ExprOp.MIN)
                e.append(*limit)

            if multiply is not None:
                e.append(multiply, ExprOp.MUL)

            if clamp:
                e.append(ExprOp.clamp())

        return expr

    def _mm_func(
        self,
        clip: vs.VideoNode,
        radius: RadiusT = 1,
        thr: float | None = None,
        iterations: int = 1,
        coords: Sequence[int] | None = None,
        multiply: float | None = None,
        planes: PlanesT = None,
        *,
        func: FuncExceptT,
        mm_func: VSFunctionAllArgs,
        op: Literal[ExprOp.MIN, ExprOp.MAX],
        **kwargs: Any
    ) -> vs.VideoNode:
        if isinstance(radius, tuple):
            radius, conv_mode = radius
        else:
            conv_mode = ConvMode.SQUARE

        if not complexpr_available:
            if radius > 1:
                raise CustomValueError('If akarin plugin is not available, you must have radius=1', func, radius)

            if not coords:
                match conv_mode:
                    case ConvMode.VERTICAL:
                        coords = Coordinates.VERTICAL
                    case ConvMode.HORIZONTAL:
                        coords = Coordinates.HORIZONTAL
                    case ConvMode.HV:
                        coords = Coordinates.DIAMOND
                    case _:
                        coords = Coordinates.RECTANGLE

            if thr is not None:
                kwargs.update(threshold=scale_mask(thr, 32, clip))

            kwargs.update(coordinates=coords, planes=planes)

            if multiply is not None:
                mm_func = self._multiply_mm_func(mm_func, multiply)
        else:
            mm_func = cast(VSFunctionAllArgs, norm_expr)
            kwargs.update(
                expr=self._morpho_xx_imum(clip, (radius, conv_mode), thr, coords, multiply, False, op=op, func=func)
            )

        return iterate(clip, mm_func, iterations, **kwargs)

    def _xxpand_transform(
        self,
        clip: vs.VideoNode,
        sw: int, sh: int | None = None,
        mode: XxpandMode = XxpandMode.RECTANGLE,
        thr: float | None = None,
        planes: PlanesT = None,
        *,
        op: Literal[ExprOp.MIN, ExprOp.MAX],
        func: FuncExceptT,
        **kwargs: Any
    ) -> vs.VideoNode:
        sh = fallback(sh, sw)

        function = self.maximum if op is ExprOp.MAX else self.minimum

        for wi, hi in zip_longest(range(sw, -1, -1), range(sh, -1, -1), fillvalue=0):
            if wi > 0 and hi > 0:
                coords = Coordinates.from_xxpand_mode(mode, wi)
            elif wi > 0:
                coords = Coordinates.HORIZONTAL
            elif hi > 0:
                coords = Coordinates.VERTICAL
            else:
                break

            clip = function(clip, thr, 1, coords, planes=planes, func=func, **kwargs)

        return clip

    def _xxflate(
        self,
        clip: vs.VideoNode,
        radius: RadiusT = 1,
        thr: float | None = None,
        iterations: int = 1,
        coords: Sequence[int] | None = None,
        multiply: float | None = None,
        planes: PlanesT = None,
        *,
        func: FuncExceptT,
        inflate: bool,
        **kwargs: Any
    ) -> vs.VideoNode:
        if isinstance(radius, tuple):
            radius, conv_mode = radius
        else:
            conv_mode = ConvMode.SQUARE

        xxflate_func: VSFunctionAllArgs

        if not complexpr_available:
            if radius > 1 or conv_mode != ConvMode.SQUARE:
                raise CustomValueError(
                    'If akarin plugin is not available, you must have radius=1 and ConvMode.SQUARE',
                    func, (radius, conv_mode)
                )

            if coords:
                raise CustomValueError(
                    "If akarin plugin is not available, you can't have custom coordinates", func, coords
                )

            xxflate_func = core.std.Inflate if inflate else core.std.Deflate
            kwargs.update(planes=planes)

            if thr is not None:
                kwargs.update(threshold=scale_mask(thr, 32, clip))

            if multiply is not None:
                xxflate_func = self._multiply_mm_func(xxflate_func, multiply)
        else:
            if coords:
                radius, expr = self._get_matrix_from_coords(coords, func)
            else:
                expr = ExprOp.matrix('x', radius, conv_mode, exclude=[(0, 0)])

            for e in expr:
                e.append(ExprOp.ADD * e.mlength, len(e), ExprOp.DIV, 'x', ExprOp.MAX if inflate else ExprOp.MIN)

                if thr is not None:
                    thr = scale_value(thr, 32, clip)
                    limit = ['x', thr, ExprOp.ADD, ExprOp.MIN] if inflate else ['x', thr, ExprOp.SUB, ExprOp.MAX]
                    e.append(limit)

                if multiply is not None:
                    e.append(multiply, ExprOp.MUL)

            kwargs.update(expr=expr)

            xxflate_func = cast(VSFunctionAllArgs, norm_expr)

        return iterate(clip, xxflate_func, iterations, **kwargs)

    def _multiply_mm_func(self, func: VSFunctionAllArgs, multiply: float) -> VSFunctionAllArgs:
        def mm_func(clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
            return func(clip, *args, **kwargs).std.Expr(f'x {multiply} *')
        return mm_func

    @staticmethod
    def _get_matrix_from_coords(coords: Sequence[int], func: FuncExceptT) -> tuple[int, TupleExprList]:
        lc = len(coords)

        if lc < 8:
            raise CustomValueError('coords must have more than 8 elements!', func, coords)

        sq_lc = sqrt(lc + 1)

        if not (sq_lc.is_integer() and sq_lc % 2 != 0):
            raise CustomValueError(
                'coords must contain exactly (radius * 2 + 1) ** 2 - 1 numbers.\neg. 8, 24, 48...', func, coords
            )

        coords = list(coords)
        coords.insert(lc // 2, 1)

        r = int(sq_lc // 2)

        expr, = ExprOp.matrix("x", r, ConvMode.SQUARE, exclude=[(0, 0)])
        expr = ExprList([x for x, coord in zip(expr, coords) if coord])

        return r, TupleExprList([expr])

    @inject_self
    @copy_signature(_minmax_method)
    def maximum(
        self, src: vs.VideoNode, thr: float | None = None, coords: CoordsT | None = None,
        iterations: int = 1, multiply: float | None = None, planes: PlanesT = None,
        *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        return self.dilation(src, iterations, planes, thr, coords or ([1] * 8), multiply, func=func, **kwargs)

    @inject_self
    @copy_signature(_minmax_method)
    def minimum(
        self, src: vs.VideoNode, thr: float | None = None, coords: CoordsT | None = None,
        iterations: int = 1, multiply: float | None = None, planes: PlanesT = None,
        *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        return self.erosion(src, iterations, planes, thr, coords or ([1] * 8), multiply, func=func, **kwargs)

    @inject_self
    def inflate(
        self: Morpho, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        iterations: int = 1, multiply: float | None = None, *, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        for _ in range(iterations):
            src = self._xxflate(True, src, radius, planes, thr, multiply, func=func or self.inflate)
        return src

    @inject_self
    def deflate(
        self: Morpho, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        iterations: int = 1, multiply: float | None = None, *, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        for _ in range(iterations):
            src = self._xxflate(False, src, radius, planes, thr, multiply, func=func or self.deflate)
        return src

    @inject_self
    @copy_signature(_morpho_method)
    def dilation(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self._mm_func(*args, func=func or self.dilation, mm_func=core.std.Maximum, op=ExprOp.MAX, **kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def erosion(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self._mm_func(*args, func=func or self.erosion, mm_func=core.std.Minimum, op=ExprOp.MIN, **kwargs)

    @inject_self
    @copy_signature(_morpho_method2)
    def expand(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self.xxpand_transform(clip, ExprOp.MAX, *args, func=func, **kwargs)

    @inject_self
    @copy_signature(_morpho_method2)
    def inpand(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self.xxpand_transform(clip, ExprOp.MIN, *args, func=func, **kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def closing(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        func = func or self.closing

        dilated = self.dilation(src, *args, func=func, **kwargs)
        eroded = self.erosion(dilated, *args, func=func, **kwargs)

        return eroded

    @inject_self
    @copy_signature(_morpho_method)
    def opening(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        func = func or self.closing

        eroded = self.erosion(src, *args, func=func, **kwargs)
        dilated = self.dilation(eroded, *args, func=func, **kwargs)

        return dilated

    @inject_self
    def gradient(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        coords: CoordsT = 5, multiply: float | None = None, *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coords, planes, func or self.gradient)

        if radius == 1 and self._fast:
            return norm_expr(
                src, '{dilated} {eroded} - {multiply}', planes,
                dilated=self._morpho_xx_imum(src, thr, ExprOp.MAX, coords, None, True),
                eroded=self._morpho_xx_imum(src, thr, ExprOp.MIN, coords, None, True),
                multiply='' if multiply is None else f'{multiply} *'
            )

        eroded = self.erosion(src, radius, planes, thr, coords, multiply, func=func, **kwargs)
        dilated = self.dilation(src, radius, planes, thr, coords, multiply, func=func, **kwargs)

        return norm_expr([dilated, eroded], 'x y -', planes)

    @inject_self
    @copy_signature(_morpho_method)
    def top_hat(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        opened = self.opening(src, *args, func=func or self.top_hat, **kwargs)

        return norm_expr([src, opened], 'x y -', kwargs.get('planes', args[1] if len(args) > 1 else None))

    @inject_self
    @copy_signature(_morpho_method)
    def black_hat(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        closed = self.closing(src, *args, func=func or self.black_hat, **kwargs)

        return norm_expr([closed, src], 'x y -', kwargs.get('planes', args[1] if len(args) > 1 else None))

    @inject_self
    def outer_hat(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        coords: CoordsT = 5, multiply: float | None = None, *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coords, planes, func or self.outer_hat)

        if radius == 1 and self._fast:
            return norm_expr(
                src, '{dilated} {multiply} x -', planes,
                dilated=self._morpho_xx_imum(src, thr, ExprOp.MAX, coords, None, True),
                multiply='' if multiply is None else f'{multiply} *'
            )

        dilated = self.dilation(src, radius, planes, thr, coords, multiply, func=func, **kwargs)

        return norm_expr([dilated, src], 'x y -', planes)

    @inject_self
    def inner_hat(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        coords: CoordsT = 5, multiply: float | None = None, *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coords, planes, func or self.inner_hat)

        if radius == 1 and self._fast:
            return norm_expr(
                src, '{eroded} {multiply} x -', planes,
                eroded=self._morpho_xx_imum(src, thr, ExprOp.MIN, coords),
                multiply='' if multiply is None else f'{multiply} *'
            )

        eroded = self.erosion(src, radius, planes, thr, coords, multiply, func=func, **kwargs)

        return norm_expr([src, eroded], 'x y -', planes)

    @inject_self
    def binarize(
        self, src: vs.VideoNode, midthr: float | list[float] | None = None,
        lowval: float | list[float] | None = None, highval: float | list[float] | None = None,
        planes: PlanesT = None
    ) -> vs.VideoNode:
        midthr, lowval, highval = (
            thr and list(
                scale_value(t, 32, src)
                for i, t in enumerate(to_arr(thr))
            ) for thr in (midthr, lowval, highval)
        )

        return src.std.Binarize(midthr, lowval, highval, planes)


def grow_mask(
    mask: vs.VideoNode, radius: int = 1, multiply: float = 1.0,
    planes: PlanesT = None, coords: CoordsT = 5, thr: float | None = None,
    *, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = func or grow_mask

    assert check_variable(mask, func)

    morpho = Morpho(planes, func)

    kwargs.update(thr=thr, coords=coords)

    closed = morpho.closing(mask, **kwargs)
    dilated = morpho.dilation(closed, **kwargs)
    outer = morpho.outer_hat(dilated, radius, **kwargs)

    blurred = BlurMatrix.BINOMIAL()(outer, planes=planes)

    if multiply != 1.0:
        return blurred.std.Expr(f'x {multiply} *')

    return blurred
