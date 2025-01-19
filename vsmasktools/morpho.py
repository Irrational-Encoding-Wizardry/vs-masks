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
    """Collection of morphological operations"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @inject_self
    def maximum(
        self,
        clip: vs.VideoNode,
        thr: float | None = None,
        iterations: int = 1,
        coords: Sequence[int] | None = None,
        multiply: float | None = None,
        planes: PlanesT = None,
        *,
        func: FuncExceptT | None = None,
        **kwargs: Any
    ) -> vs.VideoNode:
        """
        Replaces each pixel with the largest value in its 3x3 neighbourhood.
        This operation is also known as dilation with a radius of 1.

        :param clip:            Clip to process.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become greater than input + threshold
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the 3x3 neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly 8 numbers.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        return self.dilation(clip, 1, thr, iterations, coords, multiply, planes, func=func, **kwargs)

    @inject_self
    def minimum(
        self,
        clip: vs.VideoNode,
        thr: float | None = None,
        iterations: int = 1,
        coords: Sequence[int] | None = None,
        multiply: float | None = None,
        planes: PlanesT = None,
        *,
        func: FuncExceptT | None = None,
        **kwargs: Any
    ) -> vs.VideoNode:
        """
        Replaces each pixel with the smallest value in its 3x3 neighbourhood.
        This operation is also known as erosion with a radius of 1.

        :param clip:            Clip to process.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the 3x3 neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value. This must contain exactly 8 numbers.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        return self.erosion(clip, 1, thr, iterations, coords, multiply, planes, func=func, **kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def inflate(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        Replaces each pixel with the average of the (radius * 2 + 1) ** 2 - 1 pixels
        in its (radius * 2 + 1)x(radius * 2 + 1) neighbourhood, but only if that average
        is greater than the center pixel.

        A radius of 1 will replace each pixel with the average of the 8 pixels in its 3x3 neighbourhood.
        A radius of 2 will replace each pixel with the average of the 24 pixels in its 5x5 neighbourhood.

        :param clip:            Clip to process.
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become greater than input + thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        return self._xxflate(*args, func=func or self.inflate, inflate=True, **kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def deflate(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        Replaces each pixel with the average of the (radius * 2 + 1) ** 2 - 1 pixels
        in its (radius * 2 + 1)x(radius * 2 + 1) neighbourhood, but only if that average
        is less than the center pixel.

        A radius of 1 will replace each pixel with the average of the 8 pixels in its 3x3 neighbourhood.
        A radius of 2 will replace each pixel with the average of the 24 pixels in its 5x5 neighbourhood.

        :param clip:            Clip to process.
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        return self._xxflate(*args, func=func or self.deflate, inflate=False, **kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def dilation(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        Replaces each pixel with the largest value in its (radius * 2 + 1)x(radius * 2 + 1) neighbourhood.

        :param clip:            Clip to process.
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        return self._mm_func(*args, func=func or self.dilation, mm_func=core.std.Maximum, op=ExprOp.MAX, **kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def erosion(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        Replaces each pixel with the smallest value in its (radius * 2 + 1)x(radius * 2 + 1) neighbourhood.

        :param clip:            Clip to process.
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        return self._mm_func(*args, func=func or self.erosion, mm_func=core.std.Minimum, op=ExprOp.MIN, **kwargs)

    @inject_self
    @copy_signature(_xxpand_method)
    def expand(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        Replaces multiple times each pixel with the largest value in its 3x3 neighbourhood.
        Specifying a mode will allow custom growing mode.

        :param clip:            Clip to process.
        :param sw:              Number of horizontal iterations.
        :param sh:              Number of vertical iterations.
        :param mode:            Specifies the expand mode shape.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param planes:          Which plane to process.
        """
        return self._xxpand_transform(clip, *args, op=ExprOp.MAX, func=func or self.expand, **kwargs)

    @inject_self
    @copy_signature(_xxpand_method)
    def inpand(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        Replaces multiple times each pixel with the smallest value in its 3x3 neighbourhood.
        Specifying a mode will allow custom growing mode.

        :param clip:            Clip to process.
        :param sw:              Number of horizontal iterations.
        :param sh:              Number of vertical iterations.
        :param mode:            Specifies the expand mode shape.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param planes:          Which plane to process.
        """
        return self._xxpand_transform(clip, *args, op=ExprOp.MIN, func=func or self.inpand, **kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def closing(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        A closing is an dilation followed by an erosion.

        :param clip:            Clip to process
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        func = func or self.closing

        dilated = self.dilation(clip, *args, func=func, **kwargs)
        eroded = self.erosion(dilated, *args, func=func, **kwargs)

        return eroded

    @inject_self
    @copy_signature(_morpho_method)
    def opening(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        An opening is an erosion followed by an dilation.

        :param clip:            Clip to process
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        func = func or self.opening

        eroded = self.erosion(clip, *args, func=func, **kwargs)
        dilated = self.dilation(eroded, *args, func=func, **kwargs)

        return dilated

    @inject_self
    @copy_signature(_morpho_method)
    def gradient(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        A morphological gradient is the difference between a dilation and erosion.

        :param clip:            Clip to process.
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        func = func or self.gradient

        eroded = self.erosion(clip, *args, func=func, **kwargs)
        dilated = self.dilation(clip, *args, func=func, **kwargs)

        return norm_expr(
            [dilated, eroded], 'x y -', kwargs.get('planes', args[5] if len(args) > 5 else None), func=func
        )

    @inject_self
    @copy_signature(_morpho_method)
    def top_hat(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        A top hat or a white hat is the difference of the original clip and the opening.

        :param clip:            Clip to process.
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        func = func or self.top_hat

        opened = self.opening(clip, *args, func=func, **kwargs)

        return norm_expr(
            [clip, opened], 'x y -', kwargs.get('planes', args[5] if len(args) > 5 else None), func=func
        )

    @copy_signature(top_hat)
    @inject_self
    def white_hate(self, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return self.top_hat(*args, **dict(func=self.white_hate) | kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def bottom_hat(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        A bottom hat or a black hat is the difference of the closing and the original clip.

        :param clip:            Clip to process.
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        func = func or self.bottom_hat

        closed = self.closing(clip, *args, func=func, **kwargs)

        return norm_expr(
            [closed, clip], 'x y -', kwargs.get('planes', args[5] if len(args) > 5 else None), func=func
        )

    @copy_signature(bottom_hat)
    @inject_self
    def black_hat(self, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return self.top_hat(*args, **dict(func=self.black_hat) | kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def outer_hat(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        An outer hat is the difference of the dilation and the original clip.

        :param clip:            Clip to process.
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        func = func or self.outer_hat

        dilated = self.dilation(clip, *args, func=func, **kwargs)

        return norm_expr(
            [dilated, clip], 'x y -', kwargs.get('planes', args[5] if len(args) > 5 else None), func=func
        )

    @inject_self
    def inner_hat(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        An inner hat is the difference of the original clip and the erosion.

        :param clip:            Clip to process.
        :param radius:          A single integer specifies the size of the square matrix.
                                A tuple of an integer and a ConvMode allows specification
                                of the matrix type dimension as well.
        :param thr:             Threshold (32-bit scale) to limit how much pixels are changed.
                                Output pixels will not become less than input - thr.
                                The default is no limit.
        :param iterations:      Number of times to execute the function.
        :param coords:          Specifies which pixels from the neighbourhood are considered.
                                If an element of this array is 0, the corresponding pixel is not considered
                                when finding the maximum value.
                                This must contain exactly (radius * 2 + 1) ** 2 - 1 numbers eg. 8, 24, 48...
                                When specified, this parameter takes precedence over radius.
        :param multiply:        Optional multiplier of the final value.
        :param planes:          Which plane to process.
        """
        func = func or self.inner_hat

        eroded = self.erosion(clip, *args, func=func, **kwargs)

        return norm_expr(
            [clip, eroded], 'x y -', kwargs.get('planes', args[5] if len(args) > 5 else None), func=func
        )

    @inject_self
    def binarize(
        self,
        clip: vs.VideoNode,
        midthr: float | list[float] | None = None,
        lowval: float | list[float] | None = None,
        highval: float | list[float] | None = None,
        planes: PlanesT = None
    ) -> vs.VideoNode:
        """
        Turns every pixel in the image into either lowval, if it's below midthr, or highval, otherwise.

        :param clip:            Clip to process.
        :param midthr:          Defaults to the middle point of range allowed by the format.
                                Can be specified for each plane individually.
        :param lowval:          Value given to pixels that are below threshold.
                                Can be specified for each plane individually.
                                Defaults to the lower bound of the format.
        :param highval:         Value given to pixels that are greater than or equal to threshold.
                                Defaults to the maximum value allowed by the format.
                                Can be specified for each plane individually.
                                Defaults to the upper bound of the format.
        :param planes:          Specifies which planes will be processed.
                                Any unprocessed planes will be simply copied.
        """
        midthr, lowval, highval = (
            thr and list(
                scale_value(t, 32, clip)
                for t in to_arr(thr)
            ) for thr in (midthr, lowval, highval)
        )

        return core.std.Binarize(clip, midthr, lowval, highval, planes)

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

        matrix = list(coords)
        matrix.insert(lc // 2, 0)

        r = int(sq_lc // 2)

        expr, = ExprOp.matrix("x", r, ConvMode.SQUARE, exclude=[(0, 0)])
        expr = ExprList([x for x, coord in zip(expr, coords) if coord])

        return r, TupleExprList([expr])


def grow_mask(
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
    func = func or grow_mask

    morpho = Morpho()

    closed = morpho.closing(clip, radius, thr, 1, coords, None, planes, func=func, **kwargs)
    dilated = morpho.dilation(closed, radius, thr, 1, coords, None, planes, func=func, **kwargs)
    outer = morpho.outer_hat(dilated, radius, thr, iterations, coords, None, planes, func=func, **kwargs)

    blurred = BlurMatrix.BINOMIAL()(outer, planes=planes)

    if multiply:
        return blurred.std.Expr(f'x {multiply} *')

    return blurred
