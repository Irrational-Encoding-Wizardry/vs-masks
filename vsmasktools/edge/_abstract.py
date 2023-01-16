from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, ClassVar, NoReturn, Sequence, TypeAlias, cast

from vsexprtools import ExprOp
from vstools import CustomValueError, FuncExceptT, T, check_variable, core, get_subclasses, inject_self, vs

from ..exceptions import UnknownEdgeDetectError, UnknownRidgeDetectError

__all__ = [
    'EdgeDetect', 'EdgeDetectT',
    'RidgeDetect', 'RidgeDetectT',

    'MatrixEdgeDetect', 'SingleMatrix', 'EuclidianDistance',

    'Max',

    'get_all_edge_detects',
    'get_all_ridge_detect',
]


class _Feature(Enum):
    EDGE = auto()
    RIDGE = auto()


class BaseDetect:
    @staticmethod
    def from_param(
        cls: type[T],
        value: str | type[T] | T | None,
        exception_cls: type[CustomValueError],
        excluded: Sequence[type[T]] = [],
        func_except: FuncExceptT | None = None
    ) -> type[T]:
        if isinstance(value, str):
            all_edge_detects = get_subclasses(EdgeDetect, excluded)
            search_str = value.lower().strip()

            for edge_detect_cls in all_edge_detects:
                if edge_detect_cls.__name__.lower() == search_str:
                    return edge_detect_cls  # type: ignore

            raise exception_cls(func_except or cls.from_param, value)  # type: ignore

        if issubclass(value, cls):  # type: ignore
            return value  # type: ignore

        if isinstance(value, cls):
            return value.__class__

        return cls

    @staticmethod
    def ensure_obj(
        cls: type[T],
        value: str | type[T] | T | None,
        exception_cls: type[CustomValueError],
        excluded: Sequence[type[T]] = [],
        func_except: FuncExceptT | None = None
    ) -> T:
        new_edge_detect: T | None = None

        if not isinstance(value, cls):
            try:
                new_edge_detect = cls.from_param(value, func_except)()  # type: ignore
            except Exception:
                ...
        else:
            new_edge_detect = value

        if new_edge_detect is None:
            new_edge_detect = cls()

        if new_edge_detect.__class__ in excluded:
            raise exception_cls(
                func_except or cls.ensure_obj, new_edge_detect.__class__,  # type: ignore
                'This {cls_name} can\'t be instantiated to be used!',
                cls_name=new_edge_detect.__class__
            )

        return new_edge_detect


class EdgeDetect(ABC):
    """Abstract edge detection interface."""

    _bits: int

    @classmethod
    def from_param(
        cls: type[EdgeDetect], edge_detect: EdgeDetectT | None = None, func_except: FuncExceptT | None = None
    ) -> type[EdgeDetect]:
        return BaseDetect.from_param(cls, edge_detect, UnknownEdgeDetectError, [], func_except)

    @classmethod
    def ensure_obj(
        cls: type[EdgeDetect], edge_detect: EdgeDetectT | None = None, func_except: FuncExceptT | None = None
    ) -> EdgeDetect:
        return BaseDetect.ensure_obj(cls, edge_detect, UnknownEdgeDetectError, [], func_except)

    @inject_self
    def edgemask(
        self, clip: vs.VideoNode, lthr: float = 0.0, hthr: float | None = None, multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Makes edge mask based on convolution kernel.
        The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        :param clip:            Source clip
        :param lthr:            Low threshold. Anything below lthr will be set to 0
        :param hthr:            High threshold. Anything above hthr will be set to the range max
        :param multi:           Multiply all pixels by this before thresholding
        :param clamp:           Clamp to TV or full range if True or specified range `(low, high)`

        :return:                Mask clip
        """
        return self._mask(clip, lthr, hthr, multi, clamp, _Feature.EDGE, **kwargs)

    def _mask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: float | None = None,
        multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        feature: _Feature = _Feature.EDGE, **kwargs: Any
    ) -> vs.VideoNode:
        assert check_variable(clip, self.__class__)

        self._bits = clip.format.bits_per_sample
        is_float = clip.format.sample_type == vs.FLOAT
        peak = 1.0 if is_float else (1 << self._bits) - 1
        hthr = peak if hthr is None else hthr

        clip_p = self._preprocess(clip)

        if feature == _Feature.EDGE:
            mask = self._compute_edge_mask(clip_p, **kwargs)
        elif feature == _Feature.RIDGE:
            if not isinstance(self, RidgeDetect):
                raise RuntimeError
            mask = self._compute_ridge_mask(clip_p, **kwargs)

        mask = self._postprocess(mask)

        if multi != 1:
            if is_float:
                mask = mask.std.Expr(f'x {multi} *')
            else:
                def _multi_func(x: int) -> int:
                    return round(max(min(x * multi, peak), 0))
                mask = mask.std.Lut(function=_multi_func)

        if lthr > 0 or hthr < peak:
            if is_float:
                mask = mask.std.Expr(f'x {hthr} > {peak} x {lthr} <= 0 x ? ?')
            else:
                def _thr_func(x: int) -> int | float:
                    return peak if x > hthr else 0 if x <= lthr else x  # type: ignore[operator]
                mask = mask.std.Lut(function=_thr_func)

        if clamp:
            if isinstance(clamp, list):
                mask = core.std.Expr(mask, ['x {} max {} min'.format(*c) for c in clamp])
            if isinstance(clamp, tuple):
                mask = core.std.Expr(mask, 'x {} max {} min'.format(*clamp))
            else:
                assert mask.format
                if is_float:
                    clamp_vals = [(0., 1.), (-0.5, 0.5), (-0.5, 0.5)]
                else:
                    with mask.get_frame(0) as f:
                        crange = cast(int, f.props['_ColorRange'])
                    clamp_vals = [(0, peak)] * 3 if crange == 0 else [
                        (16 << self._bits - 8, 235 << self._bits - 8),
                        (16 << self._bits - 8, 240 << self._bits - 8),
                        (16 << self._bits - 8, 240 << self._bits - 8)
                    ]

                mask = core.std.Expr(mask, ['x {} max {} min'.format(*c) for c in clamp_vals[:mask.format.num_planes]])

        return mask

    @abstractmethod
    def _compute_edge_mask(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        raise NotImplementedError

    def _preprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip

    def _postprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip


class MatrixEdgeDetect(EdgeDetect):
    matrices: ClassVar[Sequence[Sequence[float]]]
    divisors: ClassVar[Sequence[float] | None] = None
    mode_types: ClassVar[Sequence[str] | None] = None

    def _compute_edge_mask(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        return self._merge_edge([
            clip.std.Convolution(matrix=mat, divisor=div, saturate=False, mode=mode)
            for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())
        ])

    def _compute_ridge_mask(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        def _x(c: vs.VideoNode) -> vs.VideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[0], divisor=self._get_divisors()[0])

        def _y(c: vs.VideoNode) -> vs.VideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[1], divisor=self._get_divisors()[1])

        x = _x(clip)
        y = _y(clip)
        xx = _x(x)
        yy = _y(y)
        xy = _x(x)
        return self._merge_ridge([xx, yy, xy])

    @abstractmethod
    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        raise NotImplementedError

    @abstractmethod
    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return self.matrices

    def _get_divisors(self) -> Sequence[float]:
        return self.divisors if self.divisors else [0.0] * len(self._get_matrices())

    def _get_mode_types(self) -> Sequence[str]:
        return self.mode_types if self.mode_types else ['s'] * len(self._get_matrices())

    def _postprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        if len(self.matrices[0]) > 9 or (self.mode_types and self.mode_types[0] != 's'):
            clip = clip.std.Crop(
                right=clip.format.subsampling_w * 2 if clip.format and clip.format.subsampling_w != 0 else 2
            ).resize.Point(clip.width, src_width=clip.width)
        return clip


class RidgeDetect(MatrixEdgeDetect):
    @classmethod
    def from_param(  # type: ignore
        cls: type[RidgeDetect], edge_detect: RidgeDetectT | None = None, func_except: FuncExceptT | None = None
    ) -> type[RidgeDetect]:
        return BaseDetect.from_param(cls, edge_detect, UnknownRidgeDetectError, [], func_except)

    @classmethod
    def ensure_obj(  # type: ignore
        cls: type[RidgeDetect], edge_detect: RidgeDetectT | None = None, func_except: FuncExceptT | None = None
    ) -> RidgeDetect:
        return BaseDetect.ensure_obj(cls, edge_detect, UnknownRidgeDetectError, [], func_except)

    @inject_self
    def ridgemask(
        self, clip: vs.VideoNode, lthr: float = 0.0, hthr: float | None = None, multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False, **kwargs: Any
    ) -> vs.VideoNode | NoReturn:
        """
        Makes ridge mask based on convolution kernel.
        The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        :param clip:            Source clip
        :param lthr:            Low threshold. Anything below lthr will be set to 0
        :param hthr:            High threshold. Anything above hthr will be set to the range max
        :param multi:           Multiply all pixels by this before thresholding
        :param clamp:           Clamp to TV or full range if True or specified range `(low, high)`

        :return:                Mask clip
        """
        return self._mask(clip, lthr, hthr, multi, clamp, _Feature.RIDGE, **kwargs)

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return core.std.Expr(clips, 'x y * 2 * -1 * x dup * z dup * 4 * + y dup * + + sqrt x y + +')


class SingleMatrix(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return clips[0]

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError


class EuclidianDistance(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return core.std.Expr(clips, 'x x * y y * + sqrt')

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError


class Max(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return ExprOp.MAX.combine(*clips)

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError


EdgeDetectT: TypeAlias = type[EdgeDetect] | EdgeDetect | str  # type: ignore
RidgeDetectT: TypeAlias = type[RidgeDetect] | RidgeDetect | str  # type: ignore


def get_all_edge_detects(
    clip: vs.VideoNode,
    lthr: float = 0.0, hthr: float | None = None,
    multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False
) -> list[vs.VideoNode]:
    """
    Returns all the EdgeDetect subclasses

    :param clip:        Source clip
    :param lthr:        See :py:func:`EdgeDetect.get_mask`
    :param hthr:        See :py:func:`EdgeDetect.get_mask`
    :param multi:       See :py:func:`EdgeDetect.get_mask`
    :param clamp:       Clamp to TV or full range if True or specified range `(low, high)`

    :return:            A list edge masks
    """
    def _all_subclasses(cls: type[EdgeDetect] = EdgeDetect) -> set[type[EdgeDetect]]:
        return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in _all_subclasses(c))

    all_subclasses = {
        s for s in _all_subclasses()
        if s.__name__ not in {
            'MatrixEdgeDetect', 'RidgeDetect', 'SingleMatrix', 'EuclidianDistance', 'Max',
            'Matrix1D', 'SavitzkyGolay', 'SavitzkyGolayNormalise',
            'Matrix2x2', 'Matrix3x3', 'Matrix5x5'
        }
    }
    return [
        edge_detect().edgemask(clip, lthr, hthr, multi, clamp).text.Text(edge_detect.__name__)
        for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__)
    ]


def get_all_ridge_detect(
    clip: vs.VideoNode, lthr: float = 0.0, hthr: float | None = None, multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False
) -> list[vs.VideoNode]:
    """
    Returns all the RidgeDetect subclasses

    :param clip:        Source clip
    :param lthr:        See :py:func:`EdgeDetect.get_mask`
    :param hthr:        See :py:func:`EdgeDetect.get_mask`
    :param multi:       See :py:func:`EdgeDetect.get_mask`
    :param clamp:       Clamp to TV or full range if True or specified range `(low, high)`

    :return:            A list edge masks
    """
    def _all_subclasses(cls: type[RidgeDetect] = RidgeDetect) -> set[type[RidgeDetect]]:
        return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in _all_subclasses(c))

    all_subclasses = {
        s for s in _all_subclasses()
        if s.__name__ not in {
            'MatrixEdgeDetect', 'RidgeDetect', 'SingleMatrix', 'EuclidianDistance', 'Max',
            'Matrix1D', 'SavitzkyGolay', 'SavitzkyGolayNormalise',
            'Matrix2x2', 'Matrix3x3', 'Matrix5x5'
        }
    }
    return [
        edge_detect().ridgemask(clip, lthr, hthr, multi, clamp).text.Text(edge_detect.__name__)
        for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__)
    ]
