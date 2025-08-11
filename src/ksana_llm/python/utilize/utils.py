# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM Project
# [vLLM Project]
# Ref: https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/utils.py

from __future__ import annotations
import importlib
import importlib.metadata
import importlib.util
import inspect
import os
import sys
import types

from collections import UserDict, defaultdict
from collections.abc import (Hashable, Iterable, Iterator, KeysView, Mapping)
from functools import cache, lru_cache, partial, wraps
from types import MappingProxyType
from typing import (TYPE_CHECKING, Any, Callable, Generic, Literal, NamedTuple,
                    Optional, Sequence, Tuple, Type, TypeVar, Union, cast,
                    overload)
from uuid import uuid4


import regex as re
from typing_extensions import Never, TypeIs, assert_never
from .logger import init_logger

logger = init_logger(__name__)

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")
_T = TypeVar("_T")

GB_bytes = 1_000_000_000

GiB_bytes = 1 << 30


def random_uuid() -> str:
    return str(uuid4().hex)


def random_tool_call_id() -> str:
    return f"chatcmpl-tool-{random_uuid()}"


def import_from_path(module_name: str, file_path: Union[str, os.PathLike]):
    """
    Import a Python file according to its file path.

    Based on the official recipe:
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{module_name}'")

    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@cache
def get_ksana_optional_dependencies():
    """获取 Ksana 的可选依赖项
    现在返回 Ksana 相关的可选依赖。
    """
    try:
        # 尝试获取 ksana_llm 包的元数据
        metadata = importlib.metadata.metadata("ksana_llm")
        requirements = metadata.get_all("Requires-Dist", [])
        extras = metadata.get_all("Provides-Extra", [])
        
        return {
            extra: [
                re.split(r";|>=|<=|==", req)[0] for req in requirements
                if req.endswith(f'extra == "{extra}"')
            ]
            for extra in extras
        }
    except importlib.metadata.PackageNotFoundError:
        # 如果 ksana_llm 包未安装，返回默认的可选依赖
        return {
            "multimodal": ["pillow", "opencv-python", "librosa"],
            "openai": ["openai", "tiktoken"],
            "tensorrt": ["tensorrt"],
            "torch": ["torch", "torchvision"],
    }


T = TypeVar("T")
# `collections` helpers


def is_list_of(
    value: object,
    typ: Union[type[T], tuple[type[T], ...]],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[list[T]]:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)


def flatten_2d_lists(lists: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a list of lists to a single list."""
    return [item for sublist in lists for item in sublist]


import cachetools


class _MappingOrderCacheView(UserDict[_K, _V]):

    def __init__(self, data: Mapping[_K, _V], ordered_keys: Mapping[_K, None]):
        super().__init__(data)
        self.ordered_keys = ordered_keys

    def __iter__(self) -> Iterator[_K]:
        return iter(self.ordered_keys)

    def keys(self) -> KeysView[_K]:
        return KeysView(self.ordered_keys)


class _Sentinel:
    ...


ALL_PINNED_SENTINEL = _Sentinel()


class CacheInfo(NamedTuple):
    hits: int
    total: int

    @property
    def hit_ratio(self) -> float:
        if self.total == 0:
            return 0

        return self.hits / self.total

    def __sub__(self, other: CacheInfo):
        return CacheInfo(
            hits=self.hits - other.hits,
            total=self.total - other.total,
        )


class LRUCache(cachetools.LRUCache[_K, _V], Generic[_K, _V]):

    def __init__(self,
                 capacity: float,
                 getsizeof: Optional[Callable[[_V], float]] = None):
        super().__init__(capacity, getsizeof)

        self.pinned_items = set[_K]()

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    def __getitem__(self, key: _K, *, update_info: bool = True) -> _V:
        value = super().__getitem__(key)

        if update_info:
            self._hits += 1
            self._total += 1

        return value

    def __delitem__(self, key: _K) -> None:
        run_on_remove = key in self
        value = self.__getitem__(key,
                                 update_info=False)  # type: ignore[call-arg]
        super().__delitem__(key)
        if key in self.pinned_items:
            # Todo: add warning to inform that del pinned item
            self._unpin(key)
        if run_on_remove:
            self._on_remove(key, value)

    @property
    def cache(self) -> Mapping[_K, _V]:
        """Return the internal cache dictionary in order (read-only)."""
        return _MappingOrderCacheView(
            self._Cache__data,  # type: ignore
            self.order)

    @property
    def order(self) -> Mapping[_K, None]:
        """Return the internal order dictionary (read-only)."""
        return MappingProxyType(self._LRUCache__order)  # type: ignore

    @property
    def capacity(self) -> float:
        return self.maxsize

    @property
    def usage(self) -> float:
        if self.maxsize == 0:
            return 0

        return self.currsize / self.maxsize

    def stat(self, *, delta: bool = False) -> CacheInfo:
        """
        Gets the cumulative number of hits and queries against this cache.

        If `delta=True`, instead gets these statistics
        since the last call that also passed `delta=True`.
        """
        info = CacheInfo(hits=self._hits, total=self._total)

        if delta:
            info_delta = info - self._last_info
            self._last_info = info
            info = info_delta

        return info

    def touch(self, key: _K) -> None:
        try:
            self._LRUCache__order.move_to_end(key)  # type: ignore
        except KeyError:
            self._LRUCache__order[key] = None  # type: ignore

    @overload
    def get(self, key: _K, /) -> Optional[_V]:
        ...

    @overload
    def get(self, key: _K, /, default: Union[_V, _T]) -> Union[_V, _T]:
        ...

    def get(self,
            key: _K,
            /,
            default: Optional[Union[_V,
                                    _T]] = None) -> Optional[Union[_V, _T]]:
        value: Optional[Union[_V, _T]]
        if key in self:
            value = self.__getitem__(
                key, update_info=False)  # type: ignore[call-arg]

            self._hits += 1
        else:
            value = default

        self._total += 1
        return value

    @overload
    def pop(self, key: _K) -> _V:
        ...

    @overload
    def pop(self, key: _K, default: Union[_V, _T]) -> Union[_V, _T]:
        ...

    def pop(self,
            key: _K,
            default: Optional[Union[_V,
                                    _T]] = None) -> Optional[Union[_V, _T]]:
        value: Optional[Union[_V, _T]]
        if key not in self:
            return default

        value = self.__getitem__(key,
                                 update_info=False)  # type: ignore[call-arg]
        self.__delitem__(key)
        return value

    def put(self, key: _K, value: _V) -> None:
        self.__setitem__(key, value)

    def pin(self, key: _K) -> None:
        """
        Pins a key in the cache preventing it from being
        evicted in the LRU order.
        """
        if key not in self:
            raise ValueError(f"Cannot pin key: {key} not in cache.")
        self.pinned_items.add(key)

    def _unpin(self, key: _K) -> None:
        """
        Unpins a key in the cache allowing it to be
        evicted in the LRU order.
        """
        self.pinned_items.remove(key)

    def _on_remove(self, key: _K, value: Optional[_V]) -> None:
        pass

    def remove_oldest(self, *, remove_pinned: bool = False) -> None:
        if len(self) == 0:
            return

        self.popitem(remove_pinned=remove_pinned)

    def _remove_old_if_needed(self) -> None:
        while self.currsize > self.capacity:
            self.remove_oldest()

    def popitem(self, remove_pinned: bool = False):
        """Remove and return the `(key, value)` pair least recently used."""
        if not remove_pinned:
            # pop the oldest item in the cache that is not pinned
            lru_key = next(
                (key for key in self.order if key not in self.pinned_items),
                ALL_PINNED_SENTINEL)
            if lru_key is ALL_PINNED_SENTINEL:
                raise RuntimeError("All items are pinned, "
                                   "cannot remove oldest from the cache.")
        else:
            lru_key = next(iter(self.order))
        value = self.pop(cast(_K, lru_key))
        return (lru_key, value)

    def clear(self) -> None:
        while len(self) > 0:
            self.remove_oldest(remove_pinned=True)

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)


class _PlaceholderBase:
    """
    Disallows downstream usage of placeholder modules.

    We need to explicitly override each dunder method because
    :meth:`__getattr__` is not called when they are accessed.

    See also:
        [Special method lookup](https://docs.python.org/3/reference/datamodel.html#special-lookup)
    """

    def __getattr__(self, key: str) -> Never:
        """
        The main class should implement this to throw an error
        for attribute accesses representing downstream usage.
        """
        raise NotImplementedError

    # [Basic customization]

    def __lt__(self, other: object):
        return self.__getattr__("__lt__")

    def __le__(self, other: object):
        return self.__getattr__("__le__")

    def __eq__(self, other: object):
        return self.__getattr__("__eq__")

    def __ne__(self, other: object):
        return self.__getattr__("__ne__")

    def __gt__(self, other: object):
        return self.__getattr__("__gt__")

    def __ge__(self, other: object):
        return self.__getattr__("__ge__")

    def __hash__(self):
        return self.__getattr__("__hash__")

    def __bool__(self):
        return self.__getattr__("__bool__")

    # [Callable objects]

    def __call__(self, *args: object, **kwargs: object):
        return self.__getattr__("__call__")

    # [Container types]

    def __len__(self):
        return self.__getattr__("__len__")

    def __getitem__(self, key: object):
        return self.__getattr__("__getitem__")

    def __setitem__(self, key: object, value: object):
        return self.__getattr__("__setitem__")

    def __delitem__(self, key: object):
        return self.__getattr__("__delitem__")

    # __missing__ is optional according to __getitem__ specification,
    # so it is skipped

    # __iter__ and __reversed__ have a default implementation
    # based on __len__ and __getitem__, so they are skipped.

    # [Numeric Types]

    def __add__(self, other: object):
        return self.__getattr__("__add__")

    def __sub__(self, other: object):
        return self.__getattr__("__sub__")

    def __mul__(self, other: object):
        return self.__getattr__("__mul__")

    def __matmul__(self, other: object):
        return self.__getattr__("__matmul__")

    def __truediv__(self, other: object):
        return self.__getattr__("__truediv__")

    def __floordiv__(self, other: object):
        return self.__getattr__("__floordiv__")

    def __mod__(self, other: object):
        return self.__getattr__("__mod__")

    def __divmod__(self, other: object):
        return self.__getattr__("__divmod__")

    def __pow__(self, other: object, modulo: object = ...):
        return self.__getattr__("__pow__")

    def __lshift__(self, other: object):
        return self.__getattr__("__lshift__")

    def __rshift__(self, other: object):
        return self.__getattr__("__rshift__")

    def __and__(self, other: object):
        return self.__getattr__("__and__")

    def __xor__(self, other: object):
        return self.__getattr__("__xor__")

    def __or__(self, other: object):
        return self.__getattr__("__or__")

    # r* and i* methods have lower priority than
    # the methods for left operand so they are skipped

    def __neg__(self):
        return self.__getattr__("__neg__")

    def __pos__(self):
        return self.__getattr__("__pos__")

    def __abs__(self):
        return self.__getattr__("__abs__")

    def __invert__(self):
        return self.__getattr__("__invert__")

    # __complex__, __int__ and __float__ have a default implementation
    # based on __index__, so they are skipped.

    def __index__(self):
        return self.__getattr__("__index__")

    def __round__(self, ndigits: object = ...):
        return self.__getattr__("__round__")

    def __trunc__(self):
        return self.__getattr__("__trunc__")

    def __floor__(self):
        return self.__getattr__("__floor__")

    def __ceil__(self):
        return self.__getattr__("__ceil__")

    # [Context managers]

    def __enter__(self):
        return self.__getattr__("__enter__")

    def __exit__(self, *args: object, **kwargs: object):
        return self.__getattr__("__exit__")


class PlaceholderModule(_PlaceholderBase):
    """
    A placeholder object to use when a module does not exist.

    This enables more informative errors when trying to access attributes
    of a module that does not exists.
    """

    def __init__(self, name: str) -> None:
        super().__init__()

        # Apply name mangling to avoid conflicting with module attributes
        self.__name = name

    def placeholder_attr(self, attr_path: str):
        return _PlaceholderModuleAttr(self, attr_path)

    def __getattr__(self, key: str):
        name = self.__name

        try:
            importlib.import_module(name)
        except ImportError as exc:
            for extra, names in get_ksana_optional_dependencies().items():
                if name in names:
                    msg = f"Please install ksana[{extra}] for {extra} support"
                    raise ImportError(msg) from exc

            raise exc

        raise AssertionError("PlaceholderModule should not be used "
                             "when the original module can be imported")


class _PlaceholderModuleAttr(_PlaceholderBase):

    def __init__(self, module: PlaceholderModule, attr_path: str) -> None:
        super().__init__()

        # Apply name mangling to avoid conflicting with module attributes
        self.__module = module
        self.__attr_path = attr_path

    def placeholder_attr(self, attr_path: str):
        return _PlaceholderModuleAttr(self.__module,
                                      f"{self.__attr_path}.{attr_path}")

    def __getattr__(self, key: str):
        getattr(self.__module, f"{self.__attr_path}.{key}")

        raise AssertionError("PlaceholderModule should not be used "
                             "when the original module can be imported")


def full_groupby(values: Iterable[_V], *, key: Callable[[_V], _K]):
    """
    Unlike [`itertools.groupby`][], groups are not broken by
    non-contiguous data.
    """
    groups = defaultdict[_K, list[_V]](list)

    for value in values:
        groups[key(value)].append(value)

    return groups.items()


class ClassRegistry(UserDict[Type[T], _V]):

    def __getitem__(self, key: Type[T]) -> _V:
        for cls in key.mro():
            if cls in self.data:
                return self.data[cls]

        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return self.contains(key)

    def contains(self, key: object, *, strict: bool = False) -> bool:
        if not isinstance(key, type):
            return False

        if strict:
            return key in self.data

        return any(cls in self.data for cls in key.mro())


class LazyLoader(types.ModuleType):
    """
    LazyLoader module borrowed from Tensorflow
    https://github.com/tensorflow/tensorflow/blob/main/tensorflow/python/util/lazy_loader.py
    with a addition of "module caching".

    Lazily import a module, mainly to avoid pulling in large dependencies.
    Modules such as `xgrammar` might do additional side effects, so we
    only want to use this when it is needed, delaying all eager effects
    """

    def __init__(
        self,
        local_name: str,
        parent_module_globals: dict[str, Any],
        name: str,
    ):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._module: types.ModuleType | None = None

        super().__init__(str(name))

    def _load(self) -> types.ModuleType:
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
            self._parent_module_globals[self._local_name] = module
            # The additional add to sys.modules
            # ensures library is actually loaded.
            sys.modules[self._local_name] = module
        except ModuleNotFoundError as err:
            raise err from None

        # Update this object's dict so that if someone keeps a
        # reference to the LazyLoader, lookups are efficient
        # (__getattr__ is only called on lookups that fail).
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: Any) -> Any:
        if self._module is None:
            self._module = self._load()
        return getattr(self._module, item)

    def __dir__(self) -> list[str]:
        if self._module is None:
            self._module = self._load()
        return dir(self._module)


def supports_kw(
    callable: Callable[..., object],
    kw_name: str,
    *,
    requires_kw_only: bool = False,
    allow_var_kwargs: bool = True,
) -> bool:
    """Check if a keyword is a valid kwarg for a callable; if requires_kw_only
    disallows kwargs names that can also be positional arguments.
    """
    params = inspect.signature(callable).parameters
    if not params:
        return False

    param_val = params.get(kw_name)

    # Types where the it may be valid, i.e., explicitly defined & nonvariadic
    passable_kw_types = set((inspect.Parameter.POSITIONAL_ONLY,
                             inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             inspect.Parameter.KEYWORD_ONLY))

    if param_val:
        is_sig_param = param_val.kind in passable_kw_types
        # We want kwargs only, but this is passable as a positional arg
        if (requires_kw_only and is_sig_param
                and param_val.kind != inspect.Parameter.KEYWORD_ONLY):
            return False
        if ((requires_kw_only
             and param_val.kind == inspect.Parameter.KEYWORD_ONLY)
                or (not requires_kw_only and is_sig_param)):
            return True

    # If we're okay with var-kwargs, it's supported as long as
    # the kw_name isn't something like *args, **kwargs
    if allow_var_kwargs:
        # Get the last param; type is ignored here because params is a proxy
        # mapping, but it wraps an ordered dict, and they appear in order.
        # Ref: https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
        last_param = params[next(reversed(params))]  # type: ignore
        return (last_param.kind == inspect.Parameter.VAR_KEYWORD
                and last_param.name != kw_name)
    return False


def resolve_mm_processor_kwargs(
    init_kwargs: Optional[Mapping[str, object]],
    inference_kwargs: Optional[Mapping[str, object]],
    callable: Callable[..., object],
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]:
    """Applies filtering to eliminate invalid mm_processor_kwargs, i.e.,
    those who are not explicit keywords to the given callable (of one is
    given; otherwise no filtering is done), then merges the kwarg dicts,
    giving priority to inference_kwargs if there are any collisions.

    In the case that no kwarg overrides are provided, returns an empty
    dict so that it can still be kwarg expanded into the callable later on.

    If allow_var_kwargs=True, allows for things that can be expanded into
    kwargs as long as they aren't naming collision for var_kwargs or potential
    positional arguments.
    """
    # Filter inference time multimodal processor kwargs provided
    runtime_mm_kwargs = get_allowed_kwarg_only_overrides(
        callable,
        overrides=inference_kwargs,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs,
    )

    # Filter init time multimodal processor kwargs provided
    init_mm_kwargs = get_allowed_kwarg_only_overrides(
        callable,
        overrides=init_kwargs,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs,
    )

    # Merge the final processor kwargs, prioritizing inference
    # time values over the initialization time values.
    mm_processor_kwargs = {**init_mm_kwargs, **runtime_mm_kwargs}
    return mm_processor_kwargs


def get_allowed_kwarg_only_overrides(
    callable: Callable[..., object],
    overrides: Optional[Mapping[str, object]],
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]:
    """
    Given a callable which has one or more keyword only params and a dict
    mapping param names to values, drop values that can be not be kwarg
    expanded to overwrite one or more keyword-only args. This is used in a
    few places to handle custom processor overrides for multimodal models,
    e.g., for profiling when processor options provided by the user
    may affect the number of mm tokens per instance.

    Args:
        callable: Callable which takes 0 or more keyword only arguments.
                  If None is provided, all overrides names are allowed.
        overrides: Potential overrides to be used when invoking the callable.
        allow_var_kwargs: Allows overrides that are expandable for var kwargs.

    Returns:
        Dictionary containing the kwargs to be leveraged which may be used
        to overwrite one or more keyword only arguments when invoking the
        callable.
    """
    if not overrides:
        return {}

    # Drop any mm_processor_kwargs provided by the user that
    # are not kwargs, unless it can fit it var_kwargs param
    filtered_overrides = {
        kwarg_name: val
        for kwarg_name, val in overrides.items()
        if supports_kw(callable,
                       kwarg_name,
                       requires_kw_only=requires_kw_only,
                       allow_var_kwargs=allow_var_kwargs)
    }

    # If anything is dropped, log a warning
    dropped_keys = overrides.keys() - filtered_overrides.keys()
    if dropped_keys:
        if requires_kw_only:
            logger.warning(
                "The following intended overrides are not keyword-only args "
                "and will be dropped: %s", dropped_keys)
        else:
            logger.warning(
                "The following intended overrides are not keyword args "
                "and will be dropped: %s", dropped_keys)

    return filtered_overrides
