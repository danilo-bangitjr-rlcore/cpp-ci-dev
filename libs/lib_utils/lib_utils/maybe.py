from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, overload

from lib_utils.list import find


class Maybe[T]:
    def __init__(self, v: T | None):
        self._v: T | None = v


    # ------------------------
    # -- Positive Accessors --
    # ------------------------
    def map[U](self, f: Callable[[T], U | None]) -> Maybe[U]:
        """
        Transform a Maybe[T] -> Maybe[U] using a
        mapping function (T) -> U.

        If the Maybe contains a None, then this function
        is not called.
        """
        if self._v is None:
            return Maybe[U](None)

        u = f(self._v)
        return Maybe(u)


    def flat_map[U](self, f: Callable[[T], Maybe[U]]) -> Maybe[U]:
        """
        Like map, except when the mapping function itself
        returns a Maybe[U]. Using `map(f)` would result in
        a `Maybe[Maybe[U]]` which is often not desirable.

        Flat functions flatten one level of Maybe, turning
        a `Maybe[Maybe[U]]` into a `Maybe[U]`.
        """
        if self._v is None:
            return Maybe[U](None)

        return f(self._v)


    def tap(self, f: Callable[[T], Any]):
        """
        For explicitly calling side-effecty functions.
        Similar to `map(f)`, except that the given
        function _does not_ store its returned type.

        Useful, for instance, for printing:
        ```python
        Maybe(thing) \
          .tap(print)
        ```
        """
        self.map(f)
        return self

    # ------------------------
    # -- Negative Accessors --
    # ------------------------
    def otherwise[U](self, f: Callable[[], U | None]) -> Maybe[T] | Maybe[U]:
        """
        The same, but opposite, of `map(f)`.
        Whenever the Maybe[T] contains a None, the
        passed function is called with no arguments.
        """
        if self._v is None:
            return Maybe(f())

        return self


    def flat_otherwise[U](self, f: Callable[[], Maybe[U]]) -> Maybe[T] | Maybe[U]:
        """
        The same, but opposite, of `flat_map(f)`.
        Whenever the Maybe[T] contains a None, the
        passed function is called with no arguments.

        If the passed function returns a `Maybe[T]`,
        this is flattened into the return type of this
        function, giving a `Maybe[T]` instead of a
        `Maybe[Maybe[T]]`.
        """
        if self._v is None:
            return f()

        return self


    @overload
    def or_else[U](self, t: U | None, msg: str) -> T | U: ...
    # first try to match with type T or any type covariant to T
    # for instance, if T is list[A | None], then match on list[None]
    @overload
    def or_else(self, t: T) -> T: ...
    # otherwise, allow any type U that is not T
    @overload
    def or_else[U](self, t: U) -> T | U: ...

    def or_else[U](self, t: U | None, msg: str = '') -> T | U:
        """
        Pops out of the Maybe, providing back a raw
        type. If the Maybe contains a None, then
        the default value provided to `or_else` is
        returned.

        If the default value provided to `or_else`
        is also None, then an exception is raised.
        """

        if self._v is None and t is not None:
            return t

        return self.expect(msg)


    # ----------------
    # -- Unwrappers --
    # ----------------
    def expect(self, msg: str | Exception = '') -> T:
        if self._v is None:
            if isinstance(msg, Exception):
                raise msg

            raise Exception(msg)

        return self._v


    def unwrap(self) -> T | None:
        return self._v

    # ---------------------
    # -- Status Checkers --
    # ---------------------
    def is_none(self):
        return self._v is None


    def is_some(self):
        return self._v is not None

    # ---------------
    # -- Utilities --
    # ---------------

    def split[U, V](self, u: type[U], v: type[V], /) -> tuple[Maybe[U], Maybe[V]]:
        if not isinstance(self._v, Sequence): # captures self._v is None
            return Maybe[U](None), Maybe[V](None)

        left = Maybe(self._v[0]).is_instance(u)
        right = Maybe(self._v[1]).is_instance(v)
        return left, right

    @overload
    @staticmethod
    def tap_all[U, V, W](f: Callable[[U, V, W], Any], a: Maybe[U], b: Maybe[V], c: Maybe[W], /): ...
    @overload
    @staticmethod
    def tap_all[U, V](f: Callable[[U, V], Any], a: Maybe[U], b: Maybe[V], /): ...
    @staticmethod
    def tap_all(f: Callable[..., Any], *args: Maybe[Any]):
        un_args = [args.unwrap() for args in args]
        if any(arg is None for arg in un_args):
            return

        f(*un_args)

    def is_instance[U](self, typ: type[U]) -> Maybe[U]:
        if isinstance(self._v, typ):
            return Maybe[U](self._v)

        return Maybe(None)


    @staticmethod
    def from_try[U](
        f: Callable[[], U],
        e: type[Exception] = Exception,
    ) -> Maybe[U]:
        try:
            return Maybe(f())
        except e:
            logging.exception('Maybe.from_try caught an exception')
            return Maybe(None)


    @staticmethod
    def flat_from_try[U](
        f: Callable[[], Maybe[U]],
        e: type[Exception] = Exception,
    ) -> Maybe[U]:
        try:
            return f()
        except e:
            logging.exception('Maybe.flat_from_try caught an exception')
            return Maybe(None)


    @staticmethod
    def find[U](pred: Callable[[U], bool], li: Iterable[U]) -> Maybe[U]:
        return Maybe[U](find(pred, li))
