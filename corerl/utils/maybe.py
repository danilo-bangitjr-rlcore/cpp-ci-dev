from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, overload


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
    def otherwise(self, f: Callable[[], T | None]) -> Maybe[T]:
        """
        The same, but opposite, of `map(f)`.
        Whenever the Maybe[T] contains a None, the
        passed function is called with no arguments.
        """
        if self._v is None:
            return Maybe(f())

        return self


    def flat_otherwise(self, f: Callable[[], Maybe[T]]) -> Maybe[T]:
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
    def or_else(self, t: T | None, msg: str) -> T: ...
    @overload
    def or_else(self, t: T) -> T: ...

    def or_else(self, t: T | None, msg: str = '') -> T:
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
    def expect(self, msg: str = '') -> T:
        if self._v is None:
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
