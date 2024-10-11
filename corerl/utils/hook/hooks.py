from __future__ import annotations
from typing import Any, Callable, Union, List, Tuple
from corerl.utils.hook.when import Agent, Env


When = Agent | Env

class Hooks:
    def __init__(self, keys=None):
        self._hooks: dict[When, list[Callable[..., Any]]] = {}
        if keys is not None:
            for k in keys:
                self._hooks[k] = []

    def __call__(self, when: When, *args, **kwargs):
        if when not in self._hooks.keys() or len(self._hooks[when]) == 0:
            return args, kwargs
        for f in self._hooks[when]:
            # Hook forwarding --> hook[i+1] uses the output of hook[i]
            args, kwargs = f(*args, **kwargs)
        return args, kwargs

    def call(self, when: When, *args, **kwargs):
        return self(when, *args, **kwargs)

    def register(self, hook: Hook | Callable[..., Any], when: When):
        if not isinstance(hook, Hook) and callable(hook):
            hook = Hook(hook)

        if when not in self._hooks.keys():
            self._hooks[when] = [hook]
        else:
            self._hooks[when].append(hook)

        hook.register_with(self, when)

# A hook is a function which is called at a specific time (see `When` above).
# Hooks must always return their arguments and keyword arguments as a separate
# tuple, dict. It is then the job of the caller to assign these values, if
# appropriate.
#
# Right now, I am passing the agent as an argument to the hook, but I am not
# sure this is a good idea. We could just capture the agent with a closure, if
# need be...
class Hook:
    def __init__(self, f: Callable[..., Any], name: str | None = None):
        self._f = f
        if name is None:
            name = repr(self._f)
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self, *args, **kwargs):
        _args, _kwargs = self._f(*args, **kwargs)
        msg = (
            "expected hook output at position 0 to be a tuple, list, or " +
            f"None but got {type(_args)}"
        )
        assert isinstance(_args, Union[List, Tuple, None]), msg
        if _args is None:
            _args = args

        msg = (
            "expected hook output at position 1 to be a dict " +
            f"but got {type(_kwargs)}"
        )
        assert isinstance(_kwargs, dict), msg
        if len(_kwargs) == 0:
            _kwargs = kwargs

        return _args, _kwargs

    def register_with(self, hooks: Hooks, when: When):
        self._when = when
        self._hooks = hooks
        return None
