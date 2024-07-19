from enum import StrEnum
from typing import Union, List, Tuple

class When(StrEnum):
    """
    When defines a time point when a hook will be called during agent training.

    Each hook at a different time point is passed different arguments and
    keyword arguments. The hooks are required to return their (possibly
    modified) arguments and keyword arguments. The agent can then re-assign
    these returned values as it sees fit.

    Below, we list the times at which hooks are called as well as the
    functional form that the hook function should satisfy using the notation:

        f(arguments; keyword arguments) -> returned value

    where `f` is the hook function being registered.

    - AfterCreate = "after_create":
        f(agent) -> agent
    - BeforeGetAction = "before_get_action":
        f(agent, state) -> state
    - AfterGetAction = "after_get_action":
        f(agent, state, action) -> state, action
    - BeforeUpdateCriticBuffer = "before_update_critic_buffer":
        f(agent, transition) -> transition
    - AfterCriticBufferSample = "before_critic_buffer_sample":
        f(agent, batch) -> batch
    - BeforeCriticLossComputed = "before_critic_loss_computed":
        f(agent, batch, target_q, q) -> batch, target_q, q
    - AfterCriticLossComputed = "after_critic_loss_computed":
        f(agent, batch, loss) -> batch, loss
    - AfterCriticUpdate = "after_critic_update":
        f(agent, batch, prev_loss)
    - BeforeUpdateActorBuffer = "before_update_actor_buffer":
        f(agent, transition) -> transition
    - AfterActorBufferSample = "before_actor_buffer_sample":
        f(agent, batch) -> batch
    - BeforeActorLossComputed = "before_actor_loss_computed":
        f(agent, update_info) -> update_info
    - AfterActorLossComputed = "after_actor_loss_computed":
        f(agent, batch, update_info, loss) -> batch, update_info, loss
    - AfterActorUpdate = "after_actor_update":
        f(agent, batch, prev_loss)
    """
    # agent <- f(agent)
    AfterCreate = "after_create"

    # state <- f(agent, state)
    BeforeGetAction = "before_get_action"

    # state, action <- f(agent, state, action)
    AfterGetAction = "after_get_action"

    # transition <- f(agent, transition)
    BeforeUpdateCriticBuffer = "before_update_critic_buffer"

    # batch <- f(agent, batch)
    AfterCriticBufferSample = "before_critic_buffer_sample"

    # batch, target_q, q <- f(agent, batch, target_q, q)
    BeforeCriticLossComputed = "before_critic_loss_computed"

    # batch, loss <- f(agent, batch, loss)
    AfterCriticLossComputed = "after_critic_loss_computed"

    # Ideally we would also have: AfterCriticBackward and AfterCriticOptStep,
    # but there is a separate class that takes care of the backward pass and
    # optimizer step outside of the agent, which complicates these two callback
    # times

    # <- f(agent, batch, prev_loss)
    # After the backward pass and optimizer step
    AfterCriticUpdate = "after_critic_update"


    # transition <- f(agent, transition)
    BeforeUpdateActorBuffer = "before_update_actor_buffer"

    # batch <- f(agent, batch)
    AfterActorBufferSample = "before_actor_buffer_sample"

    # update_info <- f(agent, update_info)
    #   update_info are agent-specific arguments
    BeforeActorLossComputed = "before_actor_loss_computed"

    # batch, update_info, loss <- f(agent, batch, update_info, loss)
    AfterActorLossComputed = "after_actor_loss_computed"

    # Ideally we would also have: AfterActorBackward and AfterActorOptStep,
    # but there is a separate class that takes care of the backward pass and
    # optimizer step outside of the agent, which complicates these two callback
    # times

    # <- f(agent, batch, prev_loss)
    # After the backward pass and optimizer step
    AfterActorUpdate = "after_actor_update"


class Hooks:
    def __init__(self, keys=None):
        self._hooks = {}
        if keys is not None:
            for k in keys:
                self._hooks[k] = []

    def __call__(self, when, *args, **kwargs):
        if when not in self._hooks.keys() or len(self._hooks[when]) == 0:
            return args, kwargs
        for f in self._hooks[when]:
            # Hook forwarding --> hook[i+1] uses the output of hook[i]
            args, kwargs = f(*args, **kwargs)
        return args, kwargs

    def call(self, when, *args, **kwargs):
        return self(when, *args, **kwargs)

    def register(self, hook, when):
        if not isinstance(hook, Hook) and callable(hook):
            hook = Hook(hook)

        if when not in self._hooks.keys():
            self._hooks[when] = [hook]
        else:
            self._hooks[when].append(hook)

        hook.register_with(self, when)

    def swap(self, i, j):
        self[i], self[j] = self[j], self[i]

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        out = self._hooks
        for index in indices:
            out = out[index]
        return out

# A hook is a function which is called at a specific time (see `When` above).
# Hooks must always return their arguments and keyword arguments as a separate
# tuple, dict. It is then the job of the caller to assign these values, if
# appropriate.
#
# Right now, I am passing the agent as an argument to the hook, but I am not
# sure this is a good idea. We could just capture the agent with a closure, if
# need be...
class Hook:
    def __init__(self, f, name=None):
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

    def register_with(self, hooks, when):
        self._when = when
        self._hooks = hooks
        return None

    def remove(self):
        return self._hooks[self._when].remove(self)

