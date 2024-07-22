from enum import StrEnum


class Agent(StrEnum):
    """
    when.Agent defines a time point when a hook will be called during agent
    training.

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
    AfterProposalBufferSample = "before_proposal_buffer_sample"

    # update_info <- f(agent, update_info)
    #   update_info are agent-specific arguments
    BeforeActorLossComputed = "before_actor_loss_computed"
    BeforeProposalLossComputed = "before_proposal_loss_computed"

    # batch, update_info, loss <- f(agent, batch, update_info, loss)
    AfterActorLossComputed = "after_actor_loss_computed"
    AfterProposalLossComputed = "after_proposal_loss_computed"

    # Ideally we would also have: AfterActorBackward and AfterActorOptStep,
    # but there is a separate class that takes care of the backward pass and
    # optimizer step outside of the agent, which complicates these two callback
    # times

    # <- f(agent, batch, prev_loss)
    # After the backward pass and optimizer step
    AfterActorUpdate = "after_actor_update"
    AfterProposalUpdate = "after_proposal_update"


class Env(StrEnum):
    """
    when.Env defines a time point when a hook will be called during environment
    interaction.

    Each hook at a different time point is passed different arguments and
    keyword arguments. The hooks are required to return their (possibly
    modified) arguments and keyword arguments. The environment can then
    re-assign these returned values as it sees fit.

    Below, we list the times at which hooks are called as well as the
    functional form that the hook function should satisfy using the notation:

        f(arguments; keyword arguments) -> returned value

    where `f` is the hook function being registered.

    - AfterCreate = "after_create":
        f(env) -> env
    - BeforeReset = "before_reset":
        f(env, state) -> env, state
    - AfterReset = "after_reset":
        f(env, state) -> env, state
    - BeforeStep = "before_step":
        f(env, state, action) -> env, state, action
    - AfterStep = "after_step":
        f(
            env, state, action, reward, next_state, term, trunc
        ) -> env, state, action, reward, next_state, term, trunc
    """
    AfterCreate = "after_create"
    BeforeReset = "before_reset"
    AfterReset = "after_reset"
    BeforeStep = "before_step"
    AfterStep = "after_step"
