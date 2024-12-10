from corerl.utils.hydra import Group
from corerl.data.base_tc import BaseStateConstructor, BaseTransitionCreator
from corerl.data.transition_creator import AnytimeTransitionCreator, RegularRLTransitionCreator

tc_group = Group[
    [BaseStateConstructor],
    BaseTransitionCreator,
](['agent_transition_creator', 'alert_transition_creator'])

tc_group.dispatcher(AnytimeTransitionCreator)
tc_group.dispatcher(RegularRLTransitionCreator)
