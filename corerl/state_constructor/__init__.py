from corerl.utils.hydra import Group
from corerl.state_constructor.base import CompositeStateConstructor
from corerl.state_constructor.examples import (
    MultiTrace,
    AnytimeMultiTrace,
    Identity,
    SimpleReseauAnytime,
    ReseauAnytime
)

# set up config groups
sc_group = Group[[],CompositeStateConstructor,]('state_constructor')

sc_group.dispatcher(MultiTrace)
sc_group.dispatcher(AnytimeMultiTrace)
sc_group.dispatcher(Identity)
sc_group.dispatcher(SimpleReseauAnytime)
sc_group.dispatcher(ReseauAnytime)
