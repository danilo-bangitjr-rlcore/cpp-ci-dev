from corerl.state_constructor.base import CompositeStateConstructor
from corerl.utils.hydra import Group

# set up config groups
sc_group = Group[[],CompositeStateConstructor,]('state_constructor')

def register():
    from corerl.state_constructor.examples import (
        AnytimeMultiTrace,
        Identity,
        MultiTrace,
        ReseauAnytime,
        SimpleReseauAnytime,
    )

    sc_group.dispatcher(MultiTrace)
    sc_group.dispatcher(AnytimeMultiTrace)
    sc_group.dispatcher(Identity)
    sc_group.dispatcher(SimpleReseauAnytime)
    sc_group.dispatcher(ReseauAnytime)
