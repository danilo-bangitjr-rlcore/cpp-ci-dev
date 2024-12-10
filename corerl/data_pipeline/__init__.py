from corerl.data_pipeline.base import BaseDataLoader, OldBaseDataLoader
from corerl.data_pipeline.direct_action import DirectActionDataLoader, OldDirectActionDataLoader
from corerl.utils.hydra import Group

# set up config groups
dl_group = Group[[], OldBaseDataLoader | BaseDataLoader](
    'data_loader',
)

dl_group.dispatcher(OldDirectActionDataLoader)
dl_group.dispatcher(DirectActionDataLoader)
