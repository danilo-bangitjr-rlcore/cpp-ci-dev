from corerl.utils.hydra import Group
from corerl.data_pipeline.base import OldBaseDataLoader, BaseDataLoader
from corerl.data_pipeline.direct_action import OldDirectActionDataLoader, DirectActionDataLoader
# set up config groups
dl_group = Group[[], OldBaseDataLoader | BaseDataLoader](
    'data_loader',
)

dl_group.dispatcher(OldDirectActionDataLoader)
dl_group.dispatcher(DirectActionDataLoader)
