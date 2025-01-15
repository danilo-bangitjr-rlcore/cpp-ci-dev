from torch import Tensor

from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import NullConfig
from corerl.models.dummy import DummyEndoModel, DummyEndoModelConfig
from corerl.utils.torch import tensor_allclose


def test_dummy():
    cfg = DummyEndoModelConfig()
    tags = [
        TagConfig(
            name='tag-1',
            is_endogenous=False,
        ),
        TagConfig(
            name='tag-2',
            is_endogenous=True,
        ),

        TagConfig(
            name='action',
            action_constructor=[],
            state_constructor=[NullConfig()]
        ),
    ]

    model = DummyEndoModel(cfg, tags)
    model.fit([])

    state = Tensor([2, 3])
    action = Tensor([1])
    obs, _ = model.predict(
        state=state,
        action=action,
    )
    assert tensor_allclose(obs, Tensor([2]))
