from torch import Tensor

from corerl.models.dummy import DummyEndoModel, DummyEndoModelConfig
from corerl.utils.torch import tensor_allclose
from corerl.data_pipeline.tag_config import TagConfig


def test_dummy():
    cfg = DummyEndoModelConfig()
    tags = [
        TagConfig(
            name='state1',
            is_endogenous=False,
        ),
        TagConfig(
            name='state2',
            is_endogenous=True,
        ),

        TagConfig(
            name='action',
            is_action=True,
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
