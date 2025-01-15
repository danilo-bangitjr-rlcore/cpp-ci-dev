from torch import Tensor

from corerl.models.dummy import DummyModel, DummyModelConfig
from corerl.utils.torch import tensor_allclose


def test_dummy():
    cfg = DummyModelConfig()
    model = DummyModel(cfg)
    model.fit([])

    state = Tensor([0])
    action = Tensor([1])
    obs, _ = model.predict(
        state=state,
        action=action,
    )
    assert tensor_allclose(obs, state)
