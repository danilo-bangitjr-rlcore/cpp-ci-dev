from dataclasses import dataclass, field
from typing import Any
from hydra.core.config_store import ConfigStore
from hydra import initialize, compose
from omegaconf import MISSING

from corerl.utils.hydra import Group

def test_group1():
    # -------------------------
    # -- Setup dummy configs --
    # -------------------------
    @dataclass
    class Sub1:
        name: str = 'sub1'
        test_a: str = 'a'

    @dataclass
    class Sub2:
        name: str = 'sub2'
        test_b: int = 22

    @dataclass
    class MainConfig:
        sub: Any = MISSING
        defaults: list[Any] = field(default_factory=lambda: [
            {'sub': 'sub1'},
        ])


    # --------------------
    # -- Setup handlers --
    # --------------------
    group = Group[[], str]('sub')
    @group.dispatcher
    def _sub1_handler(cfg: Sub1):
        return f'Sub1 handler <{cfg.test_a}>'


    @group.dispatcher
    def _sub2_handler(cfg: Sub2):
        return f'Sub2 handler <{cfg.test_b}>'


    # -----------------------
    # -- Setup main config --
    # -----------------------
    cs = ConfigStore.instance()
    cs.store(name='main', node=MainConfig)


    # test: if group uses config `sub1`
    # then the _sub1_handler is called
    with initialize(version_base=None):
        cfg = compose(config_name='main', overrides=['sub=sub1'])

        got = group.dispatch(cfg.sub)
        assert got == 'Sub1 handler <a>'


    # test: if group uses config `sub2`
    # then the _sub2_handler is called
    with initialize(version_base=None):
        cfg = compose(config_name='main', overrides=['sub=sub2'])

        got = group.dispatch(cfg.sub)
        assert got == 'Sub2 handler <22>'
