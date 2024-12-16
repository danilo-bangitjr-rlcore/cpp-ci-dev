from typing import Literal
from pydantic import Field
from corerl.configs.config import config
from corerl.configs.group import Group

def test_group1():
    # -------------------------
    # -- Setup dummy configs --
    # -------------------------
    @config()
    class Sub1:
        name: Literal['sub1'] = 'sub1'
        test_a: str = 'a'

    @config()
    class Sub2:
        name: Literal['sub2'] = 'sub2'
        test_b: int = 22

    @config()
    class MainConfig:
        sub: Sub1 | Sub2 = Field(default_factory=Sub1, discriminator='name')


    # --------------------
    # -- Setup handlers --
    # --------------------
    group = Group[[], str]()
    @group.dispatcher
    def _sub1_handler(cfg: Sub1):
        return f'Sub1 handler <{cfg.test_a}>'


    @group.dispatcher
    def _sub2_handler(cfg: Sub2):
        return f'Sub2 handler <{cfg.test_b}>'


    # -----------------------
    # -- Setup main config --
    # -----------------------
    # test: if group uses config `sub1`
    # then the _sub1_handler is called
    cfg = MainConfig(
        sub=Sub1(),
    )
    got = group.dispatch(cfg.sub)
    assert got == 'Sub1 handler <a>'


    # test: if group uses config `sub2`
    # then the _sub2_handler is called
    cfg = MainConfig(
        sub=Sub2(),
    )
    got = group.dispatch(cfg.sub)
    assert got == 'Sub2 handler <22>'
