from corerl.environment.wrapper.mcar import MCARWrapper
from corerl.environment.wrapper.sticky_mcar import StickyMCARWrapper

wrappers = {
    "mcar": MCARWrapper,
    "sticky_mcar": StickyMCARWrapper,
}
