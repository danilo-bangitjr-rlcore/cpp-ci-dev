from corerl.data_pipeline.transforms.add_raw import AddRawConfig
from corerl.data_pipeline.transforms.affine import AffineConfig
from corerl.data_pipeline.transforms.greater_than import GreaterThanConfig
from corerl.data_pipeline.transforms.identity import IdentityConfig
from corerl.data_pipeline.transforms.less_than import LessThanConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.null import NullConfig
from corerl.data_pipeline.transforms.scale import ScaleConfig
from corerl.data_pipeline.transforms.split import SplitConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.data_pipeline.transforms.product import ProductConfig


TransformConfig = (
    AddRawConfig
    | AffineConfig
    | GreaterThanConfig
    | IdentityConfig
    | LessThanConfig
    | NormalizerConfig
    | NullConfig
    | ProductConfig
    | ScaleConfig
    | SplitConfig
    | TraceConfig
)
