from lib_config.config import config


@config()
class ComputedTag:
    is_computed: bool = False
    """
    Kind: optional external

    Specifies whether this is a computed virtual tag.
    """

    value: str | None = None
    """
    Kind: optional external

    If this is a computed virtual tag, then a value string must be specified
    in order to construct the value of the tag as a function of other tags.
    """
