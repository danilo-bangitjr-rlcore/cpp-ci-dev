from typing import Any, Type, TypeVar, get_args, get_type_hints

T = TypeVar('T')

def cast_dict_to_config(config_dict: dict, config_class: Type[T]) -> T:
    """
    Convert a dictionary to a config object
    """

    config_obj = config_class()

    try:
        type_hints = get_type_hints(config_class)
    except (TypeError, NameError):
        type_hints = {attr: type(getattr(config_obj, attr))
                     for attr in dir(config_obj)
                     if not attr.startswith('_') and not callable(getattr(config_obj, attr))}

    for key, value in config_dict.items():
        if not hasattr(config_obj, key):
            continue
        if value is None:
            continue

        target_type = type_hints.get(key, type(getattr(config_obj, key)))

        if (not isinstance(value, (str, dict)) and
            (target_type is type(value) or
             (hasattr(target_type, "__origin__") and target_type.__origin__ is type(value)))):
            setattr(config_obj, key, value)
            continue

        if isinstance(value, str):
            try:
                if target_type is bool:
                    if value.lower() in ('true', 'yes', '1', 'y', 'on'):
                        value = True
                    elif value.lower() in ('false', 'no', '0', 'n', 'off'):
                        value = False
                elif target_type is int:
                    value = int(float(value))
                elif target_type is float:
                    value = float(value)
                elif hasattr(target_type, "__origin__") and target_type.__origin__ in (list, tuple, set):
                    container_type = target_type.__origin__ or list  # Default to list if None
                    if ',' in value:
                        items = [item.strip() for item in value.split(',')]
                        if get_args(target_type):
                            item_type = get_args(target_type)[0]
                            if item_type is not Any:
                                items = [_cast_value(item, item_type) for item in items]
                        value = container_type(items)
                    else:
                        value = container_type([value])
            except (ValueError, TypeError):
                pass

        setattr(config_obj, key, value)

    return config_obj

def _cast_value(value: Any, target_type: Type) -> Any:
    if isinstance(value, target_type):
        return value

    if isinstance(value, str):
        if target_type is bool:
            return value.lower() in ('true', 'yes', '1', 'y', 'on')
        elif target_type is int:
            return int(float(value))
        elif target_type is float:
            return float(value)

    try:
        return target_type(value)
    except (ValueError, TypeError):
        return value
