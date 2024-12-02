from typing import cast
import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import MissingType

def update_existing_missing_info_col(
        missing_info: pd.DataFrame,
        name: str,
        missing_mask: np.ndarray,
        new_val: MissingType,
):
    """
    Updates a column of a dataframe filled with MissingType's to (prev_val | new_val).
        name: determines the column to update.
        missing_mask: a mask that indicates which rows in the 'name' column to update

    example -
        with args: name='sensor_x', new_val=MissingType.OUTLIER,
        if the existing MissingType at the row indicated by missing_mask was `MissingType.BOUNDS`,
        this function will update it to `MissingType.BOUNDS | MissingType.OUTLIER`.

    """
    if not missing_mask.any():
        return

    for idx, prev_val in missing_info.loc[missing_mask, name].items():
        idx = cast(int, idx)
        updated_val = MissingType(prev_val) | new_val
        missing_info.loc[idx, name] = updated_val


def update_missing_info_col(missing_info: pd.DataFrame, name: str, missing_type_mask: np.ndarray, new_val: MissingType):
    # Update existing missing info
    existing_missing_mask = missing_info[name] != MissingType.NULL
    overlap_mask = existing_missing_mask & missing_type_mask  # <- Series & np.ndarray results in Series
    update_existing_missing_info_col(
        missing_info=missing_info,
        name=name,
        missing_mask=overlap_mask.to_numpy(),
        new_val=new_val,
    )

    # Add new missing info
    new_missing_mask = ~existing_missing_mask & missing_type_mask
    missing_info.loc[new_missing_mask, name] = [new_val] * sum(new_missing_mask)
