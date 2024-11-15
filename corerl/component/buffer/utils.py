from pathlib import Path
import pickle as pkl
import numpy
from typing import Tuple

from corerl.component.buffer.buffers import EnsembleUniformBuffer
from corerl.data.data import Transition


def load_pkl_buffer(path: Path, mode: str, **kwargs) \
        -> Tuple[EnsembleUniformBuffer, EnsembleUniformBuffer]:
    """
    The function assumes transitions in pickle file are in order.
    It could be improved by adding the timestamp info in transition object or buffer pickle.
    """
    critic_buffer_path = path / "critic_buffer.pkl"
    with open(critic_buffer_path, "rb") as f:
        critic_buffer = pkl.load(f)

    policy_buffer_path = path / "policy_buffer.pkl"
    with open(policy_buffer_path, "rb") as f:
        policy_buffer = pkl.load(f)

    critic_buffer = subsampling_buffer(critic_buffer, mode, **kwargs)
    policy_buffer = subsampling_buffer(policy_buffer, mode, **kwargs)
    return critic_buffer, policy_buffer

def subsampling_buffer(buffer: [EnsembleUniformBuffer], mode: str, **kwargs) \
        -> EnsembleUniformBuffer:
    """
    The function assumes transitions in buffer are in order.
    It could be improved by adding the timestamp info in transition object or buffer pickle.
    """
    idxs = [get_loaded_transitions_idx(size_, mode, **kwargs)
             for size_ in buffer.size
             ]
    buffer.subsampling(idxs)
    return buffer

def subsampling_transitions(trans: list[Transition], mode: str, **kwargs) -> list[Transition]:
    trans_size = len(trans)
    load_idxs = get_loaded_transitions_idx(trans_size, mode, **kwargs)
    subsampled_trans = [trans[i] for i in load_idxs]
    return subsampled_trans

def get_loaded_transitions_idx(size_: int, mode: str, **kwargs) -> list[int]:
    if mode == "all":
        idxs = list(range(size_))
    elif mode == "random":
        rng = numpy.random.RandomState(kwargs['seed'])
        idxs = rng.choice(size_,
                          size=int(kwargs['subsampling_perc'] * size_),
                          replace=False)
    elif mode == "latest":
        idxs = numpy.arange(int((1. - kwargs['subsampling_perc']) * size_), size_)
    else:
        raise NotImplementedError("Buffer: Unknown subsampling mode.")
    return idxs