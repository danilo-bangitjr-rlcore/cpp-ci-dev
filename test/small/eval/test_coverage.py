import numpy as np
import pandas as pd
import pytest

from corerl.eval.coverage import (
    CoverageProtocol,
    Dataset,
    KDECoverage,
    NeighboursCoverage,
    UniformActionSampler,
    UniformDatasetSampler,
    get_norm_const,
    sample_epsilon_ball,
)


@pytest.fixture
def dataset():
    data = pd.DataFrame({
        'tag-1':  np.arange(0, 1, 0.1),
        'tag-2':  np.array([0, 1] * 5),
    })
    dataset = Dataset(
        data=data,
    )
    return dataset


@pytest.fixture
def tiny_dataset():
    data = pd.DataFrame({
        'tag-1':  [0, 1],
    })
    dataset = Dataset(
        data=data,
    )
    return dataset



class DumbCoverage:
    def __init__(self, epsilon: float, n_samples: int):
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.normalization = None

    def unnorm_cov(self, state_action: np.ndarray) ->  np.ndarray:
        assert self.dataset is not None
        return np.array([0.5])

    def cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.normalization is not None
        return self.unnorm_cov(state_action) / self.normalization

    def fit(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.normalization = get_norm_const(self, dataset, self.epsilon, self.n_samples)


def test_sample_epsilon_ball():
    center = np.array([0.5, 0.5])
    epsilon = 0.1
    n_samples = 10
    samples = sample_epsilon_ball(center, epsilon, n_samples)
    assert samples.shape == (n_samples, len(center)), "Sample shape mismatch"
    assert np.allclose(np.linalg.norm(samples - center, axis=1), epsilon), "Samples should be within epsilon distance"


def test_get_norm_const(dataset: Dataset):
    epsilon = 0.1
    n_samples = 100
    test_coverage = DumbCoverage(epsilon, n_samples)
    test_coverage.fit(dataset)

    normalization = get_norm_const(test_coverage, dataset, epsilon, n_samples)
    assert normalization > 0, "Normalization should be greater than 0"
    assert isinstance(normalization, float), "Normalization should be a float value"


def test_kde_coverage(dataset: Dataset):
    epsilon = 0.1
    n_samples = 100
    kde_coverage = KDECoverage(epsilon, n_samples)
    kde_coverage.fit(dataset)

    # Test the __call__ method for arbitrary state_action pair
    state_action = np.array([[0.5, 0.5]])
    coverage_value = kde_coverage.cov(state_action)
    assert coverage_value > 0, "Coverage value should be greater than 0"

    # Test that the normalization is greater than 0
    normalization = get_norm_const(kde_coverage, dataset, epsilon, n_samples)
    assert normalization > 0, "Normalization should be greater than 0"



def _test_uniform_dataset_sampler(
        coverage_fn : CoverageProtocol,
        dataset: Dataset,
        n_state_samples: int,
        ):

    sampler = UniformDatasetSampler()
    # Evaluate coverage on the dataset itself (should be very well-covered)
    coverage_value = sampler.eval(dataset, coverage_fn, n_state_samples)
    assert 0 <= abs(coverage_value ) <= 1, "Coverage value should be between 0 and 1."

    # Evaluate coverage on a test dataset which is very far from the original dataset.
    test_data = pd.DataFrame({
        'tag-1':  [10, 10],
    })
    test_dataset = Dataset(
        data=test_data,
    )

    coverage_value = sampler.eval(test_dataset, coverage_fn, n_state_samples)
    assert abs(coverage_value) >= 1, "Coverage value should be larger than 1."



def test_kde_with_uniform_dataset_sampler(tiny_dataset: Dataset):
    np.random.seed(42)
    epsilon = 0.1
    n_norm_samples = 1000
    n_state_samples = 1

    kde_coverage = KDECoverage(epsilon, n_norm_samples)
    kde_coverage.fit(tiny_dataset)
    _test_uniform_dataset_sampler(kde_coverage, tiny_dataset, n_state_samples)


def test_neighbours_with_uniform_dataset_sampler(tiny_dataset: Dataset):
    np.random.seed(42)
    epsilon = 0.1
    n_norm_samples = 1000
    n_state_samples = 1

    neighbours_coverage = NeighboursCoverage(epsilon, n_norm_samples)
    neighbours_coverage.fit(tiny_dataset)
    _test_uniform_dataset_sampler(neighbours_coverage, tiny_dataset, n_state_samples)


def _test_uniform_action_sampler(
        coverage_fn: CoverageProtocol,
        dataset: Dataset,
        n_state_samples: int,
        n_action_samples: int,
    ):
    sampler = UniformActionSampler()

    # Evaluate coverage on the dataset itself
    coverage_value = sampler.eval(dataset, coverage_fn, n_state_samples, n_action_samples)
    assert coverage_value > 0

    half_dataset_size = len(dataset.data) // 2
    test_tag_2 = list(np.random.uniform(0, 1, half_dataset_size))

    # make a test dataset with only the states with high action coverage
    test_data_high_cov = pd.DataFrame({
        'tag-1':  [0]*half_dataset_size, # "observation" tag
        'tag-2': test_tag_2, # "action" tag
    })

    test_dataset_high_cov = Dataset(
        data=test_data_high_cov,
        action_tags=['tag-2']
    )

    coverage_value_1 = sampler.eval(test_dataset_high_cov, coverage_fn, n_state_samples, n_action_samples)

    # make a test dataset with only the states with low action coverage
    test_data_low_cov = pd.DataFrame({
        'tag-1':  [1]*half_dataset_size, # "observation" tag
        'tag-2':  test_tag_2 # "action" tag
    })

    test_dataset_low_cov = Dataset(
        data=test_data_low_cov,
        action_tags=['tag-2']
    )
    coverage_value_2 = sampler.eval(test_dataset_low_cov, coverage_fn, n_state_samples, n_action_samples)

    assert coverage_value_1 <= coverage_value_2


def test_kde_with_uniform_action_sampler():
    np.random.seed(42)
    epsilon = 0.1
    n_norm_samples = 1000
    n_state_samples = 1
    n_action_samples = 100

    half_dataset_size = 50
    data = pd.DataFrame({
        'tag-1':  [0]*half_dataset_size + [1]*half_dataset_size,
        'tag-2':  list(np.random.uniform(0, 1, half_dataset_size)) + [0] * half_dataset_size,
    })
    dataset = Dataset(
        data=data,
        action_tags=['tag-2']
    )

    kde_coverage = KDECoverage(epsilon, n_norm_samples)
    kde_coverage.fit(dataset)
    _test_uniform_action_sampler(kde_coverage, dataset, n_state_samples, n_action_samples)


def test_neighbours_with_uniform_action_sampler():
    np.random.seed(42)
    epsilon = 0.1
    n_norm_samples = 1000
    n_state_samples = 1
    n_action_samples = 100

    half_dataset_size = 50
    data = pd.DataFrame({
        'tag-1':  [0]*half_dataset_size + [1]*half_dataset_size,
        'tag-2':  list(np.random.uniform(0, 1, half_dataset_size)) + [0] * half_dataset_size,
    })
    dataset = Dataset(
        data=data,
        action_tags=['tag-2']
    )

    neighbours_coverage = NeighboursCoverage(epsilon, n_norm_samples)
    neighbours_coverage.fit(dataset)
    _test_uniform_action_sampler(neighbours_coverage, dataset, n_state_samples, n_action_samples)
