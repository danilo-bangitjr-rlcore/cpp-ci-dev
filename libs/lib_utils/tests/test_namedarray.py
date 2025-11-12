from datetime import UTC, datetime, timedelta

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from lib_agent.buffer.storage import ReplayStorage

import lib_utils.jax as jax_u
from lib_utils.named_array import JaxTimestamp, NamedArray


@pytest.fixture
def basic_namedarray():
    return NamedArray.from_mapping({"a": jnp.asarray(1), "b": jnp.asarray(2)})

@pytest.fixture
def matrix_namedarray():
    return NamedArray.from_mapping({'a': jnp.asarray([1,2,3]), 'b': jnp.asarray([4,5,6])})

@pytest.fixture
def tensor_namedarray():
    return NamedArray.from_mapping({'a': jnp.ones((5,4)), 'b': 2*jnp.ones((5,4)), 'c': 3*jnp.ones((5,4))})


def test_user_provided_timestamp():
    ts = np.asarray(np.datetime64(datetime.now(UTC).replace(tzinfo=None)))
    x = NamedArray.from_mapping({'a': jnp.asarray(1), 'b': jnp.asarray(2)}, ts)
    assert x.timestamps.to_datetime64() == ts

def test_user_provided_tz_naive_timestamp():
    ts = np.asarray(np.datetime64(datetime.now()))
    x = NamedArray.from_mapping({'a': jnp.asarray(1), 'b': jnp.asarray(2)}, ts)
    assert x.timestamps.to_datetime64() == ts


def test_namedarray_shapes():
    # from mapping
    assert NamedArray.from_mapping({"a": jnp.asarray(1), "b": jnp.asarray(2)}).array.shape == (2,)
    assert NamedArray.from_mapping({"a": jnp.asarray([1]), "b": jnp.asarray([2])}).array.shape == (1,2)
    assert NamedArray.from_mapping({'a': jnp.asarray([1,2,3]), 'b': jnp.asarray([4,5,6])}).array.shape == (3,2)
    assert NamedArray.from_mapping(
        {"a": jnp.ones((5, 4)), "b": 2 * jnp.ones((5, 4)), "c": 3 * jnp.ones((5, 4))},
    ).array.shape == (5, 4, 3)

    # from dataframe
    single_row_df = pd.DataFrame({'a': [1], 'b': [2]})
    multi_row_df = pd.DataFrame({'a': [1,2,3], 'b': [2,3,4]})
    assert NamedArray.from_pandas(single_row_df).array.shape == (1,2)
    assert NamedArray.from_pandas(multi_row_df).array.shape == (3,2)


def test_namedarray_timestamps_from_pandas():
    start = datetime.now(UTC)
    delta = timedelta(minutes=5)
    steps = 3
    idx = pd.DatetimeIndex([start + i * delta for i in range(steps)])
    cols = ['a', 'b']
    multi_row_df = pd.DataFrame(
        data=[
            [0, 3],
            [1, 4],
            [2, 5],
        ],
        columns=pd.Index(cols),
        index=idx,
    )

    narr = NamedArray.from_pandas(multi_row_df)
    assert narr.timestamps.to_datetime64().shape == (3,)
    expected_tz = idx.tz_convert(None).to_numpy(dtype='datetime64[us]')
    assert (narr.timestamps.to_datetime64() == expected_tz).all()


def test_namedarray_tz_naive_timestamps_from_pandas():
    start = datetime.now()
    delta = timedelta(minutes=5)
    steps = 3
    idx = pd.DatetimeIndex([start + i * delta for i in range(steps)])
    cols = ['a', 'b']
    multi_row_df = pd.DataFrame(
        data=[
            [0, 3],
            [1, 4],
            [2, 5],
        ],
        columns=pd.Index(cols),
        index=idx,
    )

    narr = NamedArray.from_pandas(multi_row_df)
    assert narr.timestamps.to_datetime64().shape == (3,)
    expected_tz = idx.to_numpy(dtype='datetime64[us]')
    assert (narr.timestamps.to_datetime64() == expected_tz).all()


def test_matrix_namedarray_access():
    narray = NamedArray.from_mapping({'a': jnp.asarray([1,2,3]), 'b': jnp.asarray([4,5,6]), 'c': jnp.asarray([7,8,9])})
    ac = narray.get_features(['a', 'c']).expect()
    expected = jnp.asarray(
        [[1, 7],
         [2, 8],
         [3, 9]],
    )
    assert jnp.allclose(ac, expected)

def test_row_namedarray_index(basic_namedarray: NamedArray):
    # basic_namedarray has shape (2,), cant be indexed
    assert basic_namedarray.array.shape == (2,)
    with pytest.raises(IndexError):
        out = basic_namedarray[0]

    # dataframe produces array with shape (1,d) for d features
    narr2d = NamedArray.from_pandas(basic_namedarray.as_pandas().expect())

    # 2D namedarray can be indexed
    out = narr2d[0]
    assert out == basic_namedarray


############################################################
### Test Compatibility with Jax Transforms ###
############################################################

def test_namedarray_flatten(basic_namedarray: NamedArray, matrix_namedarray: NamedArray, tensor_namedarray: NamedArray):
    leaves, treedef = jax.tree.flatten(basic_namedarray)
    restored: NamedArray = jax.tree.unflatten(treedef, leaves)
    assert restored == basic_namedarray

    leaves, treedef = jax.tree.flatten(matrix_namedarray)
    restored: NamedArray = jax.tree.unflatten(treedef, leaves)
    assert restored == matrix_namedarray

    leaves, treedef = jax.tree.flatten(tensor_namedarray)
    restored: NamedArray = jax.tree.unflatten(treedef, leaves)
    assert restored == tensor_namedarray


def test_namedarray_vmap(matrix_namedarray: NamedArray):
    def scale_b(x: NamedArray) -> jax.Array:
        chex.assert_rank(x.array, 1)
        return x.get_feature('b').map(lambda val: 2*val).or_else(jnp.asarray(jnp.nan))

    chex.assert_rank(matrix_namedarray, 2)
    out = jax_u.vmap(scale_b)(matrix_namedarray)
    assert jnp.allclose(out, jnp.asarray([8, 10, 12]))


def test_vmap_failed_access(matrix_namedarray: NamedArray):
    """
    The feature d doesnt exist in matrix_namedarray. The vmapped scale_d fn should return all nans
    """
    def scale_d(x: NamedArray) -> jax.Array:
        chex.assert_rank(x.array, 1)
        return x.get_feature('d').map(lambda val: 2*val).or_else(jnp.asarray(jnp.nan))

    chex.assert_rank(matrix_namedarray, 2)
    out = jax_u.vmap(scale_d)(matrix_namedarray)
    assert jnp.allclose(out, jnp.asarray([jnp.nan, jnp.nan, jnp.nan]), equal_nan=True)


def test_namedarray_jit(basic_namedarray: NamedArray):
    def sum_ab(x: NamedArray) -> jax.Array:
        chex.assert_rank(x.array, 1)
        return x.get_features(['a', 'b']).map(lambda vals: vals.sum()).or_else(jnp.asarray(jnp.nan))

    jitted_sum = jax_u.jit(sum_ab)
    # call a few times
    for _ in range(10):
        out = jitted_sum(basic_namedarray)
        assert jnp.allclose(out, jnp.asarray(3))


def test_jit_failed_access(basic_namedarray: NamedArray):
    """
    The feature d doesnt exist in basic_namedarray. The jitted sum_ab fn should return nan
    """
    def sum_ab(x: NamedArray) -> jax.Array:
        chex.assert_rank(x.array, 1)
        return x.get_features(['a', 'd']).map(lambda vals: vals.sum()).or_else(jnp.asarray(jnp.nan))

    jitted_sum = jax_u.jit(sum_ab)
    # call a few times
    for _ in range(10):
        out = jitted_sum(basic_namedarray)
        assert jnp.allclose(out, jnp.asarray(jnp.nan), equal_nan=True)


def test_namedarray_grad(basic_namedarray: NamedArray):
    """
    b in basic_namedarray has a value of 2, grad should be 2*3*theta^2
    """
    def cube_scale(theta: jax.Array, x: NamedArray) -> jax.Array:
        return x.get_features(['b']).map(lambda val: jnp.sum(val*theta**3)).or_else(jnp.asarray(jnp.nan))

    cube_scale_grad = jax_u.grad(cube_scale)
    theta = jnp.asarray([12.0])
    out = cube_scale(theta, basic_namedarray)
    assert jnp.isclose(out, 2*12**3)
    grad = cube_scale_grad(theta, basic_namedarray)
    assert jnp.allclose(grad, jnp.asarray([2.0*3*12**2]))


def test_branch_on_timestamp(basic_namedarray: NamedArray):
    time_thresh = basic_namedarray.timestamps.to_datetime64() - np.timedelta64(1, 'h')
    def sum_ab_or_a(x: NamedArray) -> jax.Array:
        chex.assert_rank(x.array, 1)
        pred = x.timestamps > JaxTimestamp.from_datetime64(time_thresh)

        return jax.lax.cond(
            pred,
            lambda _x: _x.get_features(['a', 'b']).map(lambda vals: vals.sum()).or_else(jnp.asarray(jnp.nan)),
            lambda _x: _x.get_feature('a').or_else(jnp.asarray(jnp.nan)),
            x,
        )

    jitted_sum = jax_u.jit(sum_ab_or_a)

    # call a few times taking first branch
    for _ in range(10):
        out = jitted_sum(basic_namedarray)
        assert jnp.allclose(out, jnp.asarray(3))

    # take the other branch
    earlier_t = basic_namedarray.timestamps.to_datetime64() - np.timedelta64(2, 'h')
    earlier_namedarray = NamedArray.from_mapping({'a': jnp.asarray(1), 'b': jnp.asarray(2)}, timestamps=earlier_t)
    for _ in range(10):
        out = jitted_sum(earlier_namedarray)
        assert jnp.allclose(out, jnp.asarray(1))

def test_namedarray_storage_and_retrieval():
    """
    Begin with an (n, 3) namedarray (e.g. a replay buffer that stores n states, each with 3 features)

    Sample multiple batches (collections of rows) from the buffer, vstack them to form input to an ensemble fn,
    resulting in a (ensemble, batch, features) namedarray.

    Ensure accessing a subset of k of the features results in (ensemble, batch, k) namedarray with the correct values.
    """

    state1_features = NamedArray.from_pandas(pd.DataFrame({'a': [1], 'b': [2], 'c': [3], 'd': [4]}))
    state2_features = NamedArray.from_pandas(pd.DataFrame({'a': [5], 'b': [6], 'c': [7], 'd': [8]}))
    state3_features = NamedArray.from_pandas(pd.DataFrame({'a': [9], 'b': [10], 'c': [11], 'd': [12]}))

    batch1_indices = np.asarray([0, 1, 2])
    batch2_indices = np.asarray([2, 0, 1])

    expected_batch1_vals = jnp.asarray(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]],
    )
    expected_batch2_vals = jnp.asarray(
        [[9, 10, 11, 12],
         [1, 2, 3, 4],
         [5, 6, 7, 8]],
    )

    expected = jnp.stack([expected_batch1_vals, expected_batch2_vals])
    assert expected.shape == (2,3,4)

    storage: ReplayStorage[NamedArray] = ReplayStorage(capacity=100)
    storage.add(state1_features[0]) # NOTE: ReplayStorage expects rows with dim 1
    storage.add(state2_features[0])
    storage.add(state3_features[0])

    ens_batch = storage.get_ensemble_batch([batch1_indices, batch2_indices])
    assert jnp.allclose(ens_batch.array, expected)

    # access the third feature by name
    c_feat = ens_batch.get_feature('c').expect()
    expected_c = expected[..., 2]
    assert jnp.allclose(c_feat, expected_c)

    # access the first and third features by name
    ac_feats = ens_batch.get_features(['a', 'c']).expect()
    expected_a = expected[..., 0]
    expeced_ac = jnp.concatenate([jnp.expand_dims(expected_a, axis=2), jnp.expand_dims(expected_c, axis=2)], axis=2)
    assert jnp.allclose(ac_feats, expeced_ac)

    # test that timestamps were preserved
    ens_ts = ens_batch.timestamps
    # ensure no per-feature timestamp
    assert ens_ts.high_bits.shape == (2, 3)
    assert ens_ts.low_bits.shape == (2, 3)
    assert ens_ts.to_datetime64().shape == (2, 3)
    # ensure states have different timestamps
    assert state1_features.timestamps != state2_features.timestamps
    # ensure timestamps were preserved through storage and retrieval
    # first ens dim
    got_ts_1 = ens_ts.to_datetime64()[0, 0]
    assert got_ts_1 == state1_features.timestamps.to_datetime64()
    got_ts_2 = ens_ts.to_datetime64()[0, 1]
    assert got_ts_2 == state2_features.timestamps.to_datetime64()
    got_ts_3 = ens_ts.to_datetime64()[0, 2]
    assert got_ts_3 == state3_features.timestamps.to_datetime64()
    # second ens dim, different order
    got_ts_3 = ens_ts.to_datetime64()[1, 0]
    assert got_ts_3 == state3_features.timestamps.to_datetime64()
    got_ts_1 = ens_ts.to_datetime64()[1, 1]
    assert got_ts_1 == state1_features.timestamps.to_datetime64()
    got_ts_2 = ens_ts.to_datetime64()[1, 2]
    assert got_ts_2 == state2_features.timestamps.to_datetime64()


############################################################
### Test Basic Methods ###
############################################################

def test_set_method(basic_namedarray: NamedArray):
    new_values = jnp.asarray([10.0, 20.0])
    new_na = basic_namedarray.set(new_values)
    assert isinstance(new_na, NamedArray)
    assert jnp.allclose(new_na.array, new_values)
    assert new_na.names == basic_namedarray.names
    assert new_na.timestamps == basic_namedarray.timestamps


def test_dunder_add_scalar(basic_namedarray: NamedArray):
    added = basic_namedarray + 5.0
    assert isinstance(added, NamedArray)
    assert jnp.allclose(added.array, jnp.asarray([6.0, 7.0]))
    assert added.names == basic_namedarray.names
    assert added.timestamps == basic_namedarray.timestamps

def test_add_namedarray(basic_namedarray: NamedArray, matrix_namedarray: NamedArray):
    # test successful add
    added = basic_namedarray.add(basic_namedarray)
    assert isinstance(added, NamedArray)
    assert jnp.allclose(added.array, jnp.asarray([2.0, 4.0]))
    assert added.names == basic_namedarray.names
    assert added.timestamps == basic_namedarray.timestamps

    # test mismatched names
    renamed = NamedArray(
        names=["c", "d"], values=basic_namedarray.array, timestamps=basic_namedarray.timestamps.to_datetime64(),
    )
    with pytest.raises(AssertionError):
        basic_namedarray.add(renamed)

    # test mismatched shapes
    with pytest.raises(AssertionError):
        basic_namedarray.add(matrix_namedarray)

    # test broadcasting of single feature set
    # broadcast namedarray with shape (d,)
    res = matrix_namedarray.add(basic_namedarray)
    expected = jnp.asarray(
        [
            [2, 6],
            [3, 7],
            [4, 8],
        ],
    )

    assert (res.array == expected).all()

    # broadcast namedarray with shape (1,d)
    reshaped_narr = NamedArray(
        names=basic_namedarray.names,
        values=jnp.expand_dims(basic_namedarray.array, axis=0),
    )
    assert reshaped_narr.shape == (1, 2)
    res = matrix_namedarray.add(reshaped_narr)
    assert (res.array == expected).all()

def test_dunder_sub_scalar(basic_namedarray: NamedArray):
    subbed = basic_namedarray - 1.0
    assert isinstance(subbed, NamedArray)
    assert jnp.allclose(subbed.array, jnp.asarray([0.0, 1.0]))
    assert subbed.names == basic_namedarray.names
    assert subbed.timestamps == basic_namedarray.timestamps

def test_subtract_namedarray(basic_namedarray: NamedArray):
    subbed = basic_namedarray.sub(basic_namedarray)
    assert isinstance(subbed, NamedArray)
    assert jnp.allclose(subbed.array, jnp.asarray([0.0, 0.0]))
    assert subbed.names == basic_namedarray.names
    assert subbed.timestamps == basic_namedarray.timestamps


def test_dunder_mul_scalar(basic_namedarray: NamedArray):
    mult = basic_namedarray * 2.0
    assert isinstance(mult, NamedArray)
    assert jnp.allclose(mult.array, jnp.asarray([2.0, 4.0]))
    assert mult.names == basic_namedarray.names
    assert mult.timestamps == basic_namedarray.timestamps

def test_multiply_namedarray(basic_namedarray: NamedArray):
    mult = basic_namedarray.mul(basic_namedarray)
    assert isinstance(mult, NamedArray)
    assert jnp.allclose(mult.array, jnp.asarray([1.0, 4.0]))
    assert mult.names == basic_namedarray.names
    assert mult.timestamps == basic_namedarray.timestamps


def test_dunder_truediv_scalar(basic_namedarray: NamedArray):
    div = basic_namedarray / 2.0
    assert isinstance(div, NamedArray)
    assert jnp.allclose(div.array, jnp.asarray([0.5, 1.0]))
    assert div.names == basic_namedarray.names
    assert div.timestamps == basic_namedarray.timestamps

def test_divide_namedarray(basic_namedarray: NamedArray):
    div = basic_namedarray.div(basic_namedarray)
    assert isinstance(div, NamedArray)
    assert jnp.allclose(div.array, jnp.asarray([1.0, 1.0]))
    assert div.names == basic_namedarray.names
    assert div.timestamps == basic_namedarray.timestamps

