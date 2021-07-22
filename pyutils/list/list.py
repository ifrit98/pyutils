
##############################################################################
# List and Array Tools                                                       #
##############################################################################

import os
import re
import scipy
import numpy as np
import pandas as pd
from warnings import warn


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def is_bool(x):
    if x not in [True, False, 0, 1]:
        return False
    return True
isTrueOrFalse = is_bool

def is_strlike(x):
    if type(x) == bytes:
        return type(x.decode()) == str
    if is_numpy(x):
        try:
            return 'str' in x.astype('str').dtype.name
        except:
            return False
    return type(x) == str

def regextract(x, regex):
    matches = vmatch(x, regex)
    return np.asarray(x)[matches]
extract = find = regextract

def vmatch(x, regex):
    r = re.compile(regex)
    return np.vectorize(lambda x: bool(r.match(x)))(x)
rmatch = vmatch

def lengths(x):
    def maybe_len(e):
        if type(e) == list:
            return len(e)
        else:
            return 1
    if type(x) is not list: return [1]
    if len(x) == 1: return [1]
    return(list(map(maybe_len, x)))

def is_numpy(x):
    return x.__class__ in [
        np.ndarray,
        np.rec.recarray,
        np.char.chararray,
        np.ma.masked_array
    ]

def next2pow(x):
    return 2**int(np.ceil(np.log(float(x))/np.log(2.0)))

def unnest(x, return_numpy=False):
    if return_numpy:
        return np.asarray([np.asarray(e).ravel() for e in x]).ravel()
    out = []
    for e in x:
        out.extend(np.asarray(e).ravel())
    return out
    
def unwrap_np(x):
    *y, = np.asarray(x, dtype=object)
    return y

def unwrap_df(df):
    if len(df.values.shape) >= 2:
        return df.values.flatten()
    return df.values

def df_diff(df1, df2):
	ds1 = set([tuple(line) for line in df1.values])
	ds2 = set([tuple(line) for line in df2.values])
	diff = ds1.difference(ds2)
	return pd.DataFrame(list(diff))

def summarize(x):
    x = np.asarray(x)
    x = np.squeeze(x)
    try:
        df = pd.Series(x)        
    except:
        try:
            df = pd.DataFrame(x)
        except:
            raise TypeError("`x` cannot be coerced to a pandas type.")
    return df.describe(include='all')

def list_product(els):
  prod = els[0]
  for el in els[1:]:
    prod *= el
  return prod

def get_counts(df, colname):
    return pd.DataFrame.from_dict(
        dict(
            list(
                map(
                    lambda x: (x[0], len(x[1])), 
                    df.groupby(colname)
                )
            )
        ),
        orient='index'
    )

def np_arr_to_py(x):
    x = np.unstack()
    return list(x)

def logical2idx(x):
    x = np.asarray(x)
    return np.arange(len(x))[x]

def apply_pred(x, p):
    return list(map(lambda e: p(e), x))

def extract_mask(x, m):
    if len(x) != len(m):
        raise ValueError("Shapes of `x` and `m` must be equivalent.")
    return np.asarray(x)[logical2idx(m)]

def extract_cond(x, p):
    mask = list(map(lambda e: p(e), x))
    return extract_mask(x, mask)

def import_file(filepath, ext='.py'):
    import importlib.util
    if not os.path.exists(filepath):
        raise ValueError("source `filepath` not found.")
    path = os.path.abspath(filepath)
    spec = importlib.util.spec_from_file_location(
        os.path.basename(
            path[:-len(ext)]
        ), 
        path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def maybe_list_up(x):
    # if len(np.shape(x)) == 0:
    if is_scalar(x):
        return [x]
    return x

def idx_of(arr_to_seek, vals_to_idx):
    if isinstance(vals_to_idx[0], str):
        return idx_of_str(arr_to_seek, vals_to_idx)
    vals_to_idx = maybe_list_up(vals_to_idx)
    nested_idx = list(
        map(
            lambda x: np.where(arr_to_seek == x),
            vals_to_idx
        )
    )
    return list(set(unnest(nested_idx)))

def idx_of_str(arr_to_seek, vals_to_idx):
    vals_to_idx = maybe_list_up(vals_to_idx)
    arr_to_seek = np.asarray(arr_to_seek)
    nested_idx = list(
        map(
            lambda x: np.where(x == arr_to_seek), 
            vals_to_idx
        )
    )
    return list(set(unnest(nested_idx)))

def where(arr, elem, op='in'):
    return list(
        map(
            lambda x: elem == x or elem in x,
            arr
        )
    )

index = lambda arr, elem: unlist(logical2idx(where(arr, elem)))


def where2(arr, elem, op='in'):
    return list(
        map(
            lambda x: elem == x if 'eq' in op else elem in x,
            arr
        )
    )


def search(x, e):
    """ Look for and return (if found) element `e` in data structure `x`"""
    match = vmatch if type(list(x)[0]) == str else where
    ret = unnest(logical2idx(match(list(x), e)))
    if ret == []:
        return False
    return x[e] if type(x) == dict else x[ret[0]]

def count(x):
    """Returns number of `True` elements of array `x`"""
    return sum(np.asarray(x).astype(bool))

def counts(e, x):
    """Returns number of elements of array `x` equal to element `e`"""
    arr = np.asarray(arr)
    return len(np.where(arr == x)[0])

def how_many(e, x):
    """Count how many of element `e` are in array `x`"""
    return count(np.asarray(x) == e)

def zipd(one, two):
    """zip dictionary"""
    return {**one, **two}

def maybe_unwrap(x):
    if hasattr(x, '__len__'):
        head, *_ = x
        return head
    else: 
        return x

def is_empty(x):
    if x is None: return True
    x = np.asarray(x)
    return np.equal(np.size(x), 0)



# apply callable `f()` at indices `i` on data `x`
def apply_at(f, i, x):
    """
    # usage:
    >> x = [1,2,3,4,5]
    >> apply_at(lambda x: x**2, index(2, x), x)
    >> [1,4,3,4,5]
    """
    x = np.asarray(x)
    i = np.asarray(i)
    x[i] = np.asarray(
        list(
            map(
                lambda x: f(x), x[i]
            )
        )
    )
    return x

def replace_at(x, indices, repl, colname=None):
    if is_pandas(x) and colname is None:
        if x.__class__ == pd.core.frame.DataFrame:
            raise ValueError("Must supply colname with a DataFrame object")
        return replace_at_pd(x, indices, repl)
    x = np.asarray(x)
    x[indices] = np.repeat(repl, len(indices))
    return x

def is_pandas(x):
    return x.__class__ in [
        pd.core.frame.DataFrame,
        pd.core.series.Series
    ]

def replace_at_pd(x, colname, indices, repl):
    x.loc[indices, colname] = repl
    return x

def target2int(tgt):
    ret = 0
    for i in range(len(tgt)):
        ret += tgt[i] * np.power(2, i)
    return ret

def int2target(val, n_classes):
    print("\nEntering int2target")
    print("init val:", val)
    ret = []
    for i in range(n_classes):
        if val % 2 == 1:
            ret.append(1)
            val = val - 1
            print('val -1:', val)
        else:
            ret.append(0)
        val = val / 2
        print('val /2:', val)
        print('val size:', tf.shape(val))
        print('ret @ {}: {}'.format(i, ret))
        print()
    print("Leaving int2target\n")
    return ret

def tf_int2target(val, n_classes):
    def true_fn():
        return tf.constant(1)
    def false_fn():
        return tf.constant(0)
    ret = []
    for _ in range(n_classes):
        z = tf.cond(
            tf.equal(
                tf.math.mod(tf.cast(val, 'int32'), tf.constant(2)), 
                tf.constant(1)
            ), 
            true_fn, false_fn
        )
        ret.append(z)
        if tf.equal(z, tf.constant(1)):
            val = val - tf.constant(1, dtype=val.dtype)
        val = val / tf.constant(2, dtype=val.dtype)
    return tf.stack(ret)

def test_target_conversion(arr):
    print('input array:', arr)
    b = target2int(arr)
    print('encoded:', b)
    c = int2target(b, len(arr))
    print('equality:', arr==c)

def df_target2int(df):
    if is_scalar(df['target'].iloc[0]): return df
    df.loc[:, ('target')] = df['target'].map(lambda x: target2int(x))
    return df

def df_int2target(df, n_classes):
    df.loc[:, ('target')] = df['target'].map(lambda x: int2target(x, n_classes))
    return df


def set_int_targets_from_class(df, colname='class'):
    classes = df['class'].unique()
    classes.sort()
    d = dict(zip(classes, range(len(classes))))
    df['target'] = df['class'].map(lambda x: d[x])


def within(x, y, eps=1e-3):
    ymax = y + eps
    ymin = y - eps
    return x <= ymax and x >= ymin

def within1(x, y):
    return within(x, y, 1.)

def within_vec(x, y, eps=1e-3):
    vf = np.vectorize(within)
    return np.all(vf(x, y, eps=eps))

def dim(x):
    if is_numpy(x):
        return x.shape
    return np.asarray(x).shape

def shapes(x):
    shapes_fun = FUNCS[type(x)]
    return shapes_fun(x)

def shapes_list(l, print_=False):
    r"""Grab shapes from a list of tensors or numpy arrays"""
    shps = []
    for x in l:
        if print_:
            print(np.asarray(x).shape)
        shps.append(np.asarray(x).shape)
    return shps

def shapes_dict(d, print_=False):
    r"""Recursively grab shapes from potentially nested dictionaries"""
    shps = {}
    for k,v in d.items():
        if isinstance(v, dict):
            shps.update(shapes(v))
        else:
            if print_:
                print(k, ":\t", np.asarray(v).shape)
            shps[k] = np.asarray(v).shape
    return shps

def shapes_tuple(tup, return_shapes=False):
    shps = {i: None for i in range(len(tup))}
    for i, t in enumerate(tup):
        shps[i] = np.asarray(t).shape
    print(shps)
    if return_shapes:
        return shps

FUNCS = {
    dict: shapes_dict,
    list: shapes_list,
    tuple: shapes_tuple
}

def info(d, return_dict=False, print_=True):
    r"""Recursively grab shape, dtype, and size from (nested) dictionary of tensors"""
    info_ = {}
    for k,v in d.items():
        if isinstance(v, dict):
            info_.update(info(v))
        else:
            info_[k] = {
                'size': np.asarray(v).ravel().shape,
                'shape' :np.asarray(v).shape,
                'dtype': np.asarray(v).dtype.name
            }
            if print_:
                _v = np.asarray(v)
                print('key   -', k)
                print('dtype -', _v.dtype.name)
                print('size  -', np.asarray(v).ravel().shape)
                print('shape -', _v.shape)
                print()
    if return_dict:
        return info_

def stats(x, axis=None, epsilon=1e-7):
    if not is_numpy(x):
        x = np.asarray(x)
    if np.min(x) < 0:
        _x = x + abs(np.min(x) - epsilon)
    gmn = scipy.stats.gmean(_x, axis=axis)
    hmn = scipy.stats.hmean(_x, axis=axis)
    mode = scipy.stats.mode(x, axis=axis).mode[0]
    mnt2, mnt3, mnt4 = scipy.stats.moment(x, [2,3,4], axis=axis)
    lq, med, uq = scipy.stats.mstats.hdquantiles(x, axis=axis)
    lq, med, uq = np.quantile(x, [0.25, 0.5, 0.75], axis=axis)
    var = scipy.stats.variation(x, axis=axis) # coefficient of variation
    sem = scipy.stats.sem(x, axis=axis) # std error of the means
    res = scipy.stats.describe(x, axis=axis)
    nms = ['nobs          ', 
           'minmax        ', 
           'mean          ', 
           'variance      ', 
           'skewness      ', 
           'kurtosis      ']
    description = dict(zip(nms, list(res)))
    description.update({
        'coeff_of_var  ': var,
        'std_err_means ': sem,
        'lower_quartile': lq,
        'median        ': med,
        'upper_quartile': uq,
        '2nd_moment    ': mnt2,
        '3rd_moment    ': mnt3,
        '4th_moment    ': mnt4,
        'mode          ': mode,
        'geometric_mean': gmn,
        'harmoinc_mean ': hmn
    })
    return description


def unzip(x):
    if type(x) is not list:
        raise ValueError("`x` must be a list of tuple pairs")
    return list(zip(*x))

from itertools import zip_longest
def groupl(iterable, n, padvalue=None):
  "groupl(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
  return list(zip_longest(*[iter(iterable)]*n, fillvalue=padvalue))

def merge_by_colname(df1, df2, colname='target', how='outer'):
    pt = df1.pop(colname)
    nt = df2.pop(colname)    
    targets = pt.append(nt).reset_index()
    df2 = df1.merge(df2, how=how)
    df2.loc[:, (colname)] = targets
    return df2

from functools import reduce
reduce_df = lambda dfs: reduce(lambda x, y: merge_by_colname(x, y), dfs)
merge_dict = lambda dicts: reduce(
    lambda x,y: {k: v + [y[k]] for k,v in x.items()}, 
    dicts, 
    {k: [] for k in dicts[0].keys()}
)

def switch(on, pairs, default=None):
    """ Create dict switch-case from key-word pairs, mimicks R's `switch()`

        Params:
            on: key to index OR predicate returning boolean value to index into dict
            pairs: dict k,v pairs containing predicate enumeration results
        
        Returns: 
            indexed item by `on` in `pairs` dict
        Usage:
        # Predicate
            pairs = {
                True: lambda x: x**2,
                False: lambda x: x // 2
            }
            switch(
                1 == 2, # predicate
                pairs,  # dict 
                default=lambda x: x # identity
            )

        # Index on value
            key = 2
            switch(
                key, 
                values={1:"YAML", 2:"JSON", 3:"CSV"},
                default=0
            )
    """
    if type(pairs) is tuple:
        keys, vals = unzip(pairs)
        return switch2(on, keys=keys, vals=vals, default=default)
    if type(pairs) is not dict:
        raise ValueError("`pairs` must be a list of tuple pairs or a dict")
    return pairs.get(on, default)


def switch2(on, keys, vals, default=None):
    """
    Usage:
        switch(
            'a',
            keys=['a', 'b', 'c'],
            vals=[1, 2, 3],
            default=0
        )
        >>> 1

        # Can be used to select functions
        x = 10
        func = switch(
            x == 10, # predicate
            keys=[True, False],
            vals=[lambda x: x + 1, lambda x: x -1],
            default=lambda x: x # identity
        )
        func(x)
        >>> 11
    """
    if len(keys) == len(vals):
        raise ValueError("`keys` must be same length as `vals`")
    tuples = dict(zip(keys, vals))
    return tuples.get(on, default)


def comma_sep_str_to_int_list(s):
  return [int(i) for i in s.split(",") if i]

# Useful for printing all arguments to function call, mimicks dots `...` in R.
def printa(*argv):
    [print(i) for i in argv]

def printk(**kwargs):
    [print(k, ":\t", v) for k,v in kwargs.items()]

def pyrange(x, return_type='dict'):
    return {'min': np.min(x), 'max': np.max(x)} \
        if return_type == 'dict' \
        else (np.min(x), np.max(x))

def alleq(x):
    try:
        iter(x)
    except:
        x = [x]
    current = x[0]
    for v in x:
        if v != current:
            return False
    return True

def types(x):
    if isinstance(x, dict):
        return {k: type(v) for k,v in x.items()}
    return list(map(type, x))

# nearest element to `elem` in array `x`
def nearest1d(x, elem):
    lnx = len(x)
    if lnx % 2 == 1:
        mid = (lnx + 1) // 2
    else:
        mid = lnx // 2
    if mid == 1:
        return x[0]
    if x[mid] >= elem:
        return nearest1d(x[:mid], elem)
    elif x[mid] < elem:
        return nearest1d(x[mid:], elem)
    else:
        return x[0]

def idx_of_1d(x, elem):
    if elem not in x:
        raise ValueError("`elem` not contained in `x`")
    return dict(zip(x, range(len(x))))[elem]

def unlist(x):
    return list(map(lambda l: maybe_unwrap(l), x))

import random 
def random_ints(length, lo=-1e4, hi=1e4):
    return [random.randint(lo, hi) for _ in range(length)]
randint = random_ints

def random_floats(length, lo=-1e4, hi=1e4):
    return [random.random() for _ in range(length)]
randfloat = random_floats

def is_pandas(x):
    return x.__class__ in [
        pd.core.frame.DataFrame,
        pd.core.series.Series
    ]

def replace_at_pd(x, colname, indices, repl):
    x.loc[indices, colname] = repl
    return x

def _complex(real, imag):
    """ Efficiently create a complex array from 2 floating point """
    real = np.asarray(real)
    imag = np.asarray(imag)
    cplx = 1j * imag    
    return cplx + real

# Safe log10
def log10(data):
    np.seterr(divide='ignore')
    data = np.log10(data)
    np.seterr(divide='warn')
    return data
log10_safe = log10

def is_scalar(x):
    if is_numpy(x):
        return x.ndim == 0
    if isinstance(x, str) or type(x) == bytes:
        return True
    if hasattr(x, "__len__"):
        return len(x) == 1
    try:
        x = iter(x)
    except:
        return True
    return np.asarray(x).ndim == 0

def as_scalar(x):
    if (len(np.shape(x))):
        x = np.squeeze(x)
        if len(x) > 1: warn("Cannot coerce `x` to scalar... returning head")
    return maybe_unwrap(x)

def first(x):
    if is_scalar(x):
        return x
    if not is_numpy(x):
        x = np.asarray(x)
    return x.ravel()[0]

def last(x):
    if not is_numpy(x):
        x = np.asarray(x)
    return x.ravel()[-1] 

def listmap(x, f):
    return list(map(f, x))
lmap = listmap


def _range(x):
    if np.asarray(x).dtype.name in ['complex64', 'complex128']:
        R = np.real(x)
        I = np.imag(x)
        return {  
            'real': (np.min(R), np.max(R)), 
            'imag': (np.min(I), np.max(I))
        }
    return {np.min(x), np.max(x)}