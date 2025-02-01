from collections import defaultdict
from functools import wraps
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Callable, DefaultDict, Sequence, TypeVar, cast

import polars as pl

from infrastructure.hash import sha1_hash

TFunc = TypeVar('TFunc', bound=Callable[..., pl.DataFrame])


SHM_REFERENCES: DefaultDict[str, dict[str, SharedMemory]] = defaultdict(dict)
"""Keep forever the references to shared memory arrays."""

N_BYTES_INT_32 = 4


def cache_parquet(
    path: Path | str, schema: pl.Schema | None = None, verbose: bool = False
) -> Callable[[TFunc], TFunc]:
    def decorator(fn: TFunc) -> TFunc:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal path
            if isinstance(path, str):
                path = Path(path)

            if path.exists():
                if verbose:
                    print(f'Using cached parquet file: {path}')
                df_result = pl.read_parquet(path)
                df_result = _restore_categorical_ordering(df_result, schema)
                return df_result
            else:
                if verbose:
                    print('Building dataframe')
                df_result = fn(*args, **kwargs)
                if verbose:
                    print(f'Writing parquet file: {path}')
                path.parent.mkdir(parents=True, exist_ok=True)
                df_result.write_parquet(path)
            return df_result

        return cast(TFunc, wrapper)

    return decorator


def cache_parquet_fieldwise(
    cache_dir: Path | str,
    fields_arg: int | str | None,
    schema: pl.Schema | Callable[[Sequence[str]], pl.Schema] | None = None,
    verbose: bool | None = None,
    hash_field_names: bool = False,
) -> Callable[[TFunc], TFunc]:
    if verbose is None:
        verbose = False

    def decorator(fn: TFunc) -> TFunc:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            schema_instance: pl.Schema | None
            if isinstance(schema, pl.Schema):
                schema_instance = schema
            else:
                schema_instance = None

            cols: Sequence[str] | None
            if isinstance(fields_arg, int):
                cols = cast(Sequence[str], args[fields_arg])
            elif isinstance(fields_arg, str):
                cols = cast(Sequence[str], kwargs[fields_arg])
            else:
                cols = None

            if cols is None:
                assert fields_arg is None
                assert schema_instance is not None
                cols = schema_instance.names()

            if schema_instance is None:
                assert callable(schema)
                schema_instance = schema(cols)

            nonlocal cache_dir
            if isinstance(cache_dir, str):
                cache_dir = Path(cache_dir)

            cache_dir.mkdir(parents=True, exist_ok=True)
            missing_cols = []
            df_col_list = []
            for col in cols:
                col_key = sha1_hash(col)[:16] if hash_field_names else col
                col_path = cache_dir / f'{col_key}.parquet'
                if col_path.exists():
                    if verbose:
                        print(f'Loading: {col_path}')
                    df_col = pl.read_parquet(col_path)
                    df_col = _restore_categorical_ordering(
                        df_col, schema_instance
                    )
                    df_col_list.append(df_col)
                else:
                    if verbose:
                        print(f'Cache miss: {col_path}')
                    missing_cols.append(col)

            if len(missing_cols) == 0:
                return pl.concat(df_col_list, how='horizontal')

            if verbose:
                print(f'Computing: {missing_cols}')

            if isinstance(fields_arg, int):
                args = (
                    args[:fields_arg]
                    + (missing_cols,)
                    + args[fields_arg + 1 :]
                )
            elif isinstance(fields_arg, str):
                kwargs = {**kwargs, fields_arg: missing_cols}

            df_new_cols = fn(*args, **kwargs)
            for col in df_new_cols.columns:
                col_key = sha1_hash(col)[:16] if hash_field_names else col
                col_path = cache_dir / f'{col_key}.parquet'
                if verbose:
                    print(f'Writing: {col} -> {col_path}')
                col_path.parent.mkdir(parents=True, exist_ok=True)
                df_new_cols.select(col).write_parquet(col_path)

            return pl.concat([*df_col_list, df_new_cols], how='horizontal')

        return cast(TFunc, wrapper)

    return decorator


def _restore_categorical_ordering(
    df_result: pl.DataFrame, schema: pl.Schema | None = None
) -> pl.DataFrame:
    if schema is None:
        return df_result

    col_to_dtype: dict[str, pl.DataType] = {}

    for col in df_result.columns:
        schema_dtype = schema[col]
        if not isinstance(schema_dtype, pl.Categorical):
            continue
        loaded_dtype = df_result.get_column(col).dtype
        assert isinstance(loaded_dtype, pl.Categorical)
        if schema_dtype.ordering != loaded_dtype.ordering:
            col_to_dtype[col] = schema_dtype

    if len(col_to_dtype) == 0:
        return df_result

    return df_result.with_columns(
        [pl.col(col).cast(dtype) for col, dtype in col_to_dtype.items()]
    )


__all__ = [
    'cache_parquet',
    'cache_parquet_fieldwise',
    'cache_shared_memory_fieldwise',
]
