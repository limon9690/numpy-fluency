# Tracker

One row per problem. Update **Status**, **Last solved** and **Next review** yourself.

- Status: `Todo` → `Solved` → `Review 1` → `Review 2` → `Mastered`
- Review rule: after a solve, re-solve after **3 days**, then **2 weeks**, then **6 weeks**.
  ★ Core problems are re-solved from a **blank file** (see the README).
- Dates: `YYYY-MM-DD`.

| # | Problem | Category | Difficulty | Core | Status | Last solved | Next review |
|---|---|---|---|---|---|---|---|
| 001 | [Creating arrays](problems/01_arrays_dtypes_memory/001_creating_arrays/README.md) | Arrays & memory | Easy |  | Todo | | |
| 002 | [dtypes, casting and overflow](problems/01_arrays_dtypes_memory/002_dtypes_casting_overflow/README.md) | Arrays & memory | Easy |  | Todo | | |
| 003 | [NumPy scalars and Python interop](problems/01_arrays_dtypes_memory/003_scalars_python_interop/README.md) | Arrays & memory | Easy |  | Todo | | |
| 004 | [Views vs copies](problems/01_arrays_dtypes_memory/004_views_vs_copies/README.md) | Arrays & memory | Medium | ★ | Todo | | |
| 005 | [Strides and memory layout](problems/01_arrays_dtypes_memory/005_strides_memory_layout/README.md) | Arrays & memory | Medium |  | Todo | | |
| 006 | [Slicing and basic indexing](problems/02_indexing/006_slicing_basic_indexing/README.md) | Indexing | Easy |  | Todo | | |
| 007 | [Fancy indexing](problems/02_indexing/007_fancy_indexing/README.md) | Indexing | Medium | ★ | Todo | | |
| 008 | [Boolean masking](problems/02_indexing/008_boolean_masking/README.md) | Indexing | Medium | ★ | Todo | | |
| 009 | [where, nonzero, argwhere](problems/02_indexing/009_where_nonzero_argwhere/README.md) | Indexing | Easy |  | Todo | | |
| 010 | [Mixing basic and advanced indexing, np.ix_](problems/02_indexing/010_mixed_basic_advanced_indexing/README.md) | Indexing | Hard |  | Todo | | |
| 011 | [take_along_axis and put_along_axis](problems/02_indexing/011_take_put_along_axis/README.md) | Indexing | Medium |  | Todo | | |
| 012 | [Broadcasting rules](problems/03_broadcasting_shape_ops/012_broadcasting_rules/README.md) | Broadcasting & shapes | Medium | ★ | Todo | | |
| 013 | [Row and column scaling and normalization](problems/03_broadcasting_shape_ops/013_row_column_normalization/README.md) | Broadcasting & shapes | Easy | ★ | Todo | | |
| 014 | [reshape, ravel vs flatten, transpose, moveaxis](problems/03_broadcasting_shape_ops/014_reshape_ravel_transpose/README.md) | Broadcasting & shapes | Easy |  | Todo | | |
| 015 | [newaxis, expand_dims, squeeze](problems/03_broadcasting_shape_ops/015_newaxis_expand_squeeze/README.md) | Broadcasting & shapes | Easy |  | Todo | | |
| 016 | [concatenate, stack, split](problems/03_broadcasting_shape_ops/016_concatenate_stack_split/README.md) | Broadcasting & shapes | Easy |  | Todo | | |
| 017 | [tile, repeat, pad, meshgrid](problems/03_broadcasting_shape_ops/017_tile_repeat_pad_meshgrid/README.md) | Broadcasting & shapes | Medium |  | Todo | | |
| 018 | [Reductions with axis and keepdims](problems/04_reductions_vectorized_ops/018_reductions_axis_keepdims/README.md) | Reductions & vectorized | Easy | ★ | Todo | | |
| 019 | [ufuncs: out=, where=, reduce, accumulate, outer](problems/04_reductions_vectorized_ops/019_ufuncs/README.md) | Reductions & vectorized | Medium |  | Todo | | |
| 020 | [cumsum, diff and running statistics](problems/04_reductions_vectorized_ops/020_cumsum_diff_running_stats/README.md) | Reductions & vectorized | Easy |  | Todo | | |
| 021 | [sort, stable argsort, lexsort, partition](problems/04_reductions_vectorized_ops/021_sorting_partition/README.md) | Reductions & vectorized | Medium | ★ | Todo | | |
| 022 | [unique and bincount](problems/04_reductions_vectorized_ops/022_unique_bincount/README.md) | Reductions & vectorized | Medium | ★ | Todo | | |
| 023 | [searchsorted, digitize, histogram, interp](problems/04_reductions_vectorized_ops/023_searchsorted_digitize_histogram/README.md) | Reductions & vectorized | Medium |  | Todo | | |
| 024 | [Set operations: isin, intersect1d, setdiff1d, union1d](problems/04_reductions_vectorized_ops/024_set_operations/README.md) | Reductions & vectorized | Easy |  | Todo | | |
| 025 | [np.add.at and scatter-style updates](problems/04_reductions_vectorized_ops/025_unbuffered_add_at/README.md) | Reductions & vectorized | Hard |  | Todo | | |
| 026 | [Generator API and seeding](problems/05_random_statistics/026_generator_api_seeding/README.md) | Random & stats | Easy | ★ | Todo | | |
| 027 | [Sampling: choice, permutation, shuffle](problems/05_random_statistics/027_sampling_permutations/README.md) | Random & stats | Easy |  | Todo | | |
| 028 | [Percentiles, clip and winsorize](problems/05_random_statistics/028_percentiles_clip_winsorize/README.md) | Random & stats | Medium |  | Todo | | |
| 029 | [var and std with ddof, cov, corrcoef](problems/05_random_statistics/029_var_std_cov_corrcoef/README.md) | Random & stats | Medium |  | Todo | | |
| 030 | [NaN and inf: isfinite, nan* functions, errstate](problems/05_random_statistics/030_nan_inf_handling/README.md) | Random & stats | Medium |  | Todo | | |
| 031 | [matmul, dot, @ and einsum](problems/06_linear_algebra_numerics/031_matmul_dot_einsum/README.md) | Linear algebra | Medium | ★ | Todo | | |
| 032 | [solve vs inv, condition number](problems/06_linear_algebra_numerics/032_solve_vs_inv/README.md) | Linear algebra | Medium |  | Todo | | |
| 033 | [Least squares with lstsq](problems/06_linear_algebra_numerics/033_least_squares/README.md) | Linear algebra | Medium |  | Todo | | |
| 034 | [Norms, SVD and eigh](problems/06_linear_algebra_numerics/034_norms_svd_eigh/README.md) | Linear algebra | Hard |  | Todo | | |
| 035 | [Float precision: isclose, cancellation, logsumexp](problems/06_linear_algebra_numerics/035_float_precision/README.md) | Linear algebra | Medium |  | Todo | | |
| 036 | [Images as arrays: HWC uint8 vs float, channels, grayscale](problems/07_arrays_in_practice/036_images_as_arrays/README.md) | In practice | Medium | ★ | Todo | | |
| 037 | [Sliding windows and framing](problems/07_arrays_in_practice/037_sliding_windows_framing/README.md) | In practice | Medium |  | Todo | | |
| 038 | [Saving and loading: .npy, .npz, memmap, loadtxt](problems/07_arrays_in_practice/038_saving_loading/README.md) | In practice | Easy |  | Todo | | |
| 039 | [NumPy and PyTorch bridge](problems/07_arrays_in_practice/039_numpy_torch_bridge/README.md) | In practice | Easy | ★ | Todo | | |
| 040 | [Performance: vectorized vs loops, fake vectorization, memory blow-ups](problems/07_arrays_in_practice/040_performance_pitfalls/README.md) | In practice | Medium |  | Todo | | |
