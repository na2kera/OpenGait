# 計算環境の記録（2026-08-23 02:02:44 JST）

## CPU
    CPU(s):                                  16
    Model name:                              AMD EPYC 7302 16-Core Processor
    Thread(s) per core:                      1
    Core(s) per socket:                      16
    Socket(s):                               1

## スレッド関連の環境変数（ホスト）
    （未設定 ＝ BLASは既定で全16コアを使用。実測でFID-anyジョブが約1300% CPUを占有）

## コンテナ casia-b-fid-3 内の numpy/scipy 構成
    numpy 1.23.5 / scipy 1.11.3
    openblas64__info:
        libraries = ['openblas64_', 'openblas64_']
        library_dirs = ['/usr/local/lib']
        language = c
        define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None)]
        runtime_library_dirs = ['/usr/local/lib']
    blas_ilp64_opt_info:
        libraries = ['openblas64_', 'openblas64_']
        library_dirs = ['/usr/local/lib']
        language = c
        define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None)]
        runtime_library_dirs = ['/usr/local/lib']
    openblas64__lapack_info:
        libraries = ['openblas64_', 'openblas64_']
        library_dirs = ['/usr/local/lib']
        language = c
        define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None), ('HAVE_LAPACKE', None)]
        runtime_library_dirs = ['/usr/local/lib']
    lapack_ilp64_opt_info:
        libraries = ['openblas64_', 'openblas64_']
        library_dirs = ['/usr/local/lib']
        language = c
        define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None), ('HAVE_LAPACKE', None)]
        runtime_library_dirs = ['/usr/local/lib']
    Supported SIMD extensions in this NumPy install:
        baseline = SSE,SSE2,SSE3
        found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2
        not found = AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
    コンテナ内のスレッド環境変数: （未設定）
