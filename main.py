def main():
    # import numpy
    # import pandas as pd
    # from ML_algs.utils import drop_low_var_by_col, drop_metadata
    # from mh_1m_headers import DATASET_DIR
    # import pickle

    # with open('results-mh1M.pkl', 'rb') as f:
    #     a = pickle.load(f)
    # print(a)
    # return
    from parser import run as parser_run
    from compression import run as compression_run
    from mh_1m_headers import run as mh1m_headers_run
    from mh_1m_fragments import run as mh1m_fragments_run
    
    parser_run(
        overwrite_headers=False,
        overwrite_fragments=False,
        overwrite_hdfs=False
    )

    compression_run(
        overwrite_dataset=False,
        overwrite_variances=False,
    )

    mh1m_headers_run(
        overwrite_headers_diff=False,
        overwrite_headers_diff_highest_variances=False,
        overwrite_mh1m_headers=False
    )

    mh1m_fragments_run(
        overwrite_fragments=False
    )


if __name__ == "__main__":
    main()
