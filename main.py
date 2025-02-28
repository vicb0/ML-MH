def main():
    # import numpy
    # import pandas as pd
    # from ML_algs.utils import drop_low_var_by_col, drop_metadata
    # from mh_1m_headers import DATASET_DIR
    # data = numpy.load(DATASET_DIR, allow_pickle=True)
    # print(data['metadata_columns'])
    # chunk = pd.read_hdf('./mh_1m_fragments/fragment_2.h5')
    # print(chunk, chunk.shape)
    # chunk = pd.read_hdf('./dataset.h5')
    # chunk = drop_low_var_by_col(drop_metadata(chunk))
    # print(chunk, chunk.shape)
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
