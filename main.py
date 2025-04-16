def main():
    from variances import run as generate_variances
    from parser100k import run as parser_run_100k
    from compression import run as compression
    from parser1m import run as parser_run_1m


    parser_run_100k(
        overwrite_headers=False,
        overwrite_fragments=False,
        overwrite_hdfs=False
    )

    compression(
        overwrite_dataset=False
    )

    parser_run_1m(
        overwrite_headers=False,
        overwrite_fragments=False,
        overwrite_hdfs=False
    )

    generate_variances(
        overwrite_variances_100k=False,
        overwrite_variances_1m=False
    )

    from mh_1m_headers import run as mh1m_headers
    from mh_1m_fragments_for_100k_model import run as mh_1m_fragments_for_100k_model
    from mh_100k_fragments_for_1m_model import run as mh_100k_fragments_for_1m_model

    # This was used for analyzing the differences between the two datasets,
    # which helped when building the data for testing the 100k model using 1m samples.
    # mh1m_headers(
    #     overwrite_headers_diff=False,
    #     overwrite_headers_diff_highest_variances=False,
    #     overwrite_mh1m_headers=False
    # )

    mh_1m_fragments_for_100k_model(
        overwrite_fragments=False
    )

    mh_100k_fragments_for_1m_model(
        overwrite_fragments=False
    )


if __name__ == "__main__":
    main()
