import os

import numpy


# run the following command: `git clone https://github.com/Malware-Hunter/MH-1M`
# navigate to data/compressed/zip-intents-permissions-opcodes-apicalls
# run the command `copy /b amex-1M-[intents-permissions-opcodes-apicalls].npz.7z.part* full_archive.7z`
# extract full_archive.7z, rename the .npz to dataset.npz
# run this script

lines1 = []

def get_lines():
    global lines1
    #############################################################################################
    # Loads the npz, get the columns, adapt it to be the same format as MH-100K, sort and save it
    data = numpy.load(r'.\MH-1M\data\compressed\zip-intents-permissions-opcodes-apicalls\dataset.npz', allow_pickle=True)

    lines1 = data['column_names']

    for c, line in enumerate(lines1):
        category, column = line.split("::")
        category = category[:-1]

        if line.startswith("apicall"):
            lines1[c] = f"{category}::{column}()"
        else:
            lines1[c] = f"{category}::{column}"

    lines1.sort()
    #############################################################################################

def mh_1m_headers(overwrite=False):
    if os.path.isfile('./MH-1M/headers.txt') and not overwrite:
        return

    with open('./MH-1M/headers.txt', 'w') as f:
        f.write("\n".join(lines1))


def headers_diff(overwrite=False):
    if os.path.isfile('headers_diff.txt') and not overwrite:
        return 

    # Convert the headers from MH-100K to a single file, sort and save it
    with open('./headers/apicall.txt', 'r') as f1:
        with open('./headers/intent.txt', 'r') as f2:
            with open('./headers/permission.txt', 'r') as f3:
                with open('./headers/headers.txt', 'w') as f4:
                    lines2 = []
                    lines2.extend(f1.read().lower().split("\n"))
                    lines2.extend(f2.read().lower().split("\n"))
                    lines2.extend(f3.read().lower().split("\n"))
                    lines2 = [el for el in lines2 if el != '']
                    lines2.sort()
                    f4.write("\n".join(lines2))
    #############################################################################################
    # Compare the two files
    p1, p2 = 0, 0
    logs = []
    # lines1 = MH-1M
    # lines2 = MH-100K
    while p1 < len(lines1) and p2 < len(lines2):
        if lines1[p1] > lines2[p2]:
            logs.append(f"{lines2[p2]} from MH-100K not in MH-1M")
            p2 += 1
        elif lines1[p1] < lines2[p2]:
            logs.append(f"{lines1[p1]} from MH-1M not in MH-100K")
            p1 += 1
        else:
            p1 += 1
            p2 += 1

    while p1 < len(lines1):
        logs.append(f"{lines1[p1]} from MH-1M not in MH-100K")
        p1 += 1
    while p2 < len(lines2):
        logs.append(f"{lines2[p2]} from MH-100K not in MH-1M")
        p2 += 1
    #############################################################################################

    with open('headers_diff.txt', 'w') as f:
        f.write("\n".join(logs))


def headers_diff_highest_variances(overwrite=False):
    if os.path.isfile('headers_diff_highest_variances.txt') and not overwrite:
        return 

    import pandas as pd
    from ML_algs.utils import drop_low_var_by_col_100k
    from ML_algs.utils import drop_metadata

    df = pd.read_hdf('./dataset.h5')
    df = drop_low_var_by_col_100k(drop_metadata(df))
    lines2 = sorted(map(lambda x: x.lower(), df.columns.to_list()))

    # Compare the two files
    p1, p2 = 0, 0
    logs = []
    # lines1 = MH-1M
    # lines2 = MH-100K (4000 highest variances)
    while p1 < len(lines1) and p2 < len(lines2):
        if lines1[p1] > lines2[p2]:
            logs.append(f"{lines2[p2]} from MH-100K not in MH-1M")
            p2 += 1
        elif lines1[p1] < lines2[p2]:
            logs.append(f"{lines1[p1]} from MH-1M not in MH-100K")
            p1 += 1
        else:
            p1 += 1
            p2 += 1

    while p1 < len(lines1):
        logs.append(f"{lines1[p1]} from MH-1M not in MH-100K")
        p1 += 1
    while p2 < len(lines2):
        logs.append(f"{lines2[p2]} from MH-100K not in MH-1M")
        p2 += 1

    with open('headers_diff_highest_variances.txt', 'w') as f:
        f.write("\n".join(logs))


def run(overwrite_mh1m_headers=False, overwrite_headers_diff=False, overwrite_headers_diff_highest_variances=False):
    get_lines()
    mh_1m_headers(overwrite=overwrite_mh1m_headers)
    headers_diff(overwrite=overwrite_headers_diff)
    headers_diff_highest_variances(overwrite=overwrite_headers_diff_highest_variances) 


def main():
    run(
        overwrite_mh1m_headers=True,
        overwrite_headers_diff=True,
        overwrite_headers_diff_highest_variances=True
    )


if __name__ == "__main__":
    main()
