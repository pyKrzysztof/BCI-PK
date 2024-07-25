import os
import mybci.xdfconversion

session_data_path = 'session_data/'


def convert_xdfs():
    for file in os.listdir(session_data_path):
        if not file.endswith('.xdf'):
            continue

        file_path = os.path.join(session_data_path, file)
        mybci.xdfconversion.xdf_to_csv(file_path, 0, 1, {'L': 1, 'R': 2, 'S': 3})


convert_xdfs()