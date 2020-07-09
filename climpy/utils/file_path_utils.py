import glob
import os
from natsort import natsorted

STORAGE_PATH_SHAHEEN = '/project/k1090/osipovs/'
STORAGE_PATH_MISTRAL = '/work/mm0062/b302074/'


def get_root_storage_path_on_hpc():
    path = STORAGE_PATH_SHAHEEN
    if os.path.exists(STORAGE_PATH_MISTRAL):
        path = STORAGE_PATH_MISTRAL
    return path


def get_pictures_root_folder():
    return get_root_storage_path_on_hpc() + '/Pictures/'


def get_aeronet_file_path_root():
    return get_root_storage_path_on_hpc() + '/Data/NASA/Aeronet/'


def convert_file_path_mask_to_list(file_path_mask, N_parts=None):
    files_list = glob.glob(file_path_mask)
    files_list = natsorted(files_list)

    if N_parts is not None:
        files_list = files_list[0:N_parts]

    return files_list