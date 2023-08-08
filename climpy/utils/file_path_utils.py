import glob
import os
from natsort import natsorted

STORAGE_PATH_SHAHEEN = '/project/k1090/osipovs/'
STORAGE_PATH_MISTRAL = '/work/mm0062/b302074/'


def get_root_storage_path_on_hpc():
    path = os.path.expanduser('~')  # local first
    if os.path.exists(STORAGE_PATH_SHAHEEN):
        path = STORAGE_PATH_SHAHEEN
    elif os.path.exists(STORAGE_PATH_MISTRAL + 'Data/'):
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


def make_dir_for_the_full_file_path(full_file_path):
    if not os.path.exists(os.path.dirname(full_file_path)):
        print('Path {} does not exist. Creating directory.'.format(os.path.dirname(full_file_path)))
        os.makedirs(os.path.dirname(full_file_path))