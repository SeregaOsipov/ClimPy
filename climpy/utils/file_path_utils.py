import glob
import os
from natsort import natsorted

STORAGE_PATH_SHAHEEN = '/project/k10048/osipovs/'
STORAGE_PATH_LEVANTE = '/work/mm0062/b302074/'

root_path = None
root_data_path = None


def set_env(env):
    global root_path
    global root_data_path
    if env == 'local':
        root_path = os.path.expanduser('~')
    if env == 'workstation':  # KAUST workstation in the office
        root_path = '/home/osipovs/'
        root_data_path = '/HDD2/'
    if env == 'MacBook':
        root_path = '/Users/osipovs/'
        root_data_path = '/Users/osipovs/'
    if env == 'Levante':
        root_path = STORAGE_PATH_LEVANTE
        root_data_path = STORAGE_PATH_LEVANTE

def get_root_path_on_hpc():
    global root_path

    if root_path is None:
        root_path = os.path.expanduser('~')  # local first
        if os.path.exists(STORAGE_PATH_SHAHEEN):
            root_path = STORAGE_PATH_SHAHEEN
        elif os.path.exists(STORAGE_PATH_LEVANTE + 'Data/'):
            root_path = STORAGE_PATH_LEVANTE
    return root_path


def get_root_storage_path_on_hpc():
    global root_data_path

    if root_data_path is None:
        root_data_path = os.path.expanduser('~')  # local first
        if os.path.exists(STORAGE_PATH_SHAHEEN):
            root_data_path = STORAGE_PATH_SHAHEEN
        elif os.path.exists(STORAGE_PATH_LEVANTE + 'Data/'):
            root_data_path = STORAGE_PATH_LEVANTE
    return root_data_path


def get_pictures_root_folder():
    return get_root_path_on_hpc() + '/Pictures/'


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