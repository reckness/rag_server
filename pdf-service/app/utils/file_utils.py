import os
import shutil


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_file_extension(file_path):
    return file_path.split(".")[-1].lower()


def get_file_name(file_path):
    return os.path.basename(file_path)