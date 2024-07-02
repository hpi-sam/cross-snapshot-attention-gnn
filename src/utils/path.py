import os


def remove_after_last_slash(string):
    return string.rsplit("/", 1)[0]


def get_after_last_slash(string):
    return string.rsplit("/", 1)[1]


def path_exists(path):
    return os.path.exists(path)


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def get_folders(path):
    """
    Returns a list of folder names in the specified path.
    """
    folders = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            folders.append(item)
    return folders
