import os
import glob
import argparse
import shutil


def count_files(directory, extension):
    total_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        total_count += len(glob.glob1(dirpath, "*." + extension))
    return total_count


def count_files_in_subfolders(directory, extension):
    count_dict = {}
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            count_dict[subfolder] = count_files(subfolder_path, extension)
    return count_dict


"Counts for all transformed datasets, the number of .pt files to easily see if some transformations failed."

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--remove", action="store_true",
                    help="Remove directories with missing files")
args = parser.parse_args()

data_path = './tmp'
file_extension = 'pt'
expected_size = 7
all_good = True

counts = count_files_in_subfolders(data_path, file_extension)
for folder, count in counts.items():
    if count < expected_size:
        all_good = False
        if not args.remove:
            print(f"Found missing files at {folder} ({count})")
        else:
            print(f"Found missing files at {folder} ({count}), removing...")
            shutil.rmtree(os.path.join(data_path, folder))

if all_good:
    print("All good!")
