"""Make a zip file for submission."""

import os
import zipfile

# List of relative paths of files to include in the zip
FILE_PATHS = [
    os.path.join('output', '.gitkeep'),
    'p1_nn.py',
    'p4_ica.py',
]


def make_zip():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Create zip file
    zip_path = os.path.join(script_dir, 'submission.zip')
    print('Writing {} files to {}'.format(len(FILE_PATHS), zip_path))
    with zipfile.ZipFile(zip_path, 'w') as zip_fh:
        for file_path in FILE_PATHS:
            rel_path = os.path.relpath(file_path, script_dir)
            zip_fh.write(file_path, rel_path)


if __name__ == '__main__':
    make_zip()
