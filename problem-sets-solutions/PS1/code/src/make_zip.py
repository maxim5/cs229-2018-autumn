"""Make a zip file for submission."""

import os
import re
import zipfile


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Collect file names
    file_paths = []
    for base_path, _, file_names in os.walk(script_dir):
        for file_name in file_names:
            if re.search(r'\.(py|gitkeep)$', file_name):
                file_path = os.path.join(base_path, file_name)
                file_paths.append(file_path)

    # Create zip file
    zip_path = os.path.join(script_dir, 'submission.zip')
    print('Writing {} files to {}'.format(len(file_paths), zip_path))
    with zipfile.ZipFile(zip_path, 'w') as zip_fh:
        for file_path in file_paths:
            rel_path = os.path.relpath(file_path, script_dir)
            zip_fh.write(file_path, rel_path)


if __name__ == '__main__':
    main()
