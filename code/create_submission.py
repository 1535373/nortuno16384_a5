import glob
import os
import zipfile

LAB = os.path.join('..', 'lab')


def lab_files():
    """(path, name-in-zip) for the drawing lab: your code, your strokes, your
    calibrations, the results draw.py saved, and everything in lab/handin."""
    patterns = [
        'robot.py', 'plane.py',
        os.path.join('strokes', '*.json'),
        'calibration-*.json',
        os.path.join('results', '*'),
        os.path.join('handin', '**', '*'),
    ]
    found = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(LAB, pattern), recursive=True)):
            if os.path.isfile(path) and os.path.basename(path) != 'README.md':
                found.append((path, os.path.join('lab', os.path.relpath(path, LAB))))
    return found


def create_submission():
    # List of (file, name it gets in the zip) to include in the submission
    files_to_include = [('double_pendulum.py', 'double_pendulum.py')] + lab_files()

    zip_filename = "submission.zip"

    # Create a zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file, name in files_to_include:
            if os.path.exists(file):
                zipf.write(file, name)
                print(f"Added {name} to the submission zip.")
            else:
                print(f"Warning: {file} not found and not included in the submission.")

    names = [name for _, name in files_to_include]
    for needed in (os.path.join('lab', 'plane.py'), os.path.join('lab', 'robot.py')):
        if needed not in names:
            print(f"Warning: {needed} is missing -- run this from the code folder.")
    if not any(name.startswith(os.path.join('lab', 'handin') + os.sep) for name in names):
        print("Warning: lab/handin is empty -- your drawing video is not in this submission.")

    print(f"\nSubmission zip created: {zip_filename}")
    print("Rename it to <andrew_id>.zip and upload it to Gradescope.")


if __name__ == "__main__":
    create_submission()
