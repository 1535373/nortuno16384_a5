import os
import zipfile


def create_submission():
    # List of (file, name it gets in the zip) to include in the submission
    files_to_include = [
        ('double_pendulum.py', 'double_pendulum.py'),
    ]

    zip_filename = "submission.zip"

    # Create a zip file
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file, name in files_to_include:
            if os.path.exists(file):
                zipf.write(file, name)
                print(f"Added {name} to the submission zip.")
            else:
                print(f"Warning: {file} not found and not included in the submission.")

    print(f"\nSubmission zip created: {zip_filename}")
    print("Rename it to <andrew_id>.zip and upload it to Gradescope.")


if __name__ == "__main__":
    create_submission()
