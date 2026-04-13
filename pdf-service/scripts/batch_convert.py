import os
import sys
from app.services.converter import convert_to_pdf


def batch_convert(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                print(f"Converting {file_path}...")
                pdf = convert_to_pdf(file_path)
                print(f"Converted to {pdf}")
            except Exception as e:
                print(f"Error converting {file_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_convert.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    batch_convert(directory)