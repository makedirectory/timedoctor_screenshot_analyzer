import os
import zipfile
import shutil
import re
from datetime import datetime

# --- Configuration ---
# Define the input directory where your zipped screenshot folders are located.
# For example, if your zipped files are in 'my_screenshots/zipped_archives/', set this path.
# IMPORTANT: Replace 'path/to/your/zipped_screenshot_folders' with the actual path.
INPUT_ZIPPED_DIR = './zipped_screenshot_folders'

# Define the output directory where all extracted screenshots will be placed.
# This directory will be created if it doesn't exist.
# IMPORTANT: Replace 'path/to/your/extracted_screenshots' with the desired output path.
OUTPUT_EXTRACTED_DIR = './extracted_screenshots'

# --- Functions ---

def create_directory_if_not_exists(directory_path):
    """
    Creates a directory if it does not already exist.

    Args:
        directory_path (str): The path to the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def extract_zip_file(zip_filepath, extract_to_dir):
    """
    Extracts the contents of a single zip file to a specified directory.

    Args:
        zip_filepath (str): The full path to the zip file.
        extract_to_dir (str): The directory where contents should be extracted.

    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
        print(f"Successfully extracted: {zip_filepath} to {extract_to_dir}")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_filepath} is not a valid zip file or is corrupted.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while extracting {zip_filepath}: {e}")
        return False

def rename_and_move_screenshot(src_filepath, dest_dir):
    """
    Renames a screenshot file to a standardized format (e.g., 'YYYYMMDD_HHMMSS_filename.png')
    based on its original filename (e.g., '2025-06-02T10_51_54.000Z-screen1.png')
    and moves it to the destination directory.

    Args:
        src_filepath (str): The full path to the source screenshot file.
        dest_dir (str): The destination directory where the file should be moved.

    Returns:
        str or None: The new full path of the moved file if successful, None otherwise.
    """
    # Regex to capture the date and time part from the filename
    # Example: 2025-06-02T10_51_54.000Z-screen1.png
    match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}).*\.(png|jpg|jpeg|gif)$',
                     os.path.basename(src_filepath), re.IGNORECASE)

    if match:
        timestamp_str = match.group(1)
        # Convert '2025-06-02T10_51_54' to '20250602_105154'
        try:
            dt_object = datetime.strptime(timestamp_str, '%Y-%m-%dT%H_%M_%S')
            standardized_timestamp = dt_object.strftime('%Y%m%d_%H%M%S')
        except ValueError:
            print(f"Warning: Could not parse timestamp from filename: {os.path.basename(src_filepath)}")
            standardized_timestamp = "" # Use empty if parsing fails

        original_filename = os.path.basename(src_filepath)
        # Create a new, unique filename
        new_filename = f"{standardized_timestamp}_{original_filename}" if standardized_timestamp else original_filename
        dest_filepath = os.path.join(dest_dir, new_filename)

        try:
            shutil.move(src_filepath, dest_filepath)
            print(f"Moved: {os.path.basename(src_filepath)} to {new_filename}")
            return dest_filepath
        except shutil.Error as e:
            print(f"Error moving {os.path.basename(src_filepath)}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while moving {os.path.basename(src_filepath)}: {e}")
            return None
    else:
        print(f"Warning: Filename format not recognized, skipping: {os.path.basename(src_filepath)}")
        return None

# --- Main execution logic ---
def main():
    """
    Main function to orchestrate the extraction and organization process.
    """
    print("--- Starting TimeDoctor Analyzer Setup ---")

    # 1. Create the output directory for extracted screenshots
    create_directory_if_not_exists(OUTPUT_EXTRACTED_DIR)

    # 2. Iterate through the input directory for zipped files
    if not os.path.exists(INPUT_ZIPPED_DIR):
        print(f"Error: Input directory not found: {INPUT_ZIPPED_DIR}")
        print("Please create the directory and place your zipped screenshot folders inside, then update INPUT_ZIPPED_DIR in the script.")
        return

    print(f"\nScanning for zipped files in: {INPUT_ZIPPED_DIR}")
    zip_files_found = 0
    for item in os.listdir(INPUT_ZIPPED_DIR):
        item_path = os.path.join(INPUT_ZIPPED_DIR, item)

        if os.path.isfile(item_path) and item.endswith('.zip'):
            zip_files_found += 1
            temp_extract_dir = os.path.join(INPUT_ZIPPED_DIR, "temp_extracted_" + os.path.splitext(item)[0])
            create_directory_if_not_exists(temp_extract_dir)

            if extract_zip_file(item_path, temp_extract_dir):
                # After extraction, move all image files to the central output directory
                print(f"Moving extracted images from '{temp_extract_dir}' to '{OUTPUT_EXTRACTED_DIR}'...")
                images_moved_count = 0
                for root, _, files in os.walk(temp_extract_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            src_image_path = os.path.join(root, file)
                            if rename_and_move_screenshot(src_image_path, OUTPUT_EXTRACTED_DIR):
                                images_moved_count += 1
                print(f"Finished moving {images_moved_count} images from this archive.")
            
            # Clean up the temporary extraction directory
            if os.path.exists(temp_extract_dir):
                try:
                    shutil.rmtree(temp_extract_dir)
                    print(f"Cleaned up temporary directory: {temp_extract_dir}")
                except Exception as e:
                    print(f"Error cleaning up {temp_extract_dir}: {e}")

    if zip_files_found == 0:
        print("No zip files found in the input directory. Please ensure your zipped screenshot folders are placed inside it.")
    else:
        print(f"\n--- Extraction and Organization Complete. {zip_files_found} zip files processed. ---")
        print(f"All extracted screenshots are now in: {OUTPUT_EXTRACTED_DIR}")

if __name__ == "__main__":
    main()
