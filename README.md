# Time Doctor Analyzer
This project provides a pipeline to analyze Time Doctor screenshots and compare the activity shown in them against reported work hours from a spreadsheet, helping to identify potentially unproductive or fraudulent time.

## Project Structure

`main.py`: Runs entire pipeline

`pyproject.toml`: Manages project dependencies using uv.

`reports/`: (Automatically created) Directory for output CSV reports and generated graphs.

### `src`
`extract_and_organize.py`: Script to extract screenshots from zipped archives and consolidate them into a single folder.

`analyze_screenshots.py`: Script to perform OCR, detect duplicates/idle pop-ups, categorize work activity from screenshots, integrate with time tracking data, and generate reports/graphs.

## Setup and Installation
This project uses `uv` for dependency management.

### Install uv:
If you don't have uv installed globally, you can install it via pip: `pip install uv`

### Install Dependencies:
The `pyproject.toml` file defines the project's dependencies.

`uv pip install -e .`: Installs dependencies listed in pyproject.toml

This command tells uv to install the project in "editable" mode (-e .), which is useful for development and automatically picks up dependencies from `pyproject.toml`.

### Install Tesseract OCR Engine:
pytesseract is a Python wrapper for Google's Tesseract OCR engine. You must install Tesseract separately on your system.

#### Windows
Download the installer from Tesseract-OCR GitHub. Remember the installation path (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe).

#### macOS
`brew install tesseract`

#### Linux (Debian/Ubuntu):
1. sudo apt update
2. sudo apt install tesseract-ocr
3. sudo apt install libtesseract-dev # For development headers if needed

Crucial Step: Update the pytesseract.pytesseract.tesseract_cmd variable in analyze_screenshots.py if Tesseract is not in your system's PATH. For example:

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # For Windows
# Or just 'tesseract' if it's in your PATH

### Usage
Prepare Your Data:

Add your zipped screenshot archives to `zipped_screenshot_folders`. Place all your .zip files from Time Doctor exports here.

Place your time tracking spreadsheet (CSV format) in `data` as `time_tracking.csv`. Ensure it matches the format need.

#### Format
(Name, Total (Decimal), Date columns like 'Sun, Jun 16 (Decimal)').

Reference-style: 
![format required][timedoctor_format]

[timedoctor_format]: /images/export_example.png "Format required example"

## Optional Configuration:

Open extract_and_organize.py and update INPUT_ZIPPED_DIR and OUTPUT_EXTRACTED_DIR if you'd like.

Open analyze_screenshots.py and update SCREENSHOTS_DIR (should be the same as OUTPUT_EXTRACTED_DIR), TIME_SPREADSHEET_PATH, and OUTPUT_REPORTS_DIR. Adjust pytesseract.pytesseract.tesseract_cmd if necessary.

Review WORK_CATEGORIES in analyze_screenshots.py and customize keywords to better reflect your specific work environment and what constitutes "real" vs. "fake" work for your context.

## Run the entire pipeline:
Activate your uv virtual environment if it's not already active:

`source .venv/bin/activate` # On Windows: `.venv\Scripts\activate`

Then run: `python main.py` to run the entire pipeline

### Run ONLY the Extraction Script: 

`python src/extract_and_organize.py`

This will extract all screenshots and place them into the OUTPUT_EXTRACTED_DIR.

### Run ONLY the Analysis Script: 

`python analyze_screenshots.py`

This script will:
 - Read screenshots, perform OCR.
 - Detect duplicate screenshots and Time Doctor idle pop-ups.
 - Categorize activity based on defined keywords.
 - Integrate with your time tracking data.
 - Generate a detailed CSV report (screenshot_analysis_report.csv and comprehensive_time_analysis_report.csv) in the reports/ directory.
 - Generate a bar graph visualizing reported hours vs. estimated real/fake hours (hours_comparison_graph.png) in the reports/ directory.
 - Print an overall summary of estimated fraudulent/idle time to the console.

## Further Ideas
Advanced Duplicate Detection: For even more robust duplicate detection, especially against minor screen changes, consider using image diffing libraries like ImageChops.difference from PIL, or more advanced hashing algorithms (e.g., phash, dhash). The current are_images_nearly_identical function uses SSIM and pixel difference, which is a good start.

Idle Time Logic: The current logic marks screenshots with Time Doctor idle pop-ups as "fake/idle." Expand this to calculate the duration of these idle periods more precisely.

Time Allocation Heuristics: The current method for allocating hours based on screenshots is a simple proportional distribution between unique, non-duplicate screenshots. For more accuracy:
 - Assign specific time weights to certain categories (e.g., a "meeting" screenshot might imply a longer duration).
 - Use a fixed interval (e.g., if screenshots are taken every 10 minutes, each non-duplicate screenshot accounts for 10 minutes).

User-Specific Analysis: Process data for multiple users generating individual reports.

Interactive Dashboard: Instead of static CSVs and PNGs, use a library like Dash or Streamlit to create an an interactive web dashboard for real-time analysis and filtering.

Machine Learning for Categorization: For highly nuanced activity, train a machine learning model (e.g., a text classifier) on a labeled dataset of screenshots to categorize work more accurately than simple keyword matching.

Screenshot Content Analysis: Beyond OCR, object detection models (e.g., with OpenCV and pre-trained models) could identify specific application icons or UI elements, which can be more robust than text-based OCR for identifying open applications.