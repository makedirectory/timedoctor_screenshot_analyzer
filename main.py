from src import extract_and_organize
from src import analyze_screenshots

def main():
    """
    Main function to run the entire Time Doctor Analyzer pipeline.
    It calls the main functions from extract_and_organize.py
    and analyze_screenshots.py consecutively.
    """
    print("\n--- Starting Time Doctor Analyzer Pipeline ---")

    # Step 1: Extract and organize the zipped screenshot files
    # This will unzip archives and move all screenshots into a single directory.
    # Ensure that INPUT_ZIPPED_DIR and OUTPUT_EXTRACTED_DIR are correctly
    # configured in extract_and_organize.py.
    extract_and_organize.main()

    # Step 2: Analyze the extracted screenshots
    # This will perform OCR, duplicate detection, categorize work,
    # and generate reports and graphs.
    # Ensure that SCREENSHOTS_DIR and TIME_SPREADSHEET_PATH are correctly
    # configured in analyze_screenshots.py.
    analyze_screenshots.analyze_screenshots()

    print("\n--- Time Doctor Analyzer Pipeline Complete ---")

if __name__ == "__main__":
    main()