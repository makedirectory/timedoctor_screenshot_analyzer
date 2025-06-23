import os
import re
import csv
from datetime import datetime, timedelta
from PIL import Image, ImageChops, ImageStat, UnidentifiedImageError
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import collections
import json
import multiprocessing

# --- Configuration ---
# Set the path to the directory where your extracted screenshots are located.
# This should be the same as OUTPUT_EXTRACTED_DIR from extract_and_organize.py
# IMPORTANT: Replace 'path/to/your/extracted_screenshots' with the actual path.
SCREENSHOTS_DIR = './extracted_screenshots'

# Set the path to your time tracking spreadsheet (CSV format assumed for now)
# IMPORTANT: Replace 'path/to/your/time_tracking.csv' with the actual path.
TIME_SPREADSHEET_PATH = './data/time_tracking.csv'

# Output directory for reports and graphs
OUTPUT_REPORTS_DIR = './reports'

# Path to Tesseract executable (adjust if not in your PATH)
# For Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS/Linux: Usually in PATH, or specify full path if installed elsewhere (e.g., '/usr/local/bin/tesseract')
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Path to the JSON file containing work categories
WORK_CATEGORIES_FILE = 'work_categories.json'

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

def load_work_categories(file_path):
    """
    Loads work categories from a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            categories_raw = json.load(f)
            # Reformat to match original (keywords list, is_real_work boolean/None)
            categories_formatted = {}
            for category_name, data in categories_raw.items():
                keywords = data.get("keywords", [])
                is_real_work = data.get("is_real_work", None)
                categories_formatted[category_name] = (keywords, is_real_work)
            return categories_formatted
    except FileNotFoundError:
        print(f"Error: Work categories file not found at '{file_path}'. Please ensure it exists.")
        return {} # Return empty dictionary to prevent crash
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Please check file format.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred loading work categories: {e}")
        return {}

# Load work categories at script startup
# Note: For multiprocessing, this global might not be directly inherited
# by 'spawned' child processes. It's safer to pass it as an argument if needed
# in worker functions, as done in _process_single_screenshot_worker.
WORK_CATEGORIES = load_work_categories(WORK_CATEGORIES_FILE)

# Thresholds (adjust as needed)
DUPLICATE_THRESHOLD_SSIM = 0.98 # SSIM value above which images are considered duplicates
DUPLICATE_THRESHOLD_PIXEL_DIFF = 0 # Percentage difference in pixels (0 means exact match)
IDLE_POPUP_TEXT_THRESHOLD = 0.6 # Percentage of keywords found to classify as idle popup
MIN_SCREENSHOT_INTERVAL_SECONDS = 15 # Minimum time between screenshots to consider them distinct
DUPLICATE_LOOKBACK_WINDOW = 6 # Number of previous unique screenshots to compare against for duplicates

# --- Helper Functions for Image Processing ---

def calculate_image_hash(image_path):
    """
    Calculates a perceptual hash for an image using average hashing.
    Used for quick duplicate detection.
    Args:
        image_path (str): Path to the image file.
    Returns:
        int: Integer representation of the image hash, or None if error.
    """
    try:
        # Use UnidentifiedImageError for more specific error handling if possible
        image = Image.open(image_path).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = np.array(image.getdata())
        avg = pixels.mean()
        hash_bits = (pixels > avg).astype(int)
        img_hash = "".join(str(b) for b in hash_bits)
        return int(img_hash, 2)
    except (UnidentifiedImageError, IOError) as e:
        print(f"Warning: Could not open or identify image file for hashing '{image_path}': {e}. This file might be corrupted or not a valid image. Skipping.")
        return None
    except Exception as e:
        print(f"Warning: An unexpected error occurred calculating hash for '{image_path}': {e}. Skipping.")
        return None

def are_images_nearly_identical(img_path1, img_path2, ssim_threshold=DUPLICATE_THRESHOLD_SSIM, pixel_diff_threshold=DUPLICATE_THRESHOLD_PIXEL_DIFF):
    """
    Compares two images for near-identicality using SSIM and pixel difference.
    Args:
        img_path1 (str): Path to the first image.
        img_path2 (str): Path to the second image.
        ssim_threshold (float): SSIM score above which images are considered very similar.
        pixel_diff_threshold (float): Max percentage of differing pixels.
    Returns:
        bool: True if images are nearly identical, False otherwise.
        float: SSIM score.
    """
    try:
        # Open images and convert to grayscale for SSIM
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1 is None or img2 is None:
            # This check handles cases where cv2.imread fails (e.g., file not found, corrupted)
            # This message is for specific SSIM comparison failures, not general image loading.
            # General image loading warnings are handled upstream in calculate_image_hash or _process_single_screenshot_worker.
            return False, 0.0

        # Resize images to be the same dimensions for comparison
        # Using a common smaller size to speed up comparison
        fixed_size = (200, 200) # Arbitrary smaller size
        img1_resized = cv2.resize(img1, fixed_size, interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, fixed_size, interpolation=cv2.INTER_AREA)

        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        score, _ = ssim(gray1, gray2, full=True)

        # Calculate pixel difference (for exact duplicates, if SSIM is high)
        if score >= ssim_threshold:
            diff = cv2.absdiff(img1_resized, img2_resized)
            non_zero_count = np.count_nonzero(diff)
            total_pixels = img1_resized.shape[0] * img1_resized.shape[1] * img1_resized.shape[2] # For RGB images
            pixel_diff_percentage = (non_zero_count / total_pixels) * 100

            if pixel_diff_percentage <= pixel_diff_threshold:
                return True, score
        return False, score

    except Exception as e:
        print(f"Warning: An unexpected error occurred comparing images '{img_path1}' and '{img_path2}': {e}. Skipping comparison.")
        return False, 0.0

def extract_text_from_image(image_path):
    """
    Extracts text from an image using OCR (Tesseract).
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Extracted text, or empty string if error.
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not in your PATH. Please install it or set 'pytesseract.pytesseract.tesseract_cmd'.")
        return ""
    except (UnidentifiedImageError, IOError) as e:
        print(f"Warning: Could not open or identify image file for OCR '{image_path}': {e}. This file might be corrupted or not a valid image. Skipping OCR.")
        return ""
    except Exception as e:
        print(f"Warning: An unexpected error occurred performing OCR on '{image_path}': {e}. Skipping OCR.")
        return ""

def _categorize_work_internal(extracted_text, filename, categories_data, idle_text_threshold):
    """
    Internal helper to categorize work, designed to be called by worker function.
    """
    text_lower = extracted_text.lower()
    filename_lower = filename.lower()

    idle_popup_keywords = categories_data.get("time_doctor_idle_popup", ([], None))[0]
    idle_keywords_found = 0
    total_idle_keywords = len(idle_popup_keywords)
    for keyword in idle_popup_keywords:
        if keyword in text_lower:
            idle_keywords_found += 1
    if total_idle_keywords > 0 and (idle_keywords_found / total_idle_keywords) >= idle_text_threshold:
        return "time_doctor_idle_popup", False

    for category, (keywords, is_real) in categories_data.items():
        if category == "time_doctor_idle_popup" or category == "unidentified":
            continue
        for keyword in keywords:
            if keyword.lower() in text_lower or keyword.lower() in filename_lower:
                return category, is_real
    return "unidentified", None

def parse_screenshot_timestamp(filename):
    """
    Parses the timestamp from the standardized screenshot filename.
    Assumes filename format:YYYYMMDD_HHMMSS_originalfilename.png
    Args:
        filename (str): The filename of the screenshot.
    Returns:
        datetime or None: Parsed datetime object, or None if parsing fails.
    """
    match = re.match(r'(\d{8}_\d{6})_.*', filename)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            return None
    return None

# --- Worker Function for Parallel Processing ---
def _process_single_screenshot_worker(args):
    """
    Worker function to process a single screenshot in parallel.
    It performs timestamp parsing, hash calculation, OCR, and initial categorization.
    Returns a dictionary of preliminary analysis data for one screenshot.
    """
    filepath, categories_data, idle_text_threshold = args

    filename = os.path.basename(filepath)
    timestamp = parse_screenshot_timestamp(filename)

    if timestamp is None:
        return {"filename": filename, "error": "timestamp_parse_failed"}

    # Calculate hash
    img_hash = calculate_image_hash(filepath)
    if img_hash is None:
        return {"filename": filename, "error": "image_unreadable_for_hash"}

    # Perform OCR
    extracted_text = extract_text_from_image(filepath)
    # Categorize work based on OCR text and filename
    category, is_real_work = _categorize_work_internal(extracted_text, filename, categories_data, idle_text_threshold)

    return {
        "timestamp": timestamp,
        "filename": filename,
        "filepath": filepath,
        "img_hash": img_hash,
        "extracted_text": extracted_text,
        "category": category,
        "is_real_work": is_real_work,
        "error": None
    }


# --- Main Analysis Logic ---
def analyze_screenshots():
    """
    Orchestrates the analysis of screenshots, including OCR, duplicate detection,
    and work categorization.
    """
    print("\n--- Starting Screenshot Analysis ---")

    create_directory_if_not_exists(OUTPUT_REPORTS_DIR)

    if not os.path.exists(SCREENSHOTS_DIR):
        print(f"Error: Screenshots directory not found: {SCREENSHOTS_DIR}")
        print("Please ensure you have run extract_and_organize.py and the path is correct.")
        return

    screenshot_files = sorted([f for f in os.listdir(SCREENSHOTS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    
    if not screenshot_files:
        print(f"No screenshot images found in: {SCREENSHOTS_DIR}")
        return

    num_unreadable_images = 0

    print(f"Found {len(screenshot_files)} screenshots to analyze. Starting parallel processing...")

    # Prepare arguments for multiprocessing pool
    # Pass necessary global-like variables to the worker function as arguments
    pool_args = []
    for filename in screenshot_files:
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        pool_args.append((filepath, WORK_CATEGORIES, IDLE_POPUP_TEXT_THRESHOLD))

    # Use multiprocessing Pool to parallelize initial image processing
    # The default 'fork' start method on Unix-like systems might be faster due to shared memory,
    # but 'spawn' is safer and more portable, especially on Windows.
    # We will use the default context for simplicity which is usually 'fork' on Linux/macOS and 'spawn' on Windows.
    num_processes = os.cpu_count() or 1 # Use all available CPU cores
    print(f"Using {num_processes} processes for parallel analysis.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        raw_analysis_results = pool.map(_process_single_screenshot_worker, pool_args)

    print("Parallel processing complete. Consolidating results and performing sequential duplicate checks...")

    analysis_results_pre_dedupe = []
    for res in raw_analysis_results:
        if res.get("error"):
            # Error messages are already printed by worker's helper functions (e.g., cannot identify image file)
            num_unreadable_images += 1
        else:
            analysis_results_pre_dedupe.append(res)

    # Sort results by timestamp for sequential duplicate detection and accurate time allocation
    analysis_results_pre_dedupe.sort(key=lambda x: x["timestamp"])

    # --- Second Sequential Pass for Duplicate Detection and Final Flagging ---
    final_analysis_results = []
    
    # This dictionary helps track previously seen image hashes (from unique images)
    # to efficiently find potential duplicates across the *entire* sorted dataset.
    processed_hashes_for_ssim_check = {} # {image_hash: [filepath1, filepath2, ...]}
    
    # This deque stores only the *recently processed unique* screenshots (file path and timestamp)
    # for SSIM comparison within the defined lookback window.
    recent_unique_screenshots = collections.deque(maxlen=DUPLICATE_LOOKBACK_WINDOW)

    # Stores info about the very last screenshot that was fully processed,
    # used specifically for the "consecutive" duplicate check.
    last_processed_screenshot_info = None # Stores {filepath, timestamp}

    for i, res in enumerate(analysis_results_pre_dedupe):
        current_filepath = res["filepath"]
        current_timestamp = res["timestamp"]
        current_img_hash = res["img_hash"]
        current_category = res["category"]
        current_is_real_work = res["is_real_work"]
        current_extracted_text = res["extracted_text"]

        is_duplicate = False
        duplicate_of_file = None
        duplicate_reason = None

        # --- Phase A: Hash-based (check against all *already processed unique* images by hash) ---
        # If the current image's hash matches any hash of a previously processed unique image,
        # perform a more rigorous SSIM check.
        if current_img_hash in processed_hashes_for_ssim_check:
            for prev_path_hash_match in processed_hashes_for_ssim_check[current_img_hash]:
                are_identical, ssim_score = are_images_nearly_identical(prev_path_hash_match, current_filepath, 
                                                                         ssim_threshold=DUPLICATE_THRESHOLD_SSIM,
                                                                         pixel_diff_threshold=DUPLICATE_THRESHOLD_PIXEL_DIFF)
                if are_identical:
                    is_duplicate = True
                    duplicate_of_file = os.path.basename(prev_path_hash_match)
                    duplicate_reason = f"Hash & SSIM identical (SSIM: {ssim_score:.4f})"
                    break # Found a duplicate, no need to check others with same hash

        # --- Phase B: SSIM comparison against recent unique screenshots (within lookback window) ---
        # If not already marked as a duplicate by the hash-based check,
        # compare it against the limited set of recent unique screenshots.
        if not is_duplicate:
            for prev_path_recent, prev_ts_recent in recent_unique_screenshots:
                are_identical, ssim_score = are_images_nearly_identical(prev_path_recent, current_filepath,
                                                                         ssim_threshold=DUPLICATE_THRESHOLD_SSIM,
                                                                         pixel_diff_threshold=DUPLICATE_THRESHOLD_PIXEL_DIFF)
                if are_identical:
                    is_duplicate = True
                    duplicate_of_file = os.path.basename(prev_path_recent)
                    duplicate_reason = f"Recent SSIM identical (SSIM: {ssim_score:.4f}) within window of {DUPLICATE_LOOKBACK_WINDOW} unique images"
                    break

        # --- Phase C: Consecutive (small time difference) Check ---
        # This specifically checks if the *current* image is very similar to the *absolute last processed* one,
        # and if the time between them is very short. This is useful for detecting rapid, unproductive captures.
        # This is applied even if it's already a duplicate by other means, as it adds specific context.
        if last_processed_screenshot_info and current_timestamp and last_processed_screenshot_info["timestamp"]:
            time_diff = current_timestamp - last_processed_screenshot_info["timestamp"]
            if time_diff.total_seconds() < MIN_SCREENSHOT_INTERVAL_SECONDS:
                are_identical_consecutive, _ = are_images_nearly_identical(last_processed_screenshot_info["filepath"], current_filepath,
                                                                            ssim_threshold=DUPLICATE_THRESHOLD_SSIM,
                                                                            pixel_diff_threshold=DUPLICATE_THRESHOLD_PIXEL_DIFF)
                if are_identical_consecutive:
                    if not is_duplicate:
                        duplicate_of_file = os.path.basename(last_processed_screenshot_info["filepath"])
                        duplicate_reason = f"Consecutive (time diff {time_diff.total_seconds():.1f}s) & Content Identical"
                    else:
                        if "Consecutive" not in duplicate_reason: # Append if not already part of the reason
                            duplicate_reason += f"; Consecutive (time diff {time_diff.total_seconds():.1f}s)"
                    is_duplicate = True # Ensure it's marked as duplicate

        # --- Phase D: Time Doctor Idle Popup Detection ---
        # If it's identified as an idle popup, it's inherently considered unproductive.
        if current_is_real_work is False and current_category == "time_doctor_idle_popup":
            if not is_duplicate:
                duplicate_reason = "Time Doctor Idle Popup Detected"
            else:
                if "Idle Popup" not in duplicate_reason: # Append if not already part of the reason
                    duplicate_reason += "; Time Doctor Idle Popup Detected"
            is_duplicate = True # Mark as duplicate/unproductive time

        # Append final processed result for this screenshot
        final_analysis_results.append({
            "timestamp": current_timestamp,
            "filename": res["filename"], # Original filename
            "extracted_text": current_extracted_text,
            "category": current_category,
            "is_real_work": current_is_real_work, # This is the categorization from OCR, not the final 'productive' status
            "is_duplicate": is_duplicate, # This is the final flag for whether it contributes to productive time
            "duplicate_of": duplicate_of_file,
            "duplicate_reason": duplicate_reason
        })

        # Update lists for next iteration's comparisons for unique images only
        if not is_duplicate:
            recent_unique_screenshots.append((current_filepath, current_timestamp))
            processed_hashes_for_ssim_check.setdefault(current_img_hash, []).append(current_filepath)

        # Always update last processed info, even if current is a duplicate, for the next consecutive check.
        last_processed_screenshot_info = {
            "filepath": current_filepath,
            "timestamp": current_timestamp
        }

    # Now use final_analysis_results for report generation and time integration
    
    # --- Generate Analysis Report (CSV) ---
    report_filepath = os.path.join(OUTPUT_REPORTS_DIR, "screenshot_analysis_report.csv")
    with open(report_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["timestamp", "filename", "category", "is_real_work", "is_duplicate", "duplicate_of", "duplicate_reason", "extracted_text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_analysis_results) # Use final_analysis_results
    print(f"\nScreenshot analysis report generated: {report_filepath}")

    # --- Integrate with Time Tracking Data & Generate Final Report/Graph ---
    
    # Load the time tracking data
    try:
        # Assuming the CSV has 'Name' and date columns like 'Mon, Jun 17 (Decimal)'
        # We need to reshape this data for easier merging.
        time_df_raw = pd.read_csv(TIME_SPREADSHEET_PATH)
        
        # Melt the DataFrame to unpivot date columns into rows
        id_vars = ['Name', 'Total (Decimal)'] # Columns to keep as identifiers
        value_vars = [col for col in time_df_raw.columns if col not in id_vars]
        
        time_df_melted = time_df_raw.melt(id_vars=id_vars, value_vars=value_vars, 
                                          var_name='Date_Str', value_name='Hours_Worked')
        
        # Parse date from 'Mon, Jun 17 (Decimal)' format
        # Need to infer year if not present (assuming current year for now)
        current_year = datetime.now().year
        time_df_melted['Date'] = time_df_melted['Date_Str'].apply(
            lambda x: datetime.strptime(x.split(' (')[0].strip() + f', {current_year}', '%a, %b %d, %Y').date()
        )
        
        # Create a DataFrame from analysis results
        analysis_df = pd.DataFrame(final_analysis_results) # Use final_analysis_results
        analysis_df['date'] = analysis_df['timestamp'].dt.date

        # Calculate actual vs. fake work hours from screenshots per day
        # For this, we need to make assumptions about how long a screenshot "represents".
        # A simple approach: each unique, non-duplicate screenshot represents an equal
        # share of the time between it and the next unique screenshot.
        
        # Filter for non-duplicate screenshots (representing actual activity)
        # Note: If an image is a duplicate (is_duplicate == True), it does NOT contribute to productive time.
        # Only genuinely unique and "real work" screenshots (is_duplicate == False and is_real_work == True)
        # will contribute to productive time.
        
        # A screenshot counts as "productive" if it's not a duplicate AND it's categorized as real work.
        # A screenshot counts as "unproductive" if it's a duplicate OR it's categorized as fake work (but not duplicate).
        
        # Determine effective time duration for each *unique* (non-duplicate) screenshot.
        # If consecutive screenshots are duplicates, only the first in the series represents time.
        
        # Let's adjust the logic to count time_diff only for actual transitions between *unique* activities.
        # Filter for screenshots that are NOT marked as duplicates
        unique_active_screenshots_df = analysis_df[analysis_df['is_duplicate'] == False].copy()
        
        # Calculate time difference between consecutive *unique* screenshots
        unique_active_screenshots_df['time_diff'] = unique_active_screenshots_df.groupby('date')['timestamp'].diff().dt.total_seconds() / 3600
        unique_active_screenshots_df['time_diff'].fillna(0, inplace=True) # First unique screenshot of the day has no prior diff
        
        daily_summary = []
        for date, group in unique_active_screenshots_df.groupby('date'):
            # Time for real work: unique screenshots explicitly categorized as real work
            real_work_hours = group[group['is_real_work'] == True]['time_diff'].sum()
            
            # Time for fake/idle work: unique screenshots explicitly categorized as fake work
            # OR time associated with screenshots that were identified as idle popups (but not duplicates)
            # This logic needs refinement: if it's a duplicate, it doesn't represent new time.
            # We filter by 'is_duplicate == False' above, so this 'fake_work_hours' applies to UNIQUE screenshots
            # that were categorized as unproductive content.
            fake_work_hours = group[group['is_real_work'] == False]['time_diff'].sum()

            # Time for unidentified work: unique screenshots whose work type couldn't be determined
            unidentified_hours = group[group['is_real_work'].isnull()]['time_diff'].sum()
            
            daily_summary.append({
                'Date': date,
                'Real_Work_Hours_Screenshot': real_work_hours,
                'Fake_Work_Hours_Screenshot': fake_work_hours,
                'Unidentified_Hours_Screenshot': unidentified_hours
            })
        
        daily_summary_df = pd.DataFrame(daily_summary)

        # Merge with time tracking data
        final_report_df = pd.merge(time_df_melted, daily_summary_df, on='Date', how='left')
        final_report_df.fillna(0, inplace=True) # Fill NaN for days without screenshots or without tracked hours

        # Calculate variances
        final_report_df['Total_Screenshot_Hours'] = final_report_df['Real_Work_Hours_Screenshot'] + final_report_df['Fake_Work_Hours_Screenshot'] + final_report_df['Unidentified_Hours_Screenshot']
        final_report_df['Hours_Variance'] = final_report_df['Hours_Worked'] - final_report_df['Total_Screenshot_Hours']
        
        # Percentage of actual hours worked from total screenshot time
        # Avoid division by zero
        final_report_df['Real_Work_Percentage_of_Total_Screenshot_Time'] = (final_report_df['Real_Work_Hours_Screenshot'] / final_report_df['Total_Screenshot_Hours']) * 100
        final_report_df['Real_Work_Percentage_of_Total_Screenshot_Time'].fillna(0, inplace=True) # If Total_Screenshot_Hours is 0
        
        # Percentage of reported hours that were "fake/idle" based on analysis
        final_report_df['Fraudulent_Percentage_of_Reported_Hours'] = (final_report_df['Fake_Work_Hours_Screenshot'] / final_report_df['Hours_Worked']) * 100
        final_report_df['Fraudulent_Percentage_of_Reported_Hours'].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero
        final_report_df['Fraudulent_Percentage_of_Reported_Hours'].fillna(0, inplace=True)


        # Output final comprehensive report to CSV
        final_report_csv_path = os.path.join(OUTPUT_REPORTS_DIR, "comprehensive_time_analysis_report.csv")
        final_report_df.to_csv(final_report_csv_path, index=False)
        print(f"Comprehensive time analysis report generated: {final_report_csv_path}")

        # --- Generate Graph with Matplotlib ---
        plt.figure(figsize=(12, 7))
        bar_width = 0.25 # Slightly smaller bars for 3 categories
        index = np.arange(len(final_report_df['Date']))

        plt.bar(index - bar_width, final_report_df['Hours_Worked'], bar_width, label='Reported Hours (Time Doctor)', color='skyblue')
        plt.bar(index, final_report_df['Real_Work_Hours_Screenshot'], bar_width, label='Estimated Real Work Hours (Screenshots)', color='lightgreen')
        plt.bar(index + bar_width, final_report_df['Fake_Work_Hours_Screenshot'], bar_width, label='Estimated Unproductive Hours (Screenshots)', color='salmon')

        plt.xlabel('Date')
        plt.ylabel('Hours')
        plt.title('Time Doctor Analysis: Reported vs. Estimated Productive/Unproductive Hours')
        plt.xticks(index, [d.strftime('%b %d') for d in final_report_df['Date']], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        graph_filepath = os.path.join(OUTPUT_REPORTS_DIR, "hours_comparison_graph.png")
        plt.savefig(graph_filepath)
        print(f"Hours comparison graph generated: {graph_filepath}")

        # --- Total Summary ---
        total_reported_hours = final_report_df['Hours_Worked'].sum()
        total_real_work_hours = final_report_df['Real_Work_Hours_Screenshot'].sum()
        total_fake_work_hours = final_report_df['Fake_Work_Hours_Screenshot'].sum()
        total_unidentified_hours = final_report_df['Unidentified_Hours_Screenshot'].sum()
        total_estimated_screenshot_time = total_real_work_hours + total_fake_work_hours + total_unidentified_hours

        # Fraudulent percentage based on *estimated screenshot time*
        fraudulent_percentage_of_screenshot_time = 0
        if total_estimated_screenshot_time > 0:
            fraudulent_percentage_of_screenshot_time = (total_fake_work_hours / total_estimated_screenshot_time) * 100

        # Fraudulent percentage based on *reported hours*
        fraudulent_percentage_of_reported_hours = 0
        if total_reported_hours > 0:
            fraudulent_percentage_of_reported_hours = (total_fake_work_hours / total_reported_hours) * 100


        print("\n--- Overall Summary ---")
        print(f"Total Reported Hours (Time Doctor): {total_reported_hours:.2f} hours")
        print(f"Total Estimated Productive Hours (from unique screenshots): {total_real_work_hours:.2f} hours")
        print(f"Total Estimated Unproductive/Idle Hours (from unique screenshots): {total_fake_work_hours:.2f} hours")
        print(f"Total Estimated Unidentified Hours (from unique screenshots): {total_unidentified_hours:.2f} hours")
        print(f"Total Estimated Screenshot Activity Time: {total_estimated_screenshot_time:.2f} hours")
        print(f"Estimated Percentage of Unproductive Time (out of *screenshot activity*): {fraudulent_percentage_of_screenshot_time:.2f}%")
        print(f"Estimated Percentage of Unproductive Time (out of *reported hours*): {fraudulent_percentage_of_reported_hours:.2f}%")
        print("-----------------------")

    except FileNotFoundError:
        print(f"Error: Time tracking spreadsheet not found at {TIME_SPREADSHEET_PATH}. Skipping time integration and graph generation.")
    except KeyError as e:
        print(f"Error: Missing expected column in time tracking spreadsheet: {e}. Please ensure it has 'Name', 'Total (Decimal)', and date columns.")
    except Exception as e:
        print(f"An unexpected error occurred during time data integration or graph generation: {e}")

    if num_unreadable_images > 0:
        print(f"\n--- Important Note ---")
        print(f"{num_unreadable_images} image(s) could not be fully processed due to corruption, invalid format, or unreadable filenames. Please check the files in '{SCREENSHOTS_DIR}'.")
        print(f"--------------------")
    
    print("\n--- Screenshot Analysis Complete ---")


if __name__ == "__main__":
    analyze_screenshots()
