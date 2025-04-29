import os
import glob
import argparse
import subprocess

def find_latest_submission(results_folder):
    """Finds the latest CSV file in the results folder."""
    list_of_files = glob.glob(os.path.join(results_folder, "*.csv"))  # Get all CSV files
    if not list_of_files:
        raise FileNotFoundError("❌ No CSV submission file found in the results folder!")

    latest_file = max(list_of_files, key=os.path.getctime)  # Get most recent file
    print(f"✅ Found latest submission file: {latest_file}")
    return latest_file

def submit_to_kaggle(competition, submission_file, message="Automated Submission"):
    """Submits the latest CSV to Kaggle competition using the Kaggle API."""
    
    command = [
        "kaggle", "competitions", "submit", 
        "-c", competition, 
        "-f", submission_file, 
        "-m", message
    ]
    
    print("\n🚀 Submitting to Kaggle...")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Submission successful: {result.stdout}")
    else:
        print(f"❌ Submission failed: {result.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-submit the latest Kaggle submission file.")
    parser.add_argument("competition", type=str, help="Kaggle competition name (e.g., 'titanic')")
    parser.add_argument("--results_folder", type=str, default="results", help="Folder containing CSV submissions")
    parser.add_argument("--message", type=str, default="Automated Submission", help="Submission message")
    
    args = parser.parse_args()
    
    try:
        latest_submission = find_latest_submission(args.results_folder)
        submit_to_kaggle(args.competition, latest_submission, args.message)
    except Exception as e:
        print(f"❌ Error: {e}")
