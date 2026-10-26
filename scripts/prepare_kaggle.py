"""
Prepare Kaggle GTSRB dataset into:
  data/gtsrb/Train/<0..42>/
  data/gtsrb/Test/<0..42>/
Supports downloading via kagglehub if not present locally.
"""
import os
import glob
import shutil
import zipfile
import csv

def main():
    data_dir = "data"
    target_dir = os.path.join(data_dir, "gtsrb")
    os.makedirs(target_dir, exist_ok=True)

    train_dir = os.path.join(target_dir, "Train")
    test_dir = os.path.join(target_dir, "Test")

    # If already prepared and populated, verify and return
    if os.path.exists(os.path.join(train_dir, "0")) and os.path.exists(os.path.join(test_dir, "0")):
        train_count = sum(len(files) for _, _, files in os.walk(train_dir))
        test_count = sum(len(files) for _, _, files in os.walk(test_dir))
        if train_count > 30000 and test_count > 10000:
            print(f"Dataset already prepared: {train_count} train, {test_count} test images.")
            return True

    # 1. Check for zip files in data/
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    for zf in zip_files:
        print(f"Extracting {zf} into {target_dir}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(target_dir)

    # 2. If not found, download via kagglehub
    if not os.path.exists(os.path.join(target_dir, "Train")):
        print("Checking/downloading GTSRB via kagglehub...")
        try:
            import kagglehub
            hub_path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
            print(f"kagglehub downloaded to: {hub_path}")
            # Copy files from hub_path to target_dir
            for item in os.listdir(hub_path):
                s = os.path.join(hub_path, item)
                d = os.path.join(target_dir, item)
                if not os.path.exists(d):
                    print(f"Copying {item} to {target_dir}...")
                    if os.path.isdir(s):
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
        except Exception as e:
            print(f"kagglehub download error: {e}")

    # Check for nested folders
    if not os.path.exists(os.path.join(target_dir, "Train")):
        candidates = glob.glob(os.path.join(target_dir, "*", "Train"))
        if candidates:
            inner_dir = os.path.dirname(candidates[0])
            print(f"Moving contents from {inner_dir} to {target_dir}...")
            for item in os.listdir(inner_dir):
                s = os.path.join(inner_dir, item)
                d = os.path.join(target_dir, item)
                if not os.path.exists(d):
                    shutil.move(s, d)

    test_csv = os.path.join(target_dir, "Test.csv")
    if not os.path.exists(test_csv):
        if os.path.exists(os.path.join(data_dir, "Test.csv")):
            test_csv = os.path.join(data_dir, "Test.csv")

    if not os.path.exists(test_dir) or not os.path.exists(test_csv):
        print(f"Error: Test dir ({test_dir}) or Test.csv ({test_csv}) not found.")
        return False

    # Check if Test is already sorted into subfolders
    first_sub = os.path.join(test_dir, "0")
    if os.path.isdir(first_sub):
        print("Test set is already sorted into class folders.")
    else:
        print("Sorting flat Test directory into class folders using Test.csv...")
        for c in range(43):
            os.makedirs(os.path.join(test_dir, str(c)), exist_ok=True)

        with open(test_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                rel_path = row["Path"]  # e.g. "Test/00000.png"
                class_id = str(row["ClassId"])
                filename = os.path.basename(rel_path)
                src_path = os.path.join(target_dir, rel_path)
                if not os.path.exists(src_path):
                    src_path = os.path.join(test_dir, filename)
                dst_path = os.path.join(test_dir, class_id, filename)
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
                    count += 1
            print(f"Sorted {count} test images into class folders.")

    train_count = sum(len(files) for _, _, files in os.walk(train_dir))
    test_count = sum(len(files) for _, _, files in os.walk(test_dir))
    print(f"\nGTSRB Dataset Preparation Successful!")
    print(f"Train images: {train_count}")
    print(f"Test images:  {test_count}")
    return True

if __name__ == "__main__":
    main()
