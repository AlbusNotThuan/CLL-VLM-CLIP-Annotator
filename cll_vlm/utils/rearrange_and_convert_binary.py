import os
import shutil
import subprocess
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config_path = "/tmp2/maitanha/vgu/cll_vlm/cll_vlm/config/config.yaml"
    config = load_config(config_path)
    
    binary_logs_dir = config["output"]["binary"]
    data_paths = config["data"]["paths"]
    datasets = ["cifar10", "cifar20", "cifar100", "tiny200"]
    
    # script path
    convert_script = "/tmp2/maitanha/vgu/cll_vlm/cll_vlm/utils/convert_results.py"

    print(f"--- Rearranging JSON files in {binary_logs_dir} ---")
    # 1. Rearrange files in binary_logs_dir
    # Note: We skip directories like 'csv', 'stage1', etc.
    for item in os.listdir(binary_logs_dir):
        item_path = os.path.join(binary_logs_dir, item)
        if os.path.isfile(item_path) and item.endswith(".json"):
            # Identify dataset
            dataset_found = None
            # Check for exact dataset name in the middle of filename
            # Order matters: check longer names first if there are overlaps, though here there aren't many
            for ds in datasets:
                if f"_{ds}_" in item:
                    dataset_found = ds
                    break
            
            if dataset_found:
                target_dir = os.path.join(binary_logs_dir, dataset_found)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, item)
                
                # Check if target already exists (don't overwrite unless necessary, but moving is fine)
                if os.path.exists(target_path):
                     print(f"File {item} already exists in {dataset_found}/, skipping move.")
                else:
                    shutil.move(item_path, target_path)
                    print(f"Moved {item} to {dataset_found}/")
            else:
                print(f"Skipping {item} (could not identify dataset from filename pattern '_dataset_')")

    print(f"\n--- Converting JSON files to CSV ---")
    # 2. Iterate through dataset folders and convert
    for ds in datasets:
        ds_dir = os.path.join(binary_logs_dir, ds)
        if not os.path.exists(ds_dir):
            continue
            
        csv_out_dir = os.path.join(binary_logs_dir, "csv", ds)
        os.makedirs(csv_out_dir, exist_ok=True)
        
        data_root = data_paths.get(ds)
        if not data_root:
            print(f"Warning: No data path found for {ds} in config. Skipping.")
            continue
        
        json_files = [f for f in os.listdir(ds_dir) if f.endswith(".json")]
        if not json_files:
            continue
            
        print(f"\nProcessing dataset: {ds}")
        for item in json_files:
            json_path = os.path.join(ds_dir, item)
            csv_filename = item.replace(".json", ".csv")
            csv_path = os.path.join(csv_out_dir, csv_filename)
            
            print(f"  -> Converting {item}...")
            cmd = [
                "python", convert_script,
                "--input_json", json_path,
                "--output_csv", csv_path,
                "--data_name", ds,
                "--data_root", data_root
            ]
            
            try:
                # Use subprocess to call the existing script to maintain consistency 
                # as per user request to "dùng file convert_results.py"
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"    ERROR converting {item}:")
                    print(result.stderr)
                else:
                    # print(result.stdout) # optional verbose
                    pass
            except Exception as e:
                print(f"    Failed to run command for {item}: {e}")

    print("\n[Done] Rearrangement and conversion complete.")

if __name__ == "__main__":
    main()
