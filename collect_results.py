import os
import json
import pandas as pd
from collections import defaultdict

def get_first_non_empty_line(file_path):
    """Return the first non-empty line from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                return stripped
    return None

def collect_and_group_results(root="exp_logs", out_dir="collected_results"):
    """Collect experiment results and group them by (dataset, corruption_modality)."""
    grouped_data = defaultdict(list)

    for d in sorted(os.listdir(root)):
        exp_dir = os.path.join(root, d)
        if not os.path.isdir(exp_dir):
            continue

        # Load args.json
        args_path = os.path.join(exp_dir, "args.json")
        if not os.path.exists(args_path):
            print(f"Skipping {exp_dir}: args.json not found")
            continue
        try:
            with open(args_path, "r", encoding="utf-8") as f:
                args = json.load(f)
            dataset = args.get("dataset", "unknown")
            corruption_modality = args.get("corruption_modality", "unknown")
        except Exception as e:
            print(f"Skipping {exp_dir}: failed to parse args.json: {e}")
            continue

        # Load result.csv
        result_path = os.path.join(exp_dir, "result.csv")
        if not os.path.exists(result_path):
            print(f"Skipping {exp_dir}: result.csv not found")
            continue
        if os.path.getsize(result_path) == 0:
            print(f"⚠️ Empty file: {result_path}")
            continue
        df_result = pd.read_csv(result_path, header=None)#, names=["corruption", "accuracy"])
        if df_result.empty:
            print(f"Skipping {exp_dir}: result.csv is empty")
            continue
        df_result = df_result.set_index(0).T
        
        # Load remark.md
        remark_path = os.path.join(exp_dir, "remark.md")
        remark = get_first_non_empty_line(remark_path) if os.path.exists(remark_path) else ""

        # Add metadata to result table
        df_result.insert(0, "exp_name", d)
        # df_result["exp_dir"] = d
        df_result["remark"] = remark
        
        print(df_result)

        # Store by group key
        key = (dataset, corruption_modality)
        grouped_data[key].append(df_result)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Write each group to its own CSV
    for (dataset, modality), df_list in grouped_data.items():
        group_df = pd.concat(df_list, ignore_index=True)
        file_name = f"results_{dataset}_{modality}.csv"
        out_path = os.path.join(out_dir, file_name)
        group_df.to_csv(out_path, index=False)
        print(f"✅ Saved {out_path} with {len(group_df)} rows")

    return grouped_data

# Run it
grouped = collect_and_group_results()
