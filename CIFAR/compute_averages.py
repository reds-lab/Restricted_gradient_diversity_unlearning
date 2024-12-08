import os
import argparse
import numpy as np

def compute_averages(input_dir):
    ra_values = []
    ua_values = []
    class_fid_values = []
    all_but_class_fid_values = []

    for class_id in range(10):
        class_dir = os.path.join(input_dir, f'CLASS_{class_id}')
        if not os.path.exists(class_dir):
            print(f"Warning: Directory for CLASS_{class_id} not found.")
            continue

        try:
            with open(os.path.join(class_dir, f'ra_clip_{class_id}.txt'), 'r') as f:
                ra_values.append(float(f.read().strip()))
        except FileNotFoundError:
            print(f"Warning: ra_{class_id}.txt not found for CLASS_{class_id}")

        try:
            with open(os.path.join(class_dir, f'ua_clip_{class_id}.txt'), 'r') as f:
                ua_values.append(float(f.read().strip()))
        except FileNotFoundError:
            print(f"Warning: ua_{class_id}.txt not found for CLASS_{class_id}")

        try:
            with open(os.path.join(class_dir, f'class_fid_{class_id}.txt'), 'r') as f:
                class_fid_values.append(float(f.read().strip()))
        except FileNotFoundError:
            print(f"Warning: class_fid_{class_id}.txt not found for CLASS_{class_id}")

        try:
            with open(os.path.join(class_dir, f'all_but_class_fid_{class_id}.txt'), 'r') as f:
                all_but_class_fid_values.append(float(f.read().strip()))
        except FileNotFoundError:
            print(f"Warning: all_but_class_fid_{class_id}.txt not found for CLASS_{class_id}")

    return {
        'ra': np.mean(ra_values) if ra_values else None,
        'ua': np.mean(ua_values) if ua_values else None,
        'class_fid': np.mean(class_fid_values) if class_fid_values else None,
        'all_but_class_fid': np.mean(all_but_class_fid_values) if all_but_class_fid_values else None
    }

def main():
    parser = argparse.ArgumentParser(description='Compute average metrics across all classes.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing class subdirectories')
    parser.add_argument('--output_file', type=str, required=True, help='Output file to write average results')
    
    args = parser.parse_args()
    
    averages = compute_averages(args.input_dir)
    
    with open(args.output_file, 'w') as f:
        for metric, value in averages.items():
            if value is not None:
                f.write(f'{metric}: {value}\n')
            else:
                f.write(f'{metric}: No data available\n')

if __name__ == '__main__':
    main()