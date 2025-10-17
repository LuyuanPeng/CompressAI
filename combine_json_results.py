import os
import json
import csv
import re

# Directory containing the 9 JSON files
results_dir = 'results/eval_finetuned/NVSPrior-w'
output_csv = os.path.join(results_dir, 'results_summary.csv')

# Regex to extract model, metric, and quality from filename (e.g., mbt2018-mean_mse_q1.json)
model_pattern = re.compile(r'^(mbt2018-mean|cheng2020-anchor)_(mse)_q([0-9]+)\.json')

rows = []

for fname in os.listdir(results_dir):
    if fname.endswith('.json'):
        fpath = os.path.join(results_dir, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
        # Extract model, metric, and quality from filename
        model_match = model_pattern.match(fname)
        if model_match:
            model = model_match.group(1)
            metric = model_match.group(2)
            quality = model_match.group(3)
        else:
            model = fname
            metric = ''
            quality = ''
        # Extract required fields from nested 'results' dict as lists
        results = data.get('results', {})
        psnr = results.get('psnr-rgb', [''])[0]
        ms_ssim = results.get('ms-ssim-rgb', [''])[0]
        bpp = results.get('bpp', [''])[0]
        encode_time = results.get('encoding_time', [0])[0]
        decode_time = results.get('decoding_time', [0])[0]
        total_time = encode_time + decode_time
        rows.append([model, metric, quality, psnr, ms_ssim, bpp, total_time])


# Write to CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['model', 'metric', 'quality', 'psnr', 'ms-ssim', 'bpp', 'total_time'])
    writer.writerows(rows)

print(f"Combined results written to {output_csv}")
