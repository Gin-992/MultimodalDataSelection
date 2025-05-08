import os
import json
import argparse
import glob
import sys

def split_json(input_file, output_dir, num_splits):
    """
    Split a JSON file containing a list of entries into `num_splits` files.
    If entries cannot be evenly divided, extra entries go to the last file.
    File indices start from 0.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of entries.")
    total = len(data)
    chunk_size = total // num_splits
    if chunk_size == 0:
        raise ValueError("Number of splits {} is greater than total entries {}."
                         .format(num_splits, total))

    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_splits):
        start = i * chunk_size
        chunk = data[start:] if i == num_splits - 1 else data[start:start+chunk_size]
        out_path = os.path.join(output_dir, f"part_{i}.json")
        with open(out_path, 'w', encoding='utf-8') as out_f:
            json.dump(chunk, out_f, indent=4, ensure_ascii=False)
        print(f"Written {len(chunk)} entries to {out_path}")

def aggregate_json(input_dir, output_file):
    """
    Aggregate multiple JSON files (each containing a list) in `input_dir`
    into a single JSON file containing the concatenated list.
    """
    all_data = []
    files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")
    for file in files:
        if not file.endswith('.json'):
            continue
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"File {file} does not contain a JSON list.")
        all_data.extend(data)
        print(f"Loaded {len(data)} entries from {file}")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(all_data, out_f, indent=4, ensure_ascii=False)
    print(f"Aggregated total of {len(all_data)} entries into {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Split or aggregate JSON files (top-level lists of entries)."
    )
    subparsers = parser.add_subparsers(title="commands")

    # SPLIT
    sp = subparsers.add_parser('split', help='Split a JSON into multiple parts')
    sp.add_argument('-i', '--input-file',  required=True,
                    help='Path to input JSON file')
    sp.add_argument('-n', '--num-splits', type=int, required=True,
                    help='Number of parts to split into')
    sp.add_argument('-o', '--output-dir', required=True,
                    help='Directory to save split JSON files')
    sp.set_defaults(func=lambda args: split_json(
        args.input_file, args.output_dir, args.num_splits))

    # AGGREGATE
    ap = subparsers.add_parser('aggregate', help='Combine many JSON lists into one')
    ap.add_argument('-d', '--input-dir',  required=True,
                    help='Directory containing JSON files to aggregate')
    ap.add_argument('-f', '--output-file', required=True,
                    help='Path to save the aggregated JSON file')
    ap.set_defaults(func=lambda args: aggregate_json(
        args.input_dir, args.output_file))

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == '__main__':
    main()
