import os
import json
import argparse
import glob


def split_json(input_file, output_dir, num_splits):
    """
    Split a JSON file containing a list of entries into `num_splits` files,
    each with an even number of entries. Raises an error if entries cannot be evenly divided.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of entries.")
    total = len(data)
    if total % num_splits != 0:
        raise ValueError(f"Cannot split {total} entries into {num_splits} even parts.")
    chunk_size = total // num_splits

    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_splits):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        out_path = os.path.join(output_dir, f"part_{i+1}.json")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split or aggregate JSON files (top-level lists of entries)."
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Split command
    split_parser = subparsers.add_parser('split', help='Split a JSON into multiple even-sized JSON files')
    split_parser.add_argument('--input', '-i', required=True, help='Path to input JSON file')
    split_parser.add_argument('--num', '-n', type=int, required=True,
                              help='Number of equal parts to split into')
    split_parser.add_argument('--output-dir', '-o', required=True,
                              help='Directory to save split JSON files')

    # Aggregate command
    agg_parser = subparsers.add_parser('aggregate', help='Aggregate multiple JSON files into one')
    agg_parser.add_argument('--input-dir', '-i', required=True,
                            help='Directory containing JSON files to aggregate')
    agg_parser.add_argument('--output', '-o', required=True,
                            help='Path to save the aggregated JSON file')

    args = parser.parse_args()
    if args.command == 'split':
        split_json(args.input, args.output_dir, args.num)
    elif args.command == 'aggregate':
        aggregate_json(args.input_dir, args.output)
