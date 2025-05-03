import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

def load_json_list(path: Path) -> List[Dict[str, Any]]:
    """
    从给定路径加载一个 JSON 数组文件，返回 Python 列表。
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} 中不是一个 JSON 数组")
            return data
    except Exception as e:
        print(f"❌ 读取或解析 {path} 失败: {e}", file=sys.stderr)
        sys.exit(1)

def merge_records(lists_of_records: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    按 id 合并多个记录列表。相同 id 的记录会被合并到同一个 dict，
    重复字段（键）只保留首次出现的值。
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for records in lists_of_records:
        for rec in records:
            rec_id = rec.get("id")
            if rec_id is None:
                # 跳过没有 id 的条目
                continue
            if rec_id not in merged:
                # 第一次见到该 id，拷贝整个 dict
                merged[rec_id] = rec.copy()
            else:
                # 已经存在相同 id，逐字段合并
                existing = merged[rec_id]
                for k, v in rec.items():
                    if k not in existing:
                        existing[k] = v
                    # 如果 k 已存在，跳过，保留原来值
    # 返回合并后的记录列表
    return list(merged.values())

def parse_args():
    parser = argparse.ArgumentParser(
        description="按 id 合并多个 JSON 数组文件，相同字段只保留首次出现的值。"
    )
    parser.add_argument(
        '-i', '--inputs', required=True, nargs='+',
        help="要合并的 JSON 文件列表（每个文件应为一个 JSON 数组）"
    )
    parser.add_argument(
        '-o', '--output', required=True,
        help="合并后输出的 JSON 文件路径"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 加载所有输入文件
    lists_of_records = []
    for in_path in args.inputs:
        path = Path(in_path)
        if not path.is_file():
            print(f"❌ 找不到文件: {in_path}", file=sys.stderr)
            sys.exit(1)
        lists_of_records.append(load_json_list(path))

    # 2. 合并
    merged = merge_records(lists_of_records)

    # 3. 写入输出文件
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=4)
        print(f"✅ 合并完成，共 {len(merged)} 条唯一 id，结果已保存至 {args.output}")
    except Exception as e:
        print(f"❌ 写入输出文件失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()