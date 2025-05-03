#!/usr/bin/env python3
import json
import regex as re
import argparse
import sys

# 用于匹配嵌入在字符串中的 JSON 对象（支持多层嵌套）
JSON_PATTERN = r'(?s)\{(?:[^{}]|(?R))*\}'

def extract_scores(text):
    """提取 text_quality_score 中的四项分数"""
    m = re.search(JSON_PATTERN, text or "")
    if not m:
        return None
    try:
        scores = json.loads(m.group())
        return {
            "text_rarity": scores.get("Rarity"),
            "text_complexity": scores.get("Complexity"),
            "text_informativeness": scores.get("Informativeness"),
            "text_overall_rating": scores.get("Overall rating")
        }
    except json.JSONDecodeError:
        return None

def extract_task(pred_text):
    """提取 predicted_task 中的 task 和 sub-task"""
    m = re.search(JSON_PATTERN, pred_text or "")
    if not m:
        return None
    try:
        tj = json.loads(m.group())
        return {
            "task": tj.get("task"),
            "sub_task": tj.get("sub-task")
        }
    except json.JSONDecodeError:
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="在原 JSON 对象中添加 text_quality_score 和/或 predicted_task 的解析结果。"
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help="输入 JSON 文件路径（数组格式）"
    )
    parser.add_argument(
        '-o', '--output', required=True,
        help="输出 JSON 文件路径"
    )
    parser.add_argument(
        '-m', '--mode', choices=['scores','tasks','all'], default='all',
        help="要提取的内容："
             "scores 只提取 text_quality_score；"
             "tasks 只提取 predicted_task；"
             "all 同时提取两者（默认）"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 读取原始 JSON 数组
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取输入文件: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. 遍历每个条目，在原 item 上直接添加解析字段
    for item in data:
        if args.mode in ('scores', 'all'):
            scores = extract_scores(item.get("text_quality_score"))
            if scores:
                item.update(scores)
            else:
                # 可选：如果一定要保证存在 scores，则取消下一行注释
                # print(f"⚠️ 无法提取 text_quality_score for id={item.get('id')}", file=sys.stderr)
                pass

        if args.mode in ('tasks', 'all'):
            task_info = extract_task(item.get("predicted_task"))
            if task_info:
                # 添加两个新字段
                item["task"] = task_info["task"]
                item["sub_task"] = task_info["sub_task"]
            else:
                # 可选：如果一定要保证存在 task_info，则取消下一行注释
                # print(f"⚠️ 无法提取 predicted_task for id={item.get('id')}", file=sys.stderr)
                pass

    # 3. 写入新的 JSON 文件
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ 已处理 {len(data)} 条记录，并保存至 {args.output}")
    except Exception as e:
        print(f"❌ 无法写入输出文件: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()