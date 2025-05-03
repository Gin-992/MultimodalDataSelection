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
            "rarity": scores.get("Rarity"),
            "complexity": scores.get("Complexity"),
            "informativeness": scores.get("Informativeness"),
            "overall_rating": scores.get("Overall rating")
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
        description="从 JSON 中批量提取 text_quality_score 和/或 predicted_task。"
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

    # 读取整个 JSON 数组
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取输入文件: {e}", file=sys.stderr)
        sys.exit(1)

    output = []
    for item in data:
        rec = {"id": item.get("id")}

        if args.mode in ('scores','all'):
            scores = extract_scores(item.get("text_quality_score"))
            if scores:
                rec.update(scores)
            else:
                # 如果严格要求 presence，可在这里跳过：continue
                pass

        if args.mode in ('tasks','all'):
            task_info = extract_task(item.get("predicted_task"))
            if task_info:
                rec["task"] = task_info["task"]
                rec["sub_task"] = task_info["sub_task"]
            else:
                # 同上，可根据需要跳过
                pass

        output.append(rec)

    # 写入输出
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f"✅ 成功处理 {len(output)} 条记录，结果保存至 {args.output}")
    except Exception as e:
        print(f"❌ 无法写入输出文件: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()