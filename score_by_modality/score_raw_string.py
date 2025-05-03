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

def extract_caption_score(text):
    """提取 caption_score 中的 rating 和 explanation"""
    m = re.search(JSON_PATTERN, text or "")
    if not m:
        return None
    try:
        cs = json.loads(m.group())
        return {
            "mm_rating": cs.get("rating"),
        }
    except json.JSONDecodeError:
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="在原 JSON 对象中添加解析结果：text_quality_score、predicted_task 和/或 caption_score。"
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
        '-m', '--mode', choices=['scores','tasks','captions','all'], default='all',
        help=(
            "要提取的内容：\n"
            " scores    — 只提取 text_quality_score；\n"
            " tasks     — 只提取 predicted_task；\n"
            " captions  — 只提取 caption_score；\n"
            " all       — 同时提取三者（默认）"
        )
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

        if args.mode in ('tasks', 'all'):
            task_info = extract_task(item.get("predicted_task"))
            if task_info:
                item["task"] = task_info["task"]
                item["sub_task"] = task_info["sub_task"]

        if args.mode in ('captions', 'all'):
            cap_info = extract_caption_score(item.get("caption_score"))
            if cap_info:
                item.update(cap_info)

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