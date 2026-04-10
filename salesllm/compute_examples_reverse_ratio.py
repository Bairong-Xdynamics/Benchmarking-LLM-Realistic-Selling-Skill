import argparse
import json
import os
import glob
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute user reply count and examples error ratio from JSONL results")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing JSONL files (e.g., /Users/admin/SaleLLM/data/eval_data/eval_reverse_test_data/gpt4o_as_users)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save JSON summary (e.g., /path/to/stats_examples_error_ratio.json)",
    )
    return parser.parse_args()


def process_file(file_path: str) -> Dict[str, Any]:
    total_user_replies = 0
    total_examples_errors = 0
    conversations = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {line_num} in {file_path}: {e}")
                    continue

                conversations += 1
                messages = obj.get("messages", [])
                total_user_replies += sum(1 for m in messages if m.get("role") == "user")

                ra = obj.get("reversion_analysis", {})
                examples = ra.get("examples", [])
                if isinstance(examples, list):
                    total_examples_errors += len(examples)

        return {
            "file": file_path,
            "conversations": conversations,
            "user_replies": total_user_replies,
            "examples_errors": total_examples_errors,
            "error_ratio": (total_examples_errors / total_user_replies) if total_user_replies > 0 else 0.0,
        }
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {
            "file": file_path,
            "conversations": conversations,
            "user_replies": total_user_replies,
            "examples_errors": total_examples_errors,
            "error_ratio": 0.0,
            "error": str(e),
        }


def main():
    args = parse_args()
    input_dir = args.input_dir

    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    if not files:
        logger.error(f"No JSONL files found in {input_dir}")
        return

    overall_user_replies = 0
    overall_examples_errors = 0
    overall_conversations = 0
    per_file_stats = []

    for fp in files:
        stats = process_file(fp)
        per_file_stats.append(stats)
        overall_user_replies += stats.get("user_replies", 0)
        overall_examples_errors += stats.get("examples_errors", 0)
        overall_conversations += stats.get("conversations", 0)

    overall_error_ratio = (overall_examples_errors / overall_user_replies) if overall_user_replies > 0 else 0.0

    summary = {
        "input_dir": input_dir,
        "files_processed": len(per_file_stats),
        "overall": {
            "conversations": overall_conversations,
            "user_replies": overall_user_replies,
            "examples_errors": overall_examples_errors,
            "error_ratio": overall_error_ratio,
        },
        "per_file": per_file_stats,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_path:
        try:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Summary saved to {args.output_path}")
        except Exception as e:
            logger.error(f"Failed to save summary to {args.output_path}: {e}")


if __name__ == "__main__":
    main()

