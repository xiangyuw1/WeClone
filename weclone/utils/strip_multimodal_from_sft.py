import argparse
import json
import re
from pathlib import Path

IMAGE_TOKENS = ("<image>", "<|image_pad|>", "<|imgpad|>")


def sanitize_content(content: str) -> str:
    text = content
    for token in IMAGE_TOKENS:
        text = text.replace(token, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip residual multimodal tokens/fields from WeClone SFT dataset for text-only training."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/res_csv/sft/sft-my.json",
        help="Path to input sft-my.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to output json. If empty, overwrite input file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level list in SFT JSON file.")

    placeholder_count = 0
    sample_count = 0

    for sample in data:
        sample_count += 1
        sample.pop("images", None)
        sample.pop("videos", None)
        sample.pop("audios", None)

        messages = sample.get("messages", [])
        if not isinstance(messages, list):
            continue

        for msg in messages:
            content = str(msg.get("content", ""))
            placeholder_count += sum(content.count(token) for token in IMAGE_TOKENS)
            msg["content"] = sanitize_content(content)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(
        f"Done. samples={sample_count}, placeholders_removed={placeholder_count}, output={output_path}"
    )


if __name__ == "__main__":
    main()
