import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT / "data" / "source"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "cleaned"

TEXT_REPLACEMENTS = {
    "\ufeff": "",
    "\r\n": "\n",
    "\r": "\n",
    "\t": " ",
    "\u00a0": " ",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2014": " -- ",
    "\u2013": " -- ",
    "\u2026": "...",
}

PUBLISHER_CATALOG_MARKERS = (
    "WORKS BY HENRY JAMES.",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean raw Gutenberg-style text for tiny language-model training."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Raw input file or folder. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Cleaned output file or folder. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite a single --input file. Not supported for folder input.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Optional text marker where training content should begin.",
    )
    parser.add_argument(
        "--start-occurrence",
        type=int,
        default=1,
        help="Which occurrence of --start-at to use. Default: 1.",
    )
    return parser.parse_args()


def clean_text(
    raw_text: str,
    start_at: str | None = None,
    start_occurrence: int = 1,
) -> str:
    text = normalize_characters(raw_text)
    text = strip_gutenberg_wrapper(text)
    text = remove_transcriber_note_blocks(text)
    text = cut_publisher_catalog_sections(text)
    text = trim_to_start_marker(text, start_at, start_occurrence)
    text = remove_inline_markup(text)
    text = normalize_lines(text)
    return text


def normalize_characters(text: str) -> str:
    for old, new in TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def strip_gutenberg_wrapper(text: str) -> str:
    start_match = re.search(r"^\*\*\* START OF .*?\*\*\*$", text, flags=re.MULTILINE)
    if start_match:
        text = text[start_match.end() :]

    end_match = re.search(r"^\*\*\* END OF .*?\*\*\*$", text, flags=re.MULTILINE)
    if end_match:
        text = text[: end_match.start()]

    return text


def remove_transcriber_note_blocks(text: str) -> str:
    # Notes are usually bracketed metadata, not book prose. Remove the block but
    # keep the story that follows it.
    return re.sub(
        r"\[\s*Transcriber's Notes?:.*?\]",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def cut_publisher_catalog_sections(text: str) -> str:
    cut_points = [
        idx for marker in PUBLISHER_CATALOG_MARKERS if (idx := text.find(marker)) != -1
    ]
    if not cut_points:
        return text
    return text[: min(cut_points)]


def trim_to_start_marker(
    text: str,
    marker: str | None,
    occurrence: int,
) -> str:
    if marker is None:
        return text
    if occurrence <= 0:
        raise ValueError("--start-occurrence must be greater than 0")

    search_from = 0
    marker_idx = -1
    for _ in range(occurrence):
        marker_idx = text.find(marker, search_from)
        if marker_idx == -1:
            raise ValueError(f"could not find start marker: {marker!r}")
        search_from = marker_idx + len(marker)

    return text[marker_idx:]


def remove_inline_markup(text: str) -> str:
    # Gutenberg plain-text files often encode italics as _word_ or *word*.
    # For a character model, those markers become distracting spelling noise.
    text = re.sub(r"_([^_\n]+)_", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    text = re.sub(r"=([^=\n]+)=", r"\1", text)

    # Normalize long dash runs without teaching the model strange repeated forms.
    text = re.sub(r"\s*-{2,}\s*", " -- ", text)
    return text


def normalize_lines(text: str) -> str:
    clean_lines = []
    blank_line_count = 0

    for raw_line in text.splitlines():
        line = re.sub(r" {2,}", " ", raw_line.strip())

        if line:
            clean_lines.append(line)
            blank_line_count = 0
        elif blank_line_count < 1:
            clean_lines.append("")
            blank_line_count += 1

    return "\n".join(clean_lines).strip() + "\n"


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.txt"))
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def resolve_output_path(input_path: Path, output_path: Path, source_root: Path) -> Path:
    if source_root.is_dir():
        return output_path / input_path.relative_to(source_root)
    if output_path.suffix:
        return output_path
    return output_path / input_path.name


def summarize(before: str, after: str, output_path: Path) -> None:
    before_chars = set(before)
    after_chars = set(after)
    removed_chars = sorted(before_chars - after_chars)

    print(f"wrote: {output_path}")
    print(f"characters: {len(before):,} -> {len(after):,}")
    print(f"unique chars: {len(before_chars)} -> {len(after_chars)}")
    if removed_chars:
        preview = "".join(removed_chars[:30])
        print(f"removed character types: {preview!r}")


def main():
    args = parse_args()
    input_path = args.input
    input_files = iter_input_files(input_path)
    if not input_files:
        raise FileNotFoundError(f"no .txt files found in {input_path}")
    if args.in_place and input_path.is_dir():
        raise ValueError("--in-place is only supported when --input is a file")

    for source_path in input_files:
        output_path = (
            source_path
            if args.in_place
            else resolve_output_path(source_path, args.output, input_path)
        )
        raw_text = source_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(
            raw_text,
            start_at=args.start_at,
            start_occurrence=args.start_occurrence,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_text, encoding="utf-8")
        summarize(raw_text, cleaned_text, output_path)


if __name__ == "__main__":
    main()
