import ast
import io
import sys
import tokenize
from pathlib import Path

def strip_hash_comments_from_source(src: str) -> str:
    tokens = list(tokenize.tokenize(io.BytesIO(src.encode("utf-8")).readline))
    comments_by_line = {}
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            srow, scol = tok.start
            erow, ecol = tok.end
            comments_by_line.setdefault(srow - 1, []).append((scol, ecol))

    lines = src.splitlines(keepends=True)
    new_lines = []
    for idx, line in enumerate(lines):
        if idx in comments_by_line:
            spans = sorted(comments_by_line[idx], reverse=True)
            for scol, ecol in spans:
                prefix = line[:scol]
                suffix = line[ecol:]
                if prefix.strip() == "":
                    line = ""
                else:
                    line = prefix.rstrip() + suffix
        new_lines.append(line)

    cleaned = []
    prev_blank = False
    for l in new_lines:
        if not l.strip():
            if not prev_blank:
                cleaned.append("\n")
                prev_blank = True
        else:
            cleaned.append(l.rstrip() + "\n")
            prev_blank = False

    result = "".join(cleaned).strip() + "\n"
    ast.parse(result)
    return result

def clean_file(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        cleaned = strip_hash_comments_from_source(content)
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        return True
    except Exception as e:
        print(f"Error cleaning {path}: {e}", file=sys.stderr)
        return False

def main():
    root = Path(__file__).resolve().parent.parent
    dirs = ["chemistry_framework", "benchmarks", "tests"]
    total = 0
    cleaned = 0
    for d in dirs:
        target_dir = root / d
        if not target_dir.exists():
            continue
        for p in target_dir.rglob("*.py"):
            if p.name == "strip_all_comments.py":
                continue
            total += 1
            if clean_file(p):
                cleaned += 1
    print(f"Cleaned {cleaned}/{total} Python files.")

if __name__ == "__main__":
    main()
