from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "project_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def summarize_pdf(path: Path) -> dict:
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        first = r.pages[0].extract_text() if len(r.pages) else ""
        return {"pages": len(r.pages), "first_page_preview": first[:600]}
    except Exception as e:
        return {"pages": None, "error": str(e)}


def walk_files(base: Path) -> list[dict]:
    out = []
    for p in base.rglob("*"):
        if p.is_dir():
            continue
        if "__pycache__" in p.parts:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        out.append({
            "relpath": str(p.relative_to(base)),
            "suffix": p.suffix.lower(),
            "size_bytes": size,
            "sha256_16": file_digest(p) if size < 50 * 1024 * 1024 else "",
        })
    return out


def parse_python(path: Path) -> dict:
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"error": str(e)}
    classes, functions, imports = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append({"name": node.name, "lineno": node.lineno, "methods": methods})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(node, "col_offset", 0) == 0:
            functions.append({"name": node.name, "lineno": node.lineno})
        elif isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for n in node.names:
                imports.append(f"{mod}.{n.name}" if mod else n.name)
    return {"classes": classes, "functions": functions, "imports": sorted(set(imports))}


def classify(rel: str, code_info: dict | None) -> list[str]:
    tags = []
    name = rel.lower()
    if name.endswith(".pdf"):
        tags.append("documentation")
    if "course" in name:
        tags.append("dynamical-timetabling")
    if "autoassoc" in name or "hopfield" in name:
        tags.append("autoassociation")
    if "time_table_app" in name:
        tags.append("ui-entry-point")
    if name.endswith(".stu"):
        tags.append("data-students")
    if name.endswith(".crs"):
        tags.append("data-courses")
    if code_info:
        joined = json.dumps(code_info).lower()
        if "tkinter" in joined or "tk.tk" in joined:
            tags.append("ui")
        if "clash" in joined:
            tags.append("clash-counting")
        if "matplotlib" in joined or "plt." in joined:
            tags.append("plotting")
        if "readclashes" in joined or "open(" in joined:
            tags.append("data-loading")
    return sorted(set(tags))


def main() -> None:
    inventory = walk_files(ROOT)
    df = pd.DataFrame(inventory)
    df.to_csv(OUT_DIR / "file_inventory.csv", index=False)

    docs_dir = ROOT / "docs"
    docs_summary = {}
    if docs_dir.exists():
        for pdf in sorted(docs_dir.glob("*.pdf")):
            docs_summary[pdf.name] = summarize_pdf(pdf)
    (OUT_DIR / "docs_summary.json").write_text(json.dumps(docs_summary, indent=2))

    code_structure = {}
    for py in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        if py.is_relative_to(ROOT / "scripts"):
            continue
        rel = str(py.relative_to(ROOT))
        info = parse_python(py)
        info["tags"] = classify(rel, info)
        code_structure[rel] = info
    (OUT_DIR / "code_structure.json").write_text(json.dumps(code_structure, indent=2))

    print(f"Inventoried {len(inventory)} files")
    print(f"Found {len(docs_summary)} PDFs in docs/")
    print(f"Parsed {len(code_structure)} Python source files")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
