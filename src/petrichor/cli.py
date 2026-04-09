from pathlib import Path
import argparse

from petrichor.main import run_one_site, _discover_site_jsons


def _find_site_json(input_root: Path, site_name: str) -> Path:
    site_name = site_name.strip().lower()
    matches = []

    for p in input_root.rglob("*.json"):
        if p.stem.lower() == site_name or p.parent.name.lower() == site_name:
            matches.append(p.resolve())

    matches = sorted(set(matches))

    if not matches:
        raise FileNotFoundError(f"No site JSON found for site name: {site_name}")

    if len(matches) > 1:
        raise ValueError(
            "Multiple matching site JSON files found: "
            + ", ".join(str(p) for p in matches)
        )

    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Petrichor runner")
    parser.add_argument(
        "site",
        nargs="?",
        help="Optional site name, e.g. AAC001. If omitted, run all sites."
    )
    parser.add_argument(
        "--input-root",
        default="input",
        help="Root folder containing site folders and JSON files."
    )

    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = (project_root / args.input_root).resolve()

    if args.site:
        site_json = _find_site_json(input_root, args.site)
        run_one_site(project_root, site_json)
        return

    site_jsons = _discover_site_jsons(input_root)
    for site_json in site_jsons:
        run_one_site(project_root, site_json)