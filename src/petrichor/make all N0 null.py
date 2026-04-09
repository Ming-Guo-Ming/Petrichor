from pathlib import Path
import json
import shutil


# Root input directory
INPUT_DIR = Path("input")

# Whether to create a backup file: sitename.json.bak
MAKE_BACKUP = True


def set_n0_to_null_for_all_sites(input_dir: Path) -> None:
    """
    Set config["N0"] = None for every site JSON following this structure:

        input/
            SITENAME/
                SITENAME.json

    Rules
    -----
    - Only process the JSON file whose name matches the folder name.
    - Write Python None as JSON null.
    - Optionally create a .bak backup before overwriting.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir.resolve()}")

    changed = 0
    skipped = 0
    failed = 0

    # Only check first-level site folders under input/
    for site_dir in sorted(input_dir.iterdir()):
        if not site_dir.is_dir():
            continue

        site_name = site_dir.name
        json_path = site_dir / f"{site_name}.json"

        if not json_path.exists():
            print(f"[SKIP] JSON not found: {json_path}")
            skipped += 1
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                print(f"[SKIP] Not a JSON object: {json_path}")
                skipped += 1
                continue

            config = data.get("config")
            if not isinstance(config, dict):
                print(f"[SKIP] No valid 'config' section: {json_path}")
                skipped += 1
                continue

            old_n0 = config.get("N0", "__MISSING__")

            if MAKE_BACKUP:
                backup_path = json_path.with_suffix(json_path.suffix + ".bak")
                if not backup_path.exists():
                    shutil.copy2(json_path, backup_path)

            config["N0"] = None

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[OK] {json_path} | N0: {old_n0} -> null")
            changed += 1

        except Exception as e:
            print(f"[FAIL] {json_path} | {e}")
            failed += 1

    print("\n[DONE]")
    print(f"Changed: {changed}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    set_n0_to_null_for_all_sites(INPUT_DIR)