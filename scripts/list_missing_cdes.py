#!/usr/bin/env python3
"""
Identify ANTs structures/measures that are missing from the CDE registry.

This script mirrors the logic in ``read_ants_stats`` but stops short of
modifying ``ants-cdes.json``.  It reports every key tuple that would
otherwise trigger a ``ValueError`` so we can update the mappings in one pass.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from ants_seg_to_nidm.ants_seg_to_nidm.antsutils import (
        ANTSDKT,
        cde_file,
        create_cde_graph,
        get_details,
        get_id_to_struct,
        map_file,
    )
except ImportError as exc:  # pragma: no cover - guard for missing deps
    raise SystemExit(
        "Unable to import ants_seg_to_nidm utilities. "
        "Run from a repo checkout or install the package first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labelstats",
        required=True,
        type=Path,
        help="Path to antslabelstats.csv",
    )
    parser.add_argument(
        "--brainvols",
        required=True,
        type=Path,
        help="Path to antsbrainvols.csv",
    )
    parser.add_argument(
        "--cde-json",
        default=Path(cde_file),
        type=Path,
        help="Existing ants-cdes.json to compare against",
    )
    parser.add_argument(
        "--map-json",
        default=Path(map_file),
        type=Path,
        help="Path to antsmap.json (required for --update)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Append missing entries to CDE/structure maps and regenerate ants_cde.ttl",
    )
    return parser.parse_args()


def load_existing_cdes(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise SystemExit(f"CDE mapping file not found: {path}")
    with path.open() as f:
        return json.load(f)


def load_structure_map(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise SystemExit(f"Structure map file not found: {path}")
    with path.open() as f:
        return json.load(f)


def scan_brainvols(df: pd.DataFrame) -> List[Dict]:
    missing: List[Dict] = []
    for key, row in df.T.iterrows():
        keytuple = ANTSDKT(
            structure=key if "vol" in key.lower() else "Brain",
            hemi=None,
            measure="Volume" if "vol" in key.lower() else key,
            unit="mm^3" if "vol" in key.lower() else ("mm" if "Thickness" in key else None),
        )
        missing.append({"key": keytuple, "extra": {"source": f"brainvols:{key}"}})
    return missing


def scan_labelstats(df: pd.DataFrame) -> List[Dict]:
    missing: List[Dict] = []
    for _, row in df.iterrows():
        segid = int(row["Label"])
        structure = get_id_to_struct(segid)
        if structure is None:
            continue
        for column, value in row.items():
            if column == "Label" or value == 0:
                continue
            if "VolumeInVoxels" in column or "Area" in column:
                hemi, measure, unit = get_details(column, structure)
                key_tuple = ANTSDKT(structure=structure, hemi=hemi, measure=measure, unit=unit)
                missing.append(
                    {
                        "key": key_tuple,
                        "extra": {
                            "structure_id": segid,
                            "column": column,
                        },
                    }
                )
                if "VolumeInVoxels" in column:
                    key_tuple = ANTSDKT(structure=structure, hemi=hemi, measure="Volume", unit="mm^3")
                    missing.append(
                        {
                            "key": key_tuple,
                            "extra": {
                                "structure_id": segid,
                                "column": column,
                                "derived_from": "VolumeInVoxels",
                            },
                        }
                    )
    return missing


def describe_missing(entries: List[Dict], existing: Dict[str, Dict]) -> List[Dict]:
    seen = {}
    for entry in entries:
        key = entry["key"]
        key_str = str(key)
        if key_str in existing or key_str in seen:
            continue
        info = {
            "key": key_str,
            "structure": key.structure,
            "hemi": key.hemi,
            "measure": key.measure,
            "unit": key.unit,
        }
        info.update(entry.get("extra", {}))
        seen[key_str] = info
    return list(seen.values())


def ensure_structure_mapping(structures: Dict[str, Dict], structure: str) -> bool:
    current = structures.get(structure)
    metadata = {"antskey": [structure], "isAbout": None}
    if current is None:
        structures[structure] = metadata
        return True
    changed = False
    if current.get("antskey") != metadata["antskey"]:
        current["antskey"] = metadata["antskey"]
        changed = True
    if "isAbout" not in current:
        current["isAbout"] = None
        changed = True
    return changed


def append_cde_entry(data: Dict[str, Dict], entry: Dict) -> bool:
    key = entry["key"]
    if key in data:
        return False

    data["count"] += 1
    next_id = f"{data['count']:0>6d}"
    payload = {
        "id": next_id,
        "label": f"{entry['structure']} {entry['measure']} ({entry['unit']})",
        "datumType": (
            "http://uri.interlex.org/base/ilx_0102597"
            if entry["measure"] == "VolumeInVoxels"
            else "http://uri.interlex.org/base/ilx_0738276"
        ),
        "hasUnit": entry["unit"],
        "measureOf": "http://uri.interlex.org/base/ilx_0112559",
    }
    if entry.get("structure_id") is not None:
        payload["structure_id"] = entry["structure_id"]
    data[key] = payload
    return True


def main() -> int:
    args = parse_args()
    cdes = load_existing_cdes(args.cde_json)
    structure_map = load_structure_map(args.map_json) if args.update else None

    brainvols = pd.read_csv(args.brainvols)
    labelstats = pd.read_csv(args.labelstats)

    brainvol_keys = scan_brainvols(brainvols)
    labelstat_keys = scan_labelstats(labelstats)

    missing = describe_missing(brainvol_keys + labelstat_keys, cdes)

    if not missing:
        print("No missing CDE keys detected.")
        return 0

    print(f"Missing entries ({len(missing)} total):")
    for item in missing:
        summary = item["key"]
        meta = []
        if "structure_id" in item:
            meta.append(f"id={item['structure_id']}")
        if item.get("column"):
            meta.append(f"column={item['column']}")
        if item.get("source"):
            meta.append(item["source"])
        if item.get("derived_from"):
            meta.append(f"derived_from={item['derived_from']}")
        suffix = f" ({', '.join(meta)})" if meta else ""
        print(f" - {summary}{suffix}")

    if args.update and missing:
        changed_cde = False
        changed_map = False
        structures = structure_map.setdefault("Structures", {})
        for item in missing:
            structure = item["structure"]
            if structure is None:
                continue
            if ensure_structure_mapping(structures, structure):
                changed_map = True
            if append_cde_entry(cdes, item):
                changed_cde = True

        if changed_map:
            with args.map_json.open("w") as f:
                json.dump(structure_map, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"Updated structure map: {args.map_json}")
        if changed_cde:
            with args.cde_json.open("w") as f:
                json.dump(cdes, f, indent=2)
                f.write("\n")
            ttl_path = args.cde_json.parent / "ants_cde.ttl"
            graph = create_cde_graph()
            graph.serialize(destination=str(ttl_path), format="turtle")
            print(f"Wrote updated CDE JSON and regenerated {ttl_path}")
        else:
            print("No new entries added (existing mappings already present).")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
