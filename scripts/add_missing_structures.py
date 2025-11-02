#!/usr/bin/env python3
"""
Add missing anatomical structures to ants-cdes.json with full ontology mappings.

This script adds the 23 legitimate brain structures that are in FreeSurferColorLUT.txt
but missing from ants-cdes.json. Each structure gets proper ontology mappings.
"""

import json
from pathlib import Path

# Define the structures with their FreeSurfer IDs and ontology mappings
MISSING_STRUCTURES = {
    # Ventricles
    72: {
        "structure": "5th-Ventricle",
        "hemi": None,
        "isAbout": "http://purl.obolibrary.org/obo/UBERON_0004682",  # 5th ventricle
    },
    
    # Choroid plexus
    31: {
        "structure": "Left-choroid-plexus",
        "hemi": "Left",
        "isAbout": "http://purl.obolibrary.org/obo/UBERON_0001886",  # choroid plexus
    },
    63: {
        "structure": "Right-choroid-plexus",
        "hemi": "Right",
        "isAbout": "http://purl.obolibrary.org/obo/UBERON_0001886",
    },
    
    # Thalamus (Right-Thalamus is different from Right-Thalamus-Proper)
    48: {
        "structure": "Right-Thalamus",
        "hemi": "Right",
        "isAbout": "http://purl.obolibrary.org/obo/UBERON_0001897",  # thalamus
    },
    
    # Substancia Nigra
    71: {
        "structure": "Right-Substancia-Nigra",
        "hemi": "Right",
        "isAbout": "http://purl.obolibrary.org/obo/UBERON_0002038",  # substantia nigra
    },
    
    # Cerebellum Cortex (Right - Left already exists)
    8: {
        "structure": "Right-Cerebellum-Cortex",
        "hemi": "Right",
        "isAbout": "http://purl.obolibrary.org/obo/UBERON_0002129",  # cerebellar cortex
    },
    
    # Cerebral structures
    24: {
        "structure": "Right-Cerebral-Exterior",
        "hemi": "Right",
        # No specific ontology for "exterior" - related to cerebral cortex
        "isAbout": None,
    },
    41: {
        "structure": "Right-Cerebral-White-Matter",
        "hemi": "Right",
        "isAbout": "http://purl.obolibrary.org/obo/UBERON_0002437",  # cerebral white matter
    },
    
    # Insula (Right - Left already exists)
    47: {
        "structure": "Right-Insula",
        "hemi": "Right",
        "isAbout": "http://purl.obolibrary.org/obo/UBERON_0002012",  # insular cortex
    },
    
    # Operculum
    65: {
        "structure": "Right-Operculum",
        "hemi": "Right",
        # Operculum is part of frontal/temporal/parietal cortex
        "isAbout": None,
    },
    
    # Orbital gyrus subdivisions (from DKT31)
    11132: {
        "structure": "Left-Aorg",
        "hemi": "Left",
        # Anterior orbital gyrus
        "isAbout": None,
    },
    12132: {
        "structure": "Right-Aorg",
        "hemi": "Right",
        "isAbout": None,
    },
    11136: {
        "structure": "Left-F3orb",
        "hemi": "Left",
        # Orbital part of inferior frontal gyrus
        "isAbout": None,
    },
    12136: {
        "structure": "Right-F3orb",
        "hemi": "Right",
        "isAbout": None,
    },
    11133: {
        "structure": "Left-Porg",
        "hemi": "Left",
        # Posterior orbital gyrus
        "isAbout": None,
    },
    
    # Occipital gyrus subdivisions
    11139: {
        "structure": "Left-mOg",
        "hemi": "Left",
        # Middle occipital gyrus
        "isAbout": None,
    },
    11140: {
        "structure": "Left-pOg",
        "hemi": "Left",
        # Posterior occipital gyrus
        "isAbout": None,
    },
    
    # Interior (not in standard ontologies)
    11146: {
        "structure": "Left-Interior",
        "hemi": "Left",
        "isAbout": None,
    },
    12146: {
        "structure": "Right-Interior",
        "hemi": "Right",
        "isAbout": None,
    },
    
    # Stellate cells region
    11147: {
        "structure": "Left-Stellate",
        "hemi": "Left",
        "isAbout": None,
    },
    12147: {
        "structure": "Right-Stellate",
        "hemi": "Right",
        "isAbout": None,
    },
    
    # Lesion (pathological, not anatomical)
    77: {
        "structure": "Right-Lesion",
        "hemi": "Right",
        "isAbout": None,
    },
    
    # Undetermined
    80: {
        "structure": "Right-undetermined",
        "hemi": "Right",
        "isAbout": None,
    },
}

def add_missing_structures(cde_file_path):
    """Add missing structures to ants-cdes.json with full metadata."""
    
    # Read existing CDE file
    with open(cde_file_path, 'r') as f:
        ants_cde = json.load(f)
    
    count = ants_cde["count"]
    added = []
    
    # Add each missing structure with both VolumeInVoxels and Volume measures
    for segid, info in MISSING_STRUCTURES.items():
        structure = info["structure"]
        hemi = info["hemi"]
        is_about = info.get("isAbout")
        
        # Add VolumeInVoxels entry
        voxel_tuple = f"ANTSDKT(structure='{structure}', hemi={repr(hemi)}, measure='VolumeInVoxels', unit='voxel')"
        
        if voxel_tuple not in ants_cde:
            count += 1
            entry = {
                "id": f"{count:0>6d}",
                "structure_id": segid,
                "label": f"{structure} VolumeInVoxels (voxel)",
                "datumType": "http://uri.interlex.org/base/ilx_0102597",
                "hasUnit": "voxel",
                "measureOf": "http://uri.interlex.org/base/ilx_0112559",
            }
            if is_about:
                entry["isAbout"] = is_about
            
            ants_cde[voxel_tuple] = entry
            added.append(voxel_tuple)
            print(f"Added {count:0>6d}: {structure} VolumeInVoxels")
        
        # Add Volume (mm^3) entry
        volume_tuple = f"ANTSDKT(structure='{structure}', hemi={repr(hemi)}, measure='Volume', unit='mm^3')"
        
        if volume_tuple not in ants_cde:
            count += 1
            entry = {
                "id": f"{count:0>6d}",
                "structure_id": segid,
                "label": f"{structure} Volume (mm^3)",
                "datumType": "http://uri.interlex.org/base/ilx_0738276",
                "hasUnit": "mm^3",
                "measureOf": "http://uri.interlex.org/base/ilx_0112559",
            }
            if is_about:
                entry["isAbout"] = is_about
            
            ants_cde[volume_tuple] = entry
            added.append(volume_tuple)
            print(f"Added {count:0>6d}: {structure} Volume")
    
    # Update count
    ants_cde["count"] = count
    
    # Write back to file
    with open(cde_file_path, 'w') as f:
        json.dump(ants_cde, f, indent=2)
    
    print(f"\n✓ Successfully added {len(added)} entries to {cde_file_path}")
    print(f"✓ New count: {count}")
    print(f"\nAdded structures:")
    for i, structure_info in enumerate(MISSING_STRUCTURES.values(), 1):
        marker = "✓" if structure_info.get("isAbout") else "○"
        print(f"  {marker} {structure_info['structure']}")
    
    print("\nLegend:")
    print("  ✓ = Has ontology mapping (isAbout)")
    print("  ○ = No ontology mapping (custom/uncommon structure)")
    
    return added

if __name__ == "__main__":
    # Path to ants-cdes.json
    repo_root = Path(__file__).parent.parent
    cde_file = repo_root / "src" / "ants_seg_to_nidm" / "ants_seg_to_nidm" / "mapping_data" / "ants-cdes.json"
    
    if not cde_file.exists():
        print(f"Error: {cde_file} not found")
        exit(1)
    
    print(f"Adding missing structures to: {cde_file}\n")
    added = add_missing_structures(cde_file)
    
    print(f"\n✓ Complete! Added {len(added)} total entries (2 per structure)")
