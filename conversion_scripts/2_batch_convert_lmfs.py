#!/usr/bin/env python3
"""
Batch convert GlobalWordNet XML files to Cygnet XML format.
"""

from pathlib import Path

from cyg.converters import WordNetToCygnetConverter


def should_skip_file(filepath):
    """
    Check if a file should be skipped based on its name.

    Args:
        filepath: Path object for the file

    Returns:
        bool: True if file should be skipped
    """
    stem = filepath.stem
    # Remove .xml from .xml.gz files
    if stem.endswith('.xml'):
        stem = stem[:-4]

    skip_names = ['omw-en', 'omw-ja', 'omw-id', 'omw-da', 'omw-zsm']
    return stem in skip_names


def collect_xml_files(raw_wns_dir):
    """
    Collect all XML files to process from raw_wns directory.

    Args:
        raw_wns_dir: Path to raw_wns directory

    Returns:
        list: List of Path objects for XML files to process
    """
    raw_wns_path = Path(raw_wns_dir)
    xml_files = []

    # Process top-level XML files
    for filepath in raw_wns_path.glob('*.xml'):
        if not should_skip_file(filepath):
            xml_files.append(filepath)

    for filepath in raw_wns_path.glob('*.xml.gz'):
        if not should_skip_file(filepath):
            xml_files.append(filepath)

    for filepath in raw_wns_path.glob('*.xml.xz'):
        if not should_skip_file(filepath):
            xml_files.append(filepath)

    # Process omw-1.4/*/*.xml files
    omw_pattern = raw_wns_path / 'omw-1.4' / '*' / '*.xml'
    for filepath in raw_wns_path.glob('omw-1.4/*/*.xml'):
        if not should_skip_file(filepath):
            xml_files.append(filepath)

    for filepath in raw_wns_path.glob('omw-1.4/*/*.xml.gz'):
        if not should_skip_file(filepath):
            xml_files.append(filepath)

    return sorted(xml_files)


def batch_convert(cili_file, raw_wns_dir='raw_wns', output_dir='cygnets_presynth'):
    """
    Batch convert all GWN XML files to Cygnet format.

    Args:
        raw_wns_dir: Directory containing raw WordNet XML files
        output_dir: Directory to save converted Cygnet XML files
        skip_examples: Whether to skip processing examples
        xsd_file: Optional XSD file for validation
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Collect files to process
    print(f"Scanning {raw_wns_dir} for XML files...")
    xml_files = collect_xml_files(raw_wns_dir)

    if not xml_files:
        print(f"No XML files found in {raw_wns_dir}")
        return

    print(f"Found {len(xml_files)} files to process")
    print()
    # Process each file
    success_count = 0
    error_count = 0
    skipped_count = 0

    # Reorder - find oewn and move it to the start
    oewn = next((t for t in xml_files if 'english-wordnet' in str(t)), None)
    if oewn:
        xml_files.remove(oewn)
        xml_files.insert(0, oewn)

    for i, xml_file in enumerate(xml_files, 1):
        print(f"[{i}/{len(xml_files)}] Processing {xml_file.name}...")

        try:

            if i == 1:
                # English !
                converter = WordNetToCygnetConverter(cili_path=cili_file, skip_cili_defns=True)
            else:
                converter = WordNetToCygnetConverter(cili_path=cili_file, relations_path=str(output_path/"oewn-2024.xml"),
                                                     skip_cili_defns=False)

            # Read metadata first (without converting)
            root, tree = converter.read_metadata(str(xml_file))

            # Get metadata
            lexicon_id = converter.lexicon_id
            lexicon_version = converter.lexicon_version

            if i == 1:
                assert lexicon_id == 'oewn'
            else:
                assert lexicon_id != 'oewn'

            if lexicon_id is None or lexicon_version is None:
                print(f"  ✗ Skipping {xml_file.name} (could not read metadata)")
                error_count += 1
                continue

            # Construct proper output filename: id-version.xml
            output_filename = f"{lexicon_id}-{lexicon_version}.xml"
            output_file = output_path / output_filename

            # Check if output already exists
            if output_file.exists():
                print(f"  ⊘ Skipping {xml_file.name} (output {output_filename} already exists)")
                skipped_count += 1
                continue

            # Now convert and save (using the already-parsed tree)
            print(f"  Converting {xml_file.name}...")
            converter.convert_from_tree(root)
            converter.save(str(output_file))
            print(f"  ✓ Converted to {output_filename}")

        except Exception as e:
            print(f"  ✗ Error converting {xml_file.name}: {e}")
            error_count += 1

        print()

    # Print summary
    print("=" * 70)
    print("Batch conversion complete!")
    print(f"  Successful: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(xml_files)}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch convert GlobalWordNet XML files to Cygnet XML format.'
    )
    parser.add_argument(
        '--raw-wns-dir',
        default='bin/raw_wns',
        help='Directory containing raw WordNet XML files (default: raw_wns)'
    )
    parser.add_argument(
        '--output-dir',
        default='bin/cygnets_presynth',
        help='Directory to save converted files (default: cygnets_presynth)'
    )
    parser.add_argument(
        '--cili-file',
        default='bin/cygnets_presynth/cili-1.0.xml',
    )

    args = parser.parse_args()

    batch_convert(
        raw_wns_dir=args.raw_wns_dir,
        output_dir=args.output_dir,
        cili_file=args.cili_file
    )


if __name__ == '__main__':
    main()