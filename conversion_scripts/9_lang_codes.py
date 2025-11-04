#!/usr/bin/env python3
"""
Extract language codes from a Cygnet XML file and create a mapping
to human-readable language names using langcodes.
"""

import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path
import langcodes


def extract_language_codes(xml_file):
    """
    Parse a Cygnet XML file and extract all unique language codes from wordforms.

    Args:
        xml_file: Path to the Cygnet XML file

    Returns:
        set: Unique language codes found in the file
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    language_codes = set()

    # Find the WordformLayer element
    wordform_layer = root.find('LexemeLayer')

    if wordform_layer is not None:
        # Iterate through all wordform elements (WrittenWordform and SpokenWordform)
        for wordform in wordform_layer:
            # Get the language attribute
            lang_code = wordform.get('language')
            if lang_code:
                language_codes.add(lang_code)

    return language_codes


def build_language_name_dict(language_codes):
    """
    Build a dictionary mapping language codes to human-readable names.

    Args:
        language_codes: Iterable of language codes

    Returns:
        dict: Mapping of code -> human-readable name
    """
    lang_dict = {}

    for code in sorted(language_codes):
        try:
            lang = langcodes.Language.get(code)
            lang_dict[code] = lang.display_name()
        except Exception as e:
            # If langcodes can't find the language, store an error message
            print(f"Warning: Could not resolve language code '{code}': {e}")
            lang_dict[code] = f"Unknown ({code})"

    return lang_dict


def main(xml_file, output_file='website_data/lang_codes.json'):
    """
    Main function to extract language codes and save to JSON.

    Args:
        xml_file: Path to input Cygnet XML file
        output_file: Path to output JSON file (default: website_data/lang_codes.json)
    """
    # Extract language codes from XML
    print(f"Reading XML file: {xml_file}")
    language_codes = extract_language_codes(xml_file)
    print(f"Found {len(language_codes)} unique language codes: {sorted(language_codes)}")

    # Build dictionary with human-readable names
    print("Building language name dictionary...")
    lang_dict = build_language_name_dict(language_codes)

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(lang_dict, f, indent=2, ensure_ascii=False)

    print("Done!")
    print(f"\nLanguage mapping:")
    for code, name in lang_dict.items():
        print(f"  {code} -> {name}")


if __name__ == '__main__':
    import sys

    # Default to 'cygnet.xml' if no argument provided
    xml_file = sys.argv[1] if len(sys.argv) > 1 else 'cygnet.xml'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'website_data/lang_codes.json'

    main(xml_file, output_file)