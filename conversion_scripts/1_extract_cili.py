#!/usr/bin/env python3
"""
Convert CILI TSV data to Cygnet XML format.
"""

import csv
from lxml import etree as ET
import re
import sys

NEW_POS_LABELS = {
    'n': 'NOUN',
    'v': 'VERB',
    'a': 'ADJ',
    'r': 'ADV',
    's': 'ADJ',  # adjective satellite -> adjective
    'c': 'CONJ',
    'p': 'ADP',
    'x': 'NREF',
    'u': 'UNK'
}

def normalize_whitespace(text):
    """Normalize whitespace in text."""
    if not text:
        return text
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()


def process_definition(definition):
    """Process definition text: strip tags, normalize whitespace, escape XML."""
    if not definition:
        return definition
    # Normalize whitespace
    text = normalize_whitespace(definition)
    # Note: We don't escape here because ET will handle it when creating text nodes
    return text


def get_ontological_category(origin):
    """Extract ontological category from origin field (last character)."""
    if not origin or not origin.strip():
        assert False
    category = origin.strip()[-1].lower()
    assert category in {'n', 'v', 'a', 's', 'r', 'p', 'u', 'c', 'x'}
    category = NEW_POS_LABELS[category]
    return category


def get_from(origin):

    wn_code, original_id = origin.split(':')
    wn_name, version = wn_code.split('-')
    assert wn_name == 'pwn'
    assert version == "3.0"
    return wn_name, version, original_id


def load_tsv_data(tsv_file):
    """Load CILI TSV data."""
    print(f"Loading TSV data from {tsv_file}...")

    rows = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from TSV")
    return rows


def validate_data(tsv_rows):
    """Validate TSV data."""
    print("Validating data...")

    errors = []

    for i, row in enumerate(tsv_rows, 1):
        ili_id = row['ili_id']
        definition = row['definition']

        # Check for empty definitions
        if not definition or not definition.strip():
            errors.append(f"Row {i}: ILI {ili_id} has empty definition")

    if errors:
        print(f"ERROR: Found {len(errors)} validation errors:")
        for error in errors[:20]:  # Show first 20
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)

    print("Validation passed!")


def create_cygnet_xml(tsv_rows, output_file):
    """Create Cygnet XML from TSV data."""
    print("Creating Cygnet XML...")

    # Create root element
    root = ET.Element('CygnetResource')
    root.set('id', 'cili')
    root.set('label', 'Collaborative Interlingua Index')
    root.set('version', '1.0')

    # Create ConceptLayer
    concept_layer = ET.SubElement(root, 'ConceptLayer')

    for row in tsv_rows:
        ili_id = row['ili_id']
        status = row['status']
        superseded_by = row['superseded_by']
        origin = row['origin']

        # Get ontological category from last character of origin
        ontological_category = get_ontological_category(origin)

        concept = ET.SubElement(concept_layer, 'Concept')
        concept.set('id', f'cili.{ili_id}')
        concept.set('ontological_category', ontological_category)
        concept.set('status', status)

        wn_name, version, original_id = get_from(origin)

        from_ele = ET.SubElement(concept, 'Provenance')
        from_ele.set('resource', wn_name)
        from_ele.set('version', version)
        from_ele.set('original_id', original_id)

        # Only include superseded_by if not empty
        if superseded_by and superseded_by.strip():
            concept.set('superseded_by', superseded_by)

    # Create GlossLayer
    gloss_layer = ET.SubElement(root, 'GlossLayer')

    for row in tsv_rows:
        ili_id = row['ili_id']
        definition = row['definition']
        origin = row['origin']

        # Process definition
        processed_def = process_definition(definition)

        gloss = ET.SubElement(gloss_layer, 'Gloss')
        gloss.set('definiendum', f'cili.{ili_id}')
        gloss.set('language', 'en')

        wn_name, version, original_id = get_from(origin)

        # Create AnnotatedSentence container for the definition text
        annotated_sentence = ET.SubElement(gloss, 'AnnotatedSentence')
        annotated_sentence.text = processed_def

        from_ele = ET.SubElement(gloss, 'Provenance')
        from_ele.set('resource', wn_name)
        from_ele.set('version', version)
        from_ele.set('original_id', original_id)


    # Create the tree and write to file with pretty printing
    tree = ET.ElementTree(root)
    tree.write(output_file,
               encoding='UTF-8',
               xml_declaration=True,
               pretty_print=True)

    print(f"Created {output_file}")
    print(f"  Concepts: {len(tsv_rows)}")
    print(f"  Glosses: {len(tsv_rows)}")


def main():
    """Main conversion process."""
    tsv_file = 'bin/cili.tsv'
    output_file = 'bin/cygnets_presynth/cili-1.0.xml'

    # Load data
    tsv_rows = load_tsv_data(tsv_file)

    # Validate
    validate_data(tsv_rows)

    # Create XML
    create_cygnet_xml(tsv_rows, output_file)

    print("\nConversion complete!")


if __name__ == '__main__':
    main()