#!/usr/bin/env python3
"""
Merge multiple Cygnet XML files into a single consolidated resource.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from lxml import etree

# Namespace (empty for this schema)
NS = {}


class MergeStats:
    """Track statistics during merge process."""

    def __init__(self):
        self.files_processed = 0
        self.concepts_total = 0
        self.lexemes_total = 0
        self.lexemes_merged = 0
        self.senses_total = 0
        self.senses_deduplicated = 0
        self.glosses_total = 0
        self.glosses_discarded = defaultdict(lambda: defaultdict(int))  # file -> language -> count
        self.concepts_missing_english = defaultdict(int)  # file -> count
        self.examples_total = 0
        self.sense_relations_total = 0
        self.sense_relations_deduplicated = 0
        self.concept_relations_total = 0
        self.concept_relations_deduplicated = 0

    def to_dict(self):
        """Convert stats to dictionary for JSON serialization."""
        return {
            'files_processed': self.files_processed,
            'concepts_total': self.concepts_total,
            'lexemes_total': self.lexemes_total,
            'lexemes_merged': self.lexemes_merged,
            'senses_total': self.senses_total,
            'senses_deduplicated': self.senses_deduplicated,
            'glosses_total': self.glosses_total,
            'glosses_discarded_per_file': {
                file: dict(langs) for file, langs in self.glosses_discarded.items()
            },
            'concepts_missing_english_per_file': dict(self.concepts_missing_english),
            'examples_total': self.examples_total,
            'sense_relations_total': self.sense_relations_total,
            'sense_relations_deduplicated': self.sense_relations_deduplicated,
            'concept_relations_total': self.concept_relations_total,
            'concept_relations_deduplicated': self.concept_relations_deduplicated
        }


def get_provenance_elements(element):
    """Extract all Provenance elements from an element."""
    return element.findall('Provenance')


def merge_wordform(existing_wf, new_wf):
    """Merge Scripts and Pronunciations from new_wf into existing_wf."""
    # Get existing scripts
    existing_scripts = {s.text for s in existing_wf.findall('Script') if s.text}

    # Add new scripts
    for script in new_wf.findall('Script'):
        if script.text and script.text not in existing_scripts:
            script_elem = etree.SubElement(existing_wf, 'Script')
            script_elem.text = script.text
            existing_scripts.add(script.text)

    # Get existing pronunciations as (text, variety) tuples
    existing_prons = set()
    for pron in existing_wf.findall('Pronunciation'):
        pron_text = pron.text or ''
        pron_variety = pron.get('variety')
        existing_prons.add((pron_text, pron_variety))

    # Add new pronunciations
    for pron in new_wf.findall('Pronunciation'):
        pron_text = pron.text or ''
        pron_variety = pron.get('variety')
        if (pron_text, pron_variety) not in existing_prons:
            pron_elem = etree.SubElement(existing_wf, 'Pronunciation')
            pron_elem.text = pron_text
            if pron_variety:
                pron_elem.set('variety', pron_variety)
            existing_prons.add((pron_text, pron_variety))


def merge_lexemes(existing_lexeme, new_lexeme):
    """Merge new_lexeme into existing_lexeme."""
    # Build map of form -> Wordform element for existing lexeme
    existing_wordforms = {}
    for wf in existing_lexeme.findall('Wordform'):
        form = wf.get('form')
        if form:
            existing_wordforms[form] = wf

    # Process wordforms from new lexeme
    for new_wf in new_lexeme.findall('Wordform'):
        form = new_wf.get('form')
        if not form:
            continue

        if form in existing_wordforms:
            # Merge into existing wordform
            merge_wordform(existing_wordforms[form], new_wf)
        else:
            # Add new wordform
            existing_lexeme.append(new_wf)
            existing_wordforms[form] = new_wf

    # Merge provenance (with duplicate checking)
    existing_prov_set = set()
    for prov in existing_lexeme.findall('Provenance'):
        resource = prov.get('resource', '')
        version = prov.get('version', '')
        original_id = prov.get('original_id', '')
        existing_prov_set.add((resource, version, original_id))

    for prov in new_lexeme.findall('Provenance'):
        resource = prov.get('resource', '')
        version = prov.get('version', '')
        original_id = prov.get('original_id', '')
        prov_key = (resource, version, original_id)

        if prov_key not in existing_prov_set:
            existing_lexeme.append(prov)
            existing_prov_set.add(prov_key)


def canonicalize_relation(rel_type, source, target):
    """Canonicalize symmetric relations to have source < target alphabetically."""
    symmetric_relations = {'opposite', 'antonym'}
    if rel_type in symmetric_relations and source > target:
        return target, source
    return source, target


def merge_cygnet_files(input_dir, output_file, log_file):
    """Merge all Cygnet XML files in input_dir into a single output file."""

    stats = MergeStats()

    # Get all XML files sorted alphabetically
    input_path = Path(input_dir)
    xml_files = sorted(input_path.glob('*.xml'))

    print(f"Found {len(xml_files)} XML files to merge")

    # PHASE 1: Load all data from all files
    print("\nPhase 1: Loading all data...")

    all_concepts = []  # list of (element, source_file)
    all_lexemes = []  # list of (element, source_file)
    all_senses = []  # list of (element, source_file)
    all_glosses = []  # list of (element, source_file)
    all_examples = []  # list of (element, source_file)
    all_sense_relations = []  # list of (element, source_file)
    all_concept_relations = []  # list of (element, source_file)

    for xml_file in xml_files:
        print(f"  Loading: {xml_file.name}")
        stats.files_processed += 1
        current_file = xml_file.name

        tree = etree.parse(str(xml_file))
        root = tree.getroot()

        # Load ConceptLayer
        concept_layer = root.find('ConceptLayer')
        if concept_layer is not None:
            for concept in concept_layer.findall('Concept'):
                all_concepts.append((concept, current_file))

        # Load LexemeLayer
        lexeme_layer = root.find('LexemeLayer')
        if lexeme_layer is not None:
            for lexeme in lexeme_layer.findall('Lexeme'):
                all_lexemes.append((lexeme, current_file))

        # Load SenseLayer
        sense_layer = root.find('SenseLayer')
        if sense_layer is not None:
            for sense in sense_layer.findall('Sense'):
                all_senses.append((sense, current_file))

        # Load GlossLayer
        gloss_layer = root.find('GlossLayer')
        if gloss_layer is not None:
            for gloss in gloss_layer.findall('Gloss'):
                all_glosses.append((gloss, current_file))

        # Load ExampleLayer
        example_layer = root.find('ExampleLayer')
        if example_layer is not None:
            for example in example_layer.findall('Example'):
                all_examples.append((example, current_file))

        # Load SenseRelationLayer
        sense_relation_layer = root.find('SenseRelationLayer')
        if sense_relation_layer is not None:
            for relation in sense_relation_layer.findall('SenseRelation'):
                all_sense_relations.append((relation, current_file))

        # Load ConceptRelationLayer
        concept_relation_layer = root.find('ConceptRelationLayer')
        if concept_relation_layer is not None:
            for relation in concept_relation_layer.findall('ConceptRelation'):
                all_concept_relations.append((relation, current_file))

    # PHASE 2: Process and merge all data
    print("\nPhase 2: Processing and merging data...")

    # Process concepts
    print("  Processing concepts...")
    concepts = {}  # id -> element
    for concept, source_file in all_concepts:
        concept_id = concept.get('id')
        assert concept_id not in concepts, \
            f"Duplicate concept ID '{concept_id}' found in {source_file}"
        concepts[concept_id] = concept
        stats.concepts_total += 1

    # Process lexemes
    print("  Processing lexemes...")
    lexemes = {}  # id -> element
    for lexeme, source_file in all_lexemes:
        lexeme_id = lexeme.get('id')
        if lexeme_id in lexemes:
            # Merge lexemes
            merge_lexemes(lexemes[lexeme_id], lexeme)
            stats.lexemes_merged += 1
        else:
            lexemes[lexeme_id] = lexeme
            stats.lexemes_total += 1

    # Process senses
    print("  Processing senses...")
    senses = {}  # id -> element
    sense_keys = {}  # (signifier, signified) -> id
    for sense, source_file in all_senses:
        sense_id = sense.get('id')
        signifier = sense.get('signifier')
        signified = sense.get('signified')

        sense_key = (signifier, signified)

        if sense_id in senses:
            # Duplicate sense ID - check it's identical
            existing_sense = senses[sense_id]
            assert existing_sense.get('signifier') == signifier, \
                f"Sense {sense_id} has conflicting signifier in {source_file}"
            assert existing_sense.get('signified') == signified, \
                f"Sense {sense_id} has conflicting signified in {source_file}"

            # Combine provenance (with duplicate checking)
            existing_prov_set = set()
            for prov in existing_sense.findall('Provenance'):
                resource = prov.get('resource', '')
                version = prov.get('version', '')
                original_id = prov.get('original_id', '')
                existing_prov_set.add((resource, version, original_id))

            for prov in get_provenance_elements(sense):
                resource = prov.get('resource', '')
                version = prov.get('version', '')
                original_id = prov.get('original_id', '')
                prov_key = (resource, version, original_id)

                if prov_key not in existing_prov_set:
                    existing_sense.append(prov)
                    existing_prov_set.add(prov_key)

            stats.senses_deduplicated += 1

        elif sense_key in sense_keys:
            # Same signifier/signified but different ID - this is an error
            existing_id = sense_keys[sense_key]
            assert False, \
                f"Sense with key {sense_key} exists with different IDs: {existing_id} and {sense_id} in {source_file}"
        else:
            senses[sense_id] = sense
            sense_keys[sense_key] = sense_id
            stats.senses_total += 1

    # Process glosses
    print("  Processing glosses...")
    glosses = {}  # (concept_id, language) -> element
    concept_glosses = defaultdict(set)  # concept_id -> set of languages
    for gloss, source_file in all_glosses:
        definiendum = gloss.get('definiendum')
        language = gloss.get('language')
        gloss_key = (definiendum, language)

        if gloss_key in glosses:
            # Discard duplicate gloss
            stats.glosses_discarded[source_file][language] += 1
        else:
            glosses[gloss_key] = gloss
            concept_glosses[definiendum].add(language)
            stats.glosses_total += 1

    # Process examples (no deduplication)
    print("  Processing examples...")
    examples = [example for example, _ in all_examples]
    stats.examples_total = len(examples)

    # Process sense relations
    print("  Processing sense relations...")
    sense_relations = {}  # (relation_type, source, target) -> element
    for relation, source_file in all_sense_relations:
        rel_type = relation.get('relation_type')
        source = relation.get('source')
        target = relation.get('target')

        # Canonicalize symmetric relations
        source, target = canonicalize_relation(rel_type, source, target)

        rel_key = (rel_type, source, target)

        if rel_key in sense_relations:
            # Combine provenance (with duplicate checking)
            existing = sense_relations[rel_key]
            existing_prov_set = set()
            for prov in existing.findall('Provenance'):
                resource = prov.get('resource', '')
                version = prov.get('version', '')
                original_id = prov.get('original_id', '')
                existing_prov_set.add((resource, version, original_id))

            for prov in get_provenance_elements(relation):
                resource = prov.get('resource', '')
                version = prov.get('version', '')
                original_id = prov.get('original_id', '')
                prov_key = (resource, version, original_id)

                if prov_key not in existing_prov_set:
                    existing.append(prov)
                    existing_prov_set.add(prov_key)
            stats.sense_relations_deduplicated += 1
        else:
            # Update relation with canonicalized source/target
            relation.set('source', source)
            relation.set('target', target)
            sense_relations[rel_key] = relation
            stats.sense_relations_total += 1

    # Process concept relations
    print("  Processing concept relations...")
    concept_relations = {}  # (relation_type, source, target) -> element
    for relation, source_file in all_concept_relations:
        rel_type = relation.get('relation_type')
        source = relation.get('source')
        target = relation.get('target')

        # Canonicalize symmetric relations
        source, target = canonicalize_relation(rel_type, source, target)

        rel_key = (rel_type, source, target)

        if rel_key in concept_relations:
            # Combine provenance (with duplicate checking)
            existing = concept_relations[rel_key]
            existing_prov_set = set()
            for prov in existing.findall('Provenance'):
                resource = prov.get('resource', '')
                version = prov.get('version', '')
                original_id = prov.get('original_id', '')
                existing_prov_set.add((resource, version, original_id))

            for prov in get_provenance_elements(relation):
                resource = prov.get('resource', '')
                version = prov.get('version', '')
                original_id = prov.get('original_id', '')
                prov_key = (resource, version, original_id)

                if prov_key not in existing_prov_set:
                    existing.append(prov)
                    existing_prov_set.add(prov_key)
            stats.concept_relations_deduplicated += 1
        else:
            # Update relation with canonicalized source/target
            relation.set('source', source)
            relation.set('target', target)
            concept_relations[rel_key] = relation
            stats.concept_relations_total += 1

    # PHASE 3: Validation - check for concepts missing English glosses
    print("\nPhase 3: Validation...")
    print("  Checking for concepts missing English glosses...")

    # Build map of concept_id -> source_file for better error reporting
    concept_source_map = {}
    for concept, source_file in all_concepts:
        concept_id = concept.get('id')
        concept_source_map[concept_id] = source_file

    for concept_id in concepts.keys():
        if 'en' not in concept_glosses[concept_id]:
            source_file = concept_source_map.get(concept_id, 'unknown')
            print(f"  WARNING: Concept {concept_id} missing English gloss (from {source_file})")
            stats.concepts_missing_english[source_file] += 1

    print("\nBuilding merged XML...")

    # Build the merged XML
    root = etree.Element('CygnetResource')
    root.set('id', 'cyg')
    root.set('label', 'Cygnet')
    root.set('version', '1.0')

    # Add ConceptLayer
    if concepts:
        concept_layer = etree.SubElement(root, 'ConceptLayer')
        for concept in concepts.values():
            concept_layer.append(concept)

    # Add LexemeLayer
    if lexemes:
        lexeme_layer = etree.SubElement(root, 'LexemeLayer')
        for lexeme in lexemes.values():
            lexeme_layer.append(lexeme)

    # Add SenseLayer
    if senses:
        sense_layer = etree.SubElement(root, 'SenseLayer')
        for sense in senses.values():
            sense_layer.append(sense)

    # Add GlossLayer
    if glosses:
        gloss_layer = etree.SubElement(root, 'GlossLayer')
        for gloss in glosses.values():
            gloss_layer.append(gloss)

    # Add ExampleLayer
    if examples:
        example_layer = etree.SubElement(root, 'ExampleLayer')
        for example in examples:
            example_layer.append(example)

    # Add SenseRelationLayer
    if sense_relations:
        sense_relation_layer = etree.SubElement(root, 'SenseRelationLayer')
        for relation in sense_relations.values():
            sense_relation_layer.append(relation)

    # Add ConceptRelationLayer
    if concept_relations:
        concept_relation_layer = etree.SubElement(root, 'ConceptRelationLayer')
        for relation in concept_relations.values():
            concept_relation_layer.append(relation)

    # Write output XML
    tree = etree.ElementTree(root)
    tree.write(output_file, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    print(f"\nMerged XML written to: {output_file}")

    # Write log file
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(stats.to_dict(), f, indent=2)
    print(f"Log written to: {log_file}")

    # Print summary
    print("\n=== Merge Summary ===")
    print(f"Files processed: {stats.files_processed}")
    print(f"Concepts: {stats.concepts_total}")
    print(f"Lexemes: {stats.lexemes_total} (merged: {stats.lexemes_merged})")
    print(f"Senses: {stats.senses_total} (deduplicated: {stats.senses_deduplicated})")
    print(f"Glosses: {stats.glosses_total}")
    print(f"Examples: {stats.examples_total}")
    print(f"Sense Relations: {stats.sense_relations_total} (deduplicated: {stats.sense_relations_deduplicated})")
    print(f"Concept Relations: {stats.concept_relations_total} (deduplicated: {stats.concept_relations_deduplicated})")

    if stats.glosses_discarded:
        print(
            f"\nGlosses discarded (duplicates): {sum(sum(langs.values()) for langs in stats.glosses_discarded.values())}")

    if stats.concepts_missing_english:
        print(f"\nConcepts missing English glosses: {sum(stats.concepts_missing_english.values())}")


if __name__ == '__main__':
    input_dir = 'bin/cygnets_presynth/'
    output_file = 'cygnet.xml'
    log_file = 'bin/merge_log.json'

    merge_cygnet_files(input_dir, output_file, log_file)