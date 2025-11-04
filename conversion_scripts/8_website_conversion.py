#!/usr/bin/env python3
"""
Convert Cygnet XML lexical resource to JSON format for static website.
"""

from lxml import etree as ET
from collections import defaultdict
import json
import os
from pathlib import Path
import unicodedata


def remove_accents(text):
    """Remove accents/diacritics from text for search normalization."""
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

def validate_keyrefs(root):
    """Validate that all keyref constraints are satisfied."""
    print("Validating keyrefs...")

    # Collect all IDs
    concept_ids = set()
    lexeme_ids = set()
    sense_ids = set()

    for concept in root.findall('.//Concept'):
        concept_ids.add(concept.get('id'))

    for lexeme in root.findall('.//Lexeme'):
        lexeme_ids.add(lexeme.get('id'))

    for sense in root.findall('.//Sense'):
        sense_ids.add(sense.get('id'))

    print(f"  Found {len(concept_ids)} concepts, {len(lexeme_ids)} lexemes, {len(sense_ids)} senses")

    # Validate Sense references
    for sense in root.findall('.//Sense'):
        signifier = sense.get('signifier')
        signified = sense.get('signified')

        if signifier not in lexeme_ids:
            raise ValueError(f"Sense {sense.get('id')} references non-existent lexeme {signifier}")
        if signified not in concept_ids:
            raise ValueError(f"Sense {sense.get('id')} references non-existent concept {signified}")

    # Validate Gloss references
    for gloss in root.findall('.//Gloss'):
        definiendum = gloss.get('definiendum')
        if definiendum not in concept_ids:
            raise ValueError(f"Gloss references non-existent concept {definiendum}")

    # Validate AnnotatedToken references in Glosses
    for token in root.findall('.//GlossLayer//AnnotatedToken'):
        sense_ref = token.get('sense')
        if sense_ref not in sense_ids:
            raise ValueError(f"Gloss AnnotatedToken references non-existent sense {sense_ref}")

    # Validate AnnotatedToken references in Examples
    for token in root.findall('.//ExampleLayer//AnnotatedToken'):
        sense_ref = token.get('sense')
        if sense_ref not in sense_ids:
            raise ValueError(f"Example AnnotatedToken references non-existent sense {sense_ref}")

    # Validate SenseRelation references
    for rel in root.findall('.//SenseRelation'):
        source = rel.get('source')
        target = rel.get('target')
        if source not in sense_ids:
            raise ValueError(f"SenseRelation references non-existent source sense {source}")
        if target not in sense_ids:
            raise ValueError(f"SenseRelation references non-existent target sense {target}")

    # Validate ConceptRelation references
    for rel in root.findall('.//ConceptRelation'):
        source = rel.get('source')
        target = rel.get('target')
        if source not in concept_ids:
            raise ValueError(f"ConceptRelation references non-existent source concept {source}")
        if target not in concept_ids:
            raise ValueError(f"ConceptRelation references non-existent target concept {target}")

    print("  All keyrefs validated successfully!")


def validate_unique_glosses(root):
    """Validate that there are no duplicate glosses for same concept+language."""
    print("Validating unique glosses per concept+language...")

    gloss_map = defaultdict(int)
    for gloss in root.findall('.//Gloss'):
        key = (gloss.get('definiendum'), gloss.get('language'))
        gloss_map[key] += 1

    duplicates = [(k, v) for k, v in gloss_map.items() if v > 1]
    if duplicates:
        for (concept, lang), count in duplicates[:5]:
            print(f"  ERROR: Concept {concept} has {count} glosses in language {lang}")
        raise ValueError(f"Found {len(duplicates)} concept+language pairs with multiple glosses")

    print("  All glosses are unique per concept+language!")


def parse_annotated_sentence(sentence_elem, bold_sense_id=None, return_annotations=False):
    """Parse an AnnotatedSentence element into HTML string.

    If bold_sense_id is provided, only tokens matching that sense are bolded.
    If bold_sense_id is None, no tokens are bolded (for glosses).
    If return_annotations is True, also return annotation data with offsets.
    """
    parts = []
    annotations = []
    current_offset = 0

    # Handle text and elements
    if sentence_elem.text:
        parts.append(sentence_elem.text)
        current_offset += len(sentence_elem.text)

    for child in sentence_elem:
        if child.tag.endswith('AnnotatedToken'):
            token_text = child.text or ""
            token_sense = child.get('sense')

            # Record annotation if requested
            if return_annotations and token_sense:
                annotations.append({
                    'so': current_offset,
                    'eo': current_offset + len(token_text),
                    's': token_sense
                })

            # Bold if this is the sense we're looking for (only for examples)
            if bold_sense_id and token_sense == bold_sense_id:
                parts.append(f"<b>{token_text}</b>")
            else:
                parts.append(token_text)

            current_offset += len(token_text)

        # Add tail text after element
        if child.tail:
            parts.append(child.tail)
            current_offset += len(child.tail)

    html_string = ''.join(parts)

    if return_annotations:
        return html_string, annotations
    return html_string


def shorten_relation_type(rel_type):
    """Convert relation type to short code."""
    mapping = {
        # Sense relations
        'antonym': 'ant',
        'derivation': 'der',
        'pertainym': 'ptr',
        'participle': 'prt',
        # Concept relations
        'class_hypernym': 'hyp',
        'instance_hypernym': 'ihyp',
        'member_meronym': 'mmer',
        'part_meronym': 'pmer',
        'substance_meronym': 'smer',
        'opposite': 'opp',
        'causes': 'cau',
        'entails': 'ent',
        'agent_of_action': 'aga',
        'patient_of_action': 'paa',
        'result_of_action': 'rea',
        'instrument_of_agent': 'iag',
        'instrument_of_action': 'iac',
        'result_of_agent': 'rag',
        'patient_of_agent': 'pag',
        'instrument_of_patient': 'ipa',
        'instrument_of_result': 'ire',
        'patient_of_result': 'pre',
    }
    return mapping.get(rel_type, rel_type)


def is_symmetric_relation(rel_type_short):
    """Check if a relation type is symmetric."""
    symmetric = {'opp', 'ant'}
    return rel_type_short in symmetric


def parse_cygnet_xml(xml_path):
    """Parse the Cygnet XML file and extract all data."""
    print(f"Parsing {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Validate the data
    validate_keyrefs(root)
    validate_unique_glosses(root)

    print("\nExtracting data structures...")

    # Parse concepts
    concepts = {}
    for concept_elem in root.findall('.//Concept'):
        concept_id = concept_elem.get('id')
        concepts[concept_id] = {
            'id': concept_id,
            'oc': concept_elem.get('ontological_category'),
        }
    print(f"  Parsed {len(concepts)} concepts")

    # Parse lexemes
    lexemes = {}
    for lexeme_elem in root.findall('.//Lexeme'):
        lexeme_id = lexeme_elem.get('id')
        wordforms = lexeme_elem.findall('Wordform')

        lemma = wordforms[0].get('form')
        variants = [wf.get('form') for wf in wordforms[1:]]

        lexemes[lexeme_id] = {
            'id': lexeme_id,
            'l': lemma,
            'v': variants,
            'lg': lexeme_elem.get('language'),
            'gc': lexeme_elem.get('grammatical_category'),
        }
    print(f"  Parsed {len(lexemes)} lexemes")

    # Parse senses (maintain original order for concept->sense mapping)
    senses = {}
    sense_order = []  # Track original order
    concept_to_senses = defaultdict(list)  # Track senses per concept with language

    for sense_elem in root.findall('.//Sense'):
        sense_id = sense_elem.get('id')
        signifier = sense_elem.get('signifier')
        signified = sense_elem.get('signified')

        senses[sense_id] = {
            'id': sense_id,
            'signifier': signifier,
            'signified': signified,
            'lexeme': lexemes[signifier],
            'concept': concepts[signified],
        }

        sense_order.append(sense_id)

        # Track which senses evoke each concept (with language info)
        language = lexemes[signifier]['lg']
        concept_to_senses[signified].append((sense_id, language))

    print(f"  Parsed {len(senses)} senses")

    # Parse glosses
    glosses = defaultdict(dict)
    gloss_annotations = defaultdict(dict)
    for gloss_elem in root.findall('.//Gloss'):
        concept_id = gloss_elem.get('definiendum')
        language = gloss_elem.get('language')

        sentence_elem = gloss_elem.find('AnnotatedSentence')

        if language == 'en':
            gloss_text, annotations = parse_annotated_sentence(sentence_elem, bold_sense_id=None,
                                                               return_annotations=True)
            gloss_annotations[concept_id] = annotations
        else:
            gloss_text = parse_annotated_sentence(sentence_elem, bold_sense_id=None)

        glosses[concept_id][language] = gloss_text
    print(f"  Parsed glosses for {len(glosses)} concepts")

    # Parse examples - collect all examples and their referenced senses
    examples_by_sense = defaultdict(list)
    for example_elem in root.findall('.//Example'):
        sentence_elem = example_elem.find('AnnotatedSentence')

        # Find all senses referenced in this example
        referenced_senses = set()
        for token in sentence_elem.findall('.//AnnotatedToken'):
            referenced_senses.add(token.get('sense'))

        # For each referenced sense, create the example with that sense bolded
        for sense_id in referenced_senses:
            example_text = parse_annotated_sentence(sentence_elem, bold_sense_id=sense_id)
            examples_by_sense[sense_id].append(example_text)

    total_examples = sum(len(exs) for exs in examples_by_sense.values())
    print(f"  Parsed {total_examples} example associations")

    # Parse sense relations
    sense_relations = defaultdict(list)
    for rel_elem in root.findall('.//SenseRelation'):
        rel_type = shorten_relation_type(rel_elem.get('relation_type'))
        source = rel_elem.get('source')
        target = rel_elem.get('target')

        # Add forward relation
        sense_relations[source].append({'t': rel_type, 'to': target})

        # Add inverse relation
        if is_symmetric_relation(rel_type):
            sense_relations[target].append({'t': rel_type, 'to': source})
        else:
            sense_relations[target].append({'t': f'-{rel_type}', 'to': source})
    print(f"  Parsed sense relations for {len(sense_relations)} senses")

    # Parse concept relations
    concept_relations = defaultdict(list)
    for rel_elem in root.findall('.//ConceptRelation'):
        rel_type = shorten_relation_type(rel_elem.get('relation_type'))
        source = rel_elem.get('source')
        target = rel_elem.get('target')

        # Add forward relation
        concept_relations[source].append({'t': rel_type, 'to': target})

        # Add inverse relation
        if is_symmetric_relation(rel_type):
            concept_relations[target].append({'t': rel_type, 'to': source})
        else:
            concept_relations[target].append({'t': f'-{rel_type}', 'to': source})
    print(f"  Parsed concept relations for {len(concept_relations)} concepts")

    return {
        'concepts': concepts,
        'lexemes': lexemes,
        'senses': senses,
        'sense_order': sense_order,
        'concept_to_senses': concept_to_senses,
        'glosses': glosses,
        'gloss_annotations': gloss_annotations,
        'examples_by_sense': examples_by_sense,
        'sense_relations': sense_relations,
        'concept_relations': concept_relations,
    }


def reindex_senses(data):
    """Reindex senses by language, lemma, then original order."""
    print("\nReindexing senses...")

    senses = data['senses']

    # Create sort key: (language, lemma, original_id)
    sense_list = []
    for sense_id, sense_data in senses.items():
        lexeme = sense_data['lexeme']
        sort_key = (lexeme['lg'], lexeme['l'].lower(), sense_id)
        sense_list.append((sort_key, sense_id, sense_data))

    # Sort by the key
    sense_list.sort(key=lambda x: x[0])

    # Create new index mapping
    old_to_new = {}
    new_senses = {}

    for new_idx, (sort_key, old_id, sense_data) in enumerate(sense_list):
        old_to_new[old_id] = new_idx
        new_senses[new_idx] = sense_data

    print(f"  Reindexed {len(new_senses)} senses")
    return new_senses, old_to_new


def reindex_concepts(data, old_to_new_sense):
    """Reindex concepts by order of first appearance in reindexed senses."""
    print("Reindexing concepts...")

    concepts = data['concepts']
    new_senses = data['reindexed_senses']

    # Find first appearance of each concept that has senses
    first_appearance = {}
    for new_sense_idx in sorted(new_senses.keys()):
        concept_id = new_senses[new_sense_idx]['signified']
        if concept_id not in first_appearance:
            first_appearance[concept_id] = new_sense_idx

    # Sort concepts by first appearance
    concepts_with_senses = sorted(first_appearance.items(), key=lambda x: x[1])

    # Find concepts without any senses (but may have relations)
    concepts_without_senses = sorted(
        [cid for cid in concepts.keys() if cid not in first_appearance]
    )

    # Create new index mapping: concepts with senses first, then concepts without
    old_to_new = {}
    new_concepts = {}
    new_idx = 0

    # Add concepts with senses (ordered by first appearance)
    for old_id, _ in concepts_with_senses:
        old_to_new[old_id] = new_idx
        new_concepts[new_idx] = concepts[old_id]
        new_idx += 1

    # Add concepts without senses (alphabetically ordered for consistency)
    for old_id in concepts_without_senses:
        old_to_new[old_id] = new_idx
        new_concepts[new_idx] = concepts[old_id]
        new_idx += 1

    print(
        f"  Reindexed {len(new_concepts)} concepts ({len(concepts_with_senses)} with senses, {len(concepts_without_senses)} without)")
    return new_concepts, old_to_new


def build_output_data(data, old_to_new_sense, old_to_new_concept, sense_indices):
    """Build the final output data structures with new indices."""
    print("\nBuilding output data structures...")

    senses_out = {}
    concepts_out = {}

    skipped_sense_relations = 0
    skipped_concept_relations = 0

    # Build synonym relations: for each concept+language, connect all senses
    synonym_relations = defaultdict(list)
    for concept_id, sense_lang_pairs in data['concept_to_senses'].items():
        # Group by language
        by_language = defaultdict(list)
        for old_sense_id, language in sense_lang_pairs:
            if old_sense_id in old_to_new_sense:
                by_language[language].append(old_to_new_sense[old_sense_id])

        # For each language group with 2+ senses, create syn relations
        for language, new_sense_indices in by_language.items():
            if len(new_sense_indices) >= 2:
                for i, sense_a in enumerate(new_sense_indices):
                    for sense_b in new_sense_indices[i + 1:]:
                        synonym_relations[sense_a].append({'t': 'syn', 'to': sense_b})
                        synonym_relations[sense_b].append({'t': 'syn', 'to': sense_a})

    # Build sense data
    for new_sense_idx, sense_data in data['reindexed_senses'].items():
        old_sense_id = sense_data['id']
        lexeme = sense_data['lexeme']

        sense_out = {
            'l': lexeme['l'],
            'v': lexeme['v'],
            'lg': lexeme['lg'],
            'c': old_to_new_concept[sense_data['signified']],
            'ind': sense_indices[new_sense_idx],
        }

        # Add examples
        if old_sense_id in data['examples_by_sense']:
            sense_out['ex'] = data['examples_by_sense'][old_sense_id]

        # Add sense relations (with new indices, skip missing targets)
        all_relations = []

        # Add existing sense relations
        if old_sense_id in data['sense_relations']:
            for rel in data['sense_relations'][old_sense_id]:
                target_old_id = rel['to']
                if target_old_id in old_to_new_sense:
                    all_relations.append({
                        't': rel['t'],
                        'to': old_to_new_sense[target_old_id]
                    })
                else:
                    skipped_sense_relations += 1

        # Add synonym relations
        if new_sense_idx in synonym_relations:
            all_relations.extend(synonym_relations[new_sense_idx])

        if all_relations:
            sense_out['sr'] = all_relations

        senses_out[new_sense_idx] = sense_out

    # Build concept data
    for new_concept_idx, concept_data in data['reindexed_concepts'].items():
        old_concept_id = concept_data['id']

        concept_out = {
            'oc': concept_data['oc'],
        }

        # Add definitions
        if old_concept_id in data['glosses']:
            concept_out['d'] = data['glosses'][old_concept_id]

        # Add English definition annotations
        if old_concept_id in data['gloss_annotations']:
            annotations = data['gloss_annotations'][old_concept_id]
            # Convert old sense IDs to new sense IDs
            converted_annotations = []
            for ann in annotations:
                old_sense_id = ann['s']
                if old_sense_id in old_to_new_sense:
                    converted_annotations.append({
                        'so': ann['so'],
                        'eo': ann['eo'],
                        's': old_to_new_sense[old_sense_id]
                    })
            if converted_annotations:
                concept_out['da'] = converted_annotations

        # Add sense pointers (organized by language, in original order)
        if old_concept_id in data['concept_to_senses']:
            senses_by_lang = defaultdict(list)
            for old_sense_id, language in data['concept_to_senses'][old_concept_id]:
                new_sense_idx = old_to_new_sense[old_sense_id]
                senses_by_lang[language].append(new_sense_idx)

            concept_out['s'] = dict(senses_by_lang)

        # Add concept relations (with new indices, skip missing targets)
        if old_concept_id in data['concept_relations']:
            relations = []
            for rel in data['concept_relations'][old_concept_id]:
                target_old_id = rel['to']
                if target_old_id in old_to_new_concept:
                    relations.append({
                        't': rel['t'],
                        'to': old_to_new_concept[target_old_id]
                    })
                else:
                    skipped_concept_relations += 1

            if relations:
                concept_out['cr'] = relations

        concepts_out[new_concept_idx] = concept_out

    print(f"  Built {len(senses_out)} senses and {len(concepts_out)} concepts")
    if skipped_sense_relations > 0:
        print(f"  Skipped {skipped_sense_relations} sense relations with missing targets")
    if skipped_concept_relations > 0:
        print(f"  Skipped {skipped_concept_relations} concept relations with missing targets")

    return senses_out, concepts_out


def write_chunked_data(data, output_dir, chunk_size=100):
    """Write sense and concept data to chunked JSON files."""
    print("\nWriting chunked data files...")

    senses_dir = output_dir / 'senses'
    concepts_dir = output_dir / 'concepts'

    # Write senses
    sense_indices = sorted(data['senses'].keys())
    for chunk_start in range(0, len(sense_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(sense_indices))

        # Calculate folder and file structure
        # Each folder contains 100 files (covering 10,000 senses)
        folder_num = chunk_start // 10000
        file_num = (chunk_start // chunk_size) % 100

        folder_path = senses_dir / f'{folder_num:03d}'
        folder_path.mkdir(parents=True, exist_ok=True)

        file_path = folder_path / f'{chunk_start // chunk_size:05d}.json'

        # Get chunk data
        chunk_data = {}
        for idx in range(chunk_start, chunk_end):
            if idx < len(sense_indices):
                sense_idx = sense_indices[idx]
                chunk_data[sense_idx] = data['senses'][sense_idx]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, separators=(',', ':'))

    print(f"  Wrote {len(sense_indices)} senses to {senses_dir}")

    # Write concepts
    concept_indices = sorted(data['concepts'].keys())
    for chunk_start in range(0, len(concept_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(concept_indices))

        # Calculate folder and file structure
        folder_num = chunk_start // 10000
        file_num = (chunk_start // chunk_size) % 100

        folder_path = concepts_dir / f'{folder_num:03d}'
        folder_path.mkdir(parents=True, exist_ok=True)

        file_path = folder_path / f'{chunk_start // chunk_size:05d}.json'

        # Get chunk data
        chunk_data = {}
        for idx in range(chunk_start, chunk_end):
            if idx < len(concept_indices):
                concept_idx = concept_indices[idx]
                chunk_data[concept_idx] = data['concepts'][concept_idx]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, separators=(',', ':'))

    print(f"  Wrote {len(concept_indices)} concepts to {concepts_dir}")


def build_prefix_lookups(data, old_to_new_sense):
    """Build prefix lookup data mapping wordforms to senses."""
    print("\nBuilding prefix lookups...")

    # Build wordform -> POS -> lang -> [sense_indices]
    wordform_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for new_sense_idx, sense_data in data['reindexed_senses'].items():
        lexeme = sense_data['lexeme']
        concept = sense_data['concept']

        # The POS is the ontological category of the concept
        pos = concept['oc']
        lang = lexeme['lg']

        # Add lemma (remove accents for lookup)
        lemma_lower = remove_accents(lexeme['l'].lower())
        wordform_map[lemma_lower][pos][lang].append(new_sense_idx)

        # Add variants (remove accents for lookup)
        for variant in lexeme['v']:
            variant_lower = remove_accents(variant.lower())
            wordform_map[variant_lower][pos][lang].append(new_sense_idx)

    # Convert to regular dicts
    wordform_data = {}
    for wordform, pos_dict in wordform_map.items():
        wordform_data[wordform] = {
            pos: dict(lang_dict) for pos, lang_dict in pos_dict.items()
        }

    print(f"  Built lookup data for {len(wordform_data)} unique wordforms")
    return wordform_data


def write_prefix_lookups(wordform_data, output_dir, target_size=2000):
    """Write prefix lookup files and prefix map."""
    print("\nWriting prefix lookup files...")

    lookups_dir = output_dir / 'prefix_lookups'
    lookups_dir.mkdir(parents=True, exist_ok=True)

    # Group wordforms by prefix
    prefix_groups = defaultdict(list)
    for wordform in sorted(wordform_data.keys()):
        if len(wordform) == 1:
            prefix = wordform[0]
        else:
            prefix = wordform[:2]
        prefix_groups[prefix].append(wordform)

    # Split large groups and merge small groups
    prefix_to_file = {}
    file_counter = 0
    current_batch = []
    current_batch_prefixes = []

    for prefix in sorted(prefix_groups.keys()):
        wordforms = prefix_groups[prefix]

        # If this prefix has too many wordforms, split it
        if len(wordforms) > target_size * 2:
            # Split into multiple files
            for i in range(0, len(wordforms), target_size):
                batch = wordforms[i:i + target_size]
                filename = f'{file_counter:04d}.json'
                file_counter += 1

                # Write file
                batch_data = {wf: wordform_data[wf] for wf in batch}
                with open(lookups_dir / filename, 'w', encoding='utf-8') as f:
                    json.dump(batch_data, f, ensure_ascii=False, separators=(',', ':'))

                # Map this prefix to this file (will have multiple entries)
                if prefix not in prefix_to_file:
                    prefix_to_file[prefix] = []
                prefix_to_file[prefix].append(filename)
        else:
            # Add to current batch
            current_batch.extend(wordforms)
            if prefix not in current_batch_prefixes:
                current_batch_prefixes.append(prefix)

            # If batch is large enough, write it
            if len(current_batch) >= target_size:
                filename = f'{file_counter:04d}.json'
                file_counter += 1

                # Write file
                batch_data = {wf: wordform_data[wf] for wf in current_batch}
                with open(lookups_dir / filename, 'w', encoding='utf-8') as f:
                    json.dump(batch_data, f, ensure_ascii=False, separators=(',', ':'))

                # Map all prefixes in this batch to this file
                for p in current_batch_prefixes:
                    if p not in prefix_to_file:
                        prefix_to_file[p] = []
                    prefix_to_file[p].append(filename)

                # Reset batch
                current_batch = []
                current_batch_prefixes = []

    # Write remaining batch
    if current_batch:
        filename = f'{file_counter:04d}.json'
        file_counter += 1

        batch_data = {wf: wordform_data[wf] for wf in current_batch}
        with open(lookups_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, separators=(',', ':'))

        for p in current_batch_prefixes:
            if p not in prefix_to_file:
                prefix_to_file[p] = []
            prefix_to_file[p].append(filename)

    # Convert lists to single values where possible
    prefix_map = {}
    for prefix, files in prefix_to_file.items():
        if len(files) == 1:
            prefix_map[prefix] = files[0]
        else:
            prefix_map[prefix] = files

    # Write prefix map
    with open(output_dir / 'prefix_map.json', 'w', encoding='utf-8') as f:
        json.dump(prefix_map, f, ensure_ascii=False, indent=2)

    print(f"  Wrote {file_counter} prefix lookup files")
    print(f"  Wrote prefix_map.json with {len(prefix_map)} prefixes")


def main():
    xml_path = 'cygnet.xml'
    output_dir = Path('website_data')

    # Parse the XML
    data = parse_cygnet_xml(xml_path)

    # Reindex senses
    reindexed_senses, old_to_new_sense = reindex_senses(data)
    data['reindexed_senses'] = reindexed_senses

    # Reindex concepts
    reindexed_concepts, old_to_new_concept = reindex_concepts(data, old_to_new_sense)
    data['reindexed_concepts'] = reindexed_concepts

    # Calculate sense indices
    print("\nCalculating sense indices...")
    index_counters = defaultdict(int)
    sense_indices = {}

    for new_sense_idx in sorted(reindexed_senses.keys()):
        sense_data = reindexed_senses[new_sense_idx]
        lexeme = sense_data['lexeme']
        concept = sense_data['concept']

        # Index key is (language, lemma_lowercase, POS)
        # POS is the ontological category of the concept
        index_key = (lexeme['lg'], lexeme['l'].lower(), concept['oc'])
        index_counters[index_key] += 1
        sense_indices[new_sense_idx] = index_counters[index_key]

    print(f"  Assigned indices to {len(sense_indices)} senses across {len(index_counters)} lemma groups")

    # Build output data with new indices
    senses_out, concepts_out = build_output_data(data, old_to_new_sense, old_to_new_concept, sense_indices)

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Write chunked data
    write_chunked_data({
        'senses': senses_out,
        'concepts': concepts_out,
    }, output_dir, chunk_size=100)

    # Build and write prefix lookups
    wordform_data = build_prefix_lookups(data, old_to_new_sense)
    write_prefix_lookups(wordform_data, output_dir, target_size=2000)

    print("\n✓ Conversion complete!")
    print(f"  Output written to: {output_dir}")


if __name__ == '__main__':
    main()