#!/usr/bin/env python3
"""
Cygnet Dictionary Merger

Merges multiple Cygnet lexical resources into a single unified resource.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional
from lxml import etree


class CygnetMerger:
    def __init__(self, input_dir: str = "bin/cygnets_presynth"):
        self.input_dir = Path(input_dir)
        self.concept_map: Dict[str, str] = {}  # old_id -> shared_key
        self.concept_pos: Dict[str, str] = {}  # shared_key -> pos
        self.next_n_number = 1
        self.sense_map: Dict[str, str] = {}  # old_sense_id -> new_sense_id
        self.pos_conflicts: Dict[str, Dict[str, List[str]]] = {}  # concept_id -> {pos: [dict_names]}

        # Collections for merged data
        self.concepts: Dict[str, str] = {}  # shared_key -> pos
        self.wordforms: Dict[str, etree.Element] = {}  # id -> element
        self.senses: Dict[str, Tuple[str, str]] = {}  # new_sense_id -> (signifier, signified)
        self.glosses: Dict[str, str] = {}  # concept_key -> definition
        self.examples: Dict[str, etree.Element] = {}  # new_example_id -> element
        self.wordform_relations: Set[Tuple[str, str, str]] = set()  # (type, source, target)
        self.sense_relations: Set[Tuple[str, str, str]] = set()  # (type, source, target)
        self.concept_relations: Set[Tuple[str, str, str]] = set()  # (type, source, target)

    def discover_dictionaries(self) -> List[Tuple[Path, Path]]:
        """Find all XML and corresponding JSON mapping files."""
        xml_files = sorted(self.input_dir.glob("*.xml"))
        pairs = []

        for xml_file in xml_files:
            json_file = xml_file.parent / f"{xml_file.stem}_mapping.json"
            if json_file.exists():
                pairs.append((xml_file, json_file))
            else:
                print(f"Warning: No mapping file found for {xml_file.name}")

        # Reorder - find the tuple with "oewn" and move it to the start
        oewn_tuple = next((t for t in pairs if 'oewn' in str(t[0])), None)
        if oewn_tuple:
            pairs.remove(oewn_tuple)
            pairs.insert(0, oewn_tuple)

        return pairs

    def load_mapping(self, json_path: Path) -> Dict[str, str]:
        """Load concept mapping from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_or_create_shared_key(self, old_concept_id: str, mapping: Dict[str, str]) -> str:
        """Get shared key from mapping or create new n-key."""
        if old_concept_id in mapping:
            return mapping[old_concept_id]

        # Create new n-key
        while True:
            new_key = f"n{self.next_n_number}"
            self.next_n_number += 1
            # Check for conflicts (shouldn't happen, but being safe)
            if new_key not in self.concepts and new_key not in mapping.values():
                return new_key

    def merge_concept(self, shared_key: str, pos: str, dict_name: str):
        """Add or update concept, handling POS conflicts."""
        # Initialize conflict tracking for this concept if needed
        if shared_key not in self.pos_conflicts:
            self.pos_conflicts[shared_key] = defaultdict(list)

        # Track this POS and its source
        self.pos_conflicts[shared_key][pos].append(dict_name)

        if shared_key in self.concepts:
            existing_pos = self.concepts[shared_key]
            if existing_pos != pos:
                # Split existing POS into set to check for duplicates
                existing_pos_set = set(existing_pos.split('__'))

                # Only add new POS if not already present
                if pos not in existing_pos_set:
                    merged_pos = f"{existing_pos}__{pos}"
                    self.concepts[shared_key] = merged_pos
        else:
            # First time seeing this concept
            self.concepts[shared_key] = pos
    def verify_wordform_identity(self, wf1: etree.Element, wf2: etree.Element,
                                 wf_id: str, dict1: str, dict2: str):
        """Verify two wordforms with same ID are identical."""
        # Check tag (WrittenWordform vs SpokenWordform)
        if wf1.tag != wf2.tag:
            raise ValueError(
                f"Wordform {wf_id} has different types: "
                f"{wf1.tag} in {dict1} vs {wf2.tag} in {dict2}"
            )

        # Check all attributes
        attrs1 = dict(wf1.attrib)
        attrs2 = dict(wf2.attrib)

        if attrs1 != attrs2:
            raise ValueError(
                f"Wordform {wf_id} has different attributes:\n"
                f"  {dict1}: {attrs1}\n"
                f"  {dict2}: {attrs2}"
            )

    def create_new_sense_id(self, signifier: str, signified: str) -> str:
        """Create new sense ID from signifier and signified."""
        # Change 'w' to 's' in signifier and use '.' separator
        signifier_modified = 's' + signifier[1:] if signifier.startswith('w') else signifier
        return f"{signifier_modified}.{signified}"

    def process_dictionary(self, xml_path: Path, json_path: Path):
        """Process a single dictionary and merge into global collections."""
        dict_name = xml_path.stem
        print(f"\nProcessing dictionary: {dict_name}")

        # Load mapping
        mapping = self.load_mapping(json_path)
        print(f"  Loaded {len(mapping)} concept mappings")

        # Parse XML
        tree = etree.parse(str(xml_path))
        root = tree.getroot()

        # Track mappings for this dictionary
        local_concept_map = {}  # old_id -> shared_key for this dict
        local_sense_map = {}  # old_sense_id -> new_sense_id for this dict

        # Process ConceptLayer
        concept_layer = root.find("ConceptLayer")
        if concept_layer is not None:
            for concept in concept_layer.findall("Concept"):
                old_id = concept.get("id")
                pos = concept.get("pos")
                shared_key = self.get_or_create_shared_key(old_id, mapping)
                local_concept_map[old_id] = shared_key
                self.merge_concept(shared_key, pos, dict_name)

        print(f"  Processed {len(local_concept_map)} concepts")

        # Process WordformLayer
        wordform_layer = root.find("WordformLayer")
        if wordform_layer is not None:
            for wordform in wordform_layer:
                wf_id = wordform.get("id")
                if wf_id in self.wordforms:
                    # Verify identity
                    self.verify_wordform_identity(
                        self.wordforms[wf_id], wordform, wf_id,
                        "previous", dict_name
                    )
                else:
                    # Store a copy
                    self.wordforms[wf_id] = etree.Element(wordform.tag, wordform.attrib)

        print(f"  Processed {len(wordform_layer) if wordform_layer is not None else 0} wordforms")

        # Process SenseLayer
        sense_layer = root.find("SenseLayer")
        if sense_layer is not None:
            for sense in sense_layer.findall("Sense"):
                old_sense_id = sense.get("id")
                signifier = sense.get("signifier")
                old_signified = sense.get("signified")

                # Map signified to shared key
                new_signified = local_concept_map.get(old_signified, old_signified)

                # Create new sense ID
                new_sense_id = self.create_new_sense_id(signifier, new_signified)
                local_sense_map[old_sense_id] = new_sense_id

                # Store sense (will deduplicate automatically by using dict)
                self.senses[new_sense_id] = (signifier, new_signified)

        print(f"  Processed {len(local_sense_map)} senses")

        # Process GlossLayer (first gloss only per concept)
        gloss_layer = root.find("GlossLayer")
        if gloss_layer is not None:
            for gloss in gloss_layer.findall("Gloss"):
                old_definiendum = gloss.get("definiendum")
                definition = gloss.get("definition")

                # Map to shared key
                new_definiendum = local_concept_map.get(old_definiendum, old_definiendum)

                # Only store if we don't have a gloss for this concept yet
                if new_definiendum not in self.glosses:
                    self.glosses[new_definiendum] = definition

        print(f"  Processed glosses (total unique: {len(self.glosses)})")

        # Process ExampleLayer
        example_layer = root.find("ExampleLayer")
        if example_layer is not None:
            for example in example_layer.findall("Example"):
                old_example_id = example.get("id")
                new_example_id = f"{dict_name}-{old_example_id}"

                # Create new example element with updated sense references
                new_example = etree.Element("Example", {"id": new_example_id})

                # Handle mixed content (text and AnnotatedToken elements)
                if example.text:
                    new_example.text = example.text

                for child in example:
                    if child.tag == "AnnotatedToken":
                        old_sense_ref = child.get("sense")
                        new_sense_ref = local_sense_map.get(old_sense_ref, old_sense_ref)

                        new_token = etree.SubElement(new_example, "AnnotatedToken",
                                                     {"sense": new_sense_ref})
                        new_token.text = child.text
                        if child.tail:
                            new_token.tail = child.tail

                self.examples[new_example_id] = new_example

        print(f"  Processed {len(example_layer) if example_layer is not None else 0} examples")

        # Process WordformRelationLayer
        wf_rel_layer = root.find("WordformRelationLayer")
        if wf_rel_layer is not None:
            for relation in wf_rel_layer.findall("WordformRelation"):
                rel_type = relation.get("relationType")
                source = relation.get("source")
                target = relation.get("target")
                self.wordform_relations.add((rel_type, source, target))

        # Process SenseRelationLayer
        sense_rel_layer = root.find("SenseRelationLayer")
        if sense_rel_layer is not None:
            for relation in sense_rel_layer.findall("SenseRelation"):
                rel_type = relation.get("relationType")
                old_source = relation.get("source")
                old_target = relation.get("target")

                # Map to new sense IDs
                new_source = local_sense_map.get(old_source, old_source)
                new_target = local_sense_map.get(old_target, old_target)

                self.sense_relations.add((rel_type, new_source, new_target))

        # Process ConceptRelationLayer
        concept_rel_layer = root.find("ConceptRelationLayer")
        if concept_rel_layer is not None:
            for relation in concept_rel_layer.findall("ConceptRelation"):
                rel_type = relation.get("relationType")
                old_source = relation.get("source")
                old_target = relation.get("target")

                # Map to shared keys
                new_source = local_concept_map.get(old_source, old_source)
                new_target = local_concept_map.get(old_target, old_target)

                self.concept_relations.add((rel_type, new_source, new_target))

    def build_output_xml(self) -> etree.ElementTree:
        """Build the merged XML document."""
        print("\nBuilding output XML...")

        # Create root element
        root = etree.Element("CygnetResource", {
            "id": "cyg",
            "label": "Cygnet",
            "version": "1.0.0"
        })

        # ConceptLayer
        concept_layer = etree.SubElement(root, "ConceptLayer")
        for concept_id in sorted(self.concepts.keys()):
            etree.SubElement(concept_layer, "Concept", {
                "id": concept_id,
                "pos": self.concepts[concept_id]
            })
        print(f"  Added {len(self.concepts)} concepts")

        # WordformLayer
        wordform_layer = etree.SubElement(root, "WordformLayer")
        for wf_id in sorted(self.wordforms.keys()):
            wordform = self.wordforms[wf_id]
            wordform_layer.append(wordform)
        print(f"  Added {len(self.wordforms)} wordforms")

        # SenseLayer
        sense_layer = etree.SubElement(root, "SenseLayer")
        for sense_id in sorted(self.senses.keys()):
            signifier, signified = self.senses[sense_id]
            etree.SubElement(sense_layer, "Sense", {
                "id": sense_id,
                "signifier": signifier,
                "signified": signified
            })
        print(f"  Added {len(self.senses)} senses")

        # GlossLayer
        gloss_layer = etree.SubElement(root, "GlossLayer")
        for concept_id in sorted(self.glosses.keys()):
            etree.SubElement(gloss_layer, "Gloss", {
                "definiendum": concept_id,
                "definition": self.glosses[concept_id]
            })
        print(f"  Added {len(self.glosses)} glosses")

        # ExampleLayer
        example_layer = etree.SubElement(root, "ExampleLayer")
        for example_id in sorted(self.examples.keys()):
            example_layer.append(self.examples[example_id])
        print(f"  Added {len(self.examples)} examples")

        # WordformRelationLayer
        if self.wordform_relations:
            wf_rel_layer = etree.SubElement(root, "WordformRelationLayer")
            for rel_type, source, target in sorted(self.wordform_relations):
                etree.SubElement(wf_rel_layer, "WordformRelation", {
                    "relationType": rel_type,
                    "source": source,
                    "target": target
                })
            print(f"  Added {len(self.wordform_relations)} wordform relations")

        # SenseRelationLayer
        if self.sense_relations:
            sense_rel_layer = etree.SubElement(root, "SenseRelationLayer")
            for rel_type, source, target in sorted(self.sense_relations):
                etree.SubElement(sense_rel_layer, "SenseRelation", {
                    "relationType": rel_type,
                    "source": source,
                    "target": target
                })
            print(f"  Added {len(self.sense_relations)} sense relations")

        # ConceptRelationLayer
        if self.concept_relations:
            concept_rel_layer = etree.SubElement(root, "ConceptRelationLayer")
            for rel_type, source, target in sorted(self.concept_relations):
                etree.SubElement(concept_rel_layer, "ConceptRelation", {
                    "relationType": rel_type,
                    "source": source,
                    "target": target
                })
            print(f"  Added {len(self.concept_relations)} concept relations")

        return etree.ElementTree(root)

    def merge(self, output_path: str = "bin/cygnet_prefix.xml"):
        """Main merge process."""
        print("=" * 60)
        print("Cygnet Dictionary Merger")
        print("=" * 60)

        # Discover dictionaries
        dict_pairs = self.discover_dictionaries()
        print(f"\nFound {len(dict_pairs)} dictionaries to merge")

        # Process each dictionary
        for xml_path, json_path in dict_pairs:
            self.process_dictionary(xml_path, json_path)

        # Build output
        tree = self.build_output_xml()

        # Write to file
        print(f"\nWriting merged resource to {output_path}...")
        tree.write(
            output_path,
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True
        )

        print("\n" + "=" * 60)
        print("Merge complete!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  Concepts: {len(self.concepts)}")
        print(f"  Wordforms: {len(self.wordforms)}")
        print(f"  Senses: {len(self.senses)}")
        print(f"  Glosses: {len(self.glosses)}")
        print(f"  Examples: {len(self.examples)}")
        print(f"  Wordform Relations: {len(self.wordform_relations)}")
        print(f"  Sense Relations: {len(self.sense_relations)}")
        print(f"  Concept Relations: {len(self.concept_relations)}")
        print(f"  New n-keys created: {self.next_n_number - 1}")

        # Filter to only actual conflicts (multiple different POS values)
        actual_conflicts = {
            concept_id: pos_dict
            for concept_id, pos_dict in self.pos_conflicts.items()
            if len(pos_dict) > 1
        }

        # Print conflict count
        if actual_conflicts:
            print(f"\n  POS Conflicts: {len(actual_conflicts)}")

            # Save conflicts to JSON
            conflicts_output_path = "pos_conflicts.json"
            with open(conflicts_output_path, 'w', encoding='utf-8') as f:
                json.dump(actual_conflicts, f, indent=2, ensure_ascii=False)
            print(f"  POS conflict details saved to {conflicts_output_path}")
        else:
            print(f"\n  POS Conflicts: 0")


if __name__ == "__main__":
    merger = CygnetMerger()
    merger.merge()