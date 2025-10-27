import xml.etree.ElementTree as ET
import json
from pathlib import Path
from collections import defaultdict


def parse_cygnet_xml(xml_file):
    """Parse Cygnet XML and convert to dictionary format, skipping spoken wordforms."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = {
        "concepts": {},
        "wordforms": {},
        "senses": {},
        "glosses": {},
        "examples": []
    }

    # Parse Concepts
    concept_layer = root.find('ConceptLayer')
    if concept_layer is not None:
        for concept in concept_layer.findall('Concept'):
            concept_id = concept.get('id')
            data["concepts"][concept_id] = {
                "id": concept_id,
                "pos": concept.get('pos').lower() if concept.get('pos') else ''
            }

    # Parse Wordforms - SKIP SPOKEN WORDFORMS
    wordform_layer = root.find('WordformLayer')
    if wordform_layer is not None:
        for wordform in wordform_layer:
            # Skip spoken wordforms
            if wordform.tag.endswith('SpokenWordform'):
                continue

            wordform_id = wordform.get('id')
            wordform_data = {
                "id": wordform_id,
                "form": wordform.get('form'),
                "language": wordform.get('language'),
                "type": "written"
            }

            data["wordforms"][wordform_id] = wordform_data

    # Parse Senses - only include senses that reference written wordforms
    sense_layer = root.find('SenseLayer')
    if sense_layer is not None:
        for sense in sense_layer.findall('Sense'):
            sense_id = sense.get('id')
            signifier = sense.get('signifier')

            # Only include sense if it references a written wordform
            if signifier in data["wordforms"]:
                data["senses"][sense_id] = {
                    "id": sense_id,
                    "signifier": signifier,
                    "signified": sense.get('signified')
                }

    # Parse Glosses
    gloss_layer = root.find('GlossLayer')
    if gloss_layer is not None:
        for gloss in gloss_layer.findall('Gloss'):
            definiendum = gloss.get('definiendum')
            data["glosses"][definiendum] = gloss.get('definition')

    # Parse Examples - only include examples that reference senses we kept
    example_layer = root.find('ExampleLayer')
    if example_layer is not None:
        for example in example_layer.findall('Example'):
            example_id = example.get('id')
            tokens = []
            has_valid_sense = False

            if example.text:
                tokens.append(example.text)

            for child in example:
                if child.tag == 'AnnotatedToken':
                    sense_ref = child.get('sense')
                    # Check if this sense is in our filtered senses
                    if sense_ref in data["senses"]:
                        has_valid_sense = True

                    tokens.append({
                        "text": child.text,
                        "sense": sense_ref
                    })
                    if child.tail:
                        tokens.append(child.tail)

            # Only include example if it references at least one valid sense
            if has_valid_sense:
                data["examples"].append({
                    "id": example_id,
                    "tokens": tokens
                })

    return data


def get_first_letter(word):
    """Get the first letter (normalized) for grouping."""
    if not word:
        return '_'
    first = word[0].lower()
    if first.isalpha():
        return first
    return '_'  # Non-alphabetic characters


def create_letter_chunked_structure(data, output_dir='data'):
    """
    Create a letter-chunked data structure:
    - manifest.json: List of available letters and languages
    - letters/[letter]_[lang].json: Wordforms starting with that letter in that language
    - concepts/[concept_id].json: Full details per concept (loaded on demand)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    letters_path = output_path / 'letters'
    letters_path.mkdir(exist_ok=True)
    concepts_path = output_path / 'concepts'
    concepts_path.mkdir(exist_ok=True)

    print("  Building indexes...")

    # Pre-index: Group senses by wordform (O(n))
    senses_by_wordform = defaultdict(list)
    for sense in data["senses"].values():
        senses_by_wordform[sense["signifier"]].append(sense)

    # Pre-index: Group senses by concept (O(n))
    senses_by_concept = defaultdict(list)
    for sense_id, sense in data["senses"].items():
        senses_by_concept[sense["signified"]].append((sense_id, sense))

    # Pre-index: Group examples by sense (O(n))
    examples_by_sense = defaultdict(list)
    for example in data["examples"]:
        for token in example["tokens"]:
            if isinstance(token, dict) and "sense" in token:
                examples_by_sense[token["sense"]].append(example)
                break  # Only add example once per sense

    print("  Grouping wordforms by letter and language...")

    # Group wordforms by first letter and language
    wordforms_by_letter_lang = defaultdict(lambda: defaultdict(list))
    languages = set()

    for wordform_id, wordform in data["wordforms"].items():
        letter = get_first_letter(wordform["form"])
        lang = wordform["language"]
        languages.add(lang)

        # Use pre-indexed senses
        senses = senses_by_wordform[wordform_id]

        wordform_with_senses = {
            **wordform,
            "senses": [
                {
                    "id": sense["id"],
                    "concept_id": sense["signified"],
                    "concept_pos": data["concepts"][sense["signified"]]["pos"]
                }
                for sense in senses
            ]
        }

        wordforms_by_letter_lang[letter][lang].append(wordform_with_senses)

    print("  Creating manifest...")

    # Create manifest
    manifest = {
        "languages": sorted(list(languages)),
        "letters": {}
    }

    for letter in sorted(wordforms_by_letter_lang.keys()):
        manifest["letters"][letter] = {
            lang: len(wordforms_by_letter_lang[letter][lang])
            for lang in wordforms_by_letter_lang[letter].keys()
        }

    with open(output_path / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("  Writing letter files...")

    # Create letter files
    letter_files_created = 0
    for letter, langs_dict in wordforms_by_letter_lang.items():
        for lang, wordforms in langs_dict.items():
            letter_file = letters_path / f'{letter}_{lang}.json'
            with open(letter_file, 'w', encoding='utf-8') as f:
                json.dump({"wordforms": wordforms}, f, ensure_ascii=False, separators=(',', ':'))
            letter_files_created += 1

    print(f"  Writing {len(data['concepts'])} concept files...")

    # Create concept files
    concepts_created = 0
    for i, (concept_id, concept_info) in enumerate(data["concepts"].items()):
        if i % 1000 == 0 and i > 0:
            print(f"    Progress: {i}/{len(data['concepts'])} concepts...")

        concept_data = {
            "id": concept_id,
            "pos": concept_info["pos"],
            "gloss": data["glosses"].get(concept_id, ""),
            "senses": []
        }

        # Use pre-indexed senses
        for sense_id, sense_info in senses_by_concept[concept_id]:
            sense_data = {
                "id": sense_id,
                "signifier": sense_info["signifier"],
                "wordform": data["wordforms"][sense_info["signifier"]],
                "examples": examples_by_sense.get(sense_id, [])
            }
            concept_data["senses"].append(sense_data)

        concept_file = concepts_path / f'{concept_id}.json'
        with open(concept_file, 'w', encoding='utf-8') as f:
            json.dump(concept_data, f, ensure_ascii=False, separators=(',', ':'))

        concepts_created += 1

    return {
        "manifest_size": len(json.dumps(manifest)),
        "letter_files_created": letter_files_created,
        "concepts_created": concepts_created
    }


def main():
    """Main function to generate letter-chunked dictionary structure."""
    input_file = 'cygnet.xml'
    output_dir = 'website_data'

    try:
        print(f"Reading XML file: {input_file}")
        data = parse_cygnet_xml(input_file)

        print(f"\nParsed data (written forms only):")
        print(f"  Concepts: {len(data['concepts'])}")
        print(f"  Wordforms: {len(data['wordforms'])}")
        print(f"  Senses: {len(data['senses'])}")
        print(f"  Glosses: {len(data['glosses'])}")
        print(f"  Examples: {len(data['examples'])}")

        print(f"\nCreating letter-chunked structure in '{output_dir}/'...")
        stats = create_letter_chunked_structure(data, output_dir)

        print(f"\nChunking complete!")
        print(f"  Manifest size: ~{stats['manifest_size'] / 1024:.2f} KB")
        print(f"  Letter files created: {stats['letter_files_created']}")
        print(f"  Concept files created: {stats['concepts_created']}")
        print(f"\nStructure:")
        print(f"  website_data/manifest.json         <- Load this first (~1-10 KB)")
        print(f"  website_data/letters/*.json        <- Load letter chunks as user searches")
        print(f"  website_data/concepts/*.json       <- Load individual concepts on demand")
        print(f"\nNote: All spoken wordforms have been excluded from the data.")

    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'")
        print("Please ensure your XML file exists in the current directory.")
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()