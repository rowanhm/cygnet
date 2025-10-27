#!/usr/bin/env python3
"""
Convert GlobalWordNet XML format to Cygnet XML format.
"""
import lzma

from lxml import etree
from collections import defaultdict
import json
import sys
import gzip
import spacy
import pyinflect
from nltk.stem import WordNetLemmatizer
import nltk

# ISO 639-1 language code to spaCy efficient (small) model mapping
LANGUAGE_TO_SPACY_MODEL = {
    'ca': 'ca_core_news_sm',      # Catalan
    'zh': 'zh_core_web_sm',        # Chinese
    'hr': 'hr_core_news_sm',       # Croatian
    'da': 'da_core_news_sm',       # Danish
    'nl': 'nl_core_news_sm',       # Dutch
    'en': 'en_core_web_sm',        # English
    'fi': 'fi_core_news_sm',       # Finnish
    'fr': 'fr_core_news_sm',       # French
    'de': 'de_core_news_sm',       # German
    'el': 'el_core_news_sm',       # Greek
    'it': 'it_core_news_sm',       # Italian
    'ja': 'ja_core_news_sm',       # Japanese
    'ko': 'ko_core_news_sm',       # Korean
    'lt': 'lt_core_news_sm',       # Lithuanian
    'mk': 'mk_core_news_sm',       # Macedonian
    'nb': 'nb_core_news_sm',       # Norwegian Bokmål
    'pl': 'pl_core_news_sm',       # Polish
    'pt': 'pt_core_news_sm',       # Portuguese
    'ro': 'ro_core_news_sm',       # Romanian
    'ru': 'ru_core_news_sm',       # Russian
    'sl': 'sl_core_news_sm',       # Slovenian
    'es': 'es_core_news_sm',       # Spanish
    'sv': 'sv_core_news_sm',       # Swedish
    'uk': 'uk_core_news_sm',       # Ukrainian
    'xx': 'xx_sent_ud_sm',         # Multi-language (fallback)
}

class GWNToCygnetConverter:
    def __init__(self, skip_examples=False, debug_examples=False):
        self.skip_examples = skip_examples
        self.debug_examples = debug_examples
        self.language = None
        self.lexicon_id = None
        self.lexicon_label = None
        self.lexicon_version = None
        self.concepts = {}  # synset_id -> Concept element
        self.glosses = {}  # synset_id -> definition text
        self.wordforms = {}  # (form, language, type) -> wordform_id
        self.wordform_ids = set()  # set of all wordform IDs for fast lookup
        self.synset_to_cili = {}  # synset_id -> ILI
        self.old_sense_to_new_senses = {}  # old_sense_id -> [new_sense_ids]
        self.senses = {}  # new_sense_id -> (signifier, signified)
        self.wordform_relations = []  # list of (type, source, target)
        self.sense_relations = []  # list of (type, source, target)
        self.concept_relations = []  # list of (type, source, target)
        self.examples = []  # list of (example_id, example_element)
        self.synset_pos = {}  # synset_id -> pos
        self.pos_mismatches = defaultdict(int)  # (synset_pos, lemma_pos) -> count
        self.nlp = None  # Will be loaded in convert()
        self.nltk_lemmatizer = None  # Will be loaded in convert()
        self.doc_cache = {}  # Cache for spaCy documents
        self.form_cache = {}  # Cache for word forms
        self.examples_processed = 0  # Counter for processed examples
        self.examples_skipped = 0  # Counter for skipped examples
        self.entry_written_forms = defaultdict(list)  # lexical_entry_id -> [written_wf_ids]
        self.lemma_spoken_forms = defaultdict(list)  # lexical_entry_id -> [lemma_spoken_wf_ids]
        self.form_to_spoken = {}  # form_written_wf_id -> [spoken_wf_ids]
        self.missing_pos_records = []  # List of dicts: {synset_id, assigned_pos, source}
        self.total_examples_found = 0  # Total examples before processing

    def _wn_to_ud_pos(self, wn_pos):
        """
        Map WordNet POS tag to Universal Dependencies POS tag.

        Args:
            wn_pos: Single character WordNet POS tag (n, v, a, r, s, c, p, x, u)

        Returns:
            String containing UD POS tag(s), with multiple options separated by "__"

        Raises:
            AssertionError: If wn_pos is not a valid WordNet POS tag

        """
        mapping = {
            'n': 'NOUN',
            'v': 'VERB',
            'a': 'ADJ',
            'r': 'ADV',
            's': 'ADJ',
            'c': 'CCONJ-SCONJ',
            'p': 'ADP',
            'x': 'PART-INTJ',
            'u': 'X'
        }

        assert wn_pos in mapping, f"Invalid WordNet POS tag: {wn_pos}"
        return mapping[wn_pos]
    def _get_doc(self, text):
        """Get spaCy doc with caching."""
        if text not in self.doc_cache:
            self.doc_cache[text] = self.nlp(text)
        return self.doc_cache[text]

    def read_metadata(self, input_file):
        """Read and extract metadata from input file without processing."""
        print(f"Reading metadata from {input_file}...")

        # Parse the file
        if input_file.endswith('.xml.gz'):
            with gzip.open(input_file, 'rt', encoding='utf-8') as f:
                tree = etree.parse(f)
        elif input_file.endswith('.xml.xz'):
            with lzma.open(input_file, 'rt', encoding='utf-8') as f:
                tree = etree.parse(f)
        else:
            assert input_file.endswith('.xml')
            tree = etree.parse(input_file)

        root = tree.getroot()

        # Extract metadata
        self._extract_metadata(root)

        return root, tree  # Return both for potential reuse

    def process_file(self, root, output_file, skip_examples=False, xsd_file=None):
        """Process the parsed XML tree and convert to Cygnet format."""

        # Load appropriate spaCy model based on language
        model_name = LANGUAGE_TO_SPACY_MODEL.get(self.language, 'xx_sent_ud_sm')
        print(f"Loading spaCy model '{model_name}' for language '{self.language}'...")
        try:
            self.nlp = spacy.load(model_name, disable=["parser", "ner", "tok2vec"])
        except OSError:
            print(f"Downloading spaCy model '{model_name}'...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name, disable=["parser", "ner", "tok2vec"])

        # Initialize NLTK lemmatizer
        self.nltk_lemmatizer = WordNetLemmatizer()
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK wordnet data...")
            nltk.download('wordnet')

        # Pass 1: Extract metadata and create concepts
        print("Pass 1: Creating concepts and glosses...")
        self._create_concepts_and_glosses(root)

        # Pass 2: Create wordforms from lemmas and pronunciations
        print("Pass 2: Creating wordforms...")
        self._create_wordforms(root)

        # Pass 3: Create senses
        print("Pass 3: Creating senses...")
        self._create_senses(root)
        self._report_pos_mismatches()

        # Pass 3.5: Create wordform variant relations
        print("Pass 3.5: Creating wordform variant relations...")
        self._create_wordform_variant_relations(root)

        if not skip_examples:
            print("Pass 4: Processing examples...")
            self._process_examples(root)
            print(f"  Examples processed: {self.examples_processed}")
            print(f"  Examples skipped: {self.examples_skipped}")
            print(
                f"  Success rate: {(self.examples_processed + 1 / (self.examples_processed + self.examples_skipped + 1)) * 100:.1f}%"
            )
        else:
            print("Pass 4: Skipping examples (--skip-examples flag set)")

        # Pass 5: Process relations
        print("Pass 5: Processing relations...")
        self._process_relations(root)

        self._validate_output()

        # Write output
        print(f"Writing output to {output_file}...")
        self._write_output(output_file, xsd_file)

        # Write mapping
        mapping_file = output_file.replace('.xml', '_mapping.json')
        print(f"Writing synset-to-ILI mapping to {mapping_file}...")
        with open(mapping_file, 'w') as f:
            json.dump(self.synset_to_cili, f, indent=2)

        # Write log
        log_file = output_file.replace('.xml', '_log.json')
        print(f"Writing processing log to {log_file}...")
        log_data = self._generate_log_data()
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print("Conversion complete!")

    def convert(self, input_file, output_file, skip_examples=False, xsd_file=None):
        """Main conversion function."""

        # Read metadata first
        root, tree = self.read_metadata(input_file)

        # Process the file
        self.process_file(root, output_file, skip_examples, xsd_file)

    def _extract_metadata(self, root):
        """Extract language and metadata from lexicon."""
        lexicon = root.find('.//Lexicon')
        if lexicon is not None:
            self.language = lexicon.get('language')
            self.lexicon_id = lexicon.get('id', 'converted_resource')
            self.lexicon_label = lexicon.get('label', 'Converted from GWN')
            self.lexicon_version = lexicon.get('version', '1.0')
            print(f"  Language: {self.language}")
            print(f"  ID: {self.lexicon_id}")
            print(f"  Label: {self.lexicon_label}")
            print(f"  Version: {self.lexicon_version}")
        else:
            raise ValueError("No Lexicon element found in input file")
    def _create_concepts_and_glosses(self, root):
        """Create Concepts from Synsets and Glosses from Definitions."""
        for synset in root.findall('.//Synset'):
            synset_id = synset.get('id')
            ili = synset.get('ili', None)
            pos = synset.get('partOfSpeech', '')

            # Store synset POS for later validation
            self.synset_pos[synset_id] = pos

            if ili == '' or ili == 'in':
                ili = None

            if ili is not None:
                # Store ILI mapping
                self.synset_to_cili[synset_id] = ili

            # Create concept - convert POS to UD format for storage
            self.concepts[synset_id] = self._wn_to_ud_pos(pos) if pos else ''

            # Extract definition for gloss
            definition = synset.find('Definition')
            if definition is not None and definition.text:
                self.glosses[synset_id] = definition.text.strip()

    def _validate_output(self):
        """Validate the converted data before writing output."""
        print("Validating converted data...")

        errors = []

        # Validate Concepts
        print("  Checking concepts...")
        for concept_id, pos in self.concepts.items():
            if not pos or pos == "":
                errors.append(f"Concept {concept_id} has empty POS")

        # Validate Glosses
        print("  Checking glosses...")
        for concept_id, definition in self.glosses.items():
            if not definition or definition.strip() == "":
                errors.append(f"Gloss for concept {concept_id} has empty definition")

        # Validate Wordforms
        print("  Checking wordforms...")
        for key, wf_id in self.wordforms.items():
            if not wf_id or wf_id == "":
                errors.append(f"Wordform has empty ID: {key}")

            if len(key) >= 3:
                form = key[0]
                if not form or form.strip() == "":
                    errors.append(f"Wordform {wf_id} has empty form")

        # Validate Senses
        print("  Checking senses...")
        for sense_id, (signifier, signified) in self.senses.items():
            if not signifier or signifier == "":
                errors.append(f"Sense {sense_id} has empty signifier")
            if not signified or signified == "":
                errors.append(f"Sense {sense_id} has empty signified")

        # Validate Wordform Relations
        print("  Checking wordform relations...")
        for rel_type, source, target in self.wordform_relations:
            if not rel_type or rel_type.strip() == "":
                errors.append(f"Wordform relation has empty relationType: {source} -> {target}")
            if not source or source == "":
                errors.append(f"Wordform relation has empty source: relationType={rel_type}")
            if not target or target == "":
                errors.append(f"Wordform relation has empty target: relationType={rel_type}")

        # Validate orthographic_variant relations
        print("  Checking orthographic_variant relations...")
        orthographic_variant_count = sum(1 for rt, _, _ in self.wordform_relations if rt == 'orthographic_variant')
        if orthographic_variant_count > 0:
            for rel_type, source, target in self.wordform_relations:
                if rel_type == 'orthographic_variant':
                    if source not in self.wordform_ids:
                        errors.append(f"orthographic_variant source not found: {source}")
                    if target not in self.wordform_ids:
                        errors.append(f"orthographic_variant target not found: {target}")

        # Validate pronunciation_variant relations
        print("  Checking pronunciation_variant relations...")
        pronunciation_variant_count = sum(1 for rt, _, _ in self.wordform_relations if rt == 'pronunciation_variant')
        if pronunciation_variant_count > 0:
            for rel_type, source, target in self.wordform_relations:
                if rel_type == 'pronunciation_variant':
                    if source not in self.wordform_ids:
                        errors.append(f"pronunciation_variant source not found: {source}")
                    if target not in self.wordform_ids:
                        errors.append(f"pronunciation_variant target not found: {target}")

        # Validate Sense Relations
        print("  Checking sense relations...")
        for rel_type, source, target in self.sense_relations:
            if not rel_type or rel_type.strip() == "":
                errors.append(f"Sense relation has empty relationType: {source} -> {target}")
            if not source or source == "":
                errors.append(f"Sense relation has empty source: relationType={rel_type}")
            if not target or target == "":
                errors.append(f"Sense relation has empty target: relationType={rel_type}")

        # Validate Concept Relations
        print("  Checking concept relations...")
        for rel_type, source, target in self.concept_relations:
            if not rel_type or rel_type.strip() == "":
                errors.append(f"Concept relation has empty relationType: {source} -> {target}")
            if not source or source == "":
                errors.append(f"Concept relation has empty source: relationType={rel_type}")
            if not target or target == "":
                errors.append(f"Concept relation has empty target: relationType={rel_type}")

        # Validate Examples
        print("  Checking examples...")
        for example_tuple in self.examples:
            example_id, full_text, token_start, token_end, sense_id = example_tuple

            if not example_id or example_id == "":
                errors.append(f"Example has empty ID")

            if not sense_id or sense_id == "":
                errors.append(f"Example {example_id} has empty sense")

            if token_start < 0 or token_end > len(full_text) or token_start >= token_end:
                errors.append(f"Example {example_id} has invalid token boundaries: [{token_start}, {token_end})")

        # Report results
        if errors:
            print(f"\n❌ Validation FAILED with {len(errors)} errors:")
            for error in errors[:50]:  # Show first 50 errors
                print(f"    - {error}")
            if len(errors) > 50:
                print(f"    ... and {len(errors) - 50} more errors")
            raise ValueError(f"Validation failed with {len(errors)} errors. See above for details.")
        else:
            print("  ✓ All validations passed!")

    def _encode_for_xml_id(self, text):
        """
        Convert any Unicode string to a safe XML ID format with one-to-one correspondence.
        Each unique input string produces a unique output key.

        Encoding rules:
        - ASCII letters (a-z, A-Z): pass through unchanged
        - ASCII digits (0-9): pass through unchanged
        - Everything else (including punctuation, spaces, non-ASCII Unicode): encode as xHHHH
          where HHHH is the 4-digit uppercase hex Unicode codepoint
        """
        encoded = []
        for char in text:
            # Only allow ASCII alphanumeric (a-z, A-Z, 0-9)
            if ('a' <= char <= 'z') or ('A' <= char <= 'Z') or ('0' <= char <= '9'):
                encoded.append(char)
            else:
                # Encode everything else as x followed by 4-digit hex codepoint
                hex_val = f"{ord(char):04X}"
                encoded.append(f"x{hex_val}")

        return ''.join(encoded)

    def _create_wordforms(self, root):
        """Create WrittenWordforms and SpokenWordforms from Lemmas and Forms."""
        for entry in root.findall('.//LexicalEntry'):
            entry_id = entry.get('id')
            lemma = entry.find('Lemma')
            if lemma is None:
                continue

            written_form = lemma.get('writtenForm')

            # Skip if written form is empty or whitespace-only
            if not written_form or not written_form.strip():
                print(f"  Warning: Skipping entry {entry_id} with empty writtenForm")
                continue

            # Create written wordform from Lemma with new ID format: form__lang__type
            written_key = (written_form, self.language, 'written')
            if written_key not in self.wordforms:
                wordform_id = f"{self._sanitize_wordform_id(written_form)}.{self.language}.written"
                self.wordforms[written_key] = wordform_id
                self.wordform_ids.add(wordform_id)

            lemma_written_id = self.wordforms[written_key]
            self.entry_written_forms[entry_id].append(lemma_written_id)

            # Create spoken wordforms from Lemma pronunciations
            lemma_spoken_ids = []
            for pron in lemma.findall('Pronunciation'):
                if pron.text:
                    pron_form = pron.text.strip()
                    variety = pron.get('variety', '')

                    spoken_key = (pron_form, self.language, 'spoken', variety)
                    if spoken_key not in self.wordforms:
                        # Use romanized pronunciation in ID: pron__lang_variety__spoken (if variety exists)
                        if variety:
                            spoken_id = f"{self._sanitize_wordform_id(pron_form)}.{self.language}-{variety}.spoken"
                        else:
                            spoken_id = f"{self._sanitize_wordform_id(pron_form)}.{self.language}.spoken"
                        self.wordforms[spoken_key] = spoken_id
                        self.wordform_ids.add(spoken_id)

                        # Create pronunciation_of and orthography_of relations
                        self.wordform_relations.append(
                            ('pronunciation_of', spoken_id, lemma_written_id)
                        )
                        self.wordform_relations.append(
                            ('orthography_of', lemma_written_id, spoken_id)
                        )

                    spoken_id = self.wordforms[spoken_key]
                    lemma_spoken_ids.append(spoken_id)

            self.lemma_spoken_forms[entry_id] = lemma_spoken_ids

            # Process Form elements
            for form_elem in entry.findall('Form'):
                form_written = form_elem.get('writtenForm')

                # Skip if form is empty or whitespace-only
                if not form_written or not form_written.strip():
                    continue

                # Create written wordform for Form (or reuse if same as Lemma)
                form_written_key = (form_written, self.language, 'written')
                if form_written_key not in self.wordforms:
                    form_wf_id = f"{self._sanitize_wordform_id(form_written)}.{self.language}.written"
                    self.wordforms[form_written_key] = form_wf_id
                    self.wordform_ids.add(form_wf_id)

                form_written_id = self.wordforms[form_written_key]

                # Only add to entry_written_forms if it's different from Lemma
                if form_written_id not in self.entry_written_forms[entry_id]:
                    self.entry_written_forms[entry_id].append(form_written_id)

                # Process pronunciations for this Form
                form_spoken_ids = []
                for pron in form_elem.findall('Pronunciation'):
                    if pron.text:
                        pron_form = pron.text.strip()
                        variety = pron.get('variety', '')

                        spoken_key = (pron_form, self.language, 'spoken', variety)
                        if spoken_key not in self.wordforms:
                            # Use romanized pronunciation in ID
                            if variety:
                                spoken_id = f"{self._sanitize_wordform_id(pron_form)}.{self.language}-{variety}.spoken"
                            else:
                                spoken_id = f"{self._sanitize_wordform_id(pron_form)}.{self.language}.spoken"
                            self.wordforms[spoken_key] = spoken_id
                            self.wordform_ids.add(spoken_id)

                            # Create pronunciation_of and orthography_of relations
                            self.wordform_relations.append(
                                ('pronunciation_of', spoken_id, form_written_id)
                            )
                            self.wordform_relations.append(
                                ('orthography_of', form_written_id, spoken_id)
                            )

                        spoken_id = self.wordforms[spoken_key]
                        form_spoken_ids.append(spoken_id)

                # Track Form's spoken wordforms
                if form_spoken_ids:
                    self.form_to_spoken[form_written_id] = form_spoken_ids

    def _create_senses(self, root):
        """Create Senses linking wordforms to concepts."""
        for entry in root.findall('.//LexicalEntry'):
            entry_id = entry.get('id')
            lemma = entry.find('Lemma')
            if lemma is None:
                continue

            lemma_pos = lemma.get('partOfSpeech')

            # Get all written wordforms for this entry
            all_written_wf_ids = self.entry_written_forms.get(entry_id, [])

            # Get all spoken wordforms (lemma + all forms)
            all_spoken_wf_ids = self.lemma_spoken_forms.get(entry_id, []).copy()
            for form_wf_id in all_written_wf_ids:
                if form_wf_id in self.form_to_spoken:
                    all_spoken_wf_ids.extend(self.form_to_spoken[form_wf_id])

            for sense in entry.findall('Sense'):
                old_sense_id = sense.get('id')
                synset_id = sense.get('synset')

                if synset_id not in self.concepts:
                    print(f"  Warning: Sense {old_sense_id} references unknown synset {synset_id}")
                    continue

                # Validate/update concept POS
                if not self.concepts[synset_id]:
                    if lemma_pos:
                        print(f"Warning: Concept {synset_id} has empty POS, setting to lemma pos")
                        self.concepts[synset_id] = self._wn_to_ud_pos(lemma_pos)
                        self.missing_pos_records.append({
                            'synset_id': synset_id,
                            'assigned_pos': self._wn_to_ud_pos(lemma_pos),
                            'source': 'lemma'
                        })
                    else:
                        print(f"Warning: Concept {synset_id} has empty POS, setting to unknown")
                        self.concepts[synset_id] = self._wn_to_ud_pos('u')
                        self.missing_pos_records.append({
                            'synset_id': synset_id,
                            'assigned_pos': 'X',
                            'source': 'unknown'
                        })
                elif self.synset_pos[synset_id] != lemma_pos:  # Compare using original WN POS
                    self.pos_mismatches[(self.synset_pos[synset_id], lemma_pos)] += 1

                new_sense_ids = []

                # Create senses for all written wordforms
                for written_wf_id in all_written_wf_ids:
                    written_sense_id = f"{old_sense_id}_written_{written_wf_id}"
                    self.senses[written_sense_id] = (written_wf_id, synset_id)
                    new_sense_ids.append(written_sense_id)

                # Create senses for all spoken wordforms
                for spoken_wf_id in all_spoken_wf_ids:
                    spoken_sense_id = f"{old_sense_id}_spoken_{spoken_wf_id}"
                    self.senses[spoken_sense_id] = (spoken_wf_id, synset_id)
                    new_sense_ids.append(spoken_sense_id)

                self.old_sense_to_new_senses[old_sense_id] = new_sense_ids

    def _create_wordform_variant_relations(self, root):
        """Create orthographic_variant and pronunciation_variant relations."""
        for entry in root.findall('.//LexicalEntry'):
            entry_id = entry.get('id')

            # Create orthographic_variant relations (bidirectional, all pairs)
            written_wf_ids = self.entry_written_forms.get(entry_id, [])
            if len(written_wf_ids) > 1:
                for i in range(len(written_wf_ids)):
                    for j in range(len(written_wf_ids)):
                        if i != j:
                            self.wordform_relations.append(
                                ('orthographic_variant', written_wf_ids[i], written_wf_ids[j])
                            )

            # Create pronunciation_variant relations for Lemma pronunciations
            lemma_spoken_ids = self.lemma_spoken_forms.get(entry_id, [])
            if len(lemma_spoken_ids) > 1:
                for i in range(len(lemma_spoken_ids)):
                    for j in range(len(lemma_spoken_ids)):
                        if i != j:
                            self.wordform_relations.append(
                                ('pronunciation_variant', lemma_spoken_ids[i], lemma_spoken_ids[j])
                            )

            # Create pronunciation_variant relations for each Form's pronunciations
            for form_wf_id in written_wf_ids:
                if form_wf_id in self.form_to_spoken:
                    form_spoken_ids = self.form_to_spoken[form_wf_id]
                    if len(form_spoken_ids) > 1:
                        for i in range(len(form_spoken_ids)):
                            for j in range(len(form_spoken_ids)):
                                if i != j:
                                    self.wordform_relations.append(
                                        ('pronunciation_variant', form_spoken_ids[i], form_spoken_ids[j])
                                    )
    def _report_pos_mismatches(self):
        """Report statistics on POS mismatches."""
        if not self.pos_mismatches:
            print("  No POS mismatches found.")
            return

        total = sum(self.pos_mismatches.values())
        print(f"  Found {total} POS mismatches:")
        for (synset_pos, lemma_pos), count in sorted(self.pos_mismatches.items(),
                                                     key=lambda x: x[1], reverse=True):
            print(f"    synset={synset_pos}, lemma={lemma_pos}: {count} occurrences")

    def _generate_log_data(self):
        """Generate logging statistics."""
        # Count written vs spoken wordforms
        written_count = 0
        spoken_count = 0
        for key in self.wordforms.keys():
            if len(key) == 3:  # written: (form, lang, type)
                written_count += 1
            elif len(key) == 4:  # spoken: (form, lang, type, variety)
                spoken_count += 1

        # Format pos_mismatches for output
        pos_mismatch_list = []
        for (synset_pos, lemma_pos), count in self.pos_mismatches.items():
            pos_mismatch_list.append({
                'synset_pos': synset_pos,
                'lemma_pos': lemma_pos,
                'occurrences': count
            })

        log_data = {
            'processing_summary': {
                'concepts_created': len(self.concepts),
                'wordforms_created': {
                    'written': written_count,
                    'spoken': spoken_count,
                    'total': written_count + spoken_count
                },
                'senses_created': len(self.senses),
                'glosses_created': len(self.glosses),
                'relations_created': {
                    'wordform_relations': len(self.wordform_relations),
                    'sense_relations': len(self.sense_relations),
                    'concept_relations': len(self.concept_relations),
                    'total': len(self.wordform_relations) + len(self.sense_relations) + len(self.concept_relations)
                },
                'ili_mappings': {
                    'synsets_with_ili': len(self.synset_to_cili),
                    'synsets_without_ili': len(self.concepts) - len(self.synset_to_cili)
                }
            },
            'examples': {
                'total_found': self.total_examples_found,
                'processed': self.examples_processed,
                'skipped': self.examples_skipped
            },
            'pos_issues': {
                'missing_pos': {
                    'count': len(self.missing_pos_records),
                    'details': self.missing_pos_records
                },
                'pos_mismatches': {
                    'count': sum(self.pos_mismatches.values()),
                    'by_type': pos_mismatch_list
                }
            }
        }

        return log_data

    def _process_examples(self, root):
        """Process examples from Synsets and Senses."""
        example_counter = 1
        total_examples = 0

        # Count total examples first
        for synset in root.findall('.//Synset'):
            total_examples += len(synset.findall('Example'))
        for entry in root.findall('.//LexicalEntry'):
            for sense in entry.findall('Sense'):
                total_examples += len(sense.findall('Example'))

        self.total_examples_found = total_examples
        print(f"  Total examples to process: {total_examples}")

        # Process synset examples
        for synset in root.findall('.//Synset'):
            synset_id = synset.get('id')

            for example in synset.findall('Example'):

                if example.text:
                    self._process_single_example(
                        example.text.strip(),
                        synset_id,
                        f"ex_{example_counter}",
                        is_synset_example=True
                    )
                    example_counter += 1

                    # Progress logging every 1000 examples
                    if example_counter % 1000 == 0:
                        print(f"    Processed {example_counter}/{total_examples} examples "
                              f"(skipped: {self.examples_skipped})")

        # Process sense examples
        for entry in root.findall('.//LexicalEntry'):
            for sense in entry.findall('Sense'):
                old_sense_id = sense.get('id')

                for example in sense.findall('Example'):
                    if example.text:
                        self._process_single_example(
                            example.text.strip(),
                            None,
                            f"ex_{example_counter}",
                            sense_id=old_sense_id
                        )
                        example_counter += 1

                        # Progress logging every 100 examples
                        if example_counter % 100 == 0:
                            print(f"    Processed {example_counter}/{total_examples} examples "
                                  f"(skipped: {self.examples_skipped})")

    def _expand_to_token_boundaries(self, text, match_start, match_end):
        """Expand match to complete token boundaries."""
        # Stage 1: Check if already at boundaries (CHEAP)
        token_boundaries = {' ', '.', ',', '!', '?', ';', ':', '"', "'", "'", '"', '"',
                            '(', ')', '[', ']', '{', '}', '-', '—', '–',
                            '\n', '\t', '/', '\\'}

        at_start = (match_start == 0 or text[match_start - 1] in token_boundaries)
        at_end = (match_end >= len(text) or text[match_end] in token_boundaries)

        if at_start and at_end:
            return match_start, match_end

        # Stage 2: Use spaCy to find actual token boundaries (EXPENSIVE)
        doc = self._get_doc(text)  # Use the same cached instance

        # Find tokens that overlap with our match
        overlapping_tokens = []
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)

            # Check if token overlaps with match range
            if not (token_end <= match_start or token_start >= match_end):
                overlapping_tokens.append(token)

        if not overlapping_tokens:
            return match_start, match_end

        # Expand to full span of all overlapping tokens
        new_start = overlapping_tokens[0].idx
        last_token = overlapping_tokens[-1]
        new_end = last_token.idx + len(last_token.text)

        return new_start, new_end
    def _sanitize_wordform_id(self, raw_id):
        """Ensure wordform ID conforms to xs:ID rules and is unique."""
        # URL-encode style: replace each invalid char with _xHH_ where HH is hex
        encoded = self._encode_for_xml_id(raw_id)

        # Always prefix with w.
        sanitized = 'w.' + encoded

        return sanitized

    def _process_single_example(self, text, synset_id, example_id,
                                is_synset_example=False, sense_id=None):
        """Process a single example and create annotated token."""
        # Find the appropriate sense
        target_sense_id = None
        target_wordform = None

        if is_synset_example:
            # Find all written senses for this synset
            candidates = []
            for new_sense_id, (wf_id, concept_id) in self.senses.items():
                if concept_id == synset_id and '_written' in new_sense_id:
                    # Get wordform
                    for key, stored_wf_id in self.wordforms.items():
                        if stored_wf_id == wf_id and len(key) == 3 and key[2] == 'written':
                            form = key[0]
                            candidates.append((new_sense_id, form, wf_id))

            # Find best match in text
            target_sense_id, target_wordform = self._find_best_match(
                text, candidates
            )
        else:
            # Find the written sense for this old sense
            new_sense_ids = self.old_sense_to_new_senses.get(sense_id, [])
            candidates = []
            for new_sense_id in new_sense_ids:
                if '_written' in new_sense_id:
                    wf_id, _ = self.senses[new_sense_id]
                    # Get wordform
                    for key, stored_wf_id in self.wordforms.items():
                        if stored_wf_id == wf_id and len(key) == 3 and key[2] == 'written':
                            form = key[0]
                            candidates.append((new_sense_id, form, wf_id))

            target_sense_id, target_wordform = self._find_best_match(
                text, candidates
            )


        if not target_sense_id:
            if self.debug_examples:

                # Collect all candidate wordforms for debugging
                wordforms_list = [form for _, form, _ in candidates]
                print(f"  Warning: Could not find matching wordform in example: '{text[:80]}...'")
                print(f"    Possible wordforms: {wordforms_list}")

                # Debug: show all forms that were tried for each candidate
                if candidates:
                    print(f"    Detailed form matching:")
                    for _, form, _ in candidates:
                        all_forms = self._get_all_forms(form)
                        print(f"      '{form}' -> {all_forms}")

                # Debug: show what tokens were found in the text
                doc = self._get_doc(text)
                print(f"    Tokens in text:")
                for token in doc:
                    token_forms = self._get_all_forms(token.text)
                    print(f"      '{token.text}' -> {token_forms}")

            self.examples_skipped += 1
            return

        # Create example with annotated token
        self._create_example_with_annotation(
            text, target_wordform, target_sense_id, example_id
        )
        self.examples_processed += 1

    def _find_best_match(self, text, candidates):
        """Find the best matching wordform in the text."""
        text_lower = text.lower()

        # Sort candidates by length (longest first) to prefer longer matches
        candidates_sorted = sorted(candidates, key=lambda x: len(x[1]), reverse=True)

        # Strategy 1: Try exact match first (CHEAP - no NLP processing)
        for sense_id, form, wf_id in candidates_sorted:
            if form.lower() in text_lower:
                return sense_id, form

        # Strategy 2: Try simple lowercase token matching (CHEAP - just splitting)
        # This catches many cases without expensive lemmatization
        text_tokens = set(text_lower.split())
        for sense_id, form, wf_id in candidates_sorted:
            form_words = form.lower().split()

            if len(form_words) == 1:
                # Single word - check if it's in the token set
                if form_words[0] in text_tokens:
                    return sense_id, form
            else:
                # Multi-word - check if all words are present (order doesn't matter for this quick check)
                if all(word in text_tokens for word in form_words):
                    # Do a more precise check to ensure they're reasonably close
                    text_words = text_lower.split()
                    for i in range(len(text_words)):
                        match_count = 0
                        for j in range(i, min(i + len(form_words) + 2, len(text_words))):
                            if match_count < len(form_words) and text_words[j] == form_words[match_count]:
                                match_count += 1
                        if match_count == len(form_words):
                            return sense_id, form

        # Strategy 3: Try lemmatization matching with spaCy (EXPENSIVE - full NLP pipeline)
        for sense_id, form, wf_id in candidates_sorted:
            form_words = form.split()

            if len(form_words) == 1:
                # Single word matching
                result = self._match_single_word_spacy(form, text)
                if result:
                    return sense_id, result
            else:
                # Multi-word expression matching
                result = self._match_mwe_spacy(form_words, text)
                if result:
                    return sense_id, result

        return None, None

    def _get_all_forms(self, word):
        """Get all possible forms/lemmas of a word using multiple libraries."""
        word_lower = word.lower()
        if word_lower in self.form_cache:
            return self.form_cache[word_lower]

        forms = set()
        forms.add(word_lower)

        # Call spaCy ONCE and reuse
        doc = self._get_doc(word_lower)
        if len(doc) > 0:
            token = doc[0]

            # 1. spaCy lemmatization
            forms.add(token.lemma_)

            # 3. pyinflect (reuse same token)
            if pyinflect is not None:
                for pos_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                                'NN', 'NNS',  # Nouns
                                'JJ', 'JJR', 'JJS',  # Adjectives
                                'RB', 'RBR', 'RBS']:  # Adverbs
                    inflection = token._.inflect(pos_tag)
                    if inflection:
                        forms.add(inflection.lower())

        # 2. NLTK WordNet lemmatization (all POS)
        if self.nltk_lemmatizer is not None:
            for pos in ['n', 'v', 'a', 'r']:
                lemma = self.nltk_lemmatizer.lemmatize(word_lower, pos=pos)
                forms.add(lemma)

        self.form_cache[word_lower] = forms
        return forms

    def _match_single_word_spacy(self, wordform, text):
        """Match a single word using comprehensive form matching."""
        # Get all possible forms of the wordform
        wordform_forms = self._get_all_forms(wordform)

        doc = self._get_doc(text)

        # Check each token in the document
        for token in doc:
            # Get all possible forms of the token
            token_forms = self._get_all_forms(token.text)

            # Check if there's any overlap
            if wordform_forms & token_forms:
                return token.text.lower()

        return None

    def _match_mwe_spacy(self, form_words, text):
        """Match a multi-word expression using comprehensive form matching."""
        # Identify content words (not function words)
        function_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from'}

        # Get all forms for content words in the MWE
        content_word_forms = []
        for word in form_words:
            word_lower = word.lower()
            if word_lower not in function_words:
                forms = self._get_all_forms(word_lower)
                content_word_forms.append(forms)

        # Look for matching sequences in the document
        doc = self._get_doc(text)
        tokens = list(doc)

        for i in range(len(tokens)):
            # Try to match starting from position i
            content_match_idx = 0
            matched_token_indices = []

            for j in range(i, min(i + len(form_words) + 2, len(tokens))):
                if content_match_idx >= len(content_word_forms):
                    break

                token = tokens[j]
                target_forms = content_word_forms[content_match_idx]

                # Get all forms of this token
                token_forms = self._get_all_forms(token.text)

                # Check if there's any overlap
                if target_forms & token_forms:
                    matched_token_indices.append(j)
                    content_match_idx += 1

            # If we matched all content words
            if content_match_idx == len(content_word_forms):
                # Extract the span from first to last matched token
                start_idx = matched_token_indices[0]
                end_idx = matched_token_indices[-1]

                start_token = tokens[start_idx]
                end_token = tokens[end_idx]

                # Get the span from the original text
                start_pos = start_token.idx
                end_pos = end_token.idx + len(end_token.text)

                return text[start_pos:end_pos].lower()

        return None

    def _create_example_with_annotation(self, text, wordform, sense_id, example_id):
        """Create an Example element with AnnotatedToken."""
        text_lower = text.lower()
        wordform_lower = wordform.lower()

        pos = text_lower.find(wordform_lower)
        if pos == -1:
            return

        # Expand to token boundaries - THIS IS WHERE IT'S CALLED
        match_end = pos + len(wordform)
        expanded_start, expanded_end = self._expand_to_token_boundaries(
            text_lower, pos, match_end
        )

        # Store as tuple: (id, full_text, token_start, token_end, sense_id)
        self.examples.append((
            example_id,
            text_lower,
            expanded_start,
            expanded_end,
            sense_id
        ))

    def _process_relations(self, root):
        """Process SynsetRelations and SenseRelations."""
        # Process synset relations -> concept relations
        for synset in root.findall('.//Synset'):
            source_synset = synset.get('id')

            for rel in synset.findall('SynsetRelation'):
                target_synset = rel.get('target')
                rel_type = rel.get('relType')

                if source_synset in self.concepts and target_synset in self.concepts:
                    self.concept_relations.append(
                        (rel_type, source_synset, target_synset)
                    )

        # Process sense relations -> sense relations (Cartesian product)
        for entry in root.findall('.//LexicalEntry'):
            for sense in entry.findall('Sense'):
                old_source_id = sense.get('id')

                for rel in sense.findall('SenseRelation'):
                    old_target_id = rel.get('target')
                    rel_type = rel.get('relType')

                    # Get all new sense variants for source and target
                    source_senses = self.old_sense_to_new_senses.get(old_source_id, [])
                    target_senses = self.old_sense_to_new_senses.get(old_target_id, [])

                    # Create Cartesian product: all combinations
                    for source_sense in source_senses:
                        for target_sense in target_senses:
                            if source_sense in self.senses and target_sense in self.senses:
                                self.sense_relations.append(
                                    (rel_type, source_sense, target_sense)
                                )
    def _validate(self, xml_doc, xsd_file):

        print('Checking validity...')

        # Load the XSD schema
        with open(xsd_file, 'rb') as f:
            schema_root = etree.XML(f.read())
            schema = etree.XMLSchema(schema_root)

        # Validate
        try:
            schema.assertValid(xml_doc)
            print("XML is valid!")
        except etree.DocumentInvalid as e:
            print(f"XML is invalid.")


    def _write_output(self, output_file, xsd_file=None):
        """Write the Cygnet XML output."""
        # Create root element with metadata from source
        root = etree.Element('CygnetResource', {
            'id': self.lexicon_id,
            'label': self.lexicon_label,
            'version': self.lexicon_version
        })

        # Add ConceptLayer
        concept_layer = etree.SubElement(root, 'ConceptLayer')
        for concept_id, pos in self.concepts.items():
            if pos:  # Only add if we have a POS
                etree.SubElement(concept_layer, 'Concept', {
                    'id': concept_id,
                    'pos': pos
                })

        # Add WordformLayer
        wordform_layer = etree.SubElement(root, 'WordformLayer')
        # Group by type for easier processing
        written_forms = []
        spoken_forms = []

        for key, wf_id in self.wordforms.items():
            if len(key) == 3:  # written: (form, lang, type)
                form, lang, wf_type = key
                written_forms.append((wf_id, form, lang))
            elif len(key) == 4:  # spoken: (form, lang, type, variety)
                form, lang, wf_type, variety = key
                spoken_forms.append((wf_id, form, lang, variety))

        for wf_id, form, lang in written_forms:
            etree.SubElement(wordform_layer, 'WrittenWordform', {
                'id': wf_id,
                'form': form,
                'language': lang
            })

        for wf_id, form, lang, variety in spoken_forms:
            attrs = {
                'id': wf_id,
                'form': form,
                'language': lang
            }
            if variety:
                attrs['variety'] = variety
            etree.SubElement(wordform_layer, 'SpokenWordform', attrs)

        # Add SenseLayer
        sense_layer = etree.SubElement(root, 'SenseLayer')
        for sense_id, (signifier, signified) in self.senses.items():
            etree.SubElement(sense_layer, 'Sense', {
                'id': sense_id,
                'signifier': signifier,
                'signified': signified
            })

        # Add GlossLayer
        gloss_layer = etree.SubElement(root, 'GlossLayer')
        for concept_id, definition in self.glosses.items():
            etree.SubElement(gloss_layer, 'Gloss', {
                'definiendum': concept_id,
                'definition': definition
            })

        # Add ExampleLayer
        if self.examples:
            example_layer = etree.SubElement(root, 'ExampleLayer')

            for example_id, full_text, token_start, token_end, sense_id in self.examples:
                example_elem = etree.SubElement(example_layer, 'Example', {'id': example_id})

                # Add text before token
                if token_start > 0:
                    example_elem.text = full_text[:token_start]

                # Create AnnotatedToken
                annotated_token = etree.SubElement(example_elem, 'AnnotatedToken', {'sense': sense_id})
                annotated_token.text = full_text[token_start:token_end]

                # Add text after token
                if token_end < len(full_text):
                    annotated_token.tail = full_text[token_end:]

        # Add WordformRelationLayer
        if self.wordform_relations:
            wf_rel_layer = etree.SubElement(root, 'WordformRelationLayer')
            for rel_type, source, target in self.wordform_relations:
                etree.SubElement(wf_rel_layer, 'WordformRelation', {
                    'relationType': rel_type,
                    'source': source,
                    'target': target
                })

        # Add SenseRelationLayer
        if self.sense_relations:
            sense_rel_layer = etree.SubElement(root, 'SenseRelationLayer')
            for rel_type, source, target in self.sense_relations:
                etree.SubElement(sense_rel_layer, 'SenseRelation', {
                    'relationType': rel_type,
                    'source': source,
                    'target': target
                })

        # Add ConceptRelationLayer
        if self.concept_relations:
            concept_rel_layer = etree.SubElement(root, 'ConceptRelationLayer')
            for rel_type, source, target in self.concept_relations:
                etree.SubElement(concept_rel_layer, 'ConceptRelation', {
                    'relationType': rel_type,
                    'source': source,
                    'target': target
                })

        # Write with lxml's pretty_print - it preserves inline mixed content!
        tree = etree.ElementTree(root)
        tree.write(output_file,
                   encoding='utf-8',
                   xml_declaration=True,
                   pretty_print=True)

        # Validate if needed
        if xsd_file is not None:
            self._validate(tree, xsd_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert GlobalWordNet XML format to Cygnet XML format.')
    parser.add_argument('input_file', help='Input GWN XML file')
    parser.add_argument('output_file', help='Output Cygnet XML file')
    parser.add_argument('--xsd-file', default=None, help='XSD file path (optional)')
    parser.add_argument('--skip-examples', action='store_true',
                        help='Skip processing the example layer')
    parser.add_argument('--debug-examples', action='store_true',
                        help='Print detailed error messages for individual examples')

    args = parser.parse_args()

    converter = GWNToCygnetConverter(debug_examples=args.debug_examples)
    converter.convert(args.input_file, args.output_file, skip_examples=args.skip_examples, xsd_file=args.xsd_file)


if __name__ == '__main__':
    main()