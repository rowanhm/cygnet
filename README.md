# Cygnet

A fast Python library for browsing Cygnet lexical resources, inspired by NLTK's WordNet interface.

## Features

- **Fast**: Uses lxml for efficient XML parsing and comprehensive indexing for O(1) lookups
- **Intuitive API**: Similar to NLTK WordNet for easy adoption
- **Type-safe**: Full type hints for better IDE support
- **Comprehensive**: Access concepts, senses, wordforms, examples, and relations

## Installation

### Local Development

```bash
pip install -e .
```

### From PyPI (when published)

```bash
pip install cygnet
```

## Quick Start

```python
import cygnet

# Load a Cygnet lexical resource
cyg = cygnet.Cygnet("path/to/cygnet_resource.xml")

# Access resource metadata
print(cyg.id, cyg.label, cyg.version)

# Query senses by wordform
senses = cyg.senses(wordform="bank", lang="en")
for sense in senses:
    print(f"{sense.form()} -> {sense.concept().definition()}")

# Get a concept by ID and explore it
concept = cyg.concept("c_12345")
print(f"POS: {concept.pos()}")
print(f"Definition: {concept.definition()}")

# Get all senses for a concept
for sense in concept.senses(lang="en"):
    print(f"  - {sense.form()}")

# Explore relations
for related_concept in concept.relations(relation_type="hypernym"):
    print(f"Hypernym: {related_concept.definition()}")

# View usage examples
sense = cyg.sense("s_67890")
for example in sense.examples():
    print(example.format())  # Highlights annotated tokens
```

## API Reference

### Cygnet Class

Main entry point for loading and querying lexical resources.

**Initialization:**
```python
cyg = cygnet.Cygnet("path/to/file.xml")
```

**Factory Methods:**
- `cyg.concept(concept_id)` - Get Concept by ID
- `cyg.sense(sense_id)` - Get Sense by ID
- `cyg.wordform(wordform_id)` - Get Wordform by ID
- `cyg.example(example_id)` - Get Example by ID

**Query Methods:**
- `cyg.concepts()` - Get all concepts
- `cyg.senses(wordform=None, lang=None, variety=None, type=None)` - Query senses with filters

### Concept Class

Represents an abstract concept.

**Methods:**
- `concept.id()` - Get concept ID
- `concept.pos()` - Get part of speech
- `concept.definition()` - Get definition/gloss
- `concept.senses(lang=None, variety=None, type=None)` - Get all senses for this concept
- `concept.relations(relation_type=None)` - Get related concepts
- `concept.relation_pairs(relation_type=None)` - Get (relation_type, concept) pairs

### Sense Class

Represents a word sense linking a wordform to a concept.

**Methods:**
- `sense.id()` - Get sense ID
- `sense.wordform()` - Get Wordform object
- `sense.concept()` - Get Concept object
- `sense.examples()` - Get usage examples
- `sense.relations(relation_type=None)` - Get related senses

**Convenience methods (delegate to wordform):**
- `sense.form()` - Get wordform string
- `sense.type()` - Get type ('written' or 'spoken')
- `sense.language()` - Get language code
- `sense.variety()` - Get variety (for spoken forms)

### Wordform Class

Represents a written or spoken wordform.

**Methods:**
- `wordform.id()` - Get wordform ID
- `wordform.form()` - Get wordform string
- `wordform.language()` - Get language code
- `wordform.type()` - Get type ('written' or 'spoken')
- `wordform.variety()` - Get variety (for spoken forms)

### Example Class

Represents a usage example with annotated tokens.

**Methods:**
- `example.id()` - Get example ID
- `example.text()` - Get plain text
- `example.tokens()` - Get list of AnnotatedToken objects
- `example.annotated_pairs()` - Get (text, Sense) pairs
- `example.senses()` - Get all referenced senses
- `example.format(highlight_char='**')` - Get formatted text with highlighted annotations

## Performance

The library is optimized for speed:

- **Fast initialization**: XML parsed once with lxml
- **O(1) lookups**: Comprehensive indexing for all ID-based access
- **Efficient filtering**: Multi-key indexes for common query patterns
- **Object caching**: Created objects are cached to avoid redundant instantiation

Benchmark on a typical lexical resource (10,000 concepts):
- Load time: ~0.5 seconds
- Concept lookup: ~0.0001 seconds
- Query with filters: ~0.01 seconds

## Project Structure

```
cygnet/
├── __init__.py          # Package initialization
├── cygnet.py            # Main Cygnet class
├── concept.py           # Concept class
├── sense.py             # Sense class
├── wordform.py          # Wordform class
└── example.py           # Example and AnnotatedToken classes
```

## Development

### Running Tests

```bash
pytest tests/
```

### Type Checking

```bash
mypy cygnet/
```

### Formatting

```bash
black cygnet/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.