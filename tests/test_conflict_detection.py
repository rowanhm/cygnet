"""Tests for duplicate/reversed relation conflict detection in MergeBuilder."""

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

# Import MergeBuilder from the conversion script
_script = Path(__file__).parent.parent / 'conversion_scripts' / '6_synthesise.py'
_spec = importlib.util.spec_from_file_location('synthesise', _script)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['synthesise'] = _mod
_spec.loader.exec_module(_mod)
MergeBuilder = _mod.MergeBuilder


def _wn_xml(wn_id: str, body: str, language: str = 'en', version: str = '1.0') -> str:
    """Wrap body XML in a CygnetResource root element."""
    inner = textwrap.indent(textwrap.dedent(body).strip(), '  ')
    return (
        f"<?xml version='1.0' encoding='UTF-8'?>\n"
        f'<CygnetResource id="{wn_id}" label="Test {wn_id}"'
        f' language="{language}" version="{version}">\n'
        f'{inner}\n'
        f'</CygnetResource>\n'
    )


# ---------------------------------------------------------------------------
# Shared XML snippets
# ---------------------------------------------------------------------------

_BASE_CONCEPTS = """\
<Concept id="cili.i1" ontological_category="NOUN" status="1">
  <Provenance resource="wn-a" version="1.0" original_id="t-1"/>
</Concept>
<Concept id="cili.i2" ontological_category="NOUN" status="1">
  <Provenance resource="wn-a" version="1.0" original_id="t-2"/>
</Concept>
<Gloss definiendum="cili.i1" language="en">
  <AnnotatedSentence>dog</AnnotatedSentence>
  <Provenance resource="wn-a" version="1.0"/>
</Gloss>
<Gloss definiendum="cili.i2" language="en">
  <AnnotatedSentence>animal</AnnotatedSentence>
  <Provenance resource="wn-a" version="1.0"/>
</Gloss>
"""

_BASE_LEXEMES_AND_SENSES = """\
<Lexeme id="en.NOUN.dog" language="en" grammatical_category="NOUN">
  <Wordform form="dog"/>
  <Provenance resource="wn-a" version="1.0" original_id="dog-n"/>
</Lexeme>
<Lexeme id="en.NOUN.animal" language="en" grammatical_category="NOUN">
  <Wordform form="animal"/>
  <Provenance resource="wn-a" version="1.0" original_id="animal-n"/>
</Lexeme>
<Sense id="sense.dog" signifier="en.NOUN.dog" signified="cili.i1">
  <Provenance resource="wn-a" version="1.0"/>
</Sense>
<Sense id="sense.animal" signifier="en.NOUN.animal" signified="cili.i2">
  <Provenance resource="wn-a" version="1.0"/>
</Sense>
"""


@pytest.fixture()
def builder(tmp_path):
    """MergeBuilder with temporary DB paths."""
    db = tmp_path / 'cygnet.db'
    prov = tmp_path / 'provenance.db'
    return MergeBuilder(db, prov)


# ---------------------------------------------------------------------------
# Synset-relation conflict tests
# ---------------------------------------------------------------------------

class TestSynsetRelationConflict:
    def test_reversed_hypernym_is_skipped(self, builder, tmp_path, caplog):
        """
        wn-a adds hypernym(dog→animal), which auto-generates hyponym(animal→dog).
        wn-b then explicitly adds hypernym(animal→dog) — reversed — which must be
        skipped with a warning.
        """
        wn_a = tmp_path / 'wn-a.xml'
        wn_a.write_text(_wn_xml('wn-a', _BASE_CONCEPTS + _BASE_LEXEMES_AND_SENSES + """\
<ConceptRelation relation_type="class_hypernym" source="cili.i1" target="cili.i2">
  <Provenance resource="wn-a" version="1.0"/>
</ConceptRelation>
"""))

        wn_b = tmp_path / 'wn-b.xml'
        wn_b.write_text(_wn_xml('wn-b', """\
<ConceptRelation relation_type="class_hypernym" source="cili.i2" target="cili.i1">
  <Provenance resource="wn-b" version="1.0"/>
</ConceptRelation>
"""))

        import logging
        with caplog.at_level(logging.WARNING, logger='synthesise'):
            builder.process_file(wn_a)
            builder.process_file(wn_b)

        assert builder.n_rel_conflicts == 1
        assert 'wn-b' in caplog.text
        assert 'class_hypernym' in caplog.text
        assert 'cili.i1' in caplog.text
        assert 'cili.i2' in caplog.text

    def test_reversed_hypernym_relation_not_stored(self, builder, tmp_path):
        """The conflicting reversed relation must not be persisted to the DB."""
        wn_a = tmp_path / 'wn-a.xml'
        wn_a.write_text(_wn_xml('wn-a', _BASE_CONCEPTS + _BASE_LEXEMES_AND_SENSES + """\
<ConceptRelation relation_type="class_hypernym" source="cili.i1" target="cili.i2">
  <Provenance resource="wn-a" version="1.0"/>
</ConceptRelation>
"""))
        wn_b = tmp_path / 'wn-b.xml'
        wn_b.write_text(_wn_xml('wn-b', """\
<ConceptRelation relation_type="class_hypernym" source="cili.i2" target="cili.i1">
  <Provenance resource="wn-b" version="1.0"/>
</ConceptRelation>
"""))

        builder.process_file(wn_a)
        builder.process_file(wn_b)

        rows = builder.cur.execute(
            'SELECT source_rowid, target_rowid FROM synset_relations'
        ).fetchall()
        # Expect exactly two rows: the forward hypernym and the auto-inverse hyponym
        assert len(rows) == 2

    def test_symmetric_relation_not_flagged(self, builder, tmp_path, caplog):
        """
        Symmetric relations (opposite, antonym at concept level) with both
        directions explicitly asserted should NOT trigger a conflict warning.
        """
        wn_a = tmp_path / 'wn-a.xml'
        wn_a.write_text(_wn_xml('wn-a', _BASE_CONCEPTS + _BASE_LEXEMES_AND_SENSES + """\
<ConceptRelation relation_type="opposite" source="cili.i1" target="cili.i2">
  <Provenance resource="wn-a" version="1.0"/>
</ConceptRelation>
"""))
        wn_b = tmp_path / 'wn-b.xml'
        wn_b.write_text(_wn_xml('wn-b', """\
<ConceptRelation relation_type="opposite" source="cili.i2" target="cili.i1">
  <Provenance resource="wn-b" version="1.0"/>
</ConceptRelation>
"""))

        import logging
        with caplog.at_level(logging.WARNING, logger='synthesise'):
            builder.process_file(wn_a)
            builder.process_file(wn_b)

        assert builder.n_rel_conflicts == 0

    def test_no_conflict_on_normal_relations(self, builder, tmp_path, caplog):
        """Single-source wordnet with correct hypernym should produce zero conflicts."""
        wn_a = tmp_path / 'wn-a.xml'
        wn_a.write_text(_wn_xml('wn-a', _BASE_CONCEPTS + _BASE_LEXEMES_AND_SENSES + """\
<ConceptRelation relation_type="class_hypernym" source="cili.i1" target="cili.i2">
  <Provenance resource="wn-a" version="1.0"/>
</ConceptRelation>
"""))

        import logging
        with caplog.at_level(logging.WARNING, logger='synthesise'):
            builder.process_file(wn_a)

        assert builder.n_rel_conflicts == 0


# ---------------------------------------------------------------------------
# Sense-relation conflict tests
# ---------------------------------------------------------------------------

_BASE_CONCEPTS_SENSE_REL = """\
<Concept id="cili.i3" ontological_category="VERB" status="1">
  <Provenance resource="wn-a" version="1.0" original_id="t-3"/>
</Concept>
<Concept id="cili.i4" ontological_category="NOUN" status="1">
  <Provenance resource="wn-a" version="1.0" original_id="t-4"/>
</Concept>
<Gloss definiendum="cili.i3" language="en">
  <AnnotatedSentence>run</AnnotatedSentence>
  <Provenance resource="wn-a" version="1.0"/>
</Gloss>
<Gloss definiendum="cili.i4" language="en">
  <AnnotatedSentence>running</AnnotatedSentence>
  <Provenance resource="wn-a" version="1.0"/>
</Gloss>
<Lexeme id="en.VERB.run" language="en" grammatical_category="VERB">
  <Wordform form="run"/>
  <Provenance resource="wn-a" version="1.0" original_id="run-v"/>
</Lexeme>
<Lexeme id="en.NOUN.running" language="en" grammatical_category="NOUN">
  <Wordform form="running"/>
  <Provenance resource="wn-a" version="1.0" original_id="running-n"/>
</Lexeme>
<Sense id="sense.run" signifier="en.VERB.run" signified="cili.i3">
  <Provenance resource="wn-a" version="1.0"/>
</Sense>
<Sense id="sense.running" signifier="en.NOUN.running" signified="cili.i4">
  <Provenance resource="wn-a" version="1.0"/>
</Sense>
"""


class TestSenseRelationConflict:
    def test_reversed_derivation_is_skipped(self, builder, tmp_path, caplog):
        """
        wn-a adds derivation(run→running), auto-generating derivation_of(running→run).
        wn-b explicitly adds derivation(running→run), which conflicts and must be skipped.
        """
        wn_a = tmp_path / 'wn-a.xml'
        wn_a.write_text(_wn_xml('wn-a', _BASE_CONCEPTS_SENSE_REL + """\
<SenseRelation relation_type="derivation" source="sense.run" target="sense.running">
  <Provenance resource="wn-a" version="1.0"/>
</SenseRelation>
"""))
        wn_b = tmp_path / 'wn-b.xml'
        wn_b.write_text(_wn_xml('wn-b', """\
<SenseRelation relation_type="derivation" source="sense.running" target="sense.run">
  <Provenance resource="wn-b" version="1.0"/>
</SenseRelation>
"""))

        import logging
        with caplog.at_level(logging.WARNING, logger='synthesise'):
            builder.process_file(wn_a)
            builder.process_file(wn_b)

        assert builder.n_rel_conflicts == 1
        assert 'wn-b' in caplog.text
        assert 'derivation' in caplog.text

    def test_reversed_derivation_not_stored(self, builder, tmp_path):
        """The conflicting reversed sense relation must not be persisted."""
        wn_a = tmp_path / 'wn-a.xml'
        wn_a.write_text(_wn_xml('wn-a', _BASE_CONCEPTS_SENSE_REL + """\
<SenseRelation relation_type="derivation" source="sense.run" target="sense.running">
  <Provenance resource="wn-a" version="1.0"/>
</SenseRelation>
"""))
        wn_b = tmp_path / 'wn-b.xml'
        wn_b.write_text(_wn_xml('wn-b', """\
<SenseRelation relation_type="derivation" source="sense.running" target="sense.run">
  <Provenance resource="wn-b" version="1.0"/>
</SenseRelation>
"""))

        builder.process_file(wn_a)
        builder.process_file(wn_b)

        rows = builder.cur.execute(
            'SELECT source_rowid, target_rowid FROM sense_relations'
        ).fetchall()
        # derivation(run→running) + auto derivation_of(running→run) = 2 rows
        assert len(rows) == 2

    def test_symmetric_antonym_not_flagged(self, builder, tmp_path, caplog):
        """
        antonym is symmetric (inv == rel_type); both directions being explicitly
        asserted should not trigger a conflict warning.
        """
        wn_a = tmp_path / 'wn-a.xml'
        wn_a.write_text(_wn_xml('wn-a', _BASE_CONCEPTS_SENSE_REL + """\
<SenseRelation relation_type="antonym" source="sense.run" target="sense.running">
  <Provenance resource="wn-a" version="1.0"/>
</SenseRelation>
"""))
        wn_b = tmp_path / 'wn-b.xml'
        wn_b.write_text(_wn_xml('wn-b', """\
<SenseRelation relation_type="antonym" source="sense.running" target="sense.run">
  <Provenance resource="wn-b" version="1.0"/>
</SenseRelation>
"""))

        import logging
        with caplog.at_level(logging.WARNING, logger='synthesise'):
            builder.process_file(wn_a)
            builder.process_file(wn_b)

        assert builder.n_rel_conflicts == 0
