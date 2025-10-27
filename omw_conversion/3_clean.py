# !/usr/bin/env python3
"""
Cygnet XML Sanity Checker and Corrector
Performs validation and corrections on Cygnet lexical resource XML files.
"""

from lxml import etree
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional


sense_relation_pairs = {
    # Princeton WordNet Relations

    # Antonym and similar relations (symmetric)
    'antonym': 'antonym',  # An opposite and inherently incompatible word
    'also': 'also',  # See also, a reference of weak meaning
    'similar': 'similar',  # Similar, though not necessarily interchangeable

    # Derivation (symmetric)
    'derivation': 'derivation',  # A word that is derived from some other word

    # Domain topic pairs
    'domain_topic': 'domain_member_topic',  # Indicates the category of this word
    'domain_member_topic': 'domain_topic',  # Indicates a word involved in this category described by this word

    # Domain region pairs
    'domain_region': 'domain_member_region',  # Indicates the region of this word
    'domain_member_region': 'domain_region',  # Indicates a word involved in the region described by this word

    # Exemplification pairs
    'exemplifies': 'is_exemplified_by',  # Indicates the usage of this word
    'is_exemplified_by': 'exemplifies',  # Indicates a word involved in the usage described by this word

    # Unpaired Princeton WordNet relations
    'participle': None,  # An adjective that is a participle form a verb
    'pertainym': None,
    # A relational adjective. Adjectives that are pertainyms are usually defined by such phrases as "of or pertaining to" and do not have antonyms

    # Morphosemantic relations (all unpaired)
    'agent': None,  # A word which is typically the one/that who/which does the action denoted by a given word
    'material': None,  # A word which is typically the material of a given word
    'event': None,  # An noun representing the event of a verb
    'instrument': None,  # An instrument for doing a task
    'location': None,  # A verb derived from the action performed at a place
    'by_means_of': None,  # A word which is typically the means by which something is done
    'undergoer': None,  # A word which is typically the undergoer of a given word
    'property': None,  # Cause something to have a particular property
    'result': None,  # A word which is typically the result of a given word
    'state': None,  # A state caused by the verb
    'uses': None,  # A verb that uses a noun
    'destination': None,  # The noun indicates the destination of a verb
    'body_part': None,  # A word which is typically a body part of a given word
    'vehicle': None,  # A verb indicating movement with a particular vehicle

    # Non-Princeton WordNet Relations

    # Aspect pairs - simple
    'simple_aspect_ip': 'simple_aspect_pi',
    # A word which is linked to another through a change from imperfective to perfective aspect
    'simple_aspect_pi': 'simple_aspect_ip',
    # A word which is linked to another through a change from perfective to imperfective aspect

    # Aspect pairs - secondary
    'secondary_aspect_ip': 'secondary_aspect_pi',  # A word which is linked to another through a change in aspect (ip)
    'secondary_aspect_pi': 'secondary_aspect_ip',  # A word which is linked to another through a change in aspect (pi)

    # Gender/age/size relations - feminine
    'feminine': 'has_feminine',  # A feminine form of a word
    'has_feminine': 'feminine',  # Indicates the base form of a word with a feminine derivation

    # Gender/age/size relations - masculine
    'masculine': 'has_masculine',  # A masculine form of a word
    'has_masculine': 'masculine',  # Indicates the base form of a word with a masculine derivation

    # Gender/age/size relations - young
    'young': 'has_young',  # A form of a word with a derivation indicating the young of a species
    'has_young': 'young',  # Indicates the base form of a word with a young derivation

    # Gender/age/size relations - diminutive
    'diminutive': 'has_diminutive',  # A diminutive form of a word
    'has_diminutive': 'diminutive',  # Indicates the base form of a word with a diminutive derivation

    # Gender/age/size relations - augmentative
    'augmentative': 'has_augmentative',  # An augmentative form of a word
    'has_augmentative': 'augmentative',  # Indicates the base form of a word with an augmentative derivation

    # Antonym subtypes (symmetric)
    'anto_gradable': 'anto_gradable',  # A word pair whose meanings are opposite and which lie on a continuous spectrum
    'anto_simple': 'anto_simple',
    # A word pair whose meanings are opposite but whose meanings do not lie on a continuous spectrum
    'anto_converse': 'anto_converse',
    # A word pair that name or describe a single relationship from opposite perspectives

    # Metaphor pairs
    'metaphor': 'has_metaphor',
    # A relation between two senses, where the first sense is a metaphorical extension of the second sense
    'has_metaphor': 'metaphor',
    # A relation between two senses, where the first sense can be metaphorically extended to the second sense

    # Metonym pairs
    'metonym': 'has_metonym',
    # A relation between two senses, where the first sense is a metonymic extension of the second sense
    'has_metonym': 'metonym',
    # A relation between two senses, where the first sense can be metonymically extended to the second sense

    'other': None
}

concept_relation_pairs = {
    # Hypernym/Hyponym pairs
    'hypernym': 'hyponym',  # a concept that is more general than a given concept
    'hyponym': 'hypernym',  # a concept that is more specific than a given concept

    # Instance pairs
    'instance_hypernym': 'instance_hyponym',  # the type of an instance
    'instance_hyponym': 'instance_hypernym',  # an occurrence of something

    # Meronym/Holonym pairs - member
    'mero_member': 'holo_member',  # concept A is a member of concept B
    'holo_member': 'mero_member',  # concept B is a member of concept A

    # Meronym/Holonym pairs - part
    'mero_part': 'holo_part',  # concept A is a component of concept B
    'holo_part': 'mero_part',  # concept B is the whole where concept A is a part

    # Meronym/Holonym pairs - substance
    'mero_substance': 'holo_substance',  # concept A is made of concept B
    'holo_substance': 'mero_substance',  # concept B is a substance of concept A

    # Meronym/Holonym pairs - location
    'mero_location': 'holo_location',  # A is a place located in B
    'holo_location': 'mero_location',  # B is a place located in A

    # Meronym/Holonym pairs - portion
    'mero_portion': 'holo_portion',  # A is an amount/piece/portion of B
    'holo_portion': 'mero_portion',  # B is an amount/piece/portion of A

    # General Meronym/Holonym
    'meronym': 'holonym',  # B makes up a part of A
    'holonym': 'meronym',  # A makes up a part of B

    # Entailment pairs
    'entails': 'is_entailed_by',  # impose, involve, or imply as a necessary accompaniment or result
    'is_entailed_by': 'entails',  # opposite of entails

    # Causation pairs
    'causes': 'is_caused_by',  # concept A is an entity that produces an effect or is responsible for events or results of concept B
    'is_caused_by': 'causes',  # a comes about because of B

    # Exemplification pairs
    'exemplifies': 'is_exemplified_by',  # a concept which is the example of a given concept
    'is_exemplified_by': 'exemplifies',  # a concept which is the type of a given concept

    # Domain pairs - region
    'domain_region': 'has_domain_region',  # a concept which is a geographical / cultural domain pointer of a given concept
    'has_domain_region': 'domain_region',  # a concept which is the term in the geographical / cultural domain of a given concept

    # Domain pairs - topic
    'domain_topic': 'has_domain_topic',  # a concept which is the scientific category pointer of a given concept
    'has_domain_topic': 'domain_topic',  # a concept which is a term in the scientific category of a given concept

    # General domain
    'domain': 'has_domain',  # a concept which is a Topic, Region or Usage pointer of a given concept
    'has_domain': 'domain',

    # Agent/role pairs
    'agent': 'involved_agent',  # a concept which is typically the one/that who/which does the action denoted by a given concept
    'involved_agent': 'agent',  # a concept which is the action done by an agent expressed by a given concept

    # Patient pairs
    'patient': 'involved_patient',  # a concept which is the one/that who/which undergoes a given concept
    'involved_patient': 'patient',  # a concept which is the action that the patient expressed by a given concept undergoing

    # Instrument pairs
    'instrument': 'involved_instrument',  # a concept which is the instrument necessary for the action or event expressed by a given concept
    'involved_instrument': 'instrument',  # a concept which is typically the action with the instrument expressed by a given concept

    # Location pairs
    'location': 'involved_location',  # a concept which is the place where the event expressed by a given concept happens
    'involved_location': 'location',  # a concept which is the event happening in a place expressed by a given concept

    # Direction pairs
    'direction': 'involved_direction',  # a concept which is the direction of the action or event expressed by a given concept
    'involved_direction': 'direction',  # a concept which is the action with the direction expressed by a given concept

    # Source direction pairs
    'source_direction': 'involved_source_direction',  # a concept which is the place from where the event expressed by a given concept begins
    'involved_source_direction': 'source_direction',  # a concept which is the action beginning from a place of a given concept

    # Target direction pairs
    'target_direction': 'involved_target_direction',  # a concept which is the place where the action or event expressed by a given concept leads to
    'involved_target_direction': 'target_direction',  # a concept which is the action or event leading to a place expressed by a given concept

    # Result pairs
    'result': 'involved_result',  # a concept which comes into existence as a result of a given concept
    'involved_result': 'result',  # a concept which is the action or event with a result of a given concept comes into existence

    # Role pairs
    'role': 'involved',  # a concept which is involved in the action or event expressed by a given concept
    'involved': 'role',  # a concept which is the action or event a given concept typically involved in

    # Co-agent pairs
    'co_agent_instrument': 'co_instrument_agent',  # a concept which is the instrument used by a given concept in an action
    'co_instrument_agent': 'co_agent_instrument',  # a concept which carries out an action by using a given concept as an instrument
    'co_agent_patient': 'co_patient_agent',  # a concept which is the patient undergoing an action carried out by a given concept
    'co_patient_agent': 'co_agent_patient',  # a concept which carries out an action a given concept undergoing
    'co_agent_result': 'co_result_agent',  # a concept which is the result of an action taken by a given concept
    'co_result_agent': 'co_agent_result',  # a concept which takes an action resulting in a given concept

    # Co-instrument pairs
    'co_instrument_patient': 'co_patient_instrument',  # a concept which undergoes an action with the use of a given concept as an instrument
    'co_patient_instrument': 'co_instrument_patient',  # a concept which is used as an instrument in an action a given concept undergoes
    'co_instrument_result': 'co_result_instrument',  # a concept which is the result of an action using an instrument of a given concept
    'co_result_instrument': 'co_instrument_result',  # a concept which is used as an instrument in an action resulting in a given concept

    # State pairs
    'be_in_state': 'state_of',  # a is qualified by B
    'state_of': 'be_in_state',  # B is qualified by A

    # Manner pairs
    'in_manner': 'manner_of',  # B qualifies the manner in which an action or event expressed by A takes place
    'manner_of': 'in_manner',  # a qualifies the manner in which an action or event expressed by B takes place

    # Subevent pairs
    'subevent': 'is_subevent_of',  # B takes place during or as part of A, and whenever B takes place, A takes place
    'is_subevent_of': 'subevent',  # a takes place during or as part of B, and whenever A takes place, B takes place

    # Classification pairs
    'classified_by': 'classifies',  # concept B is modified by classifier A when it is counted
    'classifies': 'classified_by',  # a concept A used when counting concept B

    # Restriction pairs
    'restricted_by': 'restricts',  # a relation between nominal (pronominal) B and an adjectival A (quantifier/determiner)
    'restricts': 'restricted_by',  # a relation between an adjectival A (quantifier/determiner) and a nominal (pronominal) B

    # Aspect pairs - simple
    'simple_aspect_ip': 'simple_aspect_pi',  # a concept which is linked to another through a change from imperfective to perfective aspect
    'simple_aspect_pi': 'simple_aspect_ip',  # a concept which is linked to another through a change from perfective to imperfective aspect

    # Aspect pairs - secondary
    'secondary_aspect_ip': 'secondary_aspect_pi',  # a concept which is linked to another through a change in aspect (ip)
    'secondary_aspect_pi': 'secondary_aspect_ip',  # a concept which is linked to another through a change in aspect (pi)

    # Gender/age/size relations with inverse - feminine
    'feminine': 'has_feminine',  # a concept used to refer to female members of a class
    'has_feminine': 'feminine',  # a concept which has a special concept for female members of its class

    # Gender/age/size relations with inverse - masculine
    'masculine': 'has_masculine',  # a concept used to refer to male members of a class
    'has_masculine': 'masculine',  # a concept which has a special concept for male members of its class

    # Gender/age/size relations with inverse - young
    'young': 'has_young',  # a concept used to refer to young members of a class
    'has_young': 'young',  # a concept which has a special concept for young members of its class

    # Gender/age/size relations with inverse - augmentative
    'augmentative': 'has_augmentative',  # a concept used to refer to generally larger members of a class
    'has_augmentative': 'augmentative',  # a concept which has a special concept for generally larger members of its class

    # Gender/age/size relations with inverse - diminutive
    'diminutive': 'has_diminutive',  # a concept used to refer to generally smaller members of a class
    'has_diminutive': 'diminutive',  # a concept which has a special concept for generally smaller members of its class

    # Symmetric relations (map to themselves)
    'similar': 'similar',  # (of words) expressing closely related meanings
    'attribute': 'attribute',  # an abstraction belonging to or characteristic of an entity
    'antonym': 'antonym',  # an opposite and inherently incompatible word
    'anto_simple': 'anto_simple',  # word pairs whose meanings are opposite but whose meanings do not lie on a continuous spectrum
    'anto_gradable': 'anto_gradable',  # word pairs whose meanings are opposite and which lie on a continuous spectrum
    'anto_converse': 'anto_converse',  # word pairs that name or describe a single relationship from opposite perspectives
    'derivation': 'derivation',  # a concept which is a derivationally related form of a given concept
    'eq_synonym': 'eq_synonym',  # A and B are equivalent concepts but their nature requires that they remain separate
    'ir_synonym': 'ir_synonym',  # a concept that means the same except for the style or connotation
    'also': None,  # a word having a loose semantic relation to another word

    # Unpaired relations
    'participle': None,  # a concept which is a participial adjective derived from a verb expressed by a given concept
    'pertainym': None,  # a concept which is of or pertaining to a given concept
    'constitutive': None,  # core semantic relations that define synsets
    'co_role': None,  # a concept undergoes an action in which a given concept is involved
    'other': None,  # any relation not otherwise specified
}

wordform_relation_pairs = {
    # Orthographic variant (symmetric)
    'orthographic_variant': 'orthographic_variant',  # alternative spelling or written form of the same word

    # Pronunciation variant (symmetric)
    'pronunciation_variant': 'pronunciation_variant',  # alternative pronunciation of the same word

    # Orthography/Pronunciation pairs (inverses of each other)
    'orthography_of': 'pronunciation_of',  # the written form of a pronunciation/phonetic form
    'pronunciation_of': 'orthography_of',  # the phonetic/spoken form of a written form
}

class CygnetProcessor:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.tree = etree.parse(input_file)
        self.root = self.tree.getroot()

        # Statistics
        self.stats = {
            'invalid_relation_types': [],
            'added_inverse_relations': defaultdict(int),
            'dag_cycles': [],
            'removed_transitive_edges': defaultdict(int),
            'pos_changes': 0,
            'pos_critical_errors': []
        }

    def validate_relation_types(self):
        """Step 1: Validate relation types against dictionaries"""
        print("\n=== Step 1: Validating Relation Types ===")

        # Check WordformRelations
        for rel in self.root.xpath('.//WordformRelation'):
            rel_type = rel.get('relationType')
            if rel_type not in wordform_relation_pairs:
                self.stats['invalid_relation_types'].append(
                    f"Invalid WordformRelation type: {rel_type}"
                )

        # Check SenseRelations
        for rel in self.root.xpath('.//SenseRelation'):
            rel_type = rel.get('relationType')
            if rel_type not in sense_relation_pairs:
                self.stats['invalid_relation_types'].append(
                    f"Invalid SenseRelation type: {rel_type}"
                )

        # Check ConceptRelations
        for rel in self.root.xpath('.//ConceptRelation'):
            rel_type = rel.get('relationType')
            if rel_type not in concept_relation_pairs:
                self.stats['invalid_relation_types'].append(
                    f"Invalid ConceptRelation type: {rel_type}"
                )

        if self.stats['invalid_relation_types']:
            print("Invalid relation types found:")
            for error in self.stats['invalid_relation_types']:
                print(f"  - {error}")
        else:
            print("✓ All relation types are valid")

    def add_inverse_relations(self):
        """Step 2: Check and add missing inverse relations"""
        print("\n=== Step 2: Adding Missing Inverse Relations ===")

        # Process WordformRelations
        self._add_inverse_for_layer('WordformRelationLayer', 'WordformRelation',
                                    wordform_relation_pairs)

        # Process SenseRelations
        self._add_inverse_for_layer('SenseRelationLayer', 'SenseRelation',
                                    sense_relation_pairs)

        # Process ConceptRelations
        self._add_inverse_for_layer('ConceptRelationLayer', 'ConceptRelation',
                                    concept_relation_pairs)

        print("Inverse relations added:")
        for rel_type, count in self.stats['added_inverse_relations'].items():
            print(f"  - {rel_type}: {count}")
        if not self.stats['added_inverse_relations']:
            print("  - No missing inverse relations found")

    def remove_duplicate_relations(self):
        """Step 1.5: Remove duplicate relations for all relation types"""
        print("\n=== Step 1.5: Removing Duplicate Relations ===")

        # Process each relation layer
        for layer_name, layer_xpath in [
            ('ConceptRelation', './/ConceptRelationLayer/ConceptRelation'),
            ('SenseRelation', './/SenseRelationLayer/SenseRelation'),
            ('WordformRelation', './/WordformRelationLayer/WordformRelation')
        ]:
            relations = self.root.xpath(layer_xpath)

            # Track unique relations: (relType, source, target)
            seen_relations = set()
            duplicates_to_remove = []

            for rel in relations:
                rel_type = rel.get('relType')
                source = rel.get('source')
                target = rel.get('target')
                relation_tuple = (rel_type, source, target)

                if relation_tuple in seen_relations:
                    # This is a duplicate
                    duplicates_to_remove.append(rel)
                else:
                    seen_relations.add(relation_tuple)

            # Remove duplicates
            for rel in duplicates_to_remove:
                rel.getparent().remove(rel)

            if duplicates_to_remove:
                print(f"  Removed {len(duplicates_to_remove)} duplicate {layer_name}s")

            self.stats[f'removed_duplicate_{layer_name.lower()}s'] = len(duplicates_to_remove)

        total_removed = (
                self.stats.get('removed_duplicate_conceptrelations', 0) +
                self.stats.get('removed_duplicate_senserelations', 0) +
                self.stats.get('removed_duplicate_wordformrelations', 0)
        )
        print(f"\nTotal duplicate relations removed: {total_removed}")

    def _add_inverse_for_layer(self, layer_name: str, rel_element: str,
                               relation_pairs: Dict[str, Optional[str]]):
        """Helper to add inverse relations for a specific layer"""
        layers = self.root.xpath(f'.//{layer_name}')
        if not layers:
            return
        layer = layers[0]

        # Build set of existing relations
        existing_relations = set()
        relations = layer.xpath(f'.//{rel_element}')

        for rel in relations:
            source = rel.get('source')
            target = rel.get('target')
            rel_type = rel.get('relationType')
            existing_relations.add((source, target, rel_type))

        # Check for missing inverses
        relations_to_add = []
        for rel in relations:
            source = rel.get('source')
            target = rel.get('target')
            rel_type = rel.get('relationType')

            inverse_type = relation_pairs.get(rel_type)
            if inverse_type is not None:
                # Check if inverse exists
                if (target, source, inverse_type) not in existing_relations:
                    relations_to_add.append((target, source, inverse_type))
                    self.stats['added_inverse_relations'][f"{layer_name}:{inverse_type}"] += 1

        # Add missing inverse relations at the end
        for source, target, rel_type in relations_to_add:
            new_rel = etree.SubElement(layer, rel_element)
            new_rel.set('relationType', rel_type)
            new_rel.set('source', source)
            new_rel.set('target', target)

    def validate_dag(self):
        """Step 3: Validate DAG structure for hypernymy/hyponymy relations"""
        print("\n=== Step 3: Validating DAG Structure ===")

        # Build graphs
        hypernymy_graph = self._build_concept_graph(['hypernym', 'instance-hypernym'])
        hyponymy_graph = self._build_concept_graph(['hyponym', 'instance-hyponym'])

        # Check for cycles
        print("Checking hypernymy + instance-hypernymy graph...")
        hyper_cycles = self._find_cycles(hypernymy_graph)
        if hyper_cycles:
            print(f"  ✗ Found {len(hyper_cycles)} cycles in hypernymy graph:")
            for cycle in hyper_cycles[:10]:  # Show first 10
                print(f"    - {' -> '.join(cycle)}")
            if len(hyper_cycles) > 10:
                print(f"    ... and {len(hyper_cycles) - 10} more cycles")
            self.stats['dag_cycles'].extend([('hypernym', c) for c in hyper_cycles])
        else:
            print("  ✓ Hypernymy graph is a valid DAG")

        print("Checking hyponymy + instance-hyponymy graph...")
        hypo_cycles = self._find_cycles(hyponymy_graph)
        if hypo_cycles:
            print(f"  ✗ Found {len(hypo_cycles)} cycles in hyponymy graph:")
            for cycle in hypo_cycles[:10]:
                print(f"    - {' -> '.join(cycle)}")
            if len(hypo_cycles) > 10:
                print(f"    ... and {len(hypo_cycles) - 10} more cycles")
            self.stats['dag_cycles'].extend([('hyponym', c) for c in hypo_cycles])
        else:
            print("  ✓ Hyponymy graph is a valid DAG")

    def _build_concept_graph(self, relation_types: List[str]) -> Dict[str, Set[str]]:
        """Build adjacency list for concept relations of given types"""
        graph = defaultdict(set)

        for rel in self.root.xpath('.//ConceptRelation'):
            rel_type = rel.get('relationType')
            if rel_type in relation_types:
                source = rel.get('source')
                target = rel.get('target')
                graph[source].add(target)

        return graph

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find all cycles in a directed graph using DFS"""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.remove(node)
            return False

        # Check all nodes
        all_nodes = set(graph.keys())
        for edges in graph.values():
            all_nodes.update(edges)

        for node in all_nodes:
            if node not in visited:
                dfs(node)

        return cycles

    def remove_transitive_edges(self):
        """Step 4: Remove transitive edges from hierarchical relations"""
        print("\n=== Step 4: Removing Transitive Edges ===")

        # Relation types to process
        relation_types = [
            'hypernym', 'instance-hypernym',
            'hyponym', 'instance-hyponym',
            'meronym', 'holonym',
            'part-meronym', 'part-holonym',
            'member-meronym', 'member-holonym',
            'substance-meronym', 'substance-holonym'
        ]

        for rel_type in relation_types:
            count = self._remove_transitive_for_type(rel_type)
            if count > 0:
                self.stats['removed_transitive_edges'][rel_type] = count

        print("Transitive edges removed:")
        if self.stats['removed_transitive_edges']:
            for rel_type, count in sorted(self.stats['removed_transitive_edges'].items()):
                print(f"  - {rel_type}: {count}")
        else:
            print("  - No transitive edges found")

    def _remove_transitive_for_type(self, rel_type: str) -> int:
        """Remove transitive edges for a specific relation type"""
        # Build graph for this relation type
        graph = defaultdict(set)
        layers = self.root.xpath('.//ConceptRelationLayer')
        if not layers:
            return 0
        layer = layers[0]

        relations = []
        for rel in layer.xpath('.//ConceptRelation'):
            if rel.get('relationType') == rel_type:
                source = rel.get('source')
                target = rel.get('target')
                graph[source].add(target)
                relations.append(rel)

        # Find transitive edges
        edges_to_remove = set()

        for source in graph.keys():
            direct_targets = graph[source].copy()

            # For each direct edge, find what's reachable through other paths
            for direct_target in direct_targets:
                # BFS to find all reachable nodes from direct_target
                reachable = set()
                queue = deque([direct_target])
                visited = {direct_target}

                while queue:
                    node = queue.popleft()
                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            reachable.add(neighbor)
                            queue.append(neighbor)

                # Check if any other direct target is reachable
                for other_target in direct_targets:
                    if other_target != direct_target and other_target in reachable:
                        # source -> direct_target -> ... -> other_target exists
                        # So source -> other_target is transitive
                        edges_to_remove.add((source, other_target))

        # Remove transitive edges from XML
        removed_count = 0
        for rel in relations:
            source = rel.get('source')
            target = rel.get('target')
            if (source, target) in edges_to_remove:
                rel.getparent().remove(rel)
                removed_count += 1

        return removed_count

    def resolve_pos_conflicts(self):
        """Step 5: Resolve POS conflicts using hypernymy chain voting"""
        print("\n=== Step 5: Resolving POS Conflicts ===")

        # Find concepts with ambiguous POS
        ambiguous_concepts = []
        concept_layers = self.root.xpath('.//ConceptLayer')
        if not concept_layers:
            print("No ConceptLayer found")
            return
        concept_layer = concept_layers[0]

        for concept in concept_layer.xpath('.//Concept'):
            concept_id = concept.get('id')
            pos = concept.get('pos')
            if '__' in pos:
                ambiguous_concepts.append((concept, concept_id, pos))

        print(f"Found {len(ambiguous_concepts)} concepts with ambiguous POS")

        if not ambiguous_concepts:
            return

        # Build hypernymy graph (both types)
        hypernymy_graph = self._build_concept_graph(['hypernym', 'instance-hypernym'])

        # Get POS for all concepts
        concept_pos = {}
        for concept in concept_layer.xpath('.//Concept'):
            concept_pos[concept.get('id')] = concept.get('pos')

        # Resolve each ambiguous concept
        for concept_elem, concept_id, old_pos in ambiguous_concepts:
            roots = self._find_roots(concept_id, hypernymy_graph)

            if not roots:
                # No path to root - this concept itself might be a root
                print(f"  ⚠ Concept {concept_id} has no hypernyms (is a root itself)")
                continue

            # Get POS of all roots
            root_pos_set = set()
            ambiguous_roots = []

            for root in roots:
                root_pos = concept_pos.get(root, '')
                if '__' in root_pos:
                    ambiguous_roots.append(root)
                else:
                    root_pos_set.add(root_pos)

            # Check for critical errors
            if ambiguous_roots:
                error_msg = f"**CRITICAL ERROR**: Concept {concept_id} reached ambiguous root(s): {', '.join(ambiguous_roots)}"
                print(f"  {error_msg}")
                self.stats['pos_critical_errors'].append(error_msg)
                continue

            if len(root_pos_set) > 1:
                error_msg = f"**CRITICAL ERROR**: Concept {concept_id} reached roots with different POS: {root_pos_set} (roots: {roots})"
                print(f"  {error_msg}")
                self.stats['pos_critical_errors'].append(error_msg)
                continue

            if len(root_pos_set) == 1:
                new_pos = root_pos_set.pop()
                concept_elem.set('pos', new_pos)
                self.stats['pos_changes'] += 1
                print(f"  ✓ Changed {concept_id}: {old_pos} -> {new_pos}")

        print(f"\nTotal POS changes made: {self.stats['pos_changes']}")
        if self.stats['pos_critical_errors']:
            print(
                f"\n**CRITICAL ERRORS**: {len(self.stats['pos_critical_errors'])} concepts with unresolvable POS conflicts")

    def _find_roots(self, concept_id: str, graph: Dict[str, Set[str]]) -> Set[str]:
        """Find all root concepts reachable from given concept via hypernymy"""
        roots = set()
        visited = set()
        queue = deque([concept_id])
        visited.add(concept_id)

        while queue:
            node = queue.popleft()

            # Check if this is a root (no outgoing edges)
            if node not in graph or not graph[node]:
                roots.add(node)
            else:
                # Continue traversal
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        return roots

    def save(self, output_file: str):
        """Save corrected XML to output file"""
        print(f"\n=== Saving corrected XML to {output_file} ===")

        # Write to file with pretty printing
        self.tree.write(output_file, encoding='utf-8', xml_declaration=True, pretty_print=True)
        print(f"✓ Saved to {output_file}")

    def print_summary(self):
        """Print final summary of all operations"""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if self.stats['invalid_relation_types']:
            print(f"\n⚠ Invalid relation types: {len(self.stats['invalid_relation_types'])}")
        else:
            print("\n✓ All relation types valid")

        total_inverses = sum(self.stats['added_inverse_relations'].values())
        print(f"✓ Inverse relations added: {total_inverses}")

        if self.stats['dag_cycles']:
            print(f"⚠ DAG cycles found: {len(self.stats['dag_cycles'])}")
        else:
            print("✓ No DAG cycles found")

        total_transitive = sum(self.stats['removed_transitive_edges'].values())
        print(f"✓ Transitive edges removed: {total_transitive}")

        print(f"✓ POS changes made: {self.stats['pos_changes']}")

        if self.stats['pos_critical_errors']:
            print(f"\n⚠ **CRITICAL POS ERRORS**: {len(self.stats['pos_critical_errors'])}")
            for error in self.stats['pos_critical_errors']:
                print(f"  {error}")


def main():
    print("Cygnet XML Processor")
    print("=" * 60)

    # Check if relation dictionaries are populated
    if not sense_relation_pairs and not concept_relation_pairs and not wordform_relation_pairs:
        print("\n⚠ WARNING: Relation pair dictionaries are empty!")
        print("Please populate sense_relation_pairs, concept_relation_pairs, and wordform_relation_pairs")
        print("before running this script.\n")

    # Process the XML
    processor = CygnetProcessor('bin/cygnet_prefix.xml')

    # Run all steps
    processor.validate_relation_types()
    processor.remove_duplicate_relations()
    processor.add_inverse_relations()
    processor.validate_dag()
    processor.remove_transitive_edges()
    processor.resolve_pos_conflicts()

    # Save results
    processor.save('cygnet.xml')

    # Print summary
    processor.print_summary()


if __name__ == '__main__':
    main()