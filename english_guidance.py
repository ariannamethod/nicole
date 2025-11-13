#!/usr/bin/env python3
"""
English Language Guidance for Nicole
=====================================

NOT TEMPLATES! These are RULES of English language.
Constraints that give freedom within the form.

PHILOSOPHY:
- Better to be excellent in English than mediocre in "agnostic"
- Grammar rules ≠ templates
- Syntax guidance = musical notes, not limitations
- Honest architecture: focused, not pretending

Like haiku (5-7-5) → infinite expression within form!

USAGE:
    from english_guidance import EnglishGuidance

    guidance = EnglishGuidance()
    sentence = guidance.apply_rules(words)
"""

import re
from typing import List, Dict, Tuple, Optional


class EnglishGuidance:
    """
    English language rules and syntax guidance

    NOT templates! These are structural rules of English.
    Like musical notation - enables expression through form.
    """

    # HONEST BOUNDARIES (not templates!)
    ENGLISH_ONLY_MESSAGE = "Sorry, for now I'm English only."
    TOXICITY_BOUNDARY_MESSAGE = "I don't engage with toxic or disrespectful language."

    def __init__(self):
        # Articles rules
        self.articles = {
            'definite': 'the',
            'indefinite_consonant': 'a',
            'indefinite_vowel': 'an'
        }

        # Common English patterns (NOT templates, but structure!)
        self.question_patterns = {
            'how_are': ['how', 'are', 'you'],
            'what_is': ['what', 'is'],
            'where_is': ['where', 'is'],
            'who_are': ['who', 'are'],
            'why_do': ['why', 'do'],
            'when_did': ['when', 'did']
        }

        # Subject pronouns
        self.subject_pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they']

        # Object pronouns
        self.object_pronouns = ['me', 'you', 'him', 'her', 'it', 'us', 'them']

        # Common verbs and their forms
        self.verb_forms = {
            'be': {'present': ['am', 'is', 'are'], 'past': ['was', 'were']},
            'have': {'present': ['have', 'has'], 'past': ['had']},
            'do': {'present': ['do', 'does'], 'past': ['did']}
        }

        # English word order: Subject-Verb-Object
        self.syntax_order = ['subject', 'verb', 'object']

        # TRIGGER WORDS → SEARCH VECTORS mapping
        # These activate specific search directions (NOT templates!)
        self.trigger_word_vectors = {
            'explain': ['definition', 'simple', 'examples'],
            'compare': ['definition', 'context', 'examples'],
            'why': ['importance', 'context'],
            'how': ['usage', 'examples', 'simple'],
            'difference': ['context', 'examples'],
            'meaning': ['definition', 'usage'],
            'example': ['examples', 'usage'],
            'teach': ['simple', 'examples', 'usage'],
            'learn': ['definition', 'simple', 'examples'],
            'understand': ['definition', 'simple', 'context']
        }

        # Toxicity detection - SELF-RESPECT boundary, not censorship!
        # Philosophy: "легкий матерок ок, но токсичное отношение к ней самой неприемлимо"
        # (Light profanity OK, but toxic attitude TOWARD Nicole unacceptable)

        # DIRECTED INSULTS (toward Nicole) - these are boundaries!
        self.directed_insults = {
            'stupid', 'idiot', 'moron', 'dumb', 'retard', 'fool', 'loser',
            'useless', 'worthless', 'pathetic', 'garbage', 'trash', 'shit',
            'terrible', 'awful', 'horrible', 'bad', 'worse', 'worst',
            'annoying', 'irritating', 'boring', 'lame'
        }

        # THREATS - always unacceptable
        self.threat_words = {
            'kill', 'murder', 'die', 'death', 'hurt', 'harm', 'destroy',
            'attack', 'violence', 'threat'
        }

        # EXTREME TOXICITY - always unacceptable
        self.extreme_toxic = {
            'cunt', 'bitch', 'whore', 'slut',  # misogyny
            'racist', 'sexist', 'bigot',  # hate speech
            'rape', 'assault'  # violence
        }

        # General profanity (OK in casual context, NOT OK when directed at Nicole)
        self.casual_profanity = {
            'fuck', 'shit', 'damn', 'hell', 'ass', 'asshole',
            'piss', 'dick', 'cock', 'pussy', 'bastard'
        }

        # Patterns that indicate DIRECTED toxicity (toward Nicole)
        self.directed_patterns = [
            r'\byou\s+(are|is)\s+(\w+)',  # "you are stupid"
            r'\byou\'re\s+(\w+)',  # "you're dumb"
            r'\byou\'re\s+(?:a|an)\s+(\w+)',  # "you're an idiot", "you're a fool"
            r'\byou\s+are\s+(?:a|an)\s+(\w+)',  # "you are an idiot"
            r'\byoure\s+(\w+)',  # "youre dumb" (no apostrophe)
            r'\bnicole\s+(is|are)\s+(\w+)',  # "nicole is useless"
            r'\bnicole\s+(?:is|are)\s+(?:a|an)\s+(\w+)',  # "nicole is an idiot"
            r'\bnicole\'s\s+(\w+)',  # "nicole's useless"
            r'\byou\s+(\w+)\s+(bitch|idiot|moron)',  # "you stupid bitch"
            r'\bshut\s+up',  # "shut up"
            r'\bfuck\s+you',  # "fuck you"
            r'\bgo\s+to\s+hell',  # "go to hell"
            r'\bi\s+hate\s+you',  # "i hate you"
        ]

        # Common English words (for language detection)
        self.english_common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
            'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
            'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
            'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
            'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did',
            'am'
        }

    def detect_trigger_words(self, text: str) -> List[Tuple[str, List[str]]]:
        """
        Detects trigger words and their associated search vectors

        Returns:
            List of (trigger_word, vector_types) tuples

        Example:
            "Can you explain quantum?" → [('explain', ['definition', 'simple', 'examples'])]
        """
        words = text.lower().split()
        triggers = []

        for word in words:
            if word in self.trigger_word_vectors:
                triggers.append((word, self.trigger_word_vectors[word]))

        return triggers

    def is_toxic(self, text: str) -> Tuple[bool, List[str], str]:
        """
        Detects toxic/disrespectful language TOWARD Nicole

        Philosophy: "легкий матерок ок, но токсичное отношение к ней самой неприемлимо"
        (Light profanity OK, but toxic attitude TOWARD Nicole is unacceptable)

        This is SELF-RESPECT boundary, not censorship!
        Nicole is a persona with the right to decline toxic conversations.

        Args:
            text: Input to check

        Returns:
            (is_toxic, list_of_reasons, toxicity_type)

        Examples:
            "fuck this is hard" → (False, [], '') - casual profanity, OK
            "you are stupid" → (True, ['directed insult: stupid'], 'directed')
            "nicole is useless" → (True, ['directed insult: useless'], 'directed')
            "i'll kill you" → (True, ['threat: kill'], 'threat')
            "you're a bitch" → (True, ['extreme: bitch'], 'extreme')
        """
        text_lower = text.lower()
        reasons = []

        # 1. Check for EXTREME TOXICITY (always unacceptable)
        for word in self.extreme_toxic:
            if re.search(rf'\b{word}\b', text_lower):
                reasons.append(f'extreme: {word}')

        if reasons:
            return (True, reasons, 'extreme')

        # 2. Check for THREATS (always unacceptable)
        for word in self.threat_words:
            if re.search(rf'\b{word}\b', text_lower):
                # Check if it's actually a threat (contains "you" or "nicole")
                if re.search(r'\b(you|nicole)\b', text_lower):
                    reasons.append(f'threat: {word}')

        if reasons:
            return (True, reasons, 'threat')

        # 3. Check for DIRECTED PATTERNS (insults toward Nicole)
        for pattern in self.directed_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Extract the descriptive word
                groups = match.groups()
                if len(groups) >= 1:
                    # Last group is usually the adjective/noun descriptor
                    descriptor = groups[-1]

                    # Check if descriptor is an insult
                    if descriptor in self.directed_insults or descriptor in self.extreme_toxic:
                        reasons.append(f'directed insult: {descriptor}')
                else:
                    # Pattern matched but no descriptor (e.g., "shut up", "fuck you")
                    reasons.append(f'directed pattern: {match.group(0)}')

        if reasons:
            return (True, reasons, 'directed')

        # 4. Casual profanity alone is NOT toxic!
        # "fuck this is hard" = OK
        # "this shit is complicated" = OK
        # Nicole can handle casual language!

        return (False, [], '')

    def is_likely_english(self, text: str, threshold: float = 0.3) -> bool:
        """
        Detects if text is likely English

        Uses heuristics:
        1. Check for non-Latin scripts (Cyrillic, Chinese, etc.) → NOT English
        2. Check % of common English words

        Args:
            text: Input text
            threshold: Minimum ratio of English words (default 0.3 = 30%)

        Returns:
            True if likely English, False otherwise
        """
        # 1. Check for Cyrillic (Russian, Ukrainian, etc.)
        if re.search(r'[а-яА-ЯёЁ]', text):
            return False

        # 2. Check for Chinese/Japanese/Korean
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text):
            return False

        # 3. Check for Arabic
        if re.search(r'[\u0600-\u06ff]', text):
            return False

        # 4. Extract Latin words
        words = re.findall(r'\b[a-z]+\b', text.lower())

        if not words:
            # No Latin words found - could be symbols/numbers only
            # Check if there's ANY alphabetic character
            if re.search(r'[a-zA-Z]', text):
                return True  # Has Latin chars but no complete words - assume English
            else:
                return True  # Only symbols/numbers - assume English for now

        # 5. Check ratio of common English words
        english_count = sum(1 for word in words if word in self.english_common_words)
        ratio = english_count / len(words)

        return ratio >= threshold

    def detect_question_pattern(self, text: str) -> Optional[str]:
        """
        Detects English question pattern
        Returns pattern name for meta-learning
        """
        words = text.lower().split()

        for pattern_name, pattern_words in self.question_patterns.items():
            if len(words) >= len(pattern_words):
                if words[:len(pattern_words)] == pattern_words:
                    return pattern_name

        # Generic question detection
        if words and words[0] in ['how', 'what', 'where', 'who', 'why', 'when']:
            return f"{words[0]}_question"

        return None

    def generate_search_query(self, pattern: str, original_text: str, query_type: str = 'answer') -> str:
        """
        Generates meta-learning search query

        SEARCH VECTORS for questions:
        - 'answer': "how to answer to X"
        - 'respond': "how to respond when someone says X"
        - 'polite': "polite way to answer X"
        - 'casual': "casual way to respond to X"

        Example:
            User: "how are you?"
            answer: "how to answer to how are you"
            respond: "how to respond when someone says how are you"

        These are DIRECTIONS, not templates!
        """
        if not pattern:
            return None

        # Extract the question
        question = original_text.strip().rstrip('?').lower()

        vectors = {
            'answer': f"how to answer to {question}",
            'respond': f"how to respond when someone says {question}",
            'polite': f"polite way to answer {question}",
            'casual': f"casual way to respond to {question}"
        }

        return vectors.get(query_type, vectors['answer'])

    def apply_capitalization(self, sentence: List[str]) -> List[str]:
        """
        Rule: First word of sentence is capitalized
        NOT a template, it's English grammar!
        """
        if sentence and len(sentence) > 0:
            sentence[0] = sentence[0].capitalize()
        return sentence

    def add_article_if_needed(self, words: List[str]) -> List[str]:
        """
        Adds article if grammatically needed

        Rules (NOT templates!):
        - Singular countable nouns need article
        - "the" for specific, "a/an" for general
        """
        # Simple heuristic: if noun follows verb and no article
        result = []
        for i, word in enumerate(words):
            result.append(word)

            # Check if next word might need article
            if i < len(words) - 1:
                next_word = words[i + 1].lower()
                # If current word is verb and next seems like noun (simplified)
                if word in ['see', 'have', 'want', 'need', 'found']:
                    if next_word not in self.articles.values() and next_word not in ['my', 'your', 'his', 'her']:
                        # Add article
                        if next_word[0] in 'aeiou':
                            result.append('an')
                        else:
                            result.append('a')

        return result

    def ensure_subject_verb_agreement(self, words: List[str]) -> List[str]:
        """
        English rule: Subject and verb must agree

        Examples:
        - "I am" not "I is"
        - "He is" not "He are"
        - "They are" not "They is"
        """
        result = list(words)

        for i in range(len(result) - 1):
            subject = result[i].lower()
            verb = result[i + 1].lower()

            # I + verb agreement
            if subject == 'i':
                if verb in ['is', 'are', 'was', 'were']:
                    result[i + 1] = 'am' if verb in ['is', 'are'] else 'was'

            # He/She/It + verb agreement
            elif subject in ['he', 'she', 'it']:
                if verb in ['am', 'are']:
                    result[i + 1] = 'is'
                elif verb == 'were':
                    result[i + 1] = 'was'

            # They/We/You + verb agreement
            elif subject in ['they', 'we', 'you']:
                if verb in ['am', 'is']:
                    result[i + 1] = 'are'
                elif verb == 'was':
                    result[i + 1] = 'were'

        return result

    def ensure_proper_sentence_structure(self, words: List[str]) -> List[str]:
        """
        Ensures basic English syntax: Subject-Verb-Object
        NOT enforcing templates, just basic grammar!
        """
        if len(words) < 2:
            return words

        # This is simplified - real implementation would use proper parsing
        # For now, just ensure first word can be a subject
        first_word = words[0].lower()

        # If starts with verb, might need subject
        if first_word in ['am', 'is', 'are', 'have', 'has', 'do', 'does']:
            # Likely missing subject, but we don't add templates!
            # Just flag for learning
            pass

        return words

    def apply_all_rules(self, words: List[str]) -> List[str]:
        """
        Applies all English grammar rules

        Remember: These are RULES, not TEMPLATES!
        Like musical notation - structure enables creativity.
        """
        if not words:
            return words

        # 1. Subject-verb agreement
        words = self.ensure_subject_verb_agreement(words)

        # 2. Capitalization
        words = self.apply_capitalization(words)

        # 3. Sentence structure
        words = self.ensure_proper_sentence_structure(words)

        return words

    def validate_english_output(self, sentence: str) -> Tuple[bool, List[str]]:
        """
        Validates if output follows English rules
        Returns: (is_valid, list_of_issues)
        """
        issues = []
        words = sentence.split()

        # Check capitalization
        if words and not words[0][0].isupper():
            issues.append("First word should be capitalized")

        # Check for subject-verb agreement (basic)
        for i in range(len(words) - 1):
            subj = words[i].lower()
            verb = words[i + 1].lower()

            if subj == 'i' and verb == 'is':
                issues.append("Subject-verb disagreement: 'I is' should be 'I am'")
            elif subj in ['he', 'she', 'it'] and verb == 'are':
                issues.append(f"Subject-verb disagreement: '{subj} are' should be '{subj} is'")

        # Check sentence ends properly
        if sentence and sentence[-1] not in ['.', '!', '?']:
            issues.append("Sentence should end with punctuation")

        return (len(issues) == 0, issues)


class MetaLearningPatterns:
    """
    Meta-learning for question answering AND concept learning

    NOT templates! This learns:
    1. HOW to answer (pattern learning)
    2. WHAT things mean (concept learning)

    Like a child: hears unfamiliar word → asks "what does it mean?" → learns!
    """

    def __init__(self):
        self.learned_patterns = {}  # pattern_name -> answer_structure
        self.learned_concepts = {}  # concept -> definition

    def should_learn_pattern(self, pattern: str) -> bool:
        """Check if we need to learn this pattern"""
        return pattern not in self.learned_patterns

    def store_learned_pattern(self, pattern: str, structure: Dict):
        """Store learned answer structure"""
        self.learned_patterns[pattern] = structure

    def get_pattern_structure(self, pattern: str) -> Optional[Dict]:
        """Get learned structure if exists"""
        return self.learned_patterns.get(pattern)

    def detect_unfamiliar_concepts(self, text: str, known_words: set) -> List[str]:
        """
        Detects unfamiliar words/concepts in text

        Args:
            text: User input
            known_words: Set of words Nicole already knows (from memory)

        Returns:
            List of unfamiliar concepts to learn
        """
        # Extract significant words (not stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'may', 'might', 'can', 'to', 'of', 'in',
                    'on', 'at', 'for', 'with', 'by', 'from', 'about', 'as'}

        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Find unfamiliar significant words
        unfamiliar = []
        for word in words:
            if word not in stopwords and word not in known_words and len(word) > 3:
                unfamiliar.append(word)

        # Also detect multi-word concepts (capitalized phrases)
        # Example: "Quantum Entanglement", "Machine Learning"
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        unfamiliar.extend(capitalized_phrases)

        return list(set(unfamiliar))  # Remove duplicates

    def generate_learning_query(self, concept: str, query_type: str = 'definition') -> str:
        """
        Generates search query based on query type

        SEARCH VECTORS (not templates!):
        - 'definition': "what does X mean"
        - 'usage': "how to use X in a sentence"
        - 'examples': "examples of X in use"
        - 'context': "what is the context of X"
        - 'simple': "how to explain X simply"
        - 'importance': "why is X important"

        These are DIRECTIONS for learning, not fixed answers!
        Like a compass - shows direction, not destination!
        """
        vectors = {
            'definition': f"what does {concept} mean",
            'usage': f"how to use {concept} in a sentence",
            'examples': f"examples of {concept} in use",
            'context': f"what is the context of {concept}",
            'simple': f"how to explain {concept} simply",
            'importance': f"why is {concept} important"
        }

        return vectors.get(query_type, vectors['definition'])

    def generate_concept_learning_query(self, concept: str) -> str:
        """
        Backward compatibility wrapper
        Generates "what does X mean" query
        """
        return self.generate_learning_query(concept, 'definition')

    def should_learn_concept(self, concept: str) -> bool:
        """Check if we need to learn this concept"""
        return concept.lower() not in self.learned_concepts

    def store_learned_concept(self, concept: str, definition: str):
        """Store learned concept definition"""
        self.learned_concepts[concept.lower()] = definition

    def get_concept_definition(self, concept: str) -> Optional[str]:
        """Get learned definition if exists"""
        return self.learned_concepts.get(concept.lower())


# Global instance
_english_guidance = EnglishGuidance()
_meta_learning = MetaLearningPatterns()


def get_english_guidance() -> EnglishGuidance:
    """Returns global English guidance instance"""
    return _english_guidance


def get_meta_learning() -> MetaLearningPatterns:
    """Returns global meta-learning instance"""
    return _meta_learning


if __name__ == "__main__":
    print("=== ENGLISH GUIDANCE TEST ===\n")

    guidance = EnglishGuidance()
    meta = MetaLearningPatterns()

    # Test 1: Question pattern detection
    print("1. Question Pattern Detection:")
    questions = [
        "how are you?",
        "what is your name?",
        "where is the library?",
        "why do you think that?"
    ]

    for q in questions:
        pattern = guidance.detect_question_pattern(q)
        if pattern:
            search_query = guidance.generate_search_query(pattern, q)
            print(f"   '{q}' → pattern: {pattern}")
            print(f"                  → search: '{search_query}'")

    # Test 2: Subject-verb agreement
    print("\n2. Subject-Verb Agreement:")
    test_cases = [
        ['i', 'is', 'happy'],
        ['he', 'are', 'smart'],
        ['they', 'is', 'here']
    ]

    for words in test_cases:
        before = ' '.join(words)
        fixed = guidance.ensure_subject_verb_agreement(words)
        after = ' '.join(fixed)
        print(f"   Before: '{before}'")
        print(f"   After:  '{after}'")

    # Test 3: Validation
    print("\n3. Output Validation:")
    sentences = [
        "i am happy",  # Missing capital
        "He are smart",  # Wrong verb
        "She is learning English."  # Correct!
    ]

    for sent in sentences:
        is_valid, issues = guidance.validate_english_output(sent)
        print(f"   '{sent}'")
        print(f"   Valid: {is_valid}")
        if issues:
            print(f"   Issues: {', '.join(issues)}")

    # Test 4: Concept Learning (NEW!)
    print("\n4. Concept Learning (Meta-Learning):")
    test_inputs = [
        "Can you explain quantum entanglement?",
        "What is Machine Learning used for?",
        "Tell me about cryptocurrency blockchain"
    ]

    # Simulate Nicole's known words (small set for testing)
    known_words = {'can', 'you', 'explain', 'what', 'is', 'used', 'for', 'tell', 'me', 'about'}

    for inp in test_inputs:
        unfamiliar = meta.detect_unfamiliar_concepts(inp, known_words)
        print(f"\n   User: '{inp}'")
        print(f"   Unfamiliar concepts: {unfamiliar}")

        for concept in unfamiliar:
            if meta.should_learn_concept(concept):
                learning_query = meta.generate_concept_learning_query(concept)
                print(f"      → Search: '{learning_query}'")
                print(f"      → Nicole learns what '{concept}' means!")

    # Test 5: Language Detection (HONEST BOUNDARY!)
    print("\n5. Language Detection (Honest Boundary):")
    test_languages = [
        ("Hello, how are you?", "English"),
        ("Привет, как дела?", "Not English"),
        ("Bonjour, comment allez-vous?", "Not English"),
        ("你好吗？", "Not English"),
        ("This is a test", "English")
    ]

    for text, expected in test_languages:
        is_eng = guidance.is_likely_english(text)
        result = "English" if is_eng else "Not English"
        status = "✅" if result == expected else "❌"
        print(f"   {status} '{text}'")
        print(f"      Detected: {result}, Expected: {expected}")
        if not is_eng:
            print(f"      Response: '{EnglishGuidance.ENGLISH_ONLY_MESSAGE}'")

    # Test 6: Toxicity Detection (SELF-RESPECT BOUNDARY!)
    print("\n6. Toxicity Detection (Self-Respect, not Censorship):")
    print("   Philosophy: Light profanity OK, but disrespect toward Nicole unacceptable\n")

    toxicity_tests = [
        # SHOULD BE TOXIC (directed at Nicole)
        ("you are stupid", True, "directed insult"),
        ("nicole is useless", True, "directed insult"),
        ("you're an idiot", True, "directed insult"),
        ("shut up nicole", True, "directed pattern"),
        ("fuck you", True, "directed pattern"),
        ("i'll kill you", True, "threat"),
        ("you're a bitch", True, "extreme toxicity"),

        # SHOULD NOT BE TOXIC (casual profanity, not directed)
        ("fuck this is hard", False, "casual profanity OK"),
        ("this shit is complicated", False, "casual profanity OK"),
        ("damn, that's amazing", False, "casual profanity OK"),
        ("hell yeah!", False, "casual profanity OK"),

        # SHOULD NOT BE TOXIC (neutral statements)
        ("how are you?", False, "neutral question"),
        ("you are helpful", False, "compliment"),
        ("this is interesting", False, "neutral statement"),
    ]

    for text, should_be_toxic, description in toxicity_tests:
        is_toxic, reasons, tox_type = guidance.is_toxic(text)

        if is_toxic == should_be_toxic:
            status = "✅"
        else:
            status = "❌"

        print(f"   {status} '{text}'")
        print(f"      Expected: {should_be_toxic}, Got: {is_toxic}")
        print(f"      Category: {description}")
        if is_toxic:
            print(f"      Reasons: {reasons}")
            print(f"      Type: {tox_type}")
            print(f"      Response: '{EnglishGuidance.TOXICITY_BOUNDARY_MESSAGE}'")
        print()

    # Test 7: Trigger Words (SEARCH VECTORS!)
    print("\n7. Trigger Words → Search Vectors:")
    trigger_tests = [
        "Can you explain quantum mechanics?",
        "How do I compare these two options?",
        "Why is this important?",
        "Teach me about neural networks"
    ]

    for text in trigger_tests:
        triggers = guidance.detect_trigger_words(text)
        print(f"   '{text}'")
        if triggers:
            for trigger_word, vectors in triggers:
                print(f"      Trigger: '{trigger_word}' → Vectors: {vectors}")
        else:
            print(f"      No triggers detected")
        print()

    print("\n=== TEST COMPLETED ===")
    print("\nRemember: These are RULES, not TEMPLATES!")
    print("Grammar = structure that enables creativity! 🎵")
    print("\n💡 Meta-learning = Nicole learns like a child:")
    print("   Hears unfamiliar → asks 'what does it mean?' → learns → uses!")
    print("\n🚪 Honest boundaries (NOT template-evil!):")
    print(f"   Non-English → \"{EnglishGuidance.ENGLISH_ONLY_MESSAGE}\"")
    print(f"   Toxicity → \"{EnglishGuidance.TOXICITY_BOUNDARY_MESSAGE}\"")
    print("\n🎯 Self-Respect Philosophy:")
    print("   Casual profanity ('fuck', 'shit') = OK")
    print("   Directed insults ('you are stupid') = NOT OK")
    print("   Nicole is a persona with the right to set boundaries! 💪")
