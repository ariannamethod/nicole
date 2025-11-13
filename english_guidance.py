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

    def generate_search_query(self, pattern: str, original_text: str) -> str:
        """
        Generates meta-learning search query

        Example:
            User: "how are you?"
            Returns: "how to answer to how are you"

        This is NOT a template! It's a meta-learning pattern.
        """
        if not pattern:
            return None

        # Extract the question
        question = original_text.strip().rstrip('?').lower()

        # Generate learning query
        return f"how to answer to {question}"

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
    Meta-learning for question answering

    NOT templates! This learns HOW to answer, not WHAT to answer.
    """

    def __init__(self):
        self.learned_patterns = {}  # pattern_name -> answer_structure

    def should_learn_pattern(self, pattern: str) -> bool:
        """Check if we need to learn this pattern"""
        return pattern not in self.learned_patterns

    def store_learned_pattern(self, pattern: str, structure: Dict):
        """Store learned answer structure"""
        self.learned_patterns[pattern] = structure

    def get_pattern_structure(self, pattern: str) -> Optional[Dict]:
        """Get learned structure if exists"""
        return self.learned_patterns.get(pattern)


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

    print("\n=== TEST COMPLETED ===")
    print("\nRemember: These are RULES, not TEMPLATES!")
    print("Grammar = structure that enables creativity! 🎵")
