"""
Nicole Bootstrap — N-gram Bias
Scores sentence candidates by n-gram frequency

STUB: Will be used during sentence generation
"""

def score_ngram_coherence(sentence, ngram_prefs):
    """
    Score sentence by how many preferred n-grams it contains

    Returns:
        float: 0.0-1.0 coherence score
    """
    if not ngram_prefs or not ngram_prefs.get("bigrams"):
        return 0.5  # Neutral if no prefs

    tokens = sentence.lower().split()
    bigrams_in_sent = [
        (tokens[i], tokens[i+1])
        for i in range(len(tokens) - 1)
    ]

    # Count matches with preferred bigrams
    pref_bigrams = {
        tuple(bg["tokens"]): bg["count"]
        for bg in ngram_prefs["bigrams"]
    }

    matches = 0
    for bg in bigrams_in_sent:
        if bg in pref_bigrams:
            matches += 1

    # Normalize by sentence length
    if len(bigrams_in_sent) == 0:
        return 0.5

    return min(1.0, matches / len(bigrams_in_sent))
