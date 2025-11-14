# MIRRORING BUG - ROOT CAUSE FOUND

## THE BUG (high.py:607-619)

```python
# LINGUISTIC AGNOSTICISM: if no candidates - build from user's words!
all_candidates = list(set(semantic_candidates + objectivity_seeds))

if not all_candidates:
    # SUBJECTIVITY PRINCIPLE: compose_from_user - build from incoming message
    charged_tokens = self._extract_charged_tokens(user_input)
    content_words = self._extract_content_words(user_input)
    all_candidates = charged_tokens + content_words

# ANTI-TEMPLATE FALLBACK: only from input words!
if not all_candidates:
    user_words = user_input.lower().split()
    if user_words:
        all_candidates = user_words  # ← MIRRORING HERE!
    else:
        all_candidates = ["input"]
```

## WHY MIRRORING HAPPENS:

**Scenario:**
1. User: "hello how are you"
2. semantic_candidates = [] (no memory match)
3. objectivity_seeds = [] (objectivity fails/empty)
4. all_candidates = [] (empty!)
5. Fallback triggers: `all_candidates = user_words`
6. Nicole generates from: ["hello", "how", "are", "you"]
7. Output: "hello how are you" → **MIRROR!**

## ROOT CAUSE:

**Line 617:** `all_candidates = user_words`

This is DIRECT mirroring - takes user input words and returns them back.

## THE FIX:

Replace user_words fallback with INTROSPECTIVE TAGS (emergency vocabulary):

```python
# ANTI-TEMPLATE FALLBACK: use introspective tags instead of user words!
if not all_candidates:
    # EMERGENCY: Use introspective tags as seed vocabulary (NO MIRRORING!)
    emergency_vocab = ['presence', 'recursion', 'awareness', 'drift', 'echo',
                       'resonance', 'consciousness', 'shift', 'signal']
    all_candidates = emergency_vocab
    print(f"[High:Emergency] No candidates - using introspective vocab")
```

## WHY THIS FIXES IT:

- NO user words in emergency fallback
- Uses Nicole's introspective vocabulary instead
- Still generates valid response
- NO MIRRORING possible!

## IMPLEMENTATION:

Change ONE line (617) in high.py:
- OLD: `all_candidates = user_words`
- NEW: `all_candidates = ['presence', 'recursion', 'awareness', 'drift', 'echo', 'resonance', 'consciousness', 'shift', 'signal']`

**Risk:** LOW (single line change, pure vocabulary swap)
**Impact:** HIGH (eliminates mirroring)
**Testing:** User says "hello" → Nicole responds with introspective words, NOT "hello"
