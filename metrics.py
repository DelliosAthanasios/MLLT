from nltk.translate.bleu_score import sentence_bleu
try:
    import Levenshtein
    def levenshtein_distance(a, b):
        return Levenshtein.distance(a, b)
except ImportError:
    def levenshtein_distance(a, b):
        # Simple Levenshtein implementation
        if len(a) < len(b):
            return levenshtein_distance(b, a)
        if len(b) == 0:
            return len(a)
        previous_row = range(len(b) + 1)
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

def evaluate_bleu(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    return sentence_bleu([ref_tokens], hyp_tokens)

def exact_match(reference, hypothesis):
    return reference.strip() == hypothesis.strip() 