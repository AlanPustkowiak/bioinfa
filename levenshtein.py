def levenshtein(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]

def normalize_levenshtein(distance, original_len, reconstructed_len):
    max_len = max(original_len, reconstructed_len)
    return distance / max_len if max_len > 0 else 0.0

if __name__ == "__main__":
    with open("dna.txt") as f:
        original = f.read().strip()
    with open("reconstructed.txt") as f:
        reconstructed = f.read().strip()
    
    dist = levenshtein(original, reconstructed)
    original_len = len(original)
    reconstructed_len = len(reconstructed)
    normalized = normalize_levenshtein(dist, original_len, reconstructed_len)
    
    with open("levenshtein.txt", "w") as f:
        f.write(f"Levenshtein distance: {dist}\n")
        f.write(f"Normalized distance: {normalized:.4f} ({normalized*100:.2f}%)\n")
        f.write(f"Original DNA length: {original_len}\n")
        f.write(f"Reconstructed DNA length: {reconstructed_len}\n")
