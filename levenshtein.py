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

if __name__ == "__main__":
    with open("dna.txt") as f:
        original = f.read().strip()
    with open("reconstructed.txt") as f:
        reconstructed = f.read().strip()
    dist = levenshtein(original, reconstructed)
    with open("levenshtein.txt", "w") as f:
        f.write(f"Levenshtein distance: {dist}\n")
