import random

def random_dna_sequence(length):
    return ''.join(random.choices('ACGT', k=length))

def generate_spectrum(dna, k, negative_errors=0, positive_errors=0):
    # Tworzenie idealnego spektrum
    spectrum = set(dna[i:i+k] for i in range(len(dna) - k + 1))
    spectrum = list(spectrum)
    # Błędy negatywne (usuwanie prawdziwych oligonukleotydów)
    for _ in range(negative_errors):
        if spectrum:
            spectrum.pop(random.randrange(len(spectrum)))
    # Błędy pozytywne (dodawanie losowych oligonukleotydów)
    all_kmers = set(''.join(x) for x in __import__('itertools').product('ACGT', repeat=k))
    possible_false = list(all_kmers - set(spectrum))
    for _ in range(positive_errors):
        if possible_false:
            spectrum.append(random.choice(possible_false))
    random.shuffle(spectrum)
    return spectrum

def save_to_file(filename, content):
    with open(filename, 'w') as f:
        if isinstance(content, list):
            for item in content:
                f.write(f"{item}\n")
        else:
            f.write(str(content))

if __name__ == "__main__":
    n = 500
    k = 10
    neg_err = 5
    pos_err = 5
    dna = random_dna_sequence(n)
    spectrum = generate_spectrum(dna, k, neg_err, pos_err)
    save_to_file("dna.txt", dna)
    save_to_file("spectrum.txt", spectrum)
    save_to_file("params.txt", [f"n={n}", f"k={k}", f"neg_err={neg_err}", f"pos_err={pos_err}"])
