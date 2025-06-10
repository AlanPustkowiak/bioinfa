import random
import math
import numpy as np
from collections import defaultdict, deque
import time
import os

class SBHSolver:
    def __init__(self, n, k, spectrum, start_oligo, negative_errors=0, positive_errors=0, use_percentage=False, original_spectrum_size=None):
        """
        Inicjalizacja solvera SBH
        n: długość DNA do rekonstrukcji
        k: długość oligonukleotydów
        spectrum: lista oligonukleotydów w spektrum
        start_oligo: oligonukleotyd startowy
        negative_errors: liczba błędów negatywnych
        positive_errors: liczba błędów pozytywnych
        """
        self.n = n
        self.k = k
        self.spectrum = spectrum
        self.start_oligo = start_oligo
        if use_percentage and original_spectrum_size is not None:
            self.negative_errors = max(0, int(original_spectrum_size * negative_errors))
            self.positive_errors = max(0, int(original_spectrum_size * positive_errors))
        else:
            self.negative_errors = int(negative_errors)
            self.positive_errors = int(positive_errors)

        
        # Parametry algorytmu mrówkowego
        self.num_ants = 20
        self.num_iterations = 100
        self.alpha = 1.0  # wpływ feromonów
        self.beta = 2.0   # wpływ heurystyki
        self.rho = 0.1    # współczynnik parowania feromonów
        self.q0 = 0.9     # próg dla eksploatacji vs eksploracji
        
        # Inicjalizacja grafu
        self.graph = self._build_graph()
        self.pheromones = self._initialize_pheromones()
        
        # Najlepsze rozwiązanie
        self.best_solution = None
        self.best_fitness = float('-inf')
        
        # Logowanie
        self.log_data = []
        
    def _build_graph(self):
        """Buduje graf z oligonukleotydów spektrum"""
        graph = defaultdict(list)
        
        for i, oligo1 in enumerate(self.spectrum):
            for j, oligo2 in enumerate(self.spectrum):
                if i != j:
                    # Sprawdź wszystkie możliwe nakładania
                    for overlap in range(1, self.k):
                        if oligo1[self.k - overlap:] == oligo2[:overlap]:
                            weight = overlap  # waga = liczba nakładających się nukleotydów
                            graph[oligo1].append((oligo2, weight))
        
        return graph
    
    def _initialize_pheromones(self):
        """Inicjalizuje poziomy feromonów"""
        pheromones = defaultdict(lambda: defaultdict(float))
        initial_pheromone = 1.0
        
        for oligo1 in self.spectrum:
            for oligo2, weight in self.graph[oligo1]:
                pheromones[oligo1][oligo2] = initial_pheromone
                
        return pheromones
    
    def _calculate_heuristic(self, current_oligo, next_oligo, weight, current_length):
        """Oblicza wartość heurystyczną dla przejścia"""
        # Preferuj większe nakładania (mniejsze wagi w kontekście rozszerzania DNA)
        overlap_bonus = weight / self.k
        
        # Preferuj oligonukleotydy prowadzące do celu
        remaining_length = self.n - current_length
        progress_bonus = min(1.0, (self.k - weight) / max(1, remaining_length))
        
        # Kara za zbyt długie rozwiązanie
        length_penalty = max(0, (current_length - self.n) / self.n) if current_length > self.n else 0
        
        return overlap_bonus + progress_bonus - length_penalty
    
    def _construct_solution(self):
        """Konstruuje rozwiązanie dla jednej mrówki"""
        path = [self.start_oligo]
        visited = {self.start_oligo}
        current_length = self.k
        
        current_oligo = self.start_oligo
        
        while current_length < self.n and len(path) < len(self.spectrum) * 2:
            if current_oligo not in self.graph or not self.graph[current_oligo]:
                # Brak dostępnych przejść - znajdź alternatywę
                available = [o for o in self.spectrum if o not in visited]
                if not available:
                    break
                current_oligo = random.choice(available)
                path.append(current_oligo)
                visited.add(current_oligo)
                current_length += self.k
                continue
            
            # Zbierz dostępne przejścia
            candidates = []
            for next_oligo, weight in self.graph[current_oligo]:
                if next_oligo not in visited or len(visited) > len(self.spectrum) * 0.8:
                    pheromone = self.pheromones[current_oligo][next_oligo]
                    heuristic = self._calculate_heuristic(current_oligo, next_oligo, weight, current_length)
                    candidates.append((next_oligo, weight, pheromone, heuristic))
            
            if not candidates:
                break
            
            # Wybór następnego oligonukleotydu
            if random.random() < self.q0:
                # Eksploatacja - wybierz najlepszy
                next_oligo, weight = max(candidates, 
                    key=lambda x: x[2] ** self.alpha * x[3] ** self.beta)[:2]
            else:
                # Eksploracja - wybór probabilistyczny
                probabilities = []
                for _, weight, pheromone, heuristic in candidates:
                    prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                    probabilities.append(prob)
                
                if sum(probabilities) == 0:
                    next_oligo, weight = random.choice(candidates)[:2]
                else:
                    probabilities = [p / sum(probabilities) for p in probabilities]
                    choice_idx = np.random.choice(len(candidates), p=probabilities)
                    next_oligo, weight = candidates[choice_idx][:2]
            
            path.append(next_oligo)
            visited.add(next_oligo)
            current_length += (self.k - weight)
            current_oligo = next_oligo
        
        return path
    
    def _evaluate_solution(self, path):
        """Ocenia jakość rozwiązania"""
        if not path:
            return float('-inf')
        
        # Rekonstruuj DNA z ścieżki
        dna = self._path_to_dna(path)
        
        # Kryteria oceny
        length_score = -abs(len(dna) - self.n) / self.n  # Kara za nieprawidłową długość
        coverage_score = len(set(path)) / len(self.spectrum)  # Pokrycie spektrum
        
        # Sprawdź poprawność nakładań
        overlap_score = 0
        for i in range(len(path) - 1):
            best_overlap = 0
            for overlap in range(1, self.k):
                if path[i][self.k - overlap:] == path[i + 1][:overlap]:
                    best_overlap = max(best_overlap, overlap)
            overlap_score += best_overlap / (self.k * (len(path) - 1)) if len(path) > 1 else 0
        
        return length_score + coverage_score + overlap_score
    
    def _path_to_dna(self, path):
        """Konwertuje ścieżkę oligonukleotydów na sekwencję DNA"""
        if not path:
            return ""
        
        dna = path[0]
        
        for i in range(1, len(path)):
            # Znajdź najlepsze nakładanie
            best_overlap = 0
            for overlap in range(1, min(self.k, len(dna), len(path[i])) + 1):
                if dna[-overlap:] == path[i][:overlap]:
                    best_overlap = overlap
            
            # Dodaj nowe nukleotydy
            dna += path[i][best_overlap:]
        
        return dna
    
    def _update_pheromones(self, all_paths, all_fitness):
        """Aktualizuje poziomy feromonów"""
        # Parowanie feromonów
        for oligo1 in self.pheromones:
            for oligo2 in self.pheromones[oligo1]:
                self.pheromones[oligo1][oligo2] *= (1 - self.rho)
        
        # Wzmocnienie najlepszych ścieżek
        best_indices = sorted(range(len(all_fitness)), key=lambda i: all_fitness[i], reverse=True)
        
        for rank, idx in enumerate(best_indices[:5]):  # Top 5 rozwiązań
            path = all_paths[idx]
            fitness = all_fitness[idx]
            delta = fitness / (rank + 1)  # Większe wzmocnienie dla lepszych rozwiązań
            
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i + 1]] += delta
    
    def solve(self):
        """Główna funkcja rozwiązująca problem"""
        print(f"Rozpoczynam rozwiązywanie problemu SBH...")
        print(f"Parametry: n={self.n}, k={self.k}, spektrum={len(self.spectrum)} oligonukleotydów")
        print(f"Błędy: negatywne={self.negative_errors}, pozytywne={self.positive_errors}")
        
        start_time = time.time()
        
        for iteration in range(self.num_iterations):
            # Konstruuj rozwiązania dla wszystkich mrówek
            all_paths = []
            all_fitness = []
            
            for ant in range(self.num_ants):
                path = self._construct_solution()
                fitness = self._evaluate_solution(path)
                
                all_paths.append(path)
                all_fitness.append(fitness)
                
                # Aktualizuj najlepsze rozwiązanie
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = path
            
            # Aktualizuj feromony
            self._update_pheromones(all_paths, all_fitness)
            
            # Logowanie
            avg_fitness = np.mean(all_fitness)
            self.log_data.append({
                'iteration': iteration,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness,
                'best_length': len(self._path_to_dna(self.best_solution)) if self.best_solution else 0
            })
            
            if (iteration + 1) % 10 == 0:
                best_dna = self._path_to_dna(self.best_solution) if self.best_solution else ""
                print(f"Iteracja {iteration + 1}: Najlepszy fitness = {self.best_fitness:.4f}, "
                      f"Długość DNA = {len(best_dna)}")
        
        end_time = time.time()
        print(f"Zakończono po {end_time - start_time:.2f} sekundach")
        
        return self.best_solution, self.best_fitness


def generate_dna(length):
    """Generuje losową sekwencję DNA"""
    nucleotides = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(nucleotides) for _ in range(length))


def create_spectrum(dna, k):
    """Tworzy idealne spektrum z sekwencji DNA"""
    spectrum = []
    for i in range(len(dna) - k + 1):
        spectrum.append(dna[i:i + k])
    return spectrum


def add_positive_errors(spectrum, num_errors, k):
    """Dodaje błędy pozytywne do spektrum"""
    if num_errors == 0 or not spectrum:
        return spectrum
    
    errors = []
    nucleotides = ['A', 'T', 'C', 'G']
    
    for _ in range(num_errors):
        # Wybierz bazowy oligonukleotyd
        base = random.choice(spectrum)
        
        # Utwórz błąd przez zmianę ostatniego lub środkowego nukleotydu
        if random.choice([True, False]):
            # Zmień ostatni nukleotyd
            new_oligo = base[:-1] + random.choice([n for n in nucleotides if n != base[-1]])
        else:
            # Zmień środkowy nukleotyd
            mid = k // 2
            new_nucleotide = random.choice([n for n in nucleotides if n != base[mid]])
            new_oligo = base[:mid] + new_nucleotide + base[mid + 1:]
        
        errors.append(new_oligo)
    
    return spectrum + errors


def add_negative_errors(spectrum, num_errors):
    """Dodaje błędy negatywne przez usunięcie elementów ze spektrum"""
    if num_errors == 0 or num_errors >= len(spectrum):
        return spectrum
    
    spectrum_copy = spectrum.copy()
    for _ in range(num_errors):
        if spectrum_copy:
            spectrum_copy.remove(random.choice(spectrum_copy))
    
    return spectrum_copy


def save_results(original_dna, reconstructed_dna, solver, filename_prefix="sbh_results"):
    """Zapisuje wyniki do plików"""
    
    # Tworzenie katalogu wyników
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Zapisz podstawowe informacje
    with open(f"results/{filename_prefix}_info.txt", "w") as f:
        f.write("=== WYNIKI REKONSTRUKCJI DNA METODĄ SBH Z ALGORYTMEM MRÓWKOWYM ===\n\n")
        f.write(f"Parametry problemu:\n")
        f.write(f"- Długość oryginalnego DNA (n): {len(original_dna)}\n")
        f.write(f"- Długość oligonukleotydów (k): {solver.k}\n")
        f.write(f"- Rozmiar spektrum: {len(solver.spectrum)}\n")
        f.write(f"- Błędy negatywne: {solver.negative_errors}\n")
        f.write(f"- Błędy pozytywne: {solver.positive_errors}\n")
        f.write(f"- Oligonukleotyd startowy: {solver.start_oligo}\n\n")
        
        f.write(f"Parametry algorytmu mrówkowego:\n")
        f.write(f"- Liczba mrówek: {solver.num_ants}\n")
        f.write(f"- Liczba iteracji: {solver.num_iterations}\n")
        f.write(f"- Alpha (wpływ feromonów): {solver.alpha}\n")
        f.write(f"- Beta (wpływ heurystyki): {solver.beta}\n")
        f.write(f"- Rho (parowanie feromonów): {solver.rho}\n\n")
        
        f.write(f"Wyniki:\n")
        f.write(f"- Długość zrekonstruowanego DNA: {len(reconstructed_dna)}\n")
        f.write(f"- Najlepszy fitness: {solver.best_fitness:.6f}\n")
        
        # Oblicz dokładność
        if original_dna and reconstructed_dna:
            min_len = min(len(original_dna), len(reconstructed_dna))
            matches = sum(1 for i in range(min_len) if original_dna[i] == reconstructed_dna[i])
            accuracy = matches / max(len(original_dna), len(reconstructed_dna))
            f.write(f"- Dokładność rekonstrukcji: {accuracy:.4f} ({matches}/{max(len(original_dna), len(reconstructed_dna))})\n")
    
    # Zapisz sekwencje DNA
    with open(f"results/{filename_prefix}_sequences.txt", "w") as f:
        f.write("ORYGINALNE DNA:\n")
        f.write(f"{original_dna}\n\n")
        f.write("ZREKONSTRUOWANE DNA:\n")
        f.write(f"{reconstructed_dna}\n\n")
        
        # Wizualne porównanie (pierwsze 100 znaków)
        f.write("PORÓWNANIE (pierwsze 100 nukleotydów):\n")
        comparison_len = min(100, len(original_dna), len(reconstructed_dna))
        f.write("ORYG: " + original_dna[:comparison_len] + "\n")
        f.write("REKN: " + reconstructed_dna[:comparison_len] + "\n")
        f.write("DOPR: " + "".join("+" if i < len(original_dna) and i < len(reconstructed_dna) 
                                   and original_dna[i] == reconstructed_dna[i] else "-" 
                                   for i in range(comparison_len)) + "\n")
    
    # Zapisz spektrum
    with open(f"results/{filename_prefix}_spectrum.txt", "w") as f:
        f.write("SPEKTRUM OLIGONUKLEOTYDÓW:\n")
        for i, oligo in enumerate(solver.spectrum):
            f.write(f"{i+1:3d}: {oligo}\n")
    
    # Zapisz log z iteracji
    with open(f"results/{filename_prefix}_log.txt", "w") as f:
        f.write("ITERACJA\tNAJLEPSZY_FITNESS\tŚREDNI_FITNESS\tDŁUGOŚĆ_DNA\n")
        for entry in solver.log_data:
            f.write(f"{entry['iteration']+1}\t{entry['best_fitness']:.6f}\t"
                   f"{entry['avg_fitness']:.6f}\t{entry['best_length']}\n")
    
    print(f"Wyniki zapisane w katalogu 'results' z prefiksem '{filename_prefix}'")


def main(seed=42, test_name="default", custom_params=None, use_percentage_errors=False):
    """
    Funkcja główna z możliwością parametryzacji
    
    Args:
        seed: Seed dla generatora liczb losowych (domyślnie 42)
        test_name: Nazwa testu dla plików wynikowych
        custom_params: Słownik z niestandardowymi parametrami algorytmu
                      np. {'num_ants': 50, 'num_iterations': 200}
    """
    # Ustawienie seed dla reprodukowalności
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"=== TEST: {test_name.upper()} (SEED: {seed}) ===\n")
    
    # Parametry problemu
    n = 50  # długość DNA
    k = 7   # długość oligonukleotydów

    if use_percentage_errors:
        negative_errors_percent = 0.05  # 5%
        positive_errors_percent = 0.05  # 7%
    else:
        negative_errors = 2
        positive_errors = 3
    
    # Generuj testowe dane
    print("Generowanie danych testowych...")
    original_dna = generate_dna(n)
    print(f"Oryginalne DNA (seed {seed}): {original_dna}")
    
    # Utwórz spektrum
    perfect_spectrum = create_spectrum(original_dna, k)
    print(f"Idealne spektrum zawiera {len(perfect_spectrum)} oligonukleotydów")

    if use_percentage_errors:
        negative_errors = max(0, int(len(perfect_spectrum) * negative_errors_percent))
        positive_errors = max(0, int(len(perfect_spectrum) * positive_errors_percent))
    
    # Dodaj błędy
    spectrum_with_errors = add_positive_errors(perfect_spectrum, positive_errors, k)
    spectrum_with_errors = add_negative_errors(spectrum_with_errors, negative_errors)
    
    # Wymieszaj spektrum (symulacja rzeczywistego eksperymentu SBH)
    random.shuffle(spectrum_with_errors)
    
    print(f"Spektrum z błędami zawiera {len(spectrum_with_errors)} oligonukleotydów")
    print(f"Błędy: {negative_errors} negatywne, {positive_errors} pozytywne")
    
    # Wybierz oligonukleotyd startowy
    start_oligo = perfect_spectrum[0]  # Pierwszy oligonukleotyd z oryginalnego DNA
    print(f"Oligonukleotyd startowy: {start_oligo}")
    
    # Utwórz solver
    print("\n" + "="*60)
    
    if use_percentage_errors:
        solver = SBHSolver(n, k, spectrum_with_errors, start_oligo, 
                        negative_errors_percent, positive_errors_percent,
                        use_percentage=True, original_spectrum_size=len(perfect_spectrum))
    else:
        solver = SBHSolver(n, k, spectrum_with_errors, start_oligo, negative_errors, positive_errors)
    
    # Zastosuj niestandardowe parametry jeśli podane
    if custom_params:
        print("Zastosowane niestandardowe parametry:")
        for param, value in custom_params.items():
            if hasattr(solver, param):
                setattr(solver, param, value)
                print(f"- {param}: {value}")
            else:
                print(f"- UWAGA: Nieznany parametr '{param}' zostanie zignorowany")
    
    # Pokaż aktualne parametry algorytmu
    print(f"\nParametry algorytmu mrówkowego:")
    print(f"- Liczba mrówek: {solver.num_ants}")
    print(f"- Liczba iteracji: {solver.num_iterations}")
    print(f"- Alpha: {solver.alpha}")
    print(f"- Beta: {solver.beta}")
    print(f"- Rho: {solver.rho}")
    print(f"- Q0: {solver.q0}")
    
    # Rozwiąż problem
    best_path, best_fitness = solver.solve()
    
    # Zrekonstruuj DNA
    if best_path:
        reconstructed_dna = solver._path_to_dna(best_path)
        print(f"\nZrekonstruowane DNA: {reconstructed_dna}")
        print(f"Długość: {len(reconstructed_dna)} (oczekiwana: {n})")
        print(f"Najlepszy fitness: {best_fitness:.6f}")
        
        # Oblicz dokładność
        if len(original_dna) > 0 and len(reconstructed_dna) > 0:
            min_len = min(len(original_dna), len(reconstructed_dna))
            matches = sum(1 for i in range(min_len) if original_dna[i] == reconstructed_dna[i])
            accuracy = matches / max(len(original_dna), len(reconstructed_dna))
            print(f"Dokładność: {accuracy:.4f} ({matches}/{max(len(original_dna), len(reconstructed_dna))})")
        
        # Zapisz wyniki z odpowiednim prefiksem
        filename_prefix = f"{test_name}_seed{seed}"
        save_results(original_dna, reconstructed_dna, solver, filename_prefix)
        
    else:
        print("Nie udało się znaleźć rozwiązania!")
        reconstructed_dna = ""
        filename_prefix = f"{test_name}_seed{seed}"
        save_results(original_dna, reconstructed_dna, solver, filename_prefix)


def run_iteration_tests():
    """Funkcja do testowania różnych liczb iteracji na tym samym DNA"""
    print("ROZPOCZYNAM TESTY LICZBY ITERACJI (SEED=42)")
    print("="*70)
    
    iteration_configs = [
        {'num_iterations': 50},
        {'num_iterations': 100},
        {'num_iterations': 150},
        {'num_iterations': 200},
        {'num_iterations': 300}
    ]
    
    for i, config in enumerate(iteration_configs, 1):
        print(f"\nTEST {i}/5: {config['num_iterations']} iteracji")
        test_name = f"iterations_test_{config['num_iterations']}"
        main(seed=42, test_name=test_name, custom_params=config)
        print("-" * 50)


def run_ants_tests():
    """Funkcja do testowania różnych liczb mrówek na innym DNA"""
    print("ROZPOCZYNAM TESTY LICZBY MRÓWEK (SEED=123)")
    print("="*70)
    
    ants_configs = [
        {'num_ants': 10},
        {'num_ants': 20},
        {'num_ants': 30},
        {'num_ants': 50},
        {'num_ants': 100}
    ]
    
    for i, config in enumerate(ants_configs, 1):
        print(f"\n🐜 TEST {i}/5: {config['num_ants']} mrówek")
        test_name = f"ants_test_{config['num_ants']}"
        main(seed=123, test_name=test_name, custom_params=config)
        print("-" * 50)


def run_alpha_beta_tests():
    """Funkcja do testowania różnych parametrów alpha i beta na kolejnym DNA"""
    print("ROZPOCZYNAM TESTY PARAMETRÓW ALPHA/BETA (SEED=456)")
    print("="*70)
    
    alpha_beta_configs = [
        {'alpha': 0.5, 'beta': 1.5},
        {'alpha': 1.0, 'beta': 2.0},  # domyślne
        {'alpha': 1.5, 'beta': 2.5},
        {'alpha': 2.0, 'beta': 1.0},
        {'alpha': 0.8, 'beta': 3.0}
    ]
    
    for i, config in enumerate(alpha_beta_configs, 1):
        print(f"\nTEST {i}/5: Alpha={config['alpha']}, Beta={config['beta']}")
        test_name = f"alpha_beta_test_a{config['alpha']}_b{config['beta']}"
        main(seed=456, test_name=test_name, custom_params=config)
        print("-" * 50)


if __name__ == "__main__":
    print("SYSTEM TESTOWANIA ALGORYTMU SBH Z ALGORYTMEM MRÓWKOWYM")
    print("="*80)
    
    # Możesz uruchomić jeden z poniższych testów:
    
    # 1. Test podstawowy
    main(seed=42, test_name="baseline")
    
    # 2. Testy liczby iteracji
    # run_iteration_tests()
    
    # 3. Testy liczby mrówek
    # run_ants_tests()
    
    # 4. Testy parametrów alpha/beta
    # run_alpha_beta_tests()
    
    # 5. Pojedynczy test z niestandardowymi parametrami
    # main(seed=789, test_name="custom_test", 
    #      custom_params={'num_ants': 50, 'num_iterations': 200, 'alpha': 1.5})