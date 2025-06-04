import random
from collections import defaultdict

def load_spectrum(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip()]

def overlap(a, b):
    # Najdłuższy możliwy suffix a == prefix b
    max_olap = 0
    for i in range(1, len(a)):
        if a[-i:] == b[:i]:
            max_olap = i
    return max_olap

def build_graph(spectrum, k):
    graph = defaultdict(dict)
    for i in range(len(spectrum)):
        for j in range(len(spectrum)):
            if i != j:
                olap = overlap(spectrum[i], spectrum[j])
                if olap > 0:
                    graph[spectrum[i]][spectrum[j]] = olap
    return graph

class AntColonyDNA:
    def __init__(self, spectrum, k, n, start_oligo, ants=100, iterations=100, alpha=1, beta=3, rho=0.1):
        self.spectrum = spectrum
        self.k = k
        self.n = n
        self.start_oligo = start_oligo
        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.graph = build_graph(spectrum, k)
        self.pheromone = defaultdict(lambda: defaultdict(lambda: 1.0))

    def heuristic(self, olap):
        return olap if olap > 0 else 0.1

    def choose_next(self, current, visited):
        neighbors = [v for v in self.graph[current] if v not in visited]
        if not neighbors:
            return None
        weights = []
        for v in neighbors:
            tau = self.pheromone[current][v] ** self.alpha
            eta = self.heuristic(self.graph[current][v]) ** self.beta
            weights.append(tau * eta)
        total = sum(weights)
        if total == 0:
            return random.choice(neighbors)
        probs = [w / total for w in weights]
        return random.choices(neighbors, probs)[0]

    def construct_solution(self):
        path = [self.start_oligo]
        current = self.start_oligo
        while len(''.join([path[0]] + [p[-1] for p in path[1:]])) < self.n:
            next_oligo = self.choose_next(current, set(path))
            if not next_oligo:
                break
            path.append(next_oligo)
            current = next_oligo
        # Zrekonstruowane DNA
        dna = path[0]
        for oligo in path[1:]:
            olap = overlap(dna[-self.k+1:], oligo)
            dna += oligo[olap:]
        return path, dna

    def update_pheromones(self, solutions):
        # Parowanie
        for u in self.pheromone:
            for v in self.pheromone[u]:
                self.pheromone[u][v] *= (1 - self.rho)
        # Wzmocnienie dla najlepszej ścieżki
        best_path, best_dna = min(solutions, key=lambda x: -len(x[1]))
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i + 1]
            self.pheromone[u][v] += 1.0 / (1 + len(best_path))

    def run(self, log_file="aco_log.txt"):
        best_solution = None
        best_dna = ""
        with open(log_file, "w") as log:
            for it in range(self.iterations):
                solutions = []
                for _ in range(self.ants):
                    path, dna = self.construct_solution()
                    solutions.append((path, dna))
                    if len(dna) > len(best_dna):
                        best_solution, best_dna = path, dna
                self.update_pheromones(solutions)
                log.write(f"Iteracja {it+1}: Najlepsza długość DNA: {len(best_dna)}\n")
        return best_solution, best_dna

if __name__ == "__main__":
    spectrum = load_spectrum("spectrum.txt")
    with open("dna.txt") as f:
        original_dna = f.read().strip()
    k = 10
    n = len(original_dna)
    start_oligo = spectrum[0]
    colony = AntColonyDNA(spectrum, k, n, start_oligo, ants=100, iterations=100)
    path, reconstructed = colony.run()
    with open("reconstructed.txt", "w") as f:
        f.write(reconstructed)
