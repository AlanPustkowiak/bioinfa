import random
import nltk
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime
import heapq
import multiprocessing as mp
import os

# Parametry instancji
dlugosc_dna = 500
dlugosc_oligo = 9
ilosc_instancji = 25

# Procenty błędów (względem długości spektrum)
procent_bledow_negatywnych = 0.02  
procent_bledow_pozytywnych = 0.02  


parametry_aco = {
    'iteracje': 200,
    'mrowki': 200,
    'alfa': 2.0, 
    'beta': 2.0, 
    'rho': 0.6,  
    'Q': 100.0    # stała jakości
}

class GrafSBH:
    def __init__(self, spektrum: List[str], k: int):
        self.spektrum = spektrum
        self.k = k
        self.graf = self._zbuduj_graf()
        self.feromony = defaultdict(lambda: defaultdict(lambda: 1.0))

    def _zbuduj_graf(self) -> Dict[str, List[Tuple[str, int]]]:
        graf = defaultdict(list)
        for oligo_a in self.spektrum:
            for oligo_b in self.spektrum:
                if oligo_a == oligo_b:
                    continue
                for wspolne in range(self.k-1, 0, -1):
                    if oligo_a[-wspolne:] == oligo_b[:wspolne]:
                        waga = self.k - wspolne
                        if waga <= 3:
                            graf[oligo_a].append((oligo_b, waga))
                            break
        return graf

    def znajdz_sciezke_dijkstra(self, start: str, cele: List[str], odwiedzone: Set[str]) -> Optional[Tuple[List[str], int]]:
        if not cele:
            return None
        pq = [(0, start, [start])]
        odwiedzone_dijkstra = set()
        while pq:
            koszt, aktualny, sciezka = heapq.heappop(pq)
            if aktualny in odwiedzone_dijkstra:
                continue
            odwiedzone_dijkstra.add(aktualny)
            if aktualny in cele and aktualny not in odwiedzone:
                return sciezka, koszt
            for sasiad, waga in self.graf.get(aktualny, []):
                if sasiad not in odwiedzone_dijkstra:
                    nowa_sciezka = sciezka + [sasiad]
                    heapq.heappush(pq, (koszt + waga, sasiad, nowa_sciezka))
        return None

    def znajdz_kandydatow_Y(self, odwiedzone: Set[str], max_kandydatow: int = 8) -> List[str]:
        kandydaci = []
        for wierzcholek in self.spektrum:
            if wierzcholek in odwiedzone:
                continue
            sasiedzi = [s for s, _ in self.graf.get(wierzcholek, [])]
            if sasiedzi and all(s not in odwiedzone for s in sasiedzi):
                kandydaci.append(wierzcholek)
                if len(kandydaci) >= max_kandydatow:
                    break
        return kandydaci

class MrowkaSBH:
    def __init__(self, graf: GrafSBH, start: str, cel_dlugosc: int, aco_params: Dict):
        self.graf = graf
        self.start = start
        self.cel_dlugosc = cel_dlugosc
        self.k = graf.k
        self.aco_params = aco_params

    def wybierz_nastepny_waga1(self, aktualny: str, odwiedzone: Set[str]) -> Optional[str]:
        kandidaci = [sasiad for sasiad, waga in self.graf.graf.get(aktualny, []) if waga == 1 and sasiad not in odwiedzone]
        if not kandidaci:
            return None
        suma = 0
        prawdopodobienstwa = []
        alfa = self.aco_params['alfa']
        beta = self.aco_params['beta']
        for sasiad in kandidaci:
            feromon = self.graf.feromony[aktualny][sasiad]
            heurystyka = 1.0
            p = (feromon ** alfa) * (heurystyka ** beta)
            prawdopodobienstwa.append((sasiad, p))
            suma += p
        if suma == 0:
            return random.choice(kandidaci)
        los = random.uniform(0, suma)
        akumulator = 0
        for sasiad, p in prawdopodobienstwa:
            akumulator += p
            if akumulator >= los:
                return sasiad
        return kandidaci[-1]

    def buduj_sciezke(self) -> Tuple[List[str], int]:
        sciezka = [self.start]
        odwiedzone = {self.start}
        aktualna_dlugosc = self.k
        aktualny = self.start
        while aktualna_dlugosc < self.cel_dlugosc:
            while True:
                nastepny = self.wybierz_nastepny_waga1(aktualny, odwiedzone)
                if nastepny is None:
                    break
                sciezka.append(nastepny)
                odwiedzone.add(nastepny)
                aktualna_dlugosc += 1
                aktualny = nastepny
                if aktualna_dlugosc >= self.cel_dlugosc:
                    break
            if aktualna_dlugosc >= self.cel_dlugosc:
                break
            kandydaci_Y = self.graf.znajdz_kandydatow_Y(odwiedzone)
            if not kandydaci_Y:
                nastepny = self._stworz_wirtualny_luk(aktualny, odwiedzone)
                if nastepny:
                    waga = self._oblicz_wage_wirtualna(aktualny, nastepny)
                    sciezka.append(nastepny)
                    odwiedzone.add(nastepny)
                    aktualna_dlugosc += waga
                    aktualny = nastepny
                else:
                    break
            else:
                wynik = self.graf.znajdz_sciezke_dijkstra(aktualny, kandydaci_Y, odwiedzone)
                if wynik:
                    sciezka_do_Y, koszt = wynik
                    for wierzcholek in sciezka_do_Y[1:]:
                        sciezka.append(wierzcholek)
                        odwiedzone.add(wierzcholek)
                    dodana_dlugosc = self._oblicz_dlugosc_sciezki(sciezka_do_Y)
                    aktualna_dlugosc += dodana_dlugosc
                    aktualny = sciezka_do_Y[-1]
                else:
                    break
        return sciezka, aktualna_dlugosc

    def _oblicz_dlugosc_sciezki(self, sciezka_fragmenty: List[str]) -> int:
        if len(sciezka_fragmenty) <= 1:
            return 0
        dlugosc = 0
        for i in range(len(sciezka_fragmenty) - 1):
            a, b = sciezka_fragmenty[i], sciezka_fragmenty[i + 1]
            for sasiad, waga in self.graf.graf.get(a, []):
                if sasiad == b:
                    dlugosc += waga
                    break
        return dlugosc

    def _stworz_wirtualny_luk(self, aktualny: str, odwiedzone: Set[str]) -> Optional[str]:
        for waga_wirtualna in range(4, min(self.k + 1, 8)):
            sufix = aktualny[-waga_wirtualna:]
            for kandydat in self.graf.spektrum:
                if kandydat not in odwiedzone and kandydat.startswith(sufix):
                    return kandydat
        return None

    def _oblicz_wage_wirtualna(self, a: str, b: str) -> int:
        for waga in range(4, self.k + 1):
            if len(a) >= waga and len(b) >= waga:
                if a[-waga:] == b[:waga]:
                    return waga
        return self.k

def generuj_DNA(n: int) -> str:
    return ''.join(random.choices('ATCG', k=n))

def generuj_spektrum(dna: str, k: int) -> List[str]:
    return [dna[i:i+k] for i in range(len(dna) - k + 1)]

def dodaj_bledy_negatywne(spektrum: List[str], ilosc_bledow: int) -> List[str]:
    spektrum = deepcopy(spektrum)
    pierwszy = spektrum[0]
    mozliwe_do_usuniecia = [i for i in range(1, len(spektrum))]
    random.shuffle(mozliwe_do_usuniecia)
    for _ in range(min(ilosc_bledow, len(mozliwe_do_usuniecia))):
        if mozliwe_do_usuniecia:
            idx = mozliwe_do_usuniecia.pop()
            spektrum.pop(idx)
            mozliwe_do_usuniecia = [i-1 if i > idx else i for i in mozliwe_do_usuniecia]
    return spektrum

def dodaj_bledy_pozytywne(spektrum: List[str], ilosc_bledow: int) -> List[str]:
    spektrum = deepcopy(spektrum)
    dodatki = []
    k = len(spektrum[0]) if spektrum else 7
    while len(dodatki) < ilosc_bledow and spektrum:
        bazowy = random.choice(spektrum)
        bledy_z_bazowego = []
        if len(bazowy) > 0:
            ostatni_char = bazowy[-1]
            nowy_ostatni = random.choice([x for x in "ACGT" if x != ostatni_char])
            blad1 = bazowy[:-1] + nowy_ostatni
            bledy_z_bazowego.append(blad1)
        if len(bazowy) > 2:
            srodek_idx = len(bazowy) // 2
            srodkowy_char = bazowy[srodek_idx]
            nowy_srodkowy = random.choice([x for x in "ACGT" if x != srodkowy_char])
            blad2 = bazowy[:srodek_idx] + nowy_srodkowy + bazowy[srodek_idx+1:]
            bledy_z_bazowego.append(blad2)
        for blad in bledy_z_bazowego:
            if (blad not in spektrum and blad not in dodatki and len(dodatki) < ilosc_bledow):
                dodatki.append(blad)
    return spektrum + dodatki

def sciezka_na_dna(sciezka: List[str]) -> str:
    if not sciezka:
        return ""
    dna = sciezka[0]
    for i in range(1, len(sciezka)):
        poprzedni, aktualny = sciezka[i-1], sciezka[i]
        max_overlap = 0
        for overlap in range(min(len(poprzedni), len(aktualny)), 0, -1):
            if poprzedni[-overlap:] == aktualny[:overlap]:
                max_overlap = overlap
                break
        if max_overlap > 0:
            dna += aktualny[max_overlap:]
        else:
            dna += aktualny
    return dna

def algorytm_aco_sbh(spektrum: List[str], start: str, k: int, cel_dlugosc: int, aco_params: Dict = None) -> str:
    test_aco_params = parametry_aco.copy()
    if aco_params is not None:
        test_aco_params.update(aco_params)
    graf = GrafSBH(spektrum, k)
    najlepszy_dna = ""
    najlepszy_wynik = float('inf')
    for iteracja in range(test_aco_params['iteracje']):
        wszystkie_sciezki = []
        for _ in range(test_aco_params['mrowki']):
            mrowka = MrowkaSBH(graf, start, cel_dlugosc, test_aco_params)
            sciezka, dlugosc = mrowka.buduj_sciezke()
            dna = sciezka_na_dna(sciezka)
            ocena = abs(len(dna) - cel_dlugosc)
            wszystkie_sciezki.append((sciezka, ocena))
            if ocena < najlepszy_wynik:
                najlepszy_wynik = ocena
                najlepszy_dna = dna
        _aktualizuj_feromony(graf, wszystkie_sciezki, test_aco_params)
    return najlepszy_dna

def _aktualizuj_feromony(graf: GrafSBH, wszystkie_sciezki: List[Tuple[List[str], float]], aco_params: Dict):
    rho = aco_params['rho']
    Q = aco_params['Q']
    for a in graf.feromony:
        for b in graf.feromony[a]:
            graf.feromony[a][b] *= (1 - rho)
    for sciezka, ocena in wszystkie_sciezki:
        if len(sciezka) > 1 and ocena > 0:
            wzmocnienie = Q / ocena
            for i in range(len(sciezka) - 1):
                a, b = sciezka[i], sciezka[i + 1]
                graf.feromony[a][b] += wzmocnienie

def lewensztajn(s1: str, s2: str) -> int:
    return nltk.edit_distance(s1, s2)

def process_instance(args):
    test_name, config, dna, procent_neg, procent_poz, pomiar_id = args
    merged_config = parametry_aco.copy()
    merged_config.update(config)
    k = merged_config.get('dlugosc_oligo', dlugosc_oligo)
    print(f"Przetwarzam instancję DNA dla testu: {test_name}, config: {merged_config}")
    spektrum = generuj_spektrum(dna, k)
    dlugosc_spektrum = len(spektrum)
    bledy_neg = int(dlugosc_spektrum * procent_neg)
    bledy_poz = int(dlugosc_spektrum * procent_poz)
    spektrum_z_bledami = dodaj_bledy_negatywne(spektrum, bledy_neg)
    spektrum_z_bledami = dodaj_bledy_pozytywne(spektrum_z_bledami, bledy_poz)
    start = spektrum[0]
    odtworzone = algorytm_aco_sbh(spektrum_z_bledami, start, k, dlugosc_dna, merged_config)
    miara = lewensztajn(dna, odtworzone)
    print(f"Test: {test_name}, instancja DNA: {miara} (odległość Levenshteina)")
    return miara

def run_test_parallel(test_name, config, instancje, procent_neg=None, procent_poz=None, pomiar_id=None):
    if procent_neg is None:
        procent_neg = procent_bledow_negatywnych
    if procent_poz is None:
        procent_poz = procent_bledow_pozytywnych
    args = [(test_name, config, dna, procent_neg, procent_poz, pomiar_id) for dna in instancje]
    print(f"Start testu równoległego {test_name} dla {len(instancje)} instancji, config: {config}")
    with mp.Pool() as pool:
        wyniki = pool.map(process_instance, args)
    suma_miar = sum(wyniki)
    srednia = suma_miar / len(instancje)
    print(f"Test {test_name} ukończony - średnia: {srednia:.2f} (odległość Levenshteina)")
    return srednia, test_name, config, procent_neg, procent_poz, pomiar_id

def run_ants_tests_parallel():
    instancje = [generuj_DNA(dlugosc_dna) for _ in range(ilosc_instancji)]
    ants_configs = [
        {'mrowki': 100, 'iteracje': 100},
        {'mrowki': 200, 'iteracje': 100},
        {'mrowki': 300, 'iteracje': 100},
        {'mrowki': 400, 'iteracje': 100},
        {'mrowki': 500, 'iteracje': 100}
    ]
    wyniki = []
    for config in ants_configs:
        srednia, test_name, _, procent_neg, procent_poz, _ = run_test_parallel(
            f"MROWKI_{config['mrowki']}", config, instancje
        )
        wyniki.append((test_name, config, procent_neg, procent_poz, srednia))
    return wyniki

def run_rho_tests_parallel():
    instancje = [generuj_DNA(dlugosc_dna) for _ in range(ilosc_instancji)]
    rho_values = [0.2, 0.4, 0.6, 0.8, 1]
    wyniki = []
    for rho in rho_values:
        config = {'rho': rho}
        srednia, test_name, _, procent_neg, procent_poz, _ = run_test_parallel(
            f"RHO_{rho:.2f}", config, instancje
        )
        wyniki.append((test_name, config, procent_neg, procent_poz, srednia))
    return wyniki

def run_beta_tests_parallel():
    instancje = [generuj_DNA(dlugosc_dna) for _ in range(ilosc_instancji)]
    beta_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    wyniki = []
    for beta in beta_values:
        config = {'beta': beta}
        srednia, test_name, _, procent_neg, procent_poz, _ = run_test_parallel(
            f"BETA_{beta:.1f}", config, instancje
        )
        wyniki.append((test_name, config, procent_neg, procent_poz, srednia))
    return wyniki

def run_alfa_tests_parallel():
    instancje = [generuj_DNA(dlugosc_dna) for _ in range(ilosc_instancji)]
    alfa_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    wyniki = []
    for alfa in alfa_values:
        config = {'alfa': alfa}
        srednia, test_name, _, procent_neg, procent_poz, _ = run_test_parallel(
            f"ALFA_{alfa:.1f}", config, instancje
        )
        wyniki.append((test_name, config, procent_neg, procent_poz, srednia))
    return wyniki

def run_all_additional_tests_parallel():
    print("=== Uruchamiam dodatkowe testy równolegle ===")
    wyniki = []
    wyniki += run_rho_tests_parallel()
    wyniki += run_beta_tests_parallel()
    wyniki += run_alfa_tests_parallel()
    print("=== Dodatkowe testy równolegle ukończone ===")
    return wyniki

def run_error_percent_tests_parallel():
    """Testy różnych procentów błędów negatywnych/pozytywnych."""
    print("=== Uruchamiam testy procentów błędów równolegle ===")
    instancje = [generuj_DNA(dlugosc_dna) for _ in range(ilosc_instancji)]
    procent_bledow_zmienne = [0.01, 0.02, 0.03, 0.04, 0.05]    # Do testów 1 i 2
    procent_bledow_obydwa = [0.02, 0.04, 0.06, 0.08, 0.1]     # Do testu 3
    wyniki = []
    
    #Stały POZ (np. 0.01) zmienne NEG (0.02, 0.04, 0.06)
    stały_poz = 0.02
    for procent in procent_bledow_zmienne:
        srednia, test_name, _, procent_neg, procent_poz, _ = run_test_parallel(
            f"BLEDY_NEG_{procent:.2f}_POZ_{stały_poz:.2f}",
            {},
            instancje,
            procent_neg=procent,
            procent_poz=stały_poz
        )
        wyniki.append((test_name, {}, procent, stały_poz, srednia))
    
    #Stały NEG (np. 0.01) zmienne POZ (0.02, 0.04, 0.06)
    stały_neg = 0.02
    for procent in procent_bledow_zmienne:
        srednia, test_name, _, procent_neg, procent_poz, _ = run_test_parallel(
            f"BLEDY_POZ_{procent:.2f}_NEG_{stały_neg:.2f}",
            {},
            instancje,
            procent_neg=stały_neg,
            procent_poz=procent
        )
        wyniki.append((test_name, {}, stały_neg, procent, srednia))
    
    #Zmienny procent dla obu błędów (0.03, 0.05, 0.07)
    for procent in procent_bledow_obydwa:
        srednia, test_name, _, procent_neg, procent_poz, _ = run_test_parallel(
            f"BLEDY_OBU_{procent:.2f}",
            {},
            instancje,
            procent_neg=procent,
            procent_poz=procent
        )
        wyniki.append((test_name, {}, procent, procent, srednia))
    
    print("=== Testy procentów błędów ukończone ===")
    return wyniki

def run_oligo_size_tests_parallel():
    """Testy różnych rozmiarów oligonukleotydów."""
    print("=== Uruchamiam testy rozmiaru oligo równolegle ===")
    instancje = [generuj_DNA(dlugosc_dna) for _ in range(ilosc_instancji)]
    rozmiary_oligo = [8, 9, 10, 11, 12]
    wyniki = []
    for k in rozmiary_oligo:
        config = {'dlugosc_oligo': k}
        srednia, test_name, _, procent_neg, procent_poz, _ = run_test_parallel(
            f"OLIGO_{k}",
            config,
            instancje
        )
        wyniki.append((test_name, config, procent_neg, procent_poz, srednia))
    print("=== Testy rozmiaru oligo ukończone ===")
    return wyniki

def main():
    print(f"Start działania - {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    with open('wyniki_dna.csv', 'w', newline='') as f:
        pass
    with open('wyniki_dna_verify.csv', 'w', newline='') as f:
        pass
    print("Pliki wynikowe wyczyszczone.")

    wyniki = []
    #print("=== Uruchamiam testy liczby mrówek równolegle ===")
    #wyniki += run_ants_tests_parallel()
    #print("=== Testy liczby mrówek równolegle ukończone ===")

    #print("=== Uruchamiam dodatkowe testy równolegle ===")
    #wyniki += run_all_additional_tests_parallel()
    #print("=== Dodatkowe testy równolegle ukończone ===")

    print("=== Uruchamiam testy rozmiaru oligo ===")
    wyniki += run_oligo_size_tests_parallel()
    print("=== Testy rozmiaru oligo ukończone ===")

    print("=== Uruchamiam testy procentów błędów ===")
    #wyniki += run_error_percent_tests_parallel()
    print("=== Testy procentów błędów ukończone ===")

    with open('wyniki_dna.csv', 'a', newline='') as plik:
        writer = csv.writer(plik)
        for (test_name, config, procent_neg, procent_poz, srednia) in wyniki:
            row_data = [
                "TEST=", test_name,
                "CONFIG=", str(config),
                "PROCENT_NEG=", f"{procent_neg:.3f}",
                "PROCENT_POZ=", f"{procent_poz:.3f}",
                "SREDNIA_LEWENSZTAJN=", srednia
            ]
            writer.writerow(row_data)
    print("Średnie wyniki zapisane do pliku wyniki_dna.csv")

    print(f"Koniec działania - {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

if __name__ == '__main__':
    main()
