# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 6 — Projekt generacyjny, część 1: przygotowanie danych dla Char-RNN
**Autor / grupa:** Damian Skorupa / GL03
**Data:** 21.11.2026

## 1. Cel ćwiczenia

Przygotowanie danych z polskiego korpusu do przewidywania następnego znaku, które posłużą do treningu Char-RNN na zajęciach 7. Zakres: wybór i oczyszczenie korpusu, podział na trening, walidację i test, słownik znaków, `Dataset` i `DataLoader` z celem przesuniętym o jeden znak, kontrola batchy oraz zbadanie, jak długość okna `SEKW` wpływa na liczbę par.

## 2. Metoda i implementacja

**Narzędzia:** Python 3, PyTorch (`Dataset`, `DataLoader`), `numpy`, `matplotlib`. Kod z wynikami: notatnik `lab06_przygotowanie_danych.ipynb`.

### 2.1. Korpus

- **Źródło:** Wolne Lektury, Bolesław Prus, *Lalka*, tom I, plik TXT (`wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt`). Tekst opracowano na podstawie wydania PIW, wyd. 31, Warszawa 1975.
- **Warunki użycia:** według stopki pliku utwór jest w domenie publicznej. Materiały dodatkowe Wolnych Lektur (przypisy, motywy) są na Licencji Wolnej Sztuki 1.3, a użycie wymaga zachowania zasad opisanych przez Fundację. Tekst wykorzystano do celów edukacyjnych, z podaniem źródła.
- **Dlaczego ten korpus:** to długi tekst (784 082 znaków przed czyszczeniem), jednorodny stylistycznie, w całości po polsku. Jest powieścią realistyczną z narracją i dialogami (w zbiorze treningowym 1 934 linii zaczyna się od myślnika dialogowego) oraz z bogatą polską fleksją i polskimi znakami. Jeden autor i jedna książka dają spójne dane, a rozmiar wystarcza do treningu modelu znakowego.
- **Przygotowanie tekstu:**

| Krok | Skutek |
|---|---|
| odczyt jako UTF-8, normalizacja NFC | tekst bez zmian po NFC; kontrola: brak znaków błędnego kodowania (`Ã`, `Ä`, `Å`), polskie znaki poprawne |
| ujednolicenie końców linii (`\r\n` → `\n`) | jednolite końce linii |
| usunięcie stopki serwisu (od zdania „Ta lektura, podobnie jak tysiące innych…”) | usunięto 1 526 znaków (informacje o licencji, wydawcy i ISBN) |
| usunięcie znaczników HTML | 0 znalezionych |
| sklejenie wielokrotnych spacji i tabulatorów, spacji przy końcach linii | 117 800 zamian |
| ograniczenie pustych linii do jednej | 37 zamian |

Po czyszczeniu tekst ma **772 633 znaków** (101 różnych znaków). Nie usuwano nagłówka (autor, tytuł, ISBN, „Tom I”), tytułów rozdziałów ani *wyróżnień* w gwiazdkach (96 znaków `*` w treningu, np. `*dziewiąta*`). Są to elementy samego tekstu pliku, a ich udział jest pomijalny (mniej niż 0,02% znaków). Model może je jednak odtwarzać przy generowaniu.

Fragmenty po oczyszczeniu:

```text
Bolesław Prus

Lalka

ISBN 978-83-288-2673-1

Tom I

I. Jak wygląda firma J. Mincel i S. Wokulski przez szkło butelek?

W początkach roku 1878, kiedy świat polityczny zajmował się pokojem san-stefańskim, [...]
```

```text
„Wariat czy geniusz?… — szepnął Wokulski czując, że sam jest w najwyższym stopniu rozstrojony. — A jeżeli geniusz?…”

Wstał i poszedł w głąb ogrodu, między spacerujących ludzi. [...]
```

### 2.2. Podział danych

Jeden długi tekst, więc podział na **trzy rozłączne, ciągłe fragmenty**, wykonany **przed** tworzeniem okien: 80% trening, 10% walidacja, 10% test. Granice wyznaczono proporcjami, a następnie przesunięto do najbliższego początku akapitu, żeby żaden fragment nie zaczynał się w środku zdania (indeksy granic: 0, 618 151, 695 626, 772 633). Okna tworzone są osobno w każdej części, więc nigdy nie przekraczają granic podziału. Po złączeniu trzech części odtwarza się cały tekst.

**Uzasadnienie proporcji:** trening ma 618 tys. znaków, wystarczająco dużo do nauki modelu. Walidacja i test mają po ok. 77 tys. znaków, czyli ponad 200 razy więcej niż najdłuższe okno (320), więc dają dziesiątki tysięcy par i stabilne oceny. Mniejsze zbiory dałyby zbyt niepewne oceny, a większe zabrałyby dane treningowi.

### 2.3. Słownik, Dataset, DataLoader

- Słownik budowany **tylko na treningu**: posortowany zbiór różnych znaków. ID 0 zarezerwowano dla znaku spoza słownika (`UNK`, symbol `�`), a walidację i test kodowano tym samym mapowaniem.
- `Dataset[i]` zwraca `x = dane[i : i+SEKW]` i `y = dane[i+1 : i+SEKW+1]`; liczba par to `max(0, N − SEKW)`.
- `DataLoader`: `batch_size = 32`; trening `shuffle=True`, `drop_last=True`; walidacja i test `shuffle=False`, `drop_last=False` (ocena obejmuje wszystkie pary). Ziarno 0.
- Wartość `SEKW` użyta w tabeli i w kontroli batchy: **40**; do eksperymentu 10, 20, 40, 80, 160, 320 przy stałym kroku okna 1.

```python
class DatasetZnakowy(Dataset):
    """Pary (wejście, cel) z jednego fragmentu tekstu; cel = wejście przesunięte o 1 znak."""

    def __init__(self, ids, sekw):
        self.ids = ids
        self.sekw = sekw

    def __len__(self):
        return max(0, len(self.ids) - self.sekw)

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        x = self.ids[i:i + self.sekw]
        y = self.ids[i + 1:i + self.sekw + 1]            # przesunięcie o 1
        return torch.from_numpy(x), torch.from_numpy(y)
```

## 3. Wyniki

### 3.1. Podział i liczebności (SEKW = 40, batch = 32)

| Zbiór | Dokumentów | Znaków | Udział w korpusie | Różnych znaków | Par | Batchy | Loader |
|---|---|---|---|---|---|---|---|
| trening | 1 | 618 151 | 80,01% | 101 | 618 111 | 19 315 | `shuffle=True`, `drop_last=True` |
| walidacja | 1 | 77 475 | 10,03% | 74 | 77 435 | 2 420 | `shuffle=False`, `drop_last=False` |
| test | 1 | 77 007 | 9,97% | 81 | 76 967 | 2 406 | `shuffle=False`, `drop_last=False` |
| razem | 1 | 772 633 | 100% | 101 | 772 513 | 24 141 | |

Liczba par zgadza się ze wzorem `N − SEKW`, a liczba batchy z `⌊pary/32⌋` dla treningu i `⌈pary/32⌉` dla walidacji i testu (sprawdzono asercjami w kodzie). W treningu `drop_last` pomija w każdej epoce 31 par.

### 3.2. Słownik i kodowanie

- **Rozmiar słownika:** V = **102** (101 znaków z treningu + `UNK`). Są to: znak nowej linii, spacja, interpunkcja i cyfry, wielkie i małe litery łacińskie, polskie litery (ą ć ę ł ń ó ś ź ż i wielkie Ł Ó Ś Ź Ż), cudzysłowy i myślniki (« » ” „ – — …) oraz rzadkie znaki à ç é ö ü.
- **Znaki spoza słownika:** w walidacji i teście **0 wystąpień**. Znaki, których nie ma w walidacji, są wyłącznie rzadkie (cyfry, `*`, `'`, Q, Y, q, v, x, « », Ó, à ç é ö ü, Ź, –), więc zbiór walidacyjny ma 74 różnych znaków wobec 101 w treningu. Odwrotna sytuacja (znak tylko w teście) nie wystąpiła, więc `UNK` nie został w tych danych użyty. Obsługa jest jednak przygotowana i przetestowana.
- **Test kodowania i dekodowania:** `dec(enc(tekst)) == tekst` dla wszystkich trzech części (w całości) oraz dodatkowo dla pierwszych 1 000 znaków każdej części. Wynik: **zgodne**. Zapisane dane `dane_lab06.npz` dekodują się do dokładnie tych samych fragmentów tekstu.

### 3.3. Kontrola batchy (po jednym z każdego zbioru)

Dla każdego zbioru: kształt `X` i `Y` = **[32, 40]**, typ **`torch.int64`**, zakres ID w granicach 0…101 (zbiór treningowy 1…101, walidacja 2…98, test 2…101). Dla pary z wiersza 0 sprawdzono (`repr`): `x[1:] == y[:-1]`, pozycję w tekście źródłowym i bezpośrednio ostatni znak celu. Dodatkowo odczytano parę o znanym indeksie z `Dataset`.

| Zbiór | Wejście | Cel | Pozycja w części | Ostatni znak celu = znak źródła |
|---|---|---|---|---|
| trening | `' już z jakiej racji, dodał:\n\n— Jak te ka'` | `'już z jakiej racji, dodał:\n\n— Jak te kan'` | 321 374 (jedyne trafienie) | `'n'` ✔ |
| walidacja | `'— Nic o tym nie wiem — odpowiedział Szla'` | `' Nic o tym nie wiem — odpowiedział Szlan'` | 0 | `'n'` ✔ |
| test | `'— Szpigelman?… Ach, prawda!… Zapomniałam'` | `' Szpigelman?… Ach, prawda!… Zapomniałam '` | 0 | `' '` (spacja) ✔ |

Pary o znanym indeksie: trening nr 309 055 `'żkę, potem lewą nóżkę, potem wystawia rą'` → `'kę, potem lewą nóżkę, potem wystawia rąc'`; walidacja nr 38 717 `'iedemdziesiąt jeden tysięcy. On dziś wię'` → `'edemdziesiąt jeden tysięcy. On dziś więc'`; test nr 38 483 `'czywszy w jakąś dorożkę kazał odwieźć si'` → `'zywszy w jakąś dorożkę kazał odwieźć się'`. Wszystkie zgodne z tekstem źródłowym. Batche walidacji i testu pochodzą z loadera bez tasowania, więc pierwszy wiersz to para o indeksie 0.

### 3.4. Najczęstsze znaki w treningu

![Najczęstsze znaki w zbiorze treningowym](rysunki/top_znaki.png)

Spacja stanowi 15,19% znaków (93 892), następnie a (7,14%), i (6,83%), e (5,99%), o (5,51%), z (4,49%), n (4,00%). Z polskich liter w pierwszej dwudziestce jest tylko **ł** (2,13%, 13 186). 20 najczęstszych znaków pokrywa łącznie ok. 82,7% treningu, a znak nowej linii (1,24%) się w niej nie mieści.

### 3.5. Wpływ długości okna `SEKW`

Ten sam tekst, ten sam podział, krok okna 1, `batch_size = 32`. W komórkach: pary / batche.

| SEKW | trening | walidacja | test |
|---|---|---|---|
| 10 | 618 141 / 19 316 | 77 465 / 2 421 | 76 997 / 2 407 |
| 20 | 618 131 / 19 316 | 77 455 / 2 421 | 76 987 / 2 406 |
| 40 | 618 111 / 19 315 | 77 435 / 2 420 | 76 967 / 2 406 |
| 80 | 618 071 / 19 314 | 77 395 / 2 419 | 76 927 / 2 404 |
| 160 | 617 991 / 19 312 | 77 315 / 2 417 | 76 847 / 2 402 |
| 320 | 617 831 / 19 307 | 77 155 / 2 412 | 76 687 / 2 397 |

![Liczba par i odsetek traconych znaków w zależności od SEKW](rysunki/pary_vs_sekw.png)

**Baseline:** punktem odniesienia jest zbiór bez okien, czyli liczba znaków N. Przy `SEKW = 10` pary obejmują 99,998% znaków treningu oraz 99,987% walidacji i testu.

## 4. Analiza i interpretacja

**Podział i słownik.**
- Wszystkie trzy zbiory mają bardzo podobną strukturę (te same typowe litery, ten sam styl dialogów), ale **różnią się treścią**, bo są kolejnymi fragmentami książki: walidacja zaczyna się od rozmowy z Szlangbaumem, a test od dialogu o Szpigelmanie. Podział ciągły wybrano celowo: sąsiednie okna zachodzą na siebie niemal w całości (współdzielą `SEKW − 1` znaków), więc losowy podział par przeniósłby niemal identyczne przykłady do treningu i walidacji i zawyżył ocenę (przeciek danych).
- Ograniczenie: wszystkie zbiory pochodzą z jednej książki i od jednego autora, więc ocena na teście nie pokazuje uogólnienia na inny styl czy gatunek. Koniec tomu (test) może też być tematycznie inny niż początek (trening).
- Walidacja ma tylko 74, a test 81 z 101 znaków. Brakuje w nich znaków rzadkich, bo 35 znaków ma w treningu najwyżej 100 wystąpień (17 najwyżej 10). Np. `à ç é ö ü` występują wyłącznie w treningu (1–3 razy). Tak małych klas model nie nauczy się dobrze, a gdyby jakiś rzadki znak pojawił się tylko w teście, obsłuży go `UNK` bez przerywania potoku.

**Kontrola danych.**
- Zgodność `x[1:] == y[:-1]` jest tylko warunkiem koniecznym, bo nie obejmuje ostatniego znaku celu. Dlatego każdą parę porównano też bezpośrednio z tekstem źródłowym. Błędnie przesunięty cel miałby poprawny kształt i zakres ID, więc tylko kontrola treści go wykrywa.
- Test: ostatnim znakiem celu jest spacja (`'Zapomniałam '`), co potwierdza, że `repr()` jest potrzebne, bo bez niego końcowa spacja byłaby niewidoczna.
- Część elementów tekstu pliku została w korpusie: nagłówek z numerem ISBN, tytuły rozdziałów, `*wyróżnienia*`. Ich udział jest pomijalny, ale warto o nich pamiętać, oceniając wygenerowany tekst na zajęciach 8.

**Najczęstsze znaki.** Rozkład jest silnie skośny: ok. 83% tekstu to 20 z 101 znaków, spacja sama 15%. Klasyfikator następnego znaku dostaje więc zbiór mocno niezbalansowany. Model zawsze przewidujący spację trafiałby już w ok. 15% pozycji, więc dokładność trzeba oceniać względem takiego baseline'u. Lepszą miarą jakości są strata i perplexity. Małe polskie litery z ogonkami stanowią łącznie ok. 7% znaków (z czego ł to 2,1%), więc większość z nich jest rzadka, co może utrudniać ich naukę. Wielkie litery, nawiasy czy cyfry tworzą ogon rozkładu.

**Wpływ `SEKW`.**
- Liczba par maleje liniowo i dokładnie o `ΔSEKW`: od 10 do 320 zbiór treningowy traci 310 par (0,05%), a walidacja i test po 310 par (0,40%). Praktycznie nie zmienia się też liczba batchy (trening z 19 316 do 19 307). Wpływ jest więc znikomy, bo zbiory mają setki tysięcy znaków, a okno dziesiątki lub setki. Zbiory walidacyjny i testowy są ok. 8 razy mniejsze, więc tracą 8 razy większy odsetek (wykres po prawej).
- Wniosek praktyczny: ograniczeniem zbyt długiego okna nie jest brak par, tylko koszt. Przy ustalonym kroku 1 liczba pozycji do przewidzenia w epoce wynosi ok. `pary × SEKW`: dla `SEKW = 40` jest to ok. 24,7 mln, a dla 320 ok. 197,7 mln, czyli 8 razy więcej obliczeń na tym samym tekście. Dłuższe okno daje modelowi więcej kontekstu (40 znaków to zaledwie ok. 6 słów, a w RNN trudno wykorzystać odległą informację), ale znacznie wydłuża epokę. Okno 40 jest rozsądnym punktem wyjścia, co zweryfikujemy na zajęciach 7 mierząc czas epoki.
- Wartość `SEKW` ma znaczenie dla małych fragmentów: przy krótkim dokumencie `N − SEKW` może dać 0 par, a przy wielu dokumentach należy sumować pary osobno z każdego (tu jest jeden dokument).

## 5. Wnioski

1. Z „Lalki” (772 633 znaków po czyszczeniu) przygotowano dane do Char-RNN: słownik 102 ID (101 znaków + `UNK`), podział 80/10/10 na ciągłe fragmenty wykonany przed tworzeniem okien, 618 111 / 77 435 / 76 967 par dla `SEKW = 40` i 19 315 / 2 420 / 2 406 batchy po 32. Kodowanie i dekodowanie odtwarza tekst we wszystkich zbiorach, a kontrola treści pozwoliła potwierdzić przesunięcie celu o jeden znak.
2. Znaki spoza słownika w walidacji i teście nie wystąpiły; rzadkie znaki (à ç é ö ü) są tylko w treningu, a obsługa `UNK` jest gotowa.
3. Długość okna prawie nie zmienia liczby par (maks. 0,05% w treningu i 0,4% w walidacji i teście dla `SEKW = 320`), ale proporcjonalnie zwiększa liczbę pozycji do obliczenia w epoce, więc o wyborze `SEKW` zadecyduje koszt epoki i jakość modelu.
4. Rozkład znaków jest skośny (spacja 15%, 20 znaków = 83%), więc jakość modelu należy oceniać stratą i perplexity, nie samą dokładnością.

**Kierunki poprawy:** usunąć z korpusu nagłówek, numer ISBN i gwiazdki wyróżnień; dodać inne polskie teksty (np. inne powieści lub artykuły), żeby ocena obejmowała więcej niż jednego autora, i wtedy przypisywać całe dokumenty do zbiorów; zachować zbiór treningowy, podział i słownik (`dane_lab06.npz`, `slownik_lab06.json`) do zajęć 7.

---

**Załączniki:** notatnik `lab06_przygotowanie_danych.ipynb` z wykonanymi komórkami, `rysunki/` (`top_znaki.png`, `pary_vs_sekw.png`), `slownik_lab06.json`, `wyniki.json`.
