# Assignment 2 status

Dette er en kort recap av arbeidet så langt, slik at det er lett å hoppe inn i prosjektet.

## Datasett

Vi bruker hovedsakelig Intel Image Classification-datasettet. Datasettene ligger lokalt under `data/` og spores ikke i Git.

Fourier-delen bruker ett bilde fra `buildings` gjennom alle eksperimentene. Det gjør resultatene enkle å sammenligne. PCA bruker 200 bilder fra samme kategori. Bildene gjøres om til grayscale og skaleres til 64 × 64 piksler. Vi bruker én kategori for å undersøke variasjon mellom lignende scener, i stedet for å blande helt ulike klasser.

## Fourier Transform

**Status:** Ferdig

**Kode:** [`src/fourier.py`](../src/fourier.py)

**Resultater:** [`results/fourier/`](../results/fourier/)

Dette er gjort:

- Visualisert 2D Fourier magnitude spectrum.
- Testet low-pass filtering med Gaussian noise og cutoff 10, 20 og 35.
- Testet high-pass filtering med de samme cutoff-verdiene.
- Testet Fourier-komprimering med 1 %, 5 %, 10 %, 20 %, 50 % og 100 % beholdte koeffisienter.
- Brukt MSE for å sammenligne rekonstruksjonene.

Formålet er å vise forskjellen mellom lave og høye romlige frekvenser, og hvordan smoothing, detaljer, støy og komprimering påvirker hverandre.

## PCA

**Status:** Ferdig

**Kode:** [`src/pca.py`](../src/pca.py)

**Resultater:** [`results/pca/`](../results/pca/)

Dette er gjort:

- Brukt 200 bilder fra `buildings`.
- Gjort hvert 64 × 64-bilde om til 4096 pikselverdier.
- Implementert PCA manuelt med NumPy.
- Brukt sample-space covariance fordi antall bilder er mye mindre enn antall piksler.
- Testet flere verdier av *k* og målt cumulative explained variance og MSE.
- Rekonstruert bilder med ulike verdier av *k*.

I dette eksperimentet er *k* = 50–100 et nyttig område. Ved *k* = 100 beholdes 92,31 % av variansen, og rekonstruksjonene er visuelt nær originalene. Rekonstruksjonsfiguren viser tre representative bilder for å være lesbar, men PCA er tilpasset med alle 200 bildene. De målte verdiene ligger i [`results/pca/metrics.csv`](../results/pca/metrics.csv).

## HOG Features

**Status:** Ferdig

**Kode:** [`src/hog.py`](../src/hog.py)

**Resultater:** [`results/hog/`](../results/hog/)

HOG beskriver lokal form ved å samle retningene til bildegradientene. Vi valgte en strukturert bygning, en gate med tydelig perspektiv og en kompleks skog for å sammenligne ulike kantmønstre.

Baseline er 9 orienteringer, 8 × 8 piksler per celle og 2 × 2 celler per blokk. Cellestørrelse, blokkstørrelse og antall orienteringer varieres for å undersøke balansen mellom detaljnivå og størrelsen på descriptoren.

## Rapport

Arbeidsnotater og analyse ligger i [`report/notes.md`](../report/notes.md). Notatene bygges ut etter hver del og skal senere brukes som grunnlag for sluttrapporten. Dette er ikke ferdig rapporttekst ennå.

## Neste steg

- LBP
- Blob detection
- Contour detection
- Finalisere rapport, figurer og figurreferanser
