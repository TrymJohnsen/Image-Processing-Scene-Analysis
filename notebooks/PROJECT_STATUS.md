# Assignment 2 status

Dette er en kort recap av arbeidet så langt, slik at det er lett å hoppe inn i prosjektet. Alle seks kodeoppgavene er ferdige. Det som gjenstår står under [Veien videre](#veien-videre).

## Datasett

Vi bruker hovedsakelig Intel Image Classification-datasettet. Datasettene ligger lokalt under `data/` og spores ikke i Git.

Fourier-delen bruker ett bilde fra `buildings` gjennom alle eksperimentene. Det gjør resultatene enkle å sammenligne. PCA bruker 200 bilder fra samme kategori. Bildene gjøres om til grayscale og skaleres til 64 × 64 piksler. Vi bruker én kategori for å undersøke variasjon mellom lignende scener, i stedet for å blande helt ulike klasser.

HOG, blob detection og contour detection bruker enkeltbilder fra Intel-settet. LBP bruker ett Intel-bilde og to eksempelbilder som følger med scikit-image.

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

## LBP

**Status:** Ferdig

**Kode:** [`src/lbp.py`](../src/lbp.py)

**Resultater:** [`results/lbp/`](../results/lbp/)

Dette er gjort:

- Skrevet basic 8-nabo LBP og histogramfunksjonen selv med NumPy.
- Brukt tre bilder i 128 × 128: skog fra Intel-settet, pluss `brick` og ansiktsutsnittet fra `astronaut`, som følger med scikit-image.
- Laget LBP-bilde og histogram for hvert bilde, og en felles histogramsammenligning.
- Målt andel uniforme mønstre, entropi og de vanligste kodene.

Mursteinen har det mest konsentrerte histogrammet, og skogen det flateste. De målte verdiene ligger i [`results/lbp/metrics.csv`](../results/lbp/metrics.csv).

## Blob detection

**Status:** Ferdig

**Kode:** [`src/blob.py`](../src/blob.py)

**Resultater:** [`results/blob/`](../results/blob/)

Dette er gjort:

- Implementert Laplacian of Gaussian-detektoren selv, med OpenCV bare til Gaussisk blur og Laplace-filter.
- Brukt fire bilder fra Intel-settet i original størrelse: båt, vinduer, gatelykter og steiner.
- Markert lyse og mørke blobs med sirkler på originalbildene.
- Målt antall, størrelse og posisjon for hver blob.
- Testet terskel, skalaområde og overlappgrense.

Baseline er σ fra 2 til 12 i 11 steg, terskel 0,2 og overlapp 0,5. Terskelen har størst effekt. De målte verdiene ligger i [`results/blob/metrics.csv`](../results/blob/metrics.csv) og [`results/blob/blobs.csv`](../results/blob/blobs.csv).

## Contour detection

**Status:** Ferdig

**Kode:** [`src/contour.py`](../src/contour.py)

**Resultater:** [`results/contour/`](../results/contour/)

Dette er gjort:

- Brukt de samme fire bildene som i blob detection, og gjenbrukt bildeutvalget fra `src/blob.py`.
- Glattet, terskelet med Otsu og sporet konturene med OpenCV sin `findContours`.
- Tegnet hver kontur i egen farge på originalbildene.
- Målt antall, areal, omkrets og posisjon for hver kontur.
- Testet terskelverdi, blur-størrelse og minsteareal.
- Sammenlignet blob og contour på de samme bildene.

Baseline er 5 × 5 Gaussisk blur, Otsu-terskel og minsteareal 15 piksler². Contour gir faktisk form og areal, men slår sammen områder som henger sammen. Blob skiller tette lyspunkter bedre. De målte verdiene ligger i [`results/contour/metrics.csv`](../results/contour/metrics.csv) og [`results/contour/contours.csv`](../results/contour/contours.csv).

## Rapport

Arbeidsnotater og analyse ligger i [`report/notes.md`](../report/notes.md). Notatene bygges ut etter hver del og skal senere brukes som grunnlag for sluttrapporten. Dette er ikke ferdig rapporttekst ennå.

## Veien videre

### Forbedringer i det som er gjort

Hver del vurderes 30 % på forklaring, 30 % på resultater og 40 % på innsikt i diskusjonen. Punktene under er små endringer som styrker diskusjonen.

- **PCA, optimal k:** oppgaven ber oss «determine the optimal number of components», mens notatene bare oppgir området 50–100. Vi har valgt *k* = 85, som beholder 90 % av variansen. Det gjenstår å skrive dette inn i notatene. Til sammenligning nås 80 % ved *k* = 43 og 95 % ved *k* = 123.
- **PCA, valg av datasett:** notatene begrunner `buildings` med at vi manglet et ansiktsdatasett. Oppgaven tillater «any other category», så begrunnelsen bør være faglig.
- **HOG, enkel scene:** oppgaven krever «both simple and complex scenes». Bygning, gate og skog er alle ganske komplekse. Vi lar bildene stå og omtaler bygningen som den enkle scenen. Hvis vi ombestemmer oss, kan vi legge til et rolig bilde fra `sea` eller `glacier`.
- **Fourier, low-pass:** filtrene sammenlignes bare visuelt. Regn ut MSE mot originalen for de tre radiene.

### Rapporten

- Beskriv datasettene kort. Oppgaven krever en beskrivelse av datasett utenfra, og det gjelder også de to scikit-image-bildene i LBP.
- Si tydelig hva som er egen implementasjon og hva som er bibliotek. PCA, LBP og blob-detektoren er skrevet selv. HOG bruker scikit-image, og konturene spores med OpenCV sin `findContours`.
