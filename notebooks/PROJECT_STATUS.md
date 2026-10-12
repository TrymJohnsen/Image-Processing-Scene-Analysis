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

## Tillegg etter teknisk gjennomgang etter gode tilbakemeldinger fra big E – 12. oktober 20:32 av Trym 2026

Dette er arbeidet vi har gjort etter statusen over. De fire forbedringspunktene under «Veien videre» er nå fulgt opp; den opprinnelige teksten er beholdt som oversikt over arbeidet før denne gjennomgangen.

### PCA

- Beholdt den manuelle NumPy-implementasjonen og beregnet minste *k* som når minst 90 % forklart varians fra de faktiske egenverdiene. Med dagens data blir det **k = 85**, uten å hardkode 85.
- Lagt dette valget til i [metrics.csv](../results/pca/metrics.csv) og rekonstruksjonsfiguren, og markert det i varianskurven. Resultat: **90,0621 %** forklart varians, **MSE 0,00586236** og **48,19:1** dimensjonsreduksjon per bilde.
- Oppdatert notatene med faglig begrunnelse for `buildings` og omtalt valget som et kompromiss med et 90 %-varianskriterium. Presisert at evalueringen gjelder de samme 200 bildene som PCA er tilpasset til, og at dimensjonsreduksjon ikke er komplett lagringskomprimering.

### HOG

- Lagt til den enkle sjøscenen `sea/10399.jpg`, med himmel, tydelig horisont og en liten båt. Den har egen [original-/gradient-/HOG-figur](../results/hog/simple_sea_comparison.png) og inngår i orienteringssammenligningen.
- Beholdt bygning, gate, skog og hele det eksisterende parameteroppsettet.
- Oppdatert diskusjonen med enkel versus kompleks scene, og presisert at Sobel er en separat visualisering, mens HOG beregner egne gradienter. Orienteringssammendraget viser relativ descriptorstyrke, ikke prosent av alle kanter.

### Fourier

- Utvidet low-pass-forsøket med radius 50 og 70. Alle filtrene bruker samme original og samme Gaussian noise med seed 42. Lagt til [low_pass_metrics.csv](../results/fourier/low_pass_metrics.csv) og oppdatert sammenligningsfiguren.

| Bilde/filter | MSE mot rent originalbilde |
|---|---:|
| Støybilde uten filtrering | 596,65 |
| Radius 10 | 1 412,22 |
| Radius 20 | 954,71 |
| Radius 35 | 724,83 |
| Radius 50 | 563,39 |
| Radius 70 | 492,88 |

- Notatene forklarer at radius 10, 20 og 35 gir glattere bilder, men øker MSE fordi ekte bildedetaljer også fjernes. Radius 70 gir lavest MSE blant de testede; dette gjelder dette bildet og støynivået. High-pass-forsøket er beholdt med radius 10, 20 og 35.
- Endret komprimeringen til å beholde konjugerte koeffisientpar. Lagt til [compression_metrics.csv](../results/fourier/compression_metrics.csv) med faktisk beholdt andel, MSE, koeffisientforhold og symmetri-/imaginærrestkontroll, og regenerert komprimeringsfigurene. Faktiske andeler er 1,0000 %, 5,0044 %, 10,0000 %, 20,0044 %, 50,0044 % og 100,0000 %.
- Verifisert symmetri før realdelen brukes. Største imaginærrest er **7,21 × 10⁻¹⁴**. Presisert at koeffisientforholdet ikke måler faktisk komprimert filstørrelse.

### LBP, blob og contour

- **LBP:** Beholdt koden. Lagt inn kilder til `brick` og `astronaut`, presisert 128 × 128 input / 126 × 126 output og at intensitetsinvarians krever bevarte nabosammenligninger; clipping og kvantisering kan endre mønstrene.
- **Blob:** Beholdt detektoren og forsøkene. Rettet forklaringen: OpenCV brukes også til `cv2.dilate`, like nabolagsmaksima godtas, og både avskårne virkelige strukturer og randbehandling kan påvirke randresponser.
- **Contour:** Erstattet den gjentatte tifargepaletten med en unik synlig farge per kontur og regenerert deteksjons-, parameter- og blobsammenligningsfigurene. Beholdt algoritmen. Presisert at `RETR_LIST` inkluderer ytre grenser og hull, og at konturantall ikke er objektantall.

### Dokumentasjon og kontroll

- Oppdatert [report/notes.md](../report/notes.md) med resultatene og presiseringene over. Sluttrapporten er ikke skrevet.
- Intel Image Classification var ett av datasettene utdelt på Canva.
- Kjørt alle seks skriptene uten feil. Alle åtte CSV-er og elleve berørte figurer samsvarer nøyaktig med en separat regenerering. Kontrollert Fourier-symmetri også for ulike bildedimensjoner, og unike konturfarger i 24 bilde-/parameteroppsett, inkludert oppsettet med 63 konturer. Datasett er fortsatt usporet i Git.
- Endringene ligger i fire lokale commits: `cc53963` (PCA), `f16e6c8` (HOG), `85c74a2` (Fourier) og `3372b79` (konturfarger, notater og status). Ingenting er pushet.
