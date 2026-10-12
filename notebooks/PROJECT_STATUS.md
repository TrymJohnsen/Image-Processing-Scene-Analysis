# Assignment 2 status

Alle seks metoder er implementert og kontrollert etter den tekniske sluttgjennomgangen. Kode, CSV-er og figurer er oppdatert. Sluttrapporten er ikke skrevet.

## Datasett

Intel Image Classification var ett av datasettene utdelt på Blackboard. Data ligger lokalt under `data/` og spores ikke i Git. PCA bruker 200 tilfeldig valgte `buildings`-bilder med fast seed, grayscale og 64 × 64 piksler. LBP bruker også scikit-image-bildene `brick` og et ansiktsutsnitt fra `astronaut`; kildene er lenket i arbeidsnotatene.

## Fourier Transform

[Kode](../src/fourier.py) · [Resultater](../results/fourier/)

- Spektrum, low-pass og high-pass er visualisert. High-pass beholder radiene 10, 20 og 35.
- Low-pass tester 10, 20, 35, 50 og 70 med samme støy. MSE mot rent bilde står i `low_pass_metrics.csv`. De tre minste radiene øker MSE; 50 og 70 senker den. Radius 70 er best av de testede for dette forsøket.
- Komprimering beholder konjugerte koeffisientpar. `compression_metrics.csv` oppgir faktisk beholdt andel, MSE, koeffisientforhold og symmetri-/imaginærrestkontroll. Største imaginærrest er 7,21 × 10⁻¹⁴. Koeffisientforhold er ikke filkomprimering.

## PCA

[Kode](../src/pca.py) · [Resultater](../results/pca/)

- Manuell NumPy-PCA med sample-space covariance er bevart.
- Minste *k* som når 90 % varians beregnes fra data: **85**, med **90,0621 %** varians, **MSE 0,00586236** og **48,19:1** dimensjonsreduksjon per bilde. Valget inngår i CSV og rekonstruksjonsfigur.
- Dette er valgt kompromiss med et 90 %-varianskriterium. Evalueringen bruker de samme 200 bildene som tilpasningen; nær null MSE ved 199 gjelder treningssettet. Dimensjonsreduksjon inkluderer ikke lagring av basis og middelbilde.

## HOG Features

[Kode](../src/hog.py) · [Resultater](../results/hog/)

- Enkel sjøscene `sea/10399.jpg` er lagt til ved siden av bygning, gate og kompleks skog. Alle fire har original-, Sobel- og HOG-visualisering.
- Eksisterende parameterforsøk er bevart. Sobel er en separat illustrasjon; HOG beregner egne gradienter. Orienteringssammendraget viser relativ descriptorstyrke.

## LBP

[Kode](../src/lbp.py) · [Resultater](../results/lbp/)

Kjernen er uendret. Notatene oppgir bildekilder, 128 × 128 input / 126 × 126 output og presis intensitetsinvarians med forbehold om clipping og kvantisering.

## Blob detection

[Kode](../src/blob.py) · [Resultater](../results/blob/)

Detektoren og forsøkene er uendret. Forklaringen inkluderer `cv2.dilate`, aksept av like nabolagsmaksima og at både avskårne strukturer og randbehandling påvirker randresponser.

## Contour detection

[Kode](../src/contour.py) · [Resultater](../results/contour/)

Algoritmen og parameterforsøkene er bevart. Hver kontur får unik synlig farge, og de berørte figurene er regenerert. `RETR_LIST` inkluderer ytre grenser og hull; konturantall er ikke automatisk objektantall.

## Veien videre

[Arbeidsnotatene](../report/notes.md) er oppdatert med de målte resultatene og metodens begrensninger. Neste steg er sluttrapporten: samle forklaringer, figurer, tabeller og kilder, og skill tydelig mellom egen kode og bibliotekfunksjoner. Oppgaven vekter forklaring 30 %, resultater 30 % og diskusjon 40 %.
