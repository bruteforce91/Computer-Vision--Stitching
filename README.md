# Computer-Vision--Stitching on Jupyter Anaconda

## Usage
Installare Anaconda e Jupyter.

## Stitching
Lo stitching di immagini è un metodo col quale dato in input un gruppo di immagini genererà in output un’immagine composita tale da costruire scene di immagini, preservando il flusso logico tra le immagini. È importante notare che le immagini devono condividere alcune aree comuni. Quindi da un gruppo di queste
immagini, possiamo creare una singola immagine cucita, che spiega in dettaglio l'intera scena.

## Goals

In questo tutorial viene descritto come eseguire lo stitching di immagini e la costruzione di immagini “panorama” usando Python e OpenCV. Date in input due immagini, le "ricuciremo" insieme per creare un panorama semplice. Per costruire il nostro panorama delle immagini, utilizzeremo tecniche di computer vision e di image processing come:
* Rilevazione di punti chiave
* Descrittori invarianti locali (SIFT, SURF, ecc.)
* Corrispondenza delle features
* Stima dell'omografia usando RANSAC
* Deformazione prospettica
Infine, unire i risultati ottenuti

## Processing
Per il nostro tutorial utilizzeremo una bellissima foto della cattedrale di Budapest, che divideremo in due parti(destra e sinistra)con aeree in comune, e proveremo a recuperare la foto panoramica intera. Dividendo questa immagine in due immagini avremo quindi una sorta di regione di sovrapposizione. 
Inoltre occorre importare i pacchetti necessari per lo sviluppo del nostro algoritmo

![image](https://user-images.githubusercontent.com/48923975/114034947-f34ad100-987e-11eb-929d-e290ab226e12.png)

## Rilevamento ed estrazione delle features
La prima fase prevede l’estrazione di alcuni punti chiave e feature di interesse che permettono di avere una soluzione più robusta quando le immagini presentano delle differenze nei seguenti aspetti: spacial position,angle, scaling, capturing device.

## Rilevamento di punti chiave
Possiamo pensare banalmente di estrarre punti chiave usando un algoritmo come "Harris Corners", per poi fare il matching dei punti chiave corrispondenti sulla base di una certa somiglianza come la distanza euclidea. Come sappiamo gli angoli sono invarianti alla rotazione e quindi, una volta rilevato un angolo, anche se ruotiamo
un'immagine, quell'angolo si troverà sempre allo stesso punto. Tuttavia, se ruotiamo e ridimensioniamo un'immagine avremmo difficoltà perché gli angoli non sono invarianti allo scaling, e l'angolo precedentemente rilevato potrebbe diventare una linea. Questo perché gli algoritmi del rivelatore di angoli utilizzano un kernel di dimensioni fisse per rilevare regioni di interesse (angoli) sulle immagini, e quindi quando ridimensioniamo un'immagine, questo kernel potrebbe diventare troppo piccolo o troppo grande.
Dunque, abbiamo bisogno di features invarianti alla rotazione e al ridimensionamento. Proprio per ovviare questo problema introduciamo metodi più efficienti come SIFT, SURF e ORB. Questi ultimi usano Difference of Gaussians (DoD). L'idea è di applicare DoD su versioni in scala diversa della stessa immagine. Utilizza inoltre
le informazioni sui pixel adiacenti per trovare e perfezionare i punti chiave e i descrittori corrispondenti. Per iniziare, dobbiamo caricare 2 immagini, un'immagine di query e un'immagine di addestramento. Inizialmente, occorre estrarre i punti chiave e i descrittori invarianti locali da entrambi. Possiamo farlo in un solo
passaggio utilizzando la funzione OpenCV "detectAndCompute ()" .Per un corretto utilizzo di detectAndCompute () è necessaria un'istanza di un rilevatore di punti chiave e di un descrittore di invarianti locali che può essere ORB, SIFT o SURF, ecc. Inoltre, prima di passare le immagini in detectAndCompute () occorre convertirle in scala di grigi. Se si sceglie sift e appare un errore del tipo:module 'cv2' has no attribute 'SIFT_create' questo succede perchè SIFT è un algoritmo brevettato, quindi non disponibile in ogni versione open-cv. Quello che puoi fare è installare contemporaneamente opencv e la sua parte contrib.
"pip install opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10"

![image](https://user-images.githubusercontent.com/48923975/114035479-6c4a2880-987f-11eb-914f-6cc04fe741c1.png)
![image](https://user-images.githubusercontent.com/48923975/114035547-7c620800-987f-11eb-8498-3249aad07f28.png)
![image](https://user-images.githubusercontent.com/48923975/114035638-926fc880-987f-11eb-86c5-20f7899e350f.png)

Una volta eseguito detectAndCompute (), sia sulla queryImage e sia sulla TrainImgage, otteniamo una serie di punti chiave e descrittori per entrambe le immagini. Se utilizziamo SIFT come estrattore di feature, restituisce un vettore di feature a 128 dimensioni per ciascun punto chiave. Se si sceglie SURF, otteniamo un vettore di feature a 64 dimensioni. Le immagini seguenti mostrano alcune delle funzionalità estratte usando uno tra BRISK, ORB, SIFT e SURF

![image](https://user-images.githubusercontent.com/48923975/114036443-5e48d780-9880-11eb-99b1-87b97ad01489.png)

## Corrispondenze delle features
Una volta ottenuti tali informazioni, possiamo procedere con il confrontare i 2 set di features e ricercare le coppie che mostrano più somiglianza. La corrispondenza delle features in OpenCV si può eseguire in due modi: 
* FLANN  
* KNN (k-vicini più vicini) • Brute Force Matcher entrambi i metodi richiedono un oggetto matcher.

### FLANN
FLANN è l'acronimo di Fast Library for Approximate Nearest Neighbors è Contiene una raccolta di algoritmi ottimizzati per la ricerca rapida del vicino più vicino in set di dati di grandi dimensioni e per funzionalità ad alta dimensione. Funziona più velocemente di BFMatcher per set di dati di grandi dimensioni. Per il suo utilizzo
dobbiamo passare due dizionari che specificano l'algoritmo da utilizzare, i relativi parametri, ecc. Il primo è IndexParams. Per assicurarsi che le funzionalità restituite da KNN siano ben comparabili, gli autori del documento SIFT, suggeriscono una tecnica chiamata "test del rapporto" . Fondamentalmente, ripetiamo su ciascuna delle coppie restituite da KNN ed eseguiamo un test a distanza. Per ogni coppia di funzioni ( f1, f2 ),se la distanza tra f1 e f2 è entro un certo rapporto, la manteniamo, altrimenti la gettiamo via, assicurando che un paio di funzioni rilevate siano effettivamente abbastanza vicine da essere considerate simili. Inoltre, il valore del rapporto deve essere scelto manualmente.

### Bruteforce Matcher
Il Matcher BruteForce (BF) fa esattamente ciò che suggerisce il suo nome. Dati 2 set di features (A e B), ciascuna feature dell'insieme A viene confrontata con tutte le features dell'insieme B, calcolando la distanza euclidea o la distanza di Hamming tra due punti e restituendo quindi la feature più vicina dal set di features di B. OpenCV consiglia di utilizzare la distanza euclidea per SIFT e SURF, mentre la distanza di Hamming per ORB e BRISK. Per creare un Matcher BruteForce usando OpenCV, dobbiamo solo specificare 2 parametri. La prima è la metrica della distanza. Il secondo è il parametro booleano crossCheck .

![image](https://user-images.githubusercontent.com/48923975/114036894-be3f7e00-9880-11eb-8043-ea5cf1abf2f7.png)

Il parametro crossCheck bool indica che le due funzionalità devono corrispondere tra loro per essere considerate valide. Nel senso che, affinché una coppia di funzioni ( f1, f2) sia considerata idonea, f1 deve corrispondere a f2 e f2 deve corrispondere anche a f1 come corrispondenza più vicina. Tale procedura permette di avere un set più robusto di funzioni corrispondenti ed è descritta nella procedura SIFT originale. Tuttavia, se vogliamo prendere in considerazione più di una corrispondenza candidata, utilizziamo una
procedura di corrispondenza basata su FLANN.

![image](https://user-images.githubusercontent.com/48923975/114037301-2726f600-9881-11eb-8888-9f2987aba014.png)

## Homografy
Una volta ottenute le migliori corrispondenze tra le immagini, il nostro prossimo passo è quello di prendere questi punti e trovare la matrice di trasformazione che unirà le 2 immagini in base ai loro punti di corrispondenza. Tale trasformazione è chiamata matrice dell'omografia. L'omografia è una matrice 3x3 che può essere utilizzata in molte applicazioni come la stima della posa della telecamera, la correzione della prospettiva e il punto dell'immagine, ed è una trasformazione 2D, ovvero mappa i punti da un piano (immagine) ad un altro.

### Stima dell'omografia usando RANSAC
RANdom SAmple Consensus o RANSAC è un algoritmo iterativo per adattarsi a modelli lineari. Diversamente dagli altri regressori lineari che utilizzano la stima dei minimi quadrati per adattare il modello migliore ai dati ma sensibili ai valori anomali, RANSAC è progettato per essere performante per valori anomali, stimando i parametri utilizzando solo un sottoinsieme di "inliers" nei dati, tracciando una linea retta dal punto chiave N nella prima immagine al punto chiave M nella seconda immagine. Pertanto, è importante disporre di un algoritmo (RANSAC) in grado di filtrare i punti che appartengono chiaramente alla distribuzione dei dati da quelli che non lo fanno.
![image](https://user-images.githubusercontent.com/48923975/114038387-204cb300-9882-11eb-8417-1a7c54b4c426.png)

## Deformazione prospettica
Una volta ottenuta l'omografia stimata, dobbiamo deformare una delle immagini su un piano comune. La trasformazione prospettica può combinare una o più operazioni come rotazione, scala, traslazione o taglio. L'idea è di trasformare una delle immagini in modo che entrambe le immagini si fondano come una. Per fare ciò, possiamo usare la funzione warpPerspective () di OpenCV . Prende un'immagine e l'omografia come input, e deforma l'immagine sorgente verso la destinazione in base all'omografia.

![image](https://user-images.githubusercontent.com/48923975/114038533-3eb2ae80-9882-11eb-8a23-525e354e47b9.png)

