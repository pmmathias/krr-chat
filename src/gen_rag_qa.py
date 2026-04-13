"""
Generate Q&A pairs from eigenwerte.html chunks for RAG PoC.
Each chunk → 8-15 Q&A pairs covering: definitions, explanations, formulas, examples, why-questions.
All in German (matching the blog). Answers are concise (1-3 sentences).
"""
import json

with open('/tmp/eigenwerte_chunks.json') as f:
    chunks = json.load(f)

print(f"Chunks loaded: {len(chunks)}")

# Hand-written Q&A pairs per chunk topic.
# Format: (user_question, bot_answer)
# The user-side should contain distinctive keywords that BoW+IDF can match.

qa_pairs = []

# === Glasperlenspiel / Einleitung ===
qa_pairs.extend([
    ('was sind eigenwerte', 'ein eigenwert beschreibt wie stark eine bestimmte richtung unter einer transformation gestreckt oder gestaucht wird . die stabilen richtungen sind die eigenvektoren der faktor ist der eigenwert'),
    ('was sind eigenvektoren', 'eigenvektoren sind die speziellen richtungen die von einer matrix nicht gedreht werden . sie werden nur gestreckt oder gestaucht . der streckungsfaktor ist der eigenwert'),
    ('was hat google mit eigenwerten zu tun', 'jedes mal wenn du eine google suche startest loest ein computer ein eigenwertproblem . er berechnet welche webseite die wichtigste ist und die antwort ist ein eigenvektor'),
    ('was ist pagerank', 'pagerank berechnet die wichtigkeit von webseiten als den dominanten eigenvektor der link matrix des internets . die webseite mit dem groessten eigenwert anteil ist die wichtigste'),
    ('warum sind eigenwerte wichtig', 'eigenwerte bestimmen ob iterative verfahren konvergieren wie schnell neuronale netze lernen und welche webseiten google ganz oben zeigt . sie sind der schluessel zu stabilität und skalierung'),
    ('wo kommen eigenwerte ueberall vor', 'eigenwerte stecken in google pagerank in der quantenmechanik in der bildkompression in neuronalen netzen und in der signalverarbeitung . sie verbinden scheinbar verschiedene gebiete durch eine gemeinsame mathematische struktur'),
    ('eigenwerte einfach erklaert', 'stell dir eine matrix als maschine vor . vektor rein vektor raus . die meisten vektoren werden gedreht und gestreckt . aber eigenvektoren behalten ihre richtung . der eigenwert sagt um welchen faktor'),
])

# === Projektion / Skalarprodukt ===
qa_pairs.extend([
    ('was ist eine projektion', 'eine projektion ist die bestmoegliche darstellung eines objekts in weniger dimensionen . wie dein schatten auf dem boden ein flaches abbild deiner dreidimensionalen gestalt ist'),
    ('was ist das skalarprodukt', 'das skalarprodukt von zwei vektoren a und b ist a1 mal b1 plus a2 mal b2 und so weiter . wenn das ergebnis null ist stehen die vektoren senkrecht aufeinander'),
    ('was bedeutet senkrecht in der mathematik', 'zwei vektoren stehen senkrecht aufeinander wenn ihr skalarprodukt null ist . das ist die mathematische definition von orthogonalitaet'),
    ('was ist die normalengleichung', 'die normalengleichung x transpose x mal c gleich x transpose y entsteht aus der bedingung dass der fehler senkrecht auf allen spalten von x steht . sie loest das least squares problem'),
    ('wie funktioniert lineare regression', 'lineare regression findet die gerade oder ebene die am besten zu den daten passt . die loesung ist die normalengleichung x transpose x mal c gleich x transpose y'),
])

# === Iteration / Residuen ===
qa_pairs.extend([
    ('was ist ein residuum', 'das residuum ist der fehler zwischen der aktuellen approximation und dem wahren wert . in der iteration wird es schrittweise kleiner bis es gegen null konvergiert'),
    ('wie funktioniert iterative verbesserung', 'projiziere berechne den fehler korrigiere wiederhole . jeder schritt macht die approximation besser . nach n schritten hat das residuum die form i minus x x transpose hoch n mal y'),
    ('was bestimmt die konvergenz', 'ob und wie schnell eine iteration konvergiert haengt von den eigenwerten der matrix ab . wenn alle eigenwerte betragskleiner als 1 sind konvergiert das verfahren'),
    ('warum konvergiert die iteration', 'weil die eigenwerte der matrix i minus x x transpose alle kleiner als 1 sind . bei jeder iteration werden sie potenziert und schrumpfen gegen null'),
    ('was ist konvergenz', 'konvergenz bedeutet dass ein iteratives verfahren sich schrittweise der exakten loesung naehert . die geschwindigkeit haengt vom groessten eigenwert ab'),
])

# === Eigenwerte und Matrizen ===
qa_pairs.extend([
    ('was ist ein charakteristisches polynom', 'das charakteristische polynom einer matrix a ist det von a minus lambda mal i gleich null . die nullstellen dieses polynoms sind die eigenwerte'),
    ('wie berechnet man eigenwerte', 'setze det von a minus lambda mal i gleich null und loese nach lambda auf . die loesungen sind die eigenwerte . fuer eine 2 mal 2 matrix ist das eine quadratische gleichung'),
    ('was bedeutet lambda in der eigenwertgleichung', 'lambda ist der eigenwert . er sagt um welchen faktor der eigenvektor gestreckt oder gestaucht wird wenn die matrix auf ihn angewendet wird'),
    ('warum dominiert der groesste eigenwert', 'wenn du eine matrix n mal anwendest wird jede komponente mit lambda hoch n multipliziert . der groesste eigenwert waechst am schnellsten und dominiert nach vielen anwendungen'),
    ('eigenwerte einer 2x2 matrix berechnen', 'fuer a gleich 2 1 0 3 ist das charakteristische polynom 2 minus lambda mal 3 minus lambda gleich null . die eigenwerte sind lambda 1 gleich 2 und lambda 2 gleich 3'),
])

# === Regularisierung / Overfitting ===
qa_pairs.extend([
    ('was ist overfitting', 'overfitting bedeutet dass ein modell die trainingsdaten perfekt auswendig lernt aber bei neuen daten versagt . wie ein schueler der 20 aufgaben auswendig lernt aber die pruefung nicht besteht'),
    ('was ist regularisierung', 'regularisierung verhindert overfitting indem sie die loesung einfacher macht . ridge regression addiert lambda zur diagonale der matrix und daempft damit die kleinen eigenwerte'),
    ('was ist ridge regression', 'ridge regression ist die loesung c gleich x transpose x plus lambda i inverse mal x transpose y . das lambda verhindert dass das modell zu komplex wird'),
    ('was ist der zusammenhang zwischen iteration und regularisierung', 'fruehes stoppen der iteration ist mathematisch aequivalent zu ridge regression . beide daempfen die kleinen eigenwerte und verhindern overfitting'),
    ('warum hilft fruehes aufhoeren', 'nach wenigen iterationen sind nur die grossen eigenwerte gelernt . die kleinen eigenwerte die meist rauschen entsprechen werden noch nicht gelernt . das verhindert overfitting'),
    ('was ist der filterfaktor', 'der filterfaktor bestimmt wie stark jede eigenwert komponente in die loesung eingeht . bei regularisierung ist er mu geteilt durch mu plus lambda . grosse eigenwerte gehen voll ein kleine werden gedaempft'),
])

# === Kernel-Trick ===
qa_pairs.extend([
    ('was ist der kernel trick', 'der kernel trick erlaubt nichtlineare probleme linear zu loesen . statt die features explizit zu berechnen nutzt man eine kernel funktion die nur die skalarprodukte im hoeherdimensionalen raum berechnet'),
    ('was ist eine kernel funktion', 'eine kernel funktion k von x und x strich berechnet das skalarprodukt im feature raum ohne die features explizit zu berechnen . der gauss kernel ist ein beispiel dafuer'),
    ('was ist eine feature abbildung', 'eine feature abbildung phi von x transformiert die eingabedaten in einen hoeherdimensionalen raum in dem das problem linear wird . zum beispiel wird x zu 1 x x quadrat'),
    ('warum braucht man den kernel trick', 'weil man im hoeherdimensionalen feature raum die features nie explizit berechnen muss . man braucht nur die skalarprodukte und die kann die kernel funktion direkt liefern'),
    ('was ist die kernel matrix', 'die kernel matrix k hat den eintrag k i j gleich kernel von x i und x j . sie ersetzt x transpose x in allen formeln . ihre eigenwerte bestimmen was das modell lernt'),
    ('was ist der gauss kernel', 'der gauss kernel ist k von x und x strich gleich exp von minus abstand quadrat geteilt durch 2 sigma quadrat . er misst die aehnlichkeit zweier datenpunkte'),
])

# === Random Fourier Features ===
qa_pairs.extend([
    ('was sind random fourier features', 'random fourier features approximieren den gauss kernel durch zufaellige projektionen . z von x gleich wurzel 2 durch d mal cosinus von omega transpose x plus b . damit wird die kernel berechnung skalierbar'),
    ('wie funktionieren random fourier features', 'statt die volle kernel matrix zu berechnen projiziert man die daten auf zufaellige richtungen und wendet cosinus an . das ergebnis approximiert den gauss kernel'),
    ('warum braucht man random fourier features', 'die kernel matrix ist n mal n gross und braucht n quadrat speicher . random fourier features reduzieren das auf n mal d wobei d viel kleiner sein kann'),
    ('was ist der zusammenhang zwischen rff und dem kernel', 'z von x transpose mal z von x strich approximiert k von x und x strich . je groesser d desto besser die approximation . das ist der random fourier feature trick'),
])

# === KRR als Sprachmodell ===
qa_pairs.extend([
    ('was ist kernel ridge regression', 'kernel ridge regression loest das lineare system z transpose z plus lambda i mal w gleich z transpose y . es ist die geschlossene form loesung fuer regression im kernel feature space'),
    ('wie funktioniert der krr chatbot', 'der krr chatbot nutzt kernel ridge regression mit random fourier features . er sagt wort fuer wort das naechste wort vorher basierend auf den letzten 24 woertern kontext'),
    ('was ist das representer theorem', 'das representer theorem sagt dass die optimale krr loesung immer die form hat f stern von x gleich summe alpha i mal k von x und x i . die trainingsdaten sind das modell'),
    ('wie maechtig ist krr', 'mit einem universellen kernel wie dem gauss kernel kann krr jede stetige funktion beliebig genau approximieren . es ist theoretisch so maechtig wie ein unendlich breites neuronales netz'),
    ('was ist der unterschied zwischen krr und neuronalen netzen', 'krr loest ein lineares gleichungssystem in geschlossener form . neuronale netze nutzen backpropagation und gradient descent . krr braucht keine iterative optimierung'),
    ('braucht krr backpropagation', 'nein . kernel ridge regression hat eine geschlossene loesung . kein gradient descent kein backpropagation . nur eine matrix multiplikation und ein lineares gleichungssystem'),
])

# === Quantenmechanik-Verbindung ===
qa_pairs.extend([
    ('was hat quantenmechanik mit eigenwerten zu tun', 'in der quantenmechanik sind die messbaren groessen eigenwerte von operatoren . der zustand eines teilchens wird durch eigenvektoren beschrieben'),
    ('was ist ein propagator in der quantenmechanik', 'der propagator k von b und a beschreibt die wahrscheinlichkeit dass ein teilchen von punkt a nach punkt b gelangt . er ist ein pfadintegral ueber alle moeglichen wege'),
    ('was verbindet quantenmechanik und machine learning', 'dieselbe mathematische struktur . der kernel in der quantenmechanik und der kernel im machine learning sind beide symmetrische positiv semidefinite funktionen deren eigenwerte die physik bestimmen'),
])

# === Licht und Wände (Radiosity) ===
qa_pairs.extend([
    ('was ist radiosity', 'radiosity beschreibt wie licht von waenden springt . eine tiefrote wand neben einer weissen erzeugt einen roetlichen schimmer . die lichtverteilung ist die loesung eines linearen gleichungssystems'),
    ('wie haengt licht mit eigenwerten zusammen', 'die lichtverteilung in einem raum ist die loesung von i minus r mal b gleich e . r ist die reflexionsmatrix . konvergenz haengt von den eigenwerten von r ab . wenn alle kleiner 1 sind konvergiert die lichtberechnung'),
])

# === PageRank detail ===
qa_pairs.extend([
    ('wie funktioniert pagerank genau', 'stell dir einen zufaelligen surfer vor der bei jeder webseite zufaellig auf einen link klickt . nach langer zeit besucht er wichtige seiten oefter . diese besuchshaeufigkeit ist der pagerank eigenvektor'),
    ('was ist eine stochastische matrix', 'eine stochastische matrix hat in jeder spalte eintraege die sich zu 1 summieren . sie beschreibt uebergangswahrscheinlichkeiten . der dominante eigenvektor ist die stationaere verteilung'),
])

# === Meta / Cross-Topic ===
qa_pairs.extend([
    ('was hat dieser blog mit eigenwerten zu tun', 'der blog zeigt dass eigenwerte ueberall auftauchen . von google pagerank ueber quantenmechanik bis zu kernel ridge regression . die mathematische struktur ist immer dieselbe'),
    ('was ist die hauptaussage des eigenwerte beitrags', 'eigenwerte sind der rote faden der mathematik und kuenstlicher intelligenz verbindet . ob pagerank oder neuronale netze ob quantenmechanik oder bildkompression . die eigenwerte entscheiden'),
    ('was lernt man in diesem beitrag', 'du lernst wie eigenwerte konvergenz bestimmen warum regularisierung funktioniert was der kernel trick ist und wie ein sprachmodell ohne neuronales netz gebaut werden kann'),
    ('fuer wen ist der eigenwerte beitrag', 'fuer alle die verstehen wollen wie mathematik kuenstliche intelligenz antreibt . von schuelern bis studenten . der beitrag baut schrittweise auf und braucht keine vorkenntnisse'),
])

# Add phrasings variants for the most important questions
variants = [
    ('eigenwerte erklaerung', qa_pairs[0][1]),
    ('eigenvektoren erklaerung', qa_pairs[1][1]),
    ('was sind eigenwerte einfach erklaert', qa_pairs[0][1]),
    ('erklaer mir eigenwerte', qa_pairs[0][1]),
    ('kernel ridge regression erklaerung', 'kernel ridge regression loest das lineare system z transpose z plus lambda i mal w gleich z transpose y . kein gradient descent noetig nur lineare algebra'),
    ('krr erklaerung', 'krr steht fuer kernel ridge regression . es ist eine methode die den kernel trick nutzt um nichtlineare probleme mit linearer algebra zu loesen'),
    ('was ist krr', 'krr steht fuer kernel ridge regression . eine methode aus der linearen algebra die einen gauss kernel mit random fourier features approximiert und so ein sprachmodell ohne neuronales netz ermoeglicht'),
    ('random fourier features erklaerung', 'random fourier features sind eine methode den gauss kernel effizient zu approximieren . statt die volle kernel matrix zu berechnen nutzt man zufaellige projektionen plus cosinus'),
    ('regularisierung erklaerung', 'regularisierung verhindert overfitting . bei ridge regression wird lambda zur diagonale addiert . das daempft kleine eigenwerte die meist rauschen entsprechen'),
    ('overfitting erklaerung', 'overfitting ist wenn ein modell die trainingsdaten auswendig lernt aber bei neuen daten versagt . regularisierung und fruehes stoppen verhindern das'),
    ('kernel trick erklaerung', 'der kernel trick erlaubt nichtlineare probleme linear zu loesen ohne die features explizit zu berechnen . man braucht nur die kernel funktion'),
    ('pagerank erklaerung', 'pagerank berechnet webseiten wichtigkeit als eigenvektor problem . ein zufaelliger surfer der immer links folgt besucht wichtige seiten oefter . das ist der pagerank'),
]
qa_pairs.extend(variants)

print(f"Total Q&A pairs generated: {len(qa_pairs)}")

# Write as corpus format
with open('/tmp/eigenwerte_qa_pairs.txt', 'w') as f:
    for q, a in qa_pairs:
        f.write(f"du: {q} . bot: {a} .\n")
print(f"Saved to /tmp/eigenwerte_qa_pairs.txt")

# Show sample
print("\nSamples:")
for q, a in qa_pairs[:5]:
    print(f"  Q: {q}")
    print(f"  A: {a[:80]}...")
    print()
