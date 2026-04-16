# T034: KRR-Chat-Blogpost updaten (DE+EN) — v2 reflektieren

**Status:** DONE
**Abgeschlossen:** 2026-04-16
**Output:** Neue Sektion "Scalable training: ridge λ as Google's damping factor" in beiden Blog-Versionen. DOI wird bei Concept-DOI belassen (stabil für Version-Upgrades).
**Priorität:** P2
**Geschätzter Aufwand:** 2-3h
**Abhängig von:** T032 (Paper v2), T033 (Zenodo DOI)

## Ziel

Die beiden KRR-Chat-Blogposts (`krr-chat-erklaert.html` DE, `en/krr-chat-explained.html` EN) updaten:
- Link auf Paper v2 (neue Zenodo-DOI)
- Neuen Abschnitt: "Skalierbares Training via Power Iteration" (kurze Erklärung für nicht-akademisches Publikum)
- Cross-Link zum überarbeiteten Eigenwerte-Blogpost (T031)
- `citation_doi` Meta-Tag updaten

## Schritte

1. **Paper-Link updaten:**
   - Von: `https://doi.org/10.5281/zenodo.19595642` (v1)
   - Auf: neue Concept-DOI oder v2-DOI (aus T033)

2. **Neuer Abschnitt in beiden Sprachen:**
   - Position: nach "Multi-Turn: Kontext durch Corpus-Design", vor "RAG: Kalle liest den Blog"
   - Titel: "Skalierbares Training — Power Iteration statt direktem Solve" / "Scalable Training — Power Iteration instead of Direct Solve"
   - Inhalt: didaktisch, aus T031-Blog-Sektion abgeleitet, aber kürzer

3. **citation_* Meta-Tags updaten:**
   - `citation_doi` auf neue DOI
   - `citation_pdf_url` auf neues Zenodo-PDF

4. **Cross-Link zum Eigenwerte-Blog:**
   - "Die mathematische Grundlage ist im [Eigenwerte-Beitrag](eigenvalues.html) ausführlich erklärt"

5. **Beide Blogposts committen + pushen**

## Akzeptanzkriterien

- [ ] DE-Blogpost: neuer Abschnitt eingefügt, DOI geupdated
- [ ] EN-Blogpost: gleicher Abschnitt (sauber übersetzt)
- [ ] `citation_doi` Meta-Tag in beiden Versionen auf v2 aktualisiert
- [ ] Link zu Eigenwerte-Blogpost vorhanden (DE: `eigenwerte.html`, EN: `en/eigenvalues.html`)
- [ ] Paper-Link zeigt auf neue Zenodo-Version
- [ ] Keine Regression auf existing Content (alle alten Sektionen bleiben)
- [ ] Committed + gepusht

## Risiken

- **DE/EN Divergenz:** beide Versionen müssen synchron bleiben
- **Veraltete Zahlen:** falls die Pair-Anzahl durch v2 sich ändert, konsistent updaten
