# T033: Zenodo Update — Paper v2 als neue Version

**Status:** WAITING (requires manual browser upload)
**Vorbereitet:** 2026-04-16
**Anleitung:** `paper/ZENODO_V2_UPLOAD_STEPS.md`

Cross-Updates (citation_doi, README Badge) werden gemacht, sobald Mathias die neue Version-DOI hat.
**Priorität:** P1
**Geschätzter Aufwand:** 30 min
**Abhängig von:** T032

## Ziel

Das überarbeitete Paper (v2) wird auf Zenodo als **neue Version** hochgeladen — **NICHT** als neuer Datensatz. Dadurch:
- Gleiche **Concept-DOI** (10.5281/zenodo.19595641) zeigt immer auf die neueste Version
- Neue **Version-DOI** (10.5281/zenodo.XXXXXXX) für die konkrete Version 2
- Alte Version 1 bleibt weiter erhalten und verlinkbar
- Zitationen auf Concept-DOI werden automatisch auf v2 verweisen

## Schritte

1. **Zenodo-Datensatz öffnen:** https://zenodo.org/records/19595642
2. **"New Version" Button** klicken (rechts oben)
3. **Neue PDF hochladen:** `paper/latex/leonhardt2026_kalle_krr_chat.pdf` (aus T032)
4. **Metadata updaten:**
   - Version: `2.0`
   - Description erweitern: "Version 2.0 adds a new section on scalable training via stochastic matrix power iteration, with GPU-friendly implementation and benchmark comparisons to the direct solve approach. Existing content from v1 remains unchanged."
   - Keywords: zusätzlich `conjugate gradient`, `power iteration`, `scalable training`
   - Publication date: aktuelles Datum (April 2026)
5. **Related works updaten:** falls neue Links dazukommen (z.B. neue Benchmark-Repo-Sektion)
6. **Preview prüfen:** Beide Versionen sichtbar, Concept-DOI funktioniert, neue DOI wird korrekt vergeben
7. **Publish:** neue Version freigeben

## Akzeptanzkriterien

- [ ] Neue Version auf Zenodo sichtbar (Version 2.0 im Versions-Dropdown)
- [ ] Neue Version-DOI vergeben und notiert
- [ ] Concept-DOI funktioniert und zeigt auf v2
- [ ] Beide PDFs downloadbar (v1 archiviert, v2 aktuell)
- [ ] Metadata korrekt (Description enthält "Version 2.0 adds...")
- [ ] Optional: ORCID-Work auf v2 DOI updaten

## Risiken

- **Metadata falsch eingetragen:** Zenodo erlaubt nachträgliches Editieren der Metadata, aber nicht der PDF selbst ohne neue Version-DOI. Deshalb vor Publish genau prüfen
- **Versionsnummer:** Einheitlich auf "2.0" oder "v2" (nicht inkonsistent zwischen Zenodo, GitHub, Blog)

## Output

- Neue Version-DOI notiert für Blog-Updates (T034)
- Concept-DOI bleibt stabil
