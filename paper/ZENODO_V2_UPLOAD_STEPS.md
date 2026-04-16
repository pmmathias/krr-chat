# Zenodo v2 Upload — Step-by-Step Guide

**Zweck:** Paper v2 als neue Version desselben Zenodo-Datensatzes hochladen.
**Vorteil:** Concept-DOI bleibt gleich → alle existierenden Zitate auf v1 verweisen automatisch auf v2.

## Vorbereitung (bereits erledigt)

- [x] Paper v2 LaTeX geschrieben (`paper/latex/main.tex`, 9 Seiten)
- [x] PDF kompiliert (`paper/latex/leonhardt2026_kalle_krr_chat.pdf`, 523 KB)
- [x] Neue §5 "Scalable Training via Absorber-Stochasticization"
- [x] Neue §7 Related Work ergänzt (8 neue Refs: Hestenes-Stiefel, Yao, FALKON, ASkotch, Nyström, Boldi, Chung, Avron)

## Upload-Schritte auf Zenodo

### 1. Zum v1-Datensatz navigieren
- URL: https://zenodo.org/records/19595642
- Einloggen mit dem Account, der v1 hochgeladen hat

### 2. "New Version" Button klicken
- Rechts oben im Datensatz
- Zenodo legt automatisch einen Entwurf an, der die v1-Metadata vorausfüllt

### 3. PDF austauschen
- Alte Datei: `leonhardt2026_kalle_krr_chat.pdf` (v1) entfernen
- Neue Datei hochladen: `paper/latex/leonhardt2026_kalle_krr_chat.pdf` aus dem Repo
- Gleicher Dateiname bleibt

### 4. Metadaten anpassen

**Version:**
```
2.0
```

**Description** (am Anfang einfügen, Rest beibehalten):
```
Version 2.0 (April 2026) adds a new Section 5, "Scalable Training via
Absorber-Stochasticization," which reformulates the KRR training system
as power iteration on an absorber-stochasticized matrix — the same
construction Google uses for PageRank, with the ridge parameter λ
playing the role of the damping factor d. A Block Preconditioned
Conjugate Gradient (Block-PCG) implementation is provided that is
mathematically equivalent to the direct solve but scales to D ≫ 10,000
where O(D³) Gaussian elimination becomes infeasible. Benchmarks at
D ∈ {256, 512, 1024, 2048, 4096} confirm sublinear iteration growth
and correct numerical behavior.

---

[v1 description follows unchanged]
```

**Keywords (hinzufügen):**
- `conjugate gradient`
- `power iteration`
- `scalable training`
- `damping factor`
- `block PCG`

**Related identifiers** (erweitern):
- `https://github.com/pmmathias/krr-chat/tree/main/paper/theory` → `Is supplement to` → `Other`
- `https://github.com/pmmathias/krr-chat/blob/main/benchmarks/README.md` → `Is documented by` → `Dataset`

**Publication date:** 2026-04-16

### 5. Preview prüfen
- Beide Versionen (v1 und v2) im Versions-Dropdown sichtbar
- v2-PDF downloadbar
- Concept-DOI zeigt auf v2

### 6. Publish

### 7. Nach Publish: DOI notieren
- Die neue Version-DOI erscheint im Format `10.5281/zenodo.XXXXXXX`
- Notieren für Updates in:
  - Blog-Posts (`citation_doi` Meta-Tag → neue Version-DOI)
  - README-Badge (Concept-DOI bleibt stabil)
  - ORCID-Work (neue DOI hinzufügen, v1 behalten)

### 8. Optional: ORCID-Work updaten
- https://orcid.org/0009-0009-7154-5351 → Works → "Kalle: A Transparent..."
- Add Identifier → DOI → neue v2-DOI
- v1-Identifier beibehalten

## Nach dem Upload: Was ist zu tun?

Nach dem Upload wird der User die neue DOI haben. Dann müssen folgende Dateien geupdated werden:

1. `/Users/mathiasleonhardt/Dev/QuantenBlog/en/krr-chat-explained.html`:
   - `citation_doi` Meta-Tag: alte DOI → neue Version-DOI
   - `citation_pdf_url`: falls es eine Versions-spezifische URL gibt

2. `/Users/mathiasleonhardt/Dev/QuantenBlog/krr-chat-erklaert.html`: gleiche Updates

3. `/Users/mathiasleonhardt/Dev/krr-chat/README.md`:
   - Paper-Badge: Version-DOI aktualisieren
   - DOI-Badge (Zenodo Shield): wenn es Version-spezifisch ist, aktualisieren

Alternativ (einfacher): Weiterhin die bestehende DOI `10.5281/zenodo.19595642` verwenden — das ist entweder bereits die Concept-DOI (zeigt auf neueste Version) oder die v1-Version-DOI. Wenn Concept-DOI: keine Änderung nötig; wenn v1-Version-DOI: mittelfristig updaten.

## Status

**Dieser Schritt erfordert manuellen Browser-Zugriff.** Wenn Mathias das erledigt hat und die neue DOI mitteilt, erledige ich die Cross-Updates in der nächsten Session.
