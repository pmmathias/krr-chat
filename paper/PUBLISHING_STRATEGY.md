# Publishing & SEO Strategy: Kalle KRR Chat Paper

## Ziel

Maximale Sichtbarkeit für den Blog (ki-mathias.de) und das Projekt (github.com/pmmathias/krr-chat) durch strategische Veröffentlichung des Papers auf mehreren Kanälen.

---

## Phase 1: Sofort (diese Woche)

### 1.1 Google Scholar Meta-Tags ✅ DONE

Die Blog-Beiträge haben jetzt `citation_*` Meta-Tags:
- `ki-mathias.de/en/krr-chat-explained.html` (EN)
- `ki-mathias.de/krr-chat-erklaert.html` (DE)

Scholar indexiert HTML-Seiten mit diesen Tags. Dauert 3-6 Monate bis zur ersten Indexierung, aber der Prozess ist automatisch — kein Submit nötig.

**Voraussetzungen die erfüllt sind:**
- ✅ `citation_title`, `citation_author`, `citation_publication_date` vorhanden
- ✅ Abstract sichtbar im HTML-Body
- ✅ Referenzen/Formeln im Body
- ✅ Seiten erreichbar von Homepage via Links
- ✅ Keine Login-Wall
- ✅ Stabile URLs

### 1.2 ORCID Profil erstellen

- https://orcid.org/register — kostenlos, 5 Minuten
- Affiliation: "Independent Researcher" oder "pmagentur.com"
- Website: ki-mathias.de
- ORCID-iD in alle zukünftigen Papers eintragen

### 1.3 Google Scholar Profil erstellen

- https://scholar.google.com/citations — mit beliebigem Google-Account
- Affiliation: "KI-Mathias / pmagentur.com"
- Homepage: ki-mathias.de
- Profil öffentlich machen
- Papers werden automatisch erscheinen sobald Scholar die Blog-Seiten indexiert

---

## Phase 2: Paper auf Zenodo (1-2 Wochen)

### Was ist Zenodo?

- Betrieben vom CERN, kostenlos, Open Access
- **Kein Review, kein Endorser, keine Affiliation nötig**
- Upload PDF → bekommst eine **DOI** (Digital Object Identifier)
- Wird von Google Scholar indexiert
- GitHub-Integration: kann automatisch Releases archivieren

### Schritte

1. LaTeX kompilieren: `paper/latex/main.tex` → PDF
2. Auf https://zenodo.org Account erstellen (GitHub-Login möglich)
3. "New Upload" → PDF hochladen
4. Metadata ausfüllen:
   - **Title:** "Kalle: A Transparent Language Model Using Kernel Ridge Regression with Random Fourier Features"
   - **Authors:** Leonhardt, Mathias (ORCID verlinken)
   - **Description:** Abstract aus dem Paper
   - **Keywords:** kernel ridge regression, random fourier features, language model, eigenvalues, RAG
   - **Related identifiers:**
     - `https://github.com/pmmathias/krr-chat` (isSupplementedBy)
     - `https://ki-mathias.de/en/krr-chat-explained.html` (isDocumentedBy)
   - **License:** CC-BY 4.0
   - **Resource type:** Preprint
5. Publish → DOI wird sofort vergeben

### SEO-Effekt

- Zenodo-DOI in Blog-Post eintragen (`citation_doi` Meta-Tag)
- Zenodo-DOI in README.md des GitHub-Repos
- Zenodo wird von Scholar indexiert → Paper taucht in Scholar auf
- DOI ist permanent und zitierbar

---

## Phase 3: arXiv (wenn Endorser gefunden, optional)

### Hürde

Seit Januar 2026 braucht man als Erstautor ohne Uni-Affiliation einen **persönlichen Endorser** — jemand der bereits in cs.CL oder cs.AI publiziert hat.

### Wie Endorser finden

- **LinkedIn:** ML-Researcher in Hamburg/Deutschland kontaktieren
- **ML-Meetups:** Hamburg hat aktive ML/AI-Meetups
- **Twitter/Mastodon:** ML-Community, Paper teilen, Kontakte knüpfen
- **Direkt fragen:** Höfliche Email an Autoren von Related Work (Rahimi, Jacot-Nachfolger)

### Falls Endorser gefunden

- Kategorie: `cs.CL` (primary), `cs.AI` (secondary)
- Format: LaTeX (`paper/latex/main.tex`)
- arXiv-ID wird sofort von Scholar + Semantic Scholar indexiert

### Falls kein Endorser

- Zenodo + Scholar-Tags reichen für SEO
- HAL (hal.science) als Alternative — kein Endorsement, europäisch, wird indexiert

---

## Phase 4: Konferenz-Einreichung (~Juli 2026)

### Bestes Ziel: EMNLP 2026 System Demonstrations

- **Deadline:** voraussichtlich Juli 2026 (noch nicht announced)
- **Format:** 6 Seiten + References, Single-blind
- **Anforderungen:** Live-Demo-Link ✅, Screencast-Video (2.5 Min)
- **Preprints erlaubt:** Ja (arXiv/Zenodo kein Problem)
- Das Paper ist bereits in der richtigen Länge und Struktur

### Alternative: JOSE (Journal of Open Source Education)

- **Rolling Submission** — kein Deadline
- Perfekter Fit: Educational Software, Open Source, Browser-basiert
- Peer-Reviewed, Open Access, kostenlos
- Kürzeres Format (2-4 Seiten), fokussiert auf das Software-Artefakt
- https://jose.theoj.org

### Backup: BEA Workshop 2027

- Building Educational Applications, co-located mit ACL
- Nächste Deadline vermutlich Anfang 2027
- 4-8 Seiten, Focus auf Educational Tools

---

## Phase 5: Laufende SEO-Optimierung

### Blog-Posts mit Scholar-Tags versehen (alle relevanten)

Auch diese Posts könnten Scholar-Tags bekommen:
- `ki-mathias.de/en/eigenvalues.html` — "Eigenvalues & AI"
- `ki-mathias.de/en/glass-bead-game.html` — "Glass Bead Game"

### Cross-Linking-Strategie

```
Zenodo (DOI)
  ↕
arXiv (wenn möglich)
  ↕
Blog: ki-mathias.de/en/krr-chat-explained.html
  ↕ (citation_doi, citation_pdf_url)
GitHub: github.com/pmmathias/krr-chat
  ↕ (README verlinkt alles)
Demo: pmmathias.github.io/krr-chat/
```

Jede Plattform verlinkt auf alle anderen → maximale Link-Equity für Google.

### Zenodo DOI in Blog eintragen

Sobald die DOI existiert:
```html
<meta name="citation_doi" content="10.5281/zenodo.XXXXXXX">
<meta name="citation_pdf_url" content="https://zenodo.org/record/XXXXXXX/files/paper.pdf">
```

### README.md Badge

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

---

## Zusammenfassung: Was bringt was?

| Aktion | Aufwand | SEO-Wert | Reichweite | Timeline |
|---|---|---|---|---|
| Scholar Meta-Tags im Blog | 30 Min ✅ | ⭐⭐⭐ | Google Scholar | 3-6 Monate |
| ORCID + Scholar Profil | 15 Min | ⭐⭐ | Identität/Credibility | Sofort |
| Zenodo Upload (DOI) | 1 Std | ⭐⭐⭐ | Scholar + DOI-System | 1-2 Wochen |
| arXiv (mit Endorser) | 2 Std + Endorser finden | ⭐⭐⭐⭐⭐ | Maximale ML-Sichtbarkeit | Unbestimmt |
| EMNLP 2026 Demo | Paper polieren + Video | ⭐⭐⭐⭐ | Top NLP-Venue | Juli 2026 |
| JOSE Submission | Paper anpassen | ⭐⭐⭐ | Peer-reviewed OA | Rolling |

**Minimaler Aufwand, maximaler Effekt:** Zenodo + Scholar-Tags + ORCID/Scholar-Profil. Das gibt dir eine DOI, Scholar-Indexierung und eine zitierbare Publikation — alles ohne Endorser, ohne Review-Wartezeit, ohne Konferenzreise.
