# Content & SEO Strategy: KI-Mathias Academic Publishing

**Stand:** 16. April 2026 (v2 Release)
**Autor:** Mathias Leonhardt
**Blog:** https://ki-mathias.de

---

## 1. Accounts & Profile

| Plattform | URL | Status | Zweck |
|---|---|---|---|
| **ORCID** | https://orcid.org/0009-0009-7154-5351 | ✅ Aktiv | Akademische Identität, verknüpft alle Publikationen |
| **Google Scholar** | https://scholar.google.com/citations (Mathias Leonhardt) | ✅ Aktiv | Scholar-Profil mit automatischer Paper-Erkennung |
| **OSF / ECSarXiv** | https://osf.io/preprints/ecsarxiv/5smqn | ✅ Eingereicht (pending moderation) | Preprint-Server, DOI-Vergabe |
| **GitHub (Blog)** | https://github.com/pmmathias/blog | ✅ Aktiv | Blog-Hosting via GitHub Pages (ki-mathias.de) |
| **GitHub (KRR Chat)** | https://github.com/pmmathias/krr-chat | ✅ Aktiv | Code-Repo + Paper + Live-Demo via GitHub Pages |
| **GitHub Pages (Demo)** | https://pmmathias.github.io/krr-chat/ | ✅ Live | Kalle-Chatbot, direkt aufrufbar |

### Vorhandene Publikation (vor diesem Projekt)

- **"Precomputed Illumination for Virtual Reality Scenarios"** (2012, VR/AR, S. 175-187, mit CA Bohn)
  - Bereits auf Google Scholar indexiert
  - Thematisch relevant: Radiosity/Global Illumination → §3.1 im Kalle-Paper

---

## 2. Pilot: Kalle-Paper — Erledigte TODOs

### 2.1 Paper geschrieben & publiziert

| TODO | Status | Details |
|---|---|---|
| Paper-Draft (Markdown) | ✅ | `paper/draft.md` — GitHub-Markdown mit Math |
| Paper (LaTeX) | ✅ | `paper/latex/main.tex` — kompilierfertig, 7 Seiten |
| Paper (PDF) | ✅ | `paper/latex/leonhardt2026_kalle_krr_chat.pdf` |
| Mathematische Herleitungen | ✅ | 19 Gleichungen: Projektion → Neumann-Reihe → Eigenwerte → Ridge → Kernel → RFF → Radiosity → PageRank → NTK |
| ECSarXiv-Einreichung | ✅ | Pending moderation, DOI kommt nach Freigabe |

### 2.2 Blog Scholar-Ready gemacht

| TODO | Status | Details |
|---|---|---|
| `citation_*` Meta-Tags (EN) | ✅ | `ki-mathias.de/en/krr-chat-explained.html` |
| `citation_*` Meta-Tags (DE) | ✅ | `ki-mathias.de/krr-chat-erklaert.html` |
| Didaktische Revisionen R1-R7 | ✅ | σ/λ Erklärung, Narrative Brücke, RAG Worked Example, n-gram Formalisierung, Zahlen-Konsistenz, bilinguales Beispiel, "Architekturelle Eigenschaften" |

### 2.3 GitHub-Repo professionalisiert

| TODO | Status | Details |
|---|---|---|
| `src/` mit Build-Pipeline | ✅ | `build.py`, `gen_corpus.py`, `gen_rag_qa.py` |
| `src/solvers.py` (v2) | ✅ | Block-PCG, Power Iteration, unified solve() entry |
| `src/build_v2.py` (v2) | ✅ | Pluggable --solver={direct,cg,power} |
| `src/benchmark.py` (v2) | ✅ | Solver comparison framework |
| `tests/` mit Regression Suite | ✅ | 34 Playwright-Szenarien |
| `benchmarks/` (v2) | ✅ | Plots + README + results.json |
| `data/` mit Corpus + Chunks | ✅ | 2174 Pairs, 29 RAG-Chunks |
| `ARCHITECTURE.md` | ✅ | + Solver-Options-Sektion (v2) |
| `paper/theory/absorber_stochastisierung.md` (v2) | ✅ | Mathematische Herleitung, 10 Referenzen |
| README mit Live-Demo-Button | ✅ | + v2 Build-Befehle |
| GitHub Pages aktiviert | ✅ | `pmmathias.github.io/krr-chat/` |
| Paper v1 (7 Seiten) | ✅ | Zenodo 10.5281/zenodo.19595642 |
| **Paper v2 (9 Seiten)** | **✅** | PDF kompiliert; **Zenodo-Upload ausstehend** (s. T033) |
| Consolidiert auf einen Chatbot | ✅ | `index.html` = Kalle (aktuell, RAG, 2178 Pairs) |

### 2.4 Profile angelegt

| TODO | Status | Details |
|---|---|---|
| ORCID-Profil | ✅ | Employment, Education, Work eingetragen |
| Google Scholar Profil | ✅ | Foto, Affiliation, Homepage, Areas of Interest |
| ORCID-Work verknüpft | ✅ | Kalle-Paper als Teaching Material eingetragen |

---

## 3. Offene Action Steps (warten auf externe Events)

### 3.1 Warten auf: ECSarXiv Moderator-Freigabe (1-3 Tage)

**Trigger:** Email von ECSarXiv "Your preprint has been accepted"

**Dann sofort tun:**

1. **DOI notieren** (Format: `10.31224/XXXX`)

2. **Blog-Posts updaten** — `citation_doi` Meta-Tag hinzufügen:
   ```html
   <meta name="citation_doi" content="10.31224/XXXX">
   <meta name="citation_pdf_url" content="https://osf.io/preprints/ecsarxiv/5smqn/download">
   ```
   In beiden Versionen: `krr-chat-erklaert.html` (DE) und `en/krr-chat-explained.html` (EN)

3. **GitHub README** — DOI-Badge hinzufügen:
   ```markdown
   [![DOI](https://img.shields.io/badge/DOI-10.31224%2FXXXX-blue?style=flat-square)](https://doi.org/10.31224/XXXX)
   ```

4. **ORCID** — Work updaten: Type von "Teaching material" auf "Preprint" ändern, DOI eintragen

5. **Google Scholar** — Paper wird automatisch erscheinen (dauert Tage bis Wochen)

### 3.2 Optional: Zenodo als Backup (wenn ECSarXiv Probleme macht)

Zenodo hatte am 15.04.2026 einen 504-Outage. Falls ECSarXiv-Moderation abgelehnt wird oder zu lange dauert:
- Zenodo-Upload als Fallback (gleiche PDF, gleiche Metadata)
- Gibt ebenfalls DOI + Scholar-Indexierung

### 3.3 Konferenz-Einreichung: EMNLP 2026 System Demonstrations (~Juli 2026)

**Trigger:** Call for Papers wird veröffentlicht (typisch April/Mai 2026)

**Dann tun:**
1. Paper auf EMNLP-Seitenformat anpassen (ACL Style, 6 Seiten)
2. 2.5-Minuten Screencast-Video aufnehmen (Demo von Kalle)
3. Einreichen via OpenReview
4. Live-Demo-Link: `https://pmmathias.github.io/krr-chat/`

### 3.4 JOSE-Einreichung (rolling, kein Deadline)

Journal of Open Source Education — für den Educational-Software-Winkel.
- Kürzeres Format (2-4 Seiten)
- Fokus auf Reproduzierbarkeit + didaktischen Nutzen
- Kann parallel zu EMNLP eingereicht werden (verschiedene Venues, verschiedener Fokus)

---

## 4. Content-Strategie: Wie funktioniert der Kreislauf

### Der SEO-Kreislauf für akademische Blog-Inhalte

```
                    ┌─────────────────────┐
                    │   Blog-Post (HTML)   │
                    │  ki-mathias.de/...   │
                    │                     │
                    │  citation_* Tags    │
                    │  Sichtbare Formeln  │
                    │  Referenzen-Liste   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────┐   ┌──────────────┐   ┌───────────┐
     │   Preprint  │   │  GitHub Repo  │   │   Demo    │
     │  (ECSarXiv/ │   │  Code + Data  │   │  (GitHub  │
     │   Zenodo)   │   │  + ARCH.md    │   │   Pages)  │
     │             │   │              │   │           │
     │  → DOI      │   │  → README    │   │  → Live   │
     └──────┬──────┘   └──────┬───────┘   └─────┬─────┘
            │                 │                  │
            └────────────────┬┘──────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Google Scholar  │
                    │  Semantic Scholar│
                    │  ORCID Profil    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Zitierbar     │
                    │   Auffindbar    │
                    │   Verlinkbar    │
                    └─────────────────┘
```

### Die drei Säulen

1. **Blog-Post** = Inhalt (didaktisch, mit Formeln, bilingual)
2. **Preprint** = Akademische Credibility (DOI, Scholar-indexiert)
3. **Code-Repo** = Reproduzierbarkeit (Source, Tests, Live-Demo)

Jede Säule verlinkt auf die anderen beiden. Google wertet diese Cross-Links als Autoritätssignal.

---

## 5. Ausblick: Ausweitung auf andere Blog-Inhalte

### 5.1 Welche Blog-Posts sind "Paper-fähig"?

Der Blog ki-mathias.de hat ~25 Posts mit ~142.000 Wörtern. Nicht jeder Post rechtfertigt ein Paper. Kriterien für Paper-Eignung:

| Kriterium | Frage |
|---|---|
| **Originalität** | Gibt es eine neue Perspektive, Methode, oder Verbindung? |
| **Reproduzierbarkeit** | Gibt es Code, Daten, oder eine interaktive Demo? |
| **Formalisierbarkeit** | Lassen sich die Ideen in Gleichungen und Theoreme fassen? |
| **Venue-Fit** | Gibt es ein Venue das genau dieses Thema sucht? |

### 5.2 Kandidaten für zukünftige Papers

| Blog-Post | Paper-Idee | Mögliches Venue |
|---|---|---|
| **Eigenwerte & KI** (`eigenwerte.html`) | "The Eigenvalue Principle: A Unified Framework from Least Squares to Neural Networks" | EMNLP Tutorial, Educational Advances in AI (EAAI) |
| **Glasperlenspiel** (`glasperlenspiel.html`) | "K = K: The Mathematical Isomorphism Between Quantum Propagators and Machine Learning Kernels" | Interdisziplinäres Journal (Foundations of Physics + ML) |
| **Gott-Post** (`gott.html`) | Bereits als EuARe-Paper eingereicht | Religionswissenschaftliches Venue |
| **Emergence-Post** | "Architectural Properties vs. Emergence in Small-Scale Language Models" | Workshop on Interpretability / BEA |

### 5.3 Template: Vom Blog-Post zum Paper (wiederholbarer Prozess)

Für jeden zukünftigen Blog-Post der Paper-Potenzial hat:

**Schritt 1: Blog Scholar-Ready machen** (30 Min)
```html
<meta name="citation_title" content="...">
<meta name="citation_author" content="Leonhardt, Mathias">
<meta name="citation_publication_date" content="YYYY/MM/DD">
<meta name="citation_language" content="en">
<meta name="citation_abstract" content="...">
<meta name="citation_keywords" content="...">
<meta name="citation_publisher" content="KI-Mathias">
```

**Schritt 2: Paper schreiben** (1-2 Tage)
- LaTeX, basierend auf Blog-Inhalt
- Formeln formalisieren, Related Work hinzufügen
- Template: `paper/latex/main.tex` als Vorlage

**Schritt 3: Preprint hochladen** (1 Stunde)
- ECSarXiv oder Zenodo
- DOI zurück in Blog-Post eintragen
- GitHub README verlinken

**Schritt 4: Konferenz/Journal einreichen** (wenn passend)
- EMNLP System Demos (NLP-Systeme)
- JOSE (Educational Software)
- BEA Workshop (Educational NLP)
- EAAI (Educational AI, co-located mit AAAI)

**Schritt 5: Cross-Links pflegen**
- ORCID Work hinzufügen
- Google Scholar Profil prüfen
- Blog ↔ Preprint ↔ Repo ↔ Demo verlinken

### 5.4 Langfristiges Ziel

Ein Google Scholar Profil das zeigt:

```
Mathias Leonhardt
CTO at pmagentur.com | KI-Mathias
Verified email at pmagentur.com — Homepage

  Precomputed Illumination for Virtual Reality Scenarios    2012    VR/AR
  Kalle: A Transparent Language Model Using KRR with RFF    2026    ECSarXiv
  The Eigenvalue Principle: From Least Squares to LLMs      2026    ECSarXiv
  K = K: Quantum Propagators and ML Kernels                 2027    tbd
```

Jedes Paper verlinkt auf ki-mathias.de → Scholar-Authority wächst → Blog rankt höher in Google → mehr Leser → mehr Zitationen → Kreislauf.

---

## 6. Kosten

| Posten | Kosten |
|---|---|
| ORCID | Kostenlos |
| Google Scholar Profil | Kostenlos |
| ECSarXiv / OSF | Kostenlos |
| Zenodo | Kostenlos |
| GitHub Pages | Kostenlos |
| JOSE Publication | Kostenlos (kein APC) |
| EMNLP Registration | ~$200-400 (nur wenn akzeptiert + teilnehmen) |

**Gesamtkosten der SEO-Strategie: €0** (außer optionale Konferenzteilnahme).
