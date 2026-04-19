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

---

## Status-Tracker (Stand: 2026-04-19)

### ✅ Erledigt

| Aktion | Datum | Details |
|--------|-------|---------|
| Scholar Meta-Tags im Blog | 2026-04-15 | `citation_*` auf eigenwerte.html + en/eigenvalues.html (Kalle) und gott.html + en/god.html (Coherence) |
| ORCID Profil | 2026-04-15 | 0009-0009-7154-5351 — Employment, Education, 3 Works gelistet |
| Google Scholar Profil | 2026-04-15 | Mit Foto, Affiliation, Homepage, Keywords |
| Zenodo: Kalle Paper | 2026-04-15 | DOI: 10.5281/zenodo.19595642 — 16 Views, 1 Download (Stand 19.04.) |
| Zenodo: Coherence Paper | 2026-04-15 | DOI: 10.5281/zenodo.19589935 — 15 Views, 2 Downloads (Stand 19.04.) |
| citation_doi in Blog | 2026-04-15 | Beide DOIs in den jeweiligen Blog-Posts als Meta-Tags |
| PhilPapers Listing | 2026-04-15 | Coherence Paper gelistet (Metaphysics & Epistemology, Value Theory) |
| Cross-Linking Blog↔Zenodo↔ORCID↔Scholar | 2026-04-15 | Kreislauf geschlossen |
| YouTube-Kanal verlinkt in ORCID | 2026-04-15 | Social Links: Blog, GitHub, YouTube, Mastodon |
| Quora Topic-Credential | 2026-04-16 | "Philosophy of Religion" mit Zenodo-Referenz |
| dev.to Cross-Posts | 2026-04-19 | Vogelsim + Deepfakes mit canonical_url auf Blog |
| SEO-Report-Automation | 2026-04-16 | `scripts/seo_report.py` — GSC + YT Analytics + Zenodo + Mastodon + Lighthouse + Bot-Filter |

### ⏳ Offen / In Arbeit

| Aktion | Status | Nächster Schritt |
|--------|--------|-----------------|
| arXiv Submission | ❌ braucht Endorser | Endorser in ML-Community suchen (LinkedIn/Mastodon/Meetups). Alternative: HAL (hal.science) als endorser-freie Alternative |
| EMNLP 2026 Demo Track | ⏳ Deadline noch nicht announced | Paper auf 6 Seiten kürzen, 2.5-Min Screencast-Video produzieren. Voraussichtlich Juli 2026 |
| JOSE Submission | ⏳ Rolling | Paper auf 2-4 Seiten kürzen, Software-Fokus. Kann jederzeit eingereicht werden |
| README.md DOI Badge | ✅ bereits vorhanden | Zeile 17 in README.md, Zenodo-Badge verlinkt auf DOI |
| KRR-Chat Demo verbessern | ⏳ in Arbeit | Eigene Claude-Code-Instanz arbeitet daran. Wenn fertig → Eigenwerte-Post auf dev.to cross-posten |
| Google Scholar Indexierung | ⏳ automatisch | Dauert 3-6 Monate. Kann nicht beschleunigt werden. |
| Quora God Debate Space | ⏳ Post submitted | "What if both sides argue about the wrong thing?" — wartet auf Moderator-Review |
| Show HN | ⏳ Account braucht Karma | Erst 3-5 echte HN-Kommentare, dann Submit. KRR-Chat oder Vogelsim als Demo-URL |
| Reddit Devvit Game | ⏳ separates Projekt | Vogelsim als eingebettetes Reddit-Game. Eigene Claude-Code-Instanz. |

### 📊 Aktuelle Metriken (19.04.2026)

| Kanal | Metrik |
|-------|--------|
| **YouTube** | 302 Lifetime Views, 264 (28d), Tommy=161 Views 🚀 |
| **Google Search** | 37 sichtbare Query-Impressionen, 4 Clicks, "mathias leonhardt" Pos 2.3 |
| **Zenodo Kalle** | 16 Views, 1 Download |
| **Zenodo Coherence** | 15 Views, 2 Downloads |
| **Mastodon** | 2 Follower, 7 Toots |
| **dev.to** | 2 Artikel live (Vogelsim + Deepfakes), 0 Reactions (frisch) |
| **Quora** | 31 Antworten (~70 Views total, low ROI), God Debate Space aktiv |
| **Reddit** | u/KI-Mathias (Premium), Karma aufbauen läuft |
| **Lighthouse** | SEO 100/100 auf allen Pages |

### 🎯 Nächste Prioritäten

1. **KRR-Chat fertig verbessern** → dann Eigenwerte-Post auf dev.to (stärkster tech-orientierter Post)
2. ~~README.md DOI Badge~~ ✅ war schon drin
3. **Reddit Karma** weiter aufbauen → r/SideProject Post wenn ≥50 Karma
4. **JOSE evaluieren** — Rolling Submission, kein Deadline-Druck, peer-reviewed
5. **arXiv Endorser suchen** — Hamburg ML-Meetups, Mastodon ML-Community
