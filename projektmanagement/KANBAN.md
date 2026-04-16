# Kalle v2: Stochastisierung + Power Iteration — Kanban Board

**Regel: IMMER nur EIN Ticket IN PROGRESS. Eins nach dem anderen.**
**Jedes Ticket hat Akzeptanzkriterien. Kein DONE ohne bestandenen Test.**
**Stand: 2026-04-16 — v2 Sprint abgeschlossen bis auf Zenodo-Upload**

## Projekt-Ziel

Kalle von Closed-Form-Solve auf iterative GPU-Methode (Power Iteration auf stochastisierter Matrix / Conjugate Gradient) umstellen. Die im Eigenwerte-Blogpost skizzierte Absorber-Idee wird als **Algorithmus-Optimierung** ausgearbeitet, nicht nur als Analogie. Paper v2 wird als neue Version auf Zenodo publiziert. Beide Blogposts werden aktualisiert.

## TODO

| # | Ticket | Prio | Aufwand |
|---|--------|------|---------|
| [T036](T036_kalle_xl_full_blog.md) | Kalle XL — experimental variant on full DE+EN blog corpus (~3K+ pairs, V≈20K) | P1 | 6–8h |

## IN PROGRESS

(leer)

## WAITING

| # | Ticket | Wartet auf |
|---|--------|------------|
| [T033](T033_zenodo_v2_upload.md) | Zenodo v2 Upload | Manueller Browser-Upload (Anleitung: `paper/ZENODO_V2_UPLOAD_STEPS.md`) |

## DONE

| # | Ticket | Abgeschlossen | Highlight |
|---|--------|---------------|-----------|
| [T028](T028_theory_absorber_stochastic.md) | Theoretische Ausarbeitung | 2026-04-16 | PCG empfohlen; Absorber als pädagogischer Rahmen |
| [T029](T029_experimental_cg_kalle.md) | Block-PCG Implementation | 2026-04-16 | `src/solvers.py`, `src/build_v2.py` — 14 iters, Top-1 62.8% |
| [T030](T030_benchmark_v1_vs_v2.md) | Benchmarks | 2026-04-16 | `benchmarks/*` — ehrliche Aussage: Direct schneller bis D ≤ 4096 auf CPU |
| [T031](T031_eigenvalues_blog_sharpening.md) | Eigenwerte-Blog schärfen | 2026-04-16 | Neue Sektion "Why stochastic matters" (DE+EN) |
| [T032](T032_paper_v2.md) | Paper v2 | 2026-04-16 | 9 Seiten, §5 "Scalable Training", 8 neue Referenzen |
| [T034](T034_krr_chat_blog_update.md) | KRR-Chat-Blog updaten | 2026-04-16 | Neue Sektion "Scalable training: ridge λ as damping factor" (DE+EN) |
| [T035](T035_repo_documentation.md) | Repo-Doku updaten | 2026-04-16 | README, ARCHITECTURE.md, CONTENT_SEO_STRATEGY.md aktualisiert |

## Abhängigkeits-Flow (final)

```
T028 (Theorie) ──► T029 (Impl.) ──► T030 (Benchmark) ─────────┐
  │                   │                                         │
  ▼                   │                                         │
T031 (Eig.-Blog)      │                                         │
  │                   │                                         │
  └───────────┬───────┴─────────────────────────────────────────┘
              ▼
            T032 (Paper v2) ──► T033 (Zenodo, pending) ──► T034 (Kalle-Blog)
                                                                  │
                                                                  ▼
                                                            T035 (Repo-Docs)
```

## Nach Zenodo-Upload (T033)

Einmalig, sobald die neue Version-DOI da ist (sollte mit Concept-DOI identisch bleiben bei New-Version-Upload):

- [ ] Blog `citation_doi` Meta-Tag prüfen (sollte automatisch auf v2 zeigen via Concept-DOI)
- [ ] ORCID-Work updaten (v2-DOI hinzufügen)
- [ ] Google Scholar re-index abwarten (3-6 Monate)
- [ ] Optional: Hacker News "Show HN" Post mit v2-Angle (PageRank↔KRR Algorithmus-Brücke als Hook)
