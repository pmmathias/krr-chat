# Kalle v2: Stochastisierung + Power Iteration — Kanban Board

**Regel: IMMER nur EIN Ticket IN PROGRESS. Eins nach dem anderen.**
**Jedes Ticket hat Akzeptanzkriterien. Kein DONE ohne bestandenen Test.**
**Stand: 2026-04-16**

## Projekt-Ziel

Kalle von Closed-Form-Solve auf iterative GPU-Methode (Power Iteration auf stochastisierter Matrix / Conjugate Gradient) umstellen. Die im Eigenwerte-Blogpost skizzierte Absorber-Idee wird als **Algorithmus-Optimierung** ausgearbeitet, nicht nur als Analogie. Paper v2 wird als neue Version auf Zenodo publiziert. Beide Blogposts werden aktualisiert.

## TODO

| # | Ticket | Prio | Abhängig von |
|---|--------|------|--------------|
| [T031](T031_eigenvalues_blog_sharpening.md) | Eigenwerte-Blogpost schärfen — Absorber als Algorithmus | P1 | T028 ✓ |
| [T032](T032_paper_v2.md) | Paper v2 — Überarbeitung mit neuer Sektion | P1 | T030, T031 |
| [T033](T033_zenodo_v2_upload.md) | Zenodo Update — Paper v2 als neue Version | P1 | T032 |
| [T034](T034_krr_chat_blog_update.md) | KRR-Chat-Blogpost updaten (DE+EN) | P2 | T032, T033 |
| [T035](T035_repo_documentation.md) | Repo-Dokumentation updaten (README, ARCHITECTURE) | P2 | T029, T030 |

## IN PROGRESS

| # | Ticket | Seit |
|---|--------|------|
| [T031](T031_eigenvalues_blog_sharpening.md) | Eigenwerte-Blog schärfen | 2026-04-16 |

## TESTING

(leer)

## DONE

| # | Ticket | Abgeschlossen | Highlight |
|---|--------|---------------|-----------|
| [T028](T028_theory_absorber_stochastic.md) | Theoretische Ausarbeitung | 2026-04-16 | PCG empfohlen; Absorber-Interpretation als pädagogischer Rahmen |
| [T029](T029_experimental_cg_kalle.md) | Block-PCG Implementation | 2026-04-16 | `src/solvers.py`, `src/build_v2.py` — 14 iters, Top-1 62.8% |
| [T030](T030_benchmark_v1_vs_v2.md) | Benchmarks | 2026-04-16 | `benchmarks/*` — ehrliche Aussage: Direct schneller bis D ≤ 4096 auf CPU |

## Reihenfolge

```
T028 (Theorie) ──► T029 (Implementation) ──► T030 (Benchmark)
  │                                             │
  ▼                                             │
T031 (Blog schärfen)                            │
  │                                             │
  └─────────────────┬───────────────────────────┘
                    ▼
                  T032 (Paper v2)
                    │
                    ▼
                  T033 (Zenodo Upload)
                    │
                    ▼
                  T034 (Blog-Links updaten)
                    │
                    ▼
                  T035 (Repo-Docs)
```
