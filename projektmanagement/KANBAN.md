# Kalle v2: Stochastisierung + Power Iteration — Kanban Board

**Regel: IMMER nur EIN Ticket IN PROGRESS. Eins nach dem anderen.**
**Jedes Ticket hat Akzeptanzkriterien. Kein DONE ohne bestandenen Test.**
**Stand: 2026-04-16**

## Projekt-Ziel

Kalle von Closed-Form-Solve auf iterative GPU-Methode (Power Iteration auf stochastisierter Matrix / Conjugate Gradient) umstellen. Die im Eigenwerte-Blogpost skizzierte Absorber-Idee wird als **Algorithmus-Optimierung** ausgearbeitet, nicht nur als Analogie. Paper v2 wird als neue Version auf Zenodo publiziert. Beide Blogposts werden aktualisiert.

## TODO

| # | Ticket | Prio | Abhängig von |
|---|--------|------|--------------|
| [T028](T028_theory_absorber_stochastic.md) | Theoretische Ausarbeitung: Absorber + Stochastisierung | P0 | — |
| [T029](T029_experimental_cg_kalle.md) | Experimentelle Kalle v2 — CG/Power Iteration Implementation | P0 | T028 |
| [T030](T030_benchmark_v1_vs_v2.md) | Benchmark v1 vs v2 (Memory, Speed, Accuracy, GPU) | P0 | T029 |
| [T031](T031_eigenvalues_blog_sharpening.md) | Eigenwerte-Blogpost schärfen — Absorber als Algorithmus | P1 | T028 |
| [T032](T032_paper_v2.md) | Paper v2 — Überarbeitung mit neuer Sektion | P1 | T030, T031 |
| [T033](T033_zenodo_v2_upload.md) | Zenodo Update — Paper v2 als neue Version | P1 | T032 |
| [T034](T034_krr_chat_blog_update.md) | KRR-Chat-Blogpost updaten (DE+EN) | P2 | T032, T033 |
| [T035](T035_repo_documentation.md) | Repo-Dokumentation updaten (README, ARCHITECTURE) | P2 | T029, T030 |

## IN PROGRESS

(noch nichts)

## TESTING

(leer)

## DONE

(leer)

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
