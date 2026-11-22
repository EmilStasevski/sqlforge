# sqlforge: Text-to-SQL auf Deutsch und Englisch, per QLoRA trainiert auf ausführungsverifizierten synthetischen Daten

[![ci](https://github.com/EmilStasevski/sqlforge/actions/workflows/ci.yml/badge.svg)](https://github.com/EmilStasevski/sqlforge/actions/workflows/ci.yml)

[English version](README.md)

sqlforge feinjustiert ein 7B-Code-Modell (Qwen2.5-Coder-7B-Instruct, 4-bit-NF4-QLoRA auf einer 16-GB-GPU), damit
es deutsche und englische Fragen mit DuckDB-SQL beantwortet. **Jedes Trainingslabel wird durch Ausführen geprüft.**
Kein Teil des Datensatzes wurde von Hand gelabelt oder kontrolliert. Bewertet wird das Modell nach
**Execution Accuracy** auf einem Datenbankschema, das es im Training nie gesehen hat. Das Serving läuft über ein
Go-Gateway mit Redis-Cache; Request-Events fließen über Kafka nach ClickHouse und Grafana.

```
Schemas (TPC-H, SMARD via dbt, Open-Meteo) ─► grammatikbasierter SQL-Sampler ─► DuckDB-Ausführungsprüfung
      ─► DE/EN-Fragevorlagen ─► Paraphrase per lokalem LLM (Ollama) + Erhaltungsfilter ─► Pandera ─► DVC/MinIO
      ─► QLoRA (TRL + PEFT + bitsandbytes; MLflow + W&B) ─► Eval (Execution Accuracy, Buckets, Fehlertaxonomie)
      ─► Merge ─► GGUF Q4_K_M ─► Ollama

Client ─► Go-Gateway :8080 ── Redis (exakter + normalisierter Frage-Cache, Rate Limit), Prometheus /metrics
             │                └─ Kafka ask.events ─► Consumer ─► ClickHouse ─► Grafana
             ▼
        FastAPI-Executor :8000 ─► Modellserver (Ollama, GGUF) ─► Rust-sqlnorm-Guard ─► DuckDB (read-only)
```

## Ergebnisse

<!-- results:start -->
| Modell (Qwen2.5-Coder-7B, 4-bit) | test_in EX | test_ood EX (unbekanntes Schema) |
|---|---|---|
| Basismodell, zero-shot | 83,0 % | 89,6 % |
| Basismodell, 3-shot | 80,7 % | 83,9 % |
| QLoRA (dieses Projekt), NF4 + Adapter | 99,5 % | 95,0 % |
| QLoRA gemergt, GGUF Q4_K_M (Serving über Ollama) | 99,2 % | 95,0 % |

n = 1000 Fragen pro Split (etwa die Hälfte auf Deutsch), Greedy-Decoding, Execution Accuracy = Gleichheit der Ergebnismengen.

| Serving (vegeta, gleicher Request-Strom) | p50 | p95 | Cache-Trefferquote |
|---|---|---|---|
| ohne Redis-Cache | 846,6 ms | 1.762,1 ms | - |
| mit Redis-Cache | 1,9 ms | 1.049,8 ms | 77,8 % |

600 Requests pro Lauf mit 2/s: 120 verschiedene test_ood-Fragen x 5, 30 % der Wiederholungen als Groß-/Klein- bzw. Leerzeichen-Varianten (maximal mögliche Trefferquote 80,0 %); Gateway -> Executor -> GGUF auf einer GPU.

Training: 12.000 Beispiele, 1 Epoche, 1,8 h auf einer RTX 5060 Ti 16 GB, VRAM-Spitze 6,1 GiB.

Rust `sqlnorm` (Parsen + Kanonisieren + Hashen): **18,0-mal** schneller als sqlglot bei 15.000 Queries.

Datensatz: 12000 train / 1000 val / 1000 test_in / 1000 test_ood, alle per Ausführung verifiziert.

<!-- results:end -->

Alle Zahlen oben erzeugt `scripts/make_results_md.py` aus `results/*.json`. Aufschlüsselung nach Buckets und die
Fehlertaxonomie stehen in [docs/results.md](docs/results.md) (englisch).

### Was die Zahlen zeigen

- **Fine-Tuning hilft am meisten auf dem unbekannten Schema.** Auf test_ood (Open-Meteo-Wetterdaten, nie im
  Training) erreicht QLoRA 95,0 % gegenüber 83,9 % für das Basismodell mit 3 Beispielen und 89,6 % zero-shot. Die
  Fehler sinken von 161 (3-shot) auf 50. Falsche Tabellen-/Spaltenreferenzen fallen von 57 auf 5: Das Modell hat
  gelernt, die mitgegebene DDL zu lesen, statt sich an TPC-H zu erinnern.
- **test_in liegt nahe an der Obergrenze und sagt weniger aus.** 99,5 % auf test_in sind erwartbar: Die Fragen
  stammen aus denselben Vorlagen und Schemas wie das Training (geteilt wird aber kein SQL, das sichert ein Leak-Check
  über kanonische Hashes). Die maßgebliche Zahl ist test_ood.
- **Deutsch ist nicht mehr schwieriger.** Das Basismodell erreicht auf test_in 76,1 % (DE) gegenüber 89,9 % (EN);
  QLoRA 99,4 % gegenüber 99,6 %.
- **Few-Shot schadet dem Basismodell** (80,7 % / 83,9 % gegenüber 83,0 % / 89,6 % zero-shot). Falsche
  Schemareferenzen steigen auf test_ood von 26 auf 57. Keine davon verwendet Tabellennamen der Trainingsschemas; es
  sind erfundene Spaltennamen (`precipitation` statt `precipitation_sum`), unqualifiziertes `city_id` in Joins und
  verschachtelte Aggregate. Der längere Prompt macht das Basismodell offenbar ungenauer im Umgang mit der DDL,
  statt es zum Kopieren der Beispiele zu verleiten.
- **Die verbleibenden Fehler folgen meist einem Muster.** 34 der 50 QLoRA-Fehler auf test_ood sind
  `wrong_aggregate`, davon 30 einfache „Zeilen auflisten“-Fragen, deren Wetterspalten wie Aggregate heißen
  („mean temperature“ für `temperature_2m_mean`, „maximale Windböe“ für `wind_gusts_10m_max`). Das Modell schreibt
  dann `AVG(...)`/`MAX(...)`. Manche dieser Fragen sind sprachlich tatsächlich mehrdeutig; das begrenzt eher die
  Vorlagen als das Modell. Abhilfe wären Trainingsspalten, deren Namen Aggregatwörter enthalten.
- **Die Quantisierung kostet hier nichts.** Das gemergte Modell als GGUF Q4_K_M (4,7 GB, Serving über Ollama)
  erreicht auf test_ood 95,0 %, genau wie das NF4-Modell mit Adapter, und auf test_in 99,2 % gegenüber 99,5 %.

## Wie die Daten entstehen (und warum keine manuelle Prüfung nötig ist)

| Schritt | Was die Korrektheit sichert |
|---|---|
| Schemas | TPC-H (von DuckDB generiert), SMARD-Stromdaten (dbt-Marts, 20 dbt-Tests), Open-Meteo-Wetterdaten für 6 NRW-Städte (**zurückgehalten**: nur Test) |
| SQL | Aus einer typisierten Grammatik gesampelt (12 Query-Arten: Filter, Aggregate, GROUP BY/HAVING, Top-k, Joins über FK-Pfade, Subqueries, Zeit-Buckets). Literale stammen aus den echten Daten |
| Verifikation | Eine Query bleibt nur, wenn sie ein einzelnes lesendes SELECT ist (Rust `sqlnorm`), innerhalb des Timeouts läuft, 1–10 000 Zeilen liefert, die nicht alle NULL sind, single- und multithreaded dasselbe Ergebnis gibt, keine Gleichstände an der ORDER-BY/LIMIT-Grenze hat und auch unter dem umbenannten/reduzierten Trainingsschema dasselbe Ergebnis liefert |
| Fragen | Aus derselben Spezifikation wie das SQL erzeugt (DE + EN). Eine Paraphrase (qwen3:8b) ersetzt die Vorlage nur, wenn sie jedes Literal, jeden Glossarbegriff, jede Zahl und jedes Operatorkonzept („Durchschnitt“, „größer als“, „nicht“, …) behält und in derselben Sprache bleibt |
| Splits | Deduplizierung über kanonische SQL-Hashes und Leak-Check zwischen den Splits; Pandera-Vertrag auf jedem Split; der Build ist deterministisch (durch Neubau geprüft) |
| Augmentierung | Zufällige Tabellen-/Spaltenteilmengen, gemischte Reihenfolge und umbenannte Bezeichner (23 % des Trainings), damit das Modell die DDL liest, statt TPC-H auswendig zu lernen |

## Schnellstart

```bash
uv sync --group train          # Python 3.12, torch cu128 (Blackwell OK), baut die Rust-Erweiterung
make env-check                 # Toolchains, GPU, Datenbanken
make data                      # dvc repro: Schemas -> verifizierter Datensatz -> Paraphrasen (braucht Ollama)
make eval-baseline             # Basismodell zero-/few-shot
make train                     # QLoRA, 1 Epoche
make eval-qlora gguf           # Adapter evaluieren; Merge -> GGUF Q4_K_M
make up                        # Gateway, Executor, Ollama, Redis, Kafka, ClickHouse, Grafana, MinIO, MLflow, Prometheus
curl -s localhost:8080/ask -d '{"question":"Wie viele Tage mit mehr als 20 mm Niederschlag gab es 2024 in Köln?","db":"weather"}'
make loadtest                  # vegeta, Redis-Cache an vs. aus
make test bench                # pytest + Benchmark sqlnorm vs. sqlglot
make register artifacts        # MLflow-Model-Registry (sqlforge-qlora); Adapter/GGUF/Eval-Reports -> MinIO
make kind-up                   # optional: Gateway/Executor/Redis auf kind per kustomize -> localhost:18080/ask
uv run --group demo streamlit run src/sqlforge/serve/demo.py   # optional: Streamlit-Oberfläche über dem Gateway
```

Grafana: http://localhost:3000 (Dashboard „sqlforge serving“). MLflow: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Repository

| Pfad | Inhalt |
|---|---|
| `src/sqlforge/schemas` | DuckDB-Builder, Schema-Karten (Rollen + DE/EN-Glossar), Schemavarianten |
| `src/sqlforge/sampler`, `nlgen` | spezifikationsbasierter SQL-Sampler, DE/EN-Fragegenerator |
| `src/sqlforge/verify` | Ausführungsprüfung, Ergebnisvergleich |
| `src/sqlforge/paraphrase` | Paraphrasieren mit Ollama + Erhaltungsfilter |
| `src/sqlforge/dataset` | Split-Builder, Pandera-Vertrag, Leak-Check |
| `src/sqlforge/train`, `eval` | QLoRA-Training, Evaluation nach Execution Accuracy |
| `src/sqlforge/serve` | FastAPI-Executor, Kafka→ClickHouse-Consumer, Adapter→GGUF-Export |
| `rust/sqlnorm` | sqlparser-rs + PyO3: `is_safe_select`, `normalize`, `canonical_hash`, `features` |
| `go/gateway` | Redis-Cache + Rate Limit, Prometheus, Kafka-Producer |
| `dbt/smard` | dbt-duckdb-Modelle und -Tests für die SMARD-Marts |
| `k8s/` | kustomize-Basis (Gateway, Executor, Redis) + kind-Overlay (Modellserver über EndpointSlice) |
| `docs/env.md` | geprüfte Versionen und Hinweise zu Blackwell / transformers 5 / GGUF-Export (englisch) |
| `docs/results.md`, `docs/cv.md` | vollständige Ergebnisse und Lebenslauf-Text, beide aus `results/*.json` erzeugt |
