# Reports

* `experiments/<run_id>/` is written by `python -m ml.experiments.run_experiments` (or the Model Laboratory's "Run full suite" button). Each run contains:
  * `results.json`: every number, the configuration, folds and environment;
  * `summary.md`: auto-generated tables;
  * `predictions.parquet`: out-of-fold scores;
  * `figures/*.png`.
* `experiments/main/` is the run cited in `docs/report.md` and `README.md`.

Nothing in this folder is written by hand. HTML reports generated through the API are stored under `STORAGE_DIR/reports/`.
