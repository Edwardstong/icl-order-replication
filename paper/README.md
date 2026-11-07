# Target values from the paper

Transcribed from the arXiv v1 HTML rendering of 2511.09700 (https://arxiv.org/html/2511.09700).
Spot-check against the PDF before quoting in the report.
Checked 2026-11-06: every value in both tables (255 numbers) was found in the PDF text extracted
with pypdf; the transcription is complete as of v1 of the paper.

- `table2_sensitivity_by_model.csv`  Table 2: ordering, selection, r per model, averaged over datasets.
- `table4_sensitivity_by_dataset.csv` Table 4 (appendix): ordering and selection per model x dataset. No r column in the paper.
- `models.csv` paper model label -> Hugging Face id. `hf_id_source` says whether the id is printed in the
  paper (Tables 5-7), inferred from family and size, or absent from the paper.

Headline averages (Sec. 3.3): selection 0.02251, ordering 0.01970; mean r 1.09 classification, 1.46 generation.
