# inverse-pinn-schrodinger

```aiignore
lf-pinn-inverse-schrodinger/
├── artifacts/
│   ├── demo_visuals/
│   │   ├── cross_overlap_heatmap.png
│   │   ├── density.png
│   │   ├── learned_energies.png
│   │   ├── learned_potential.png
│   │   ├── learned_wavefunctions.png
│   │   ├── overlap_heatmap.png
│   │   ├── pod_modes.png
│   │   ├── pod_singular_values.png
│   │   └── training_curves.png
│   ├── figures.md
│   ├── interpretability.md
│   ├── project_1_followup.md
│   └── take-home-messages.md
├── assets/
│   ├── images/
│   │   ├── loss_table.png
│   │   ├── loss_table_extended.png
│   │   ├── loss_table_numbered.png
│   │   └── project_2_architecture.png
│   ├── mermaid-diagrams/
│       ├── architecture.mmd
│       ├── blue_to_orange_gradient.mmd
│       ├── eigenscribe-theme_gradient.mmd
│       ├── purple_gradient.mmd
│       └── vanilla_architecture.mmd
├── cli/
│   ├── __init__.py
│   ├── cli_train.py                # canonical executable training workflow for generating saved experiment runs
│   └── extract_metrics.py
├── data/
│   ├── reference_runs/
│   │   ├── 20260424_205517_246896/
│   │   │   ├── config.json
│   │   │   ├── diagnostics.npz
│   │   │   ├── ground_truth.pt
│   │   │   ├── history.json
│   │   │   ├── model.pt
│   │   │   ├── pod_metrics.csv
│   │   │   ├── run_id.txt
│   │   │   └── training_analysis.csv
│   │   ├── 20260425_081239_294246/
│   │   │   ├── config.json
│   │   │   ├── diagnostics.npz
│   │   │   ├── ground_truth.pt
│   │   │   ├── history.json
│   │   │   ├── model.pt
│   │   │   ├── pod_metrics.csv
│   │   │   ├── run_id.txt
│   │   │   └── training_analysis.csv
│   │   ├── 20260425_085631_749744/
│   │       ├── config.json
│   │       ├── diagnostics.npz
│   │       ├── ground_truth.pt
│   │       ├── history.json
│   │       ├── model.pt
│   │       └── run_id.txt
│   └── README.md
├── docs/
│   ├── architecture.pdf
│   └── loss function table (extended).pdf
├── notebooks/
│   ├── demo.ipynb                   # narrative figure-generation workflow
│   └── load_run_analysis.ipynb      # validates loading and plotting CLI-generated runs
├── src/
│   ├── __init__.py
│   ├── db_logger.py
│   ├── inverse.py
│   ├── model.py
│   ├── physics.py
│   ├── pod.py
│   ├── train.py                     # canonical training source for training primitives
│   ├── utils.py
│   └── visualizations.py
├── CITATION.cff
├── LICENSE
├── README.md
├── pyproject.toml
├── references.bib
├── repo_summary.py
└── requirements.txt
 
```