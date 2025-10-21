# Use case HC2 bias detection and mitigation
This repository contains Jupyter notebooks that was used to do a toy world analysis of bias detection and mitigation in the context of use case HC2 ECG data. The approach illustrates in simple ways how to use synthetic data to repair bias observed in a small data set with only few variables.

The main analysis is based on the original data that is prepared in `incart.qmd` and `incart_repaired.qmd` files (a Quarto markdown file that is rendered using R). This data set is then (manually) adjusted to account for some oddities in the Aequitas Experimenter (mostly changing the labels of the `age_bin` factor). The analysis report from the Experimenter is stored in `aequitas HC2 original incart demographics.pdf`.

Next, the `hc2-synthetic-data-experimenter.ipynb` prepares the synthetic data component for the Experimenter that is reported in the `aequitas HC2 synthetic incart demographics.pdf`. The analyses in these PDF files are reported in deliverable D7.4.
