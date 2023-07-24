# Comparing and contrastic synthetic Standard Model background samples

This is the companion repository to the paper "The Interplay of Machine Learning--based Resonant Anomaly Detection Methods at [https://arxiv.org/abs/2212.11285](https://arxiv.org/abs/2307.11157) (authors:Tobias Golling, Gregor Kasieczka, Claudius Krause, Radha Mastandrea, Benjamin Nachman, John Andrew Raine, Debajyoti Sengupta, David Shih, Manuel Sommerhalder). All plots in the paper can be remade using these scripts and a dataset* (to be released shortly) of synthetic Standard Model background samples from the [SALAD](https://github.com/bnachman/DCTRHunting), [CATHODE](https://github.com/HEPML-AnomalyDetection/CATHODE), CURTAINs, and [FETA](https://github.com/rmastand/FETA) methods.

For questions/comments about the code contact: rmastand@berkeley.edu

## Pipeline 

### Testing the synthetic samples in anomaly detection tasks

Run the script ```final_eval_and_scatterplot_SSS.ipynb```. This will train a 5-fold binary classifier to discriminate the synthetic samples from a set ``data" (i.e. LHCO olympics background and signal). This script requires a dataset* of synthetic Standard Model background samples.

Once the above script has been run, the output can be passed through the following notebooks:

- Scatterplots: use the notebook `analyze_scatterplot_all_synth.ipynb`.
- Calculating the overlap of shared events between methods: use the notebook `analyze_scatterplot_all_synth.ipynb`.
- Classifier metrics: use the notebook `sample_combination_ensembling.ipynb.ipynb`.

### Comparing the synthetcis samples against each other

Run the script ```bk_comparison.ipynb```. This will train a classifier to discriminate the synthetic samples against each other.
Once script has been run, analyze the output with the notebook `plot_bkg_comparison.ipynb`.

