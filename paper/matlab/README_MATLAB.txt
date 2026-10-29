MATLAB figure scripts for the Layer 1 detector report
=====================================================

Requirements: MATLAB R2018b or newer (no toolboxes). The scripts also run in
GNU Octave 8 or newer. exportgraphics is used on MATLAB R2020a+; older versions
fall back to print automatically.

Files
  make_report_figures.m      Figures 2-5 of the report, from the recorded numbers.
                             Needs no data files.
  plot_from_predictions.m    Figures A1-A5 from the per-chunk predictions:
                             score distributions, ROC, precision-recall,
                             threshold sweep, per-family recall with exact
                             Wilson intervals. Also re-prints the confusion
                             matrices at threshold 0.175 as a check.
  plot_validation_history.m  Figure A6: per-epoch validation curves from
                             results/layer1_metrics.json.
  helpers                    wilson_ci, load_csv_columns, pick_column,
                             to_binary_labels, roc_curve_points, new_figure,
                             style_axes, save_figure, commas, find_field,
                             as_struct_array.

Step by step
  1. Copy this "matlab" folder into the project, for example
     <project>\paper\matlab, so that the results folder is ..\..\results.
     Otherwise edit RESULTS_DIR / METRICS_JSON at the top of each script.
  2. Start MATLAB and change the Current Folder to this folder
     (or run: cd('<project>\paper\matlab')).
  3. Type   make_report_figures        -> figures_matlab\fig2..fig5 (.png, .pdf)
  4. Type   plot_from_predictions      -> figures_matlab\figA1..figA5
     The Command Window prints the confusion matrix at tau = 0.175 for both
     test sets. They must equal the recorded ones:
        Regular  TN 5000  FP 0  FN 0    TP 5000
        Challenge TN 5000 FP 0  FN 191  TP 4809
     It also prints the per-family table (n, missed, recall, 95% CI).
     If a column is not detected, the script lists the available columns.
     Put the right names in LABEL_COLUMN / SCORE_COLUMN / FAMILY_COLUMN.
  5. Type   plot_validation_history    -> figures_matlab\figA6_validation_history
     If the history is not found, set HISTORY_FIELD to the field name that
     the script prints.
  6. Insert the PNG files into Word (Insert > Pictures) at 3.5 in width for a
     two-column IEEE paper, or 4.5-6 in for a single-column report. Use the
     PDF versions in LaTeX (\includegraphics[width=\columnwidth]{...}).

Style notes
  Colours: blue = regular test (seen families), orange = challenge (withheld
  families), grey = benign/other. The pair is safe for colour-blind readers.
  Keep the 95% x-axis note on Figure 4: the axis does not start at zero.
