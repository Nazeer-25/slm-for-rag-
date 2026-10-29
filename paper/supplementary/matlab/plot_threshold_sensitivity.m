%% plot_threshold_sensitivity.m
% Recall and false-positive rate (FPR) of the Layer 1 detector against the threshold,
% on the validation set, with 95% Wilson intervals, the FPR <= 2% constraint, the current
% threshold 0.175 (dashed vertical line) and the selected threshold tau* (orange ring).
%
% Input : threshold_sensitivity_validation.csv  (written by threshold_sensitivity.py)
% Output: fig_threshold_sensitivity.png (600 dpi) and fig_threshold_sensitivity.pdf
%
% MATLAB Online: upload this file, the four helper files (load_csv_columns.m, new_figure.m,
% style_axes.m, save_figure.m) and the CSV to one folder in MATLAB Drive, open that folder
% and press Run.

DATA_DIR  = '.';      % folder that contains the CSV
OUT_DIR   = '.';      % where the figure is written
CURRENT   = 0.175;
FPR_LIMIT = 0.02;

S = load_csv_columns(fullfile(DATA_DIR, 'threshold_sensitivity_validation.csv'));
col = @(name) S.cols{find(strcmp(S.names, name), 1)};
t   = col('threshold');
rec = 100 * col('recall');  rlo = 100 * col('recall_lo');  rhi = 100 * col('recall_hi');
fpr = 100 * col('fpr');     flo = 100 * col('fpr_lo');     fhi = 100 * col('fpr_hi');
sel = col('selected') == 1;
tau = t(sel);

NAVY = [28 92 171] / 255;  ORANGE = [235 104 52] / 255;  INK = [11 11 11] / 255;
INK2 = [82 81 78] / 255;   GRID = [225 224 217] / 255;
TXT = {'FontName', 'Arial', 'FontSize', 7};

fig = new_figure(3.45, 2.95);
ax1 = axes('Parent', fig, 'Position', [0.16 0.585 0.80 0.375]);
ax2 = axes('Parent', fig, 'Position', [0.16 0.150 0.80 0.375]);
if min(rlo) > 95
    ylim1 = [max(0, min(rlo) - 0.6), 100.4];
else
    ylim1 = [max(0, min(rlo) - 3), 100.5];
end
ylim2 = [0, max(100 * FPR_LIMIT * 1.35, 1.15 * max(fhi))];
panels = {ax1, rec, rlo, rhi, 'Recall (%)', ylim1; ax2, fpr, flo, fhi, 'FPR (%)', ylim2};
for k = 1:2
    ax = panels{k, 1};  v = panels{k, 2};  lo = panels{k, 3};  hi = panels{k, 4};  yl = panels{k, 6};
    hold(ax, 'on');
    set(ax, 'YLim', yl, 'XLim', [min(t) - 0.01, max(t) + 0.01]);
    plot(ax, [CURRENT CURRENT], yl, '--', 'Color', INK2, 'LineWidth', 0.9);
    for i = 1:numel(t)
        plot(ax, [t(i) t(i)], [lo(i) hi(i)], '-', 'Color', NAVY, 'LineWidth', 0.9);
    end
    plot(ax, t, v, '-o', 'Color', NAVY, 'MarkerFaceColor', NAVY, 'MarkerEdgeColor', 'w', ...
        'MarkerSize', 4, 'LineWidth', 1.4);
    if ~isempty(tau)
        plot(ax, tau, v(sel), 'o', 'MarkerSize', 9, 'MarkerEdgeColor', ORANGE, 'LineWidth', 1.5);
    end
    ylabel(ax, panels{k, 5}, TXT{:});
    style_axes(ax);
    set(ax, 'YGrid', 'on', 'GridColor', GRID, 'XTick', t);
end
plot(ax2, [min(t) - 0.01, max(t) + 0.01], 100 * [FPR_LIMIT FPR_LIMIT], '--', 'Color', ORANGE, 'LineWidth', 0.9);
text(max(t), 100 * FPR_LIMIT, 'FPR limit 2%', 'Parent', ax2, 'Color', INK2, ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', TXT{:});
text(CURRENT, ylim1(2), sprintf(' current %g', CURRENT), 'Parent', ax1, 'Color', INK2, ...
    'VerticalAlignment', 'top', TXT{:});
if ~isempty(tau)
    text(tau, ylim2(2) * 0.97, sprintf(' \\tau* = %g', tau), 'Parent', ax2, 'Color', INK, ...
        'VerticalAlignment', 'top', TXT{:});
end
set(ax1, 'XTickLabel', {});
set(ax2, 'XTickLabel', arrayfun(@(x) sprintf('%g', x), t, 'UniformOutput', false));
xlabel(ax2, 'Threshold \tau (validation set; whiskers: 95% Wilson intervals)', TXT{:});
save_figure(fig, OUT_DIR, 'fig_threshold_sensitivity');
fprintf('Selected threshold tau* = %g (current %g)\n', tau, CURRENT);
