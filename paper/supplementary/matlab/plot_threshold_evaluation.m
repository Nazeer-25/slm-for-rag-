%% plot_threshold_evaluation.m
% Recall (top) and false-positive rate (bottom) of the Layer 1 detector on the regular test and
% the challenge set over the whole threshold range (log-odds axis). EVALUATION ONLY: these test
% sets were not used to choose the threshold. Shading and lines as in plot_threshold_sensitivity.m
% (green band: validation-optimal interval; grey strip: requested 0.10-0.25; dashed line: 0.175;
% dotted line: midpoint of the green band). The recall axis is zoomed; curves leave it at the right.
%
% Inputs : threshold_sweep_tests.csv, threshold_final_evaluation.csv, threshold_band.csv
%          (written by threshold_sensitivity.py)
% Output : fig_threshold_evaluation.png (600 dpi) and fig_threshold_evaluation.pdf
%
% MATLAB Online: upload this file, the helpers (load_csv_columns.m, read_band.m, logodds_ticks.m,
% top_legend.m, new_figure.m, style_axes.m, save_figure.m) and the three CSV files to one folder in
% MATLAB Drive, open that folder and press Run.

DATA_DIR = '.';
OUT_DIR  = '.';

B  = read_band(fullfile(DATA_DIR, 'threshold_band.csv'));
SW = load_csv_columns(fullfile(DATA_DIR, 'threshold_sweep_tests.csv'));
FE = load_csv_columns(fullfile(DATA_DIR, 'threshold_final_evaluation.csv'));
col = @(S, name) S.cols{find(strcmp(S.names, name), 1)};
lg  = @(p) log(min(max(p, 1e-12), 1 - 1e-12) ./ (1 - min(max(p, 1e-12), 1 - 1e-12)));
tf  = @(t) regexprep(sprintf('%.3f', t), '(?<=\.\d\d)0$', '');

BLUE = [42 120 214] / 255;   ORANGE = [235 104 52] / 255;  NAVY = [28 92 171] / 255;
INK = [11 11 11] / 255;      INK2 = [82 81 78] / 255;      GRID = [225 224 217] / 255;
BAND = [223 244 236] / 255;  STRIP = [221 221 219] / 255;
TXT = {'FontName', 'Arial', 'FontSize', 7};

xl = [B.axis_min, B.axis_max];
split = col(SW, 'split');  x = col(SW, 'logit');  t = col(SW, 'threshold');
rec = 100 * col(SW, 'recall');  fpr = 100 * col(SW, 'fpr');
fsplit = col(FE, 'split');  freq = col(FE, 'requested') == 1;
fx = lg(col(FE, 'threshold'));  frec = 100 * col(FE, 'recall');
has_band = isfield(B, 'interval_lo');
show_mid = has_band && abs(lg(B.midpoint) - lg(B.current)) > 0.25;
SPLITS = {'regular', 'challenge'};  colours = {BLUE, ORANGE};

low = 100;
for s = 1:2
    low = min([low; rec(strcmp(split, SPLITS{s}) & t <= 0.99)]);
end
bottom = max(0, floor(low - 0.5));

fig = new_figure(3.45, 3.3);
ax1 = axes('Parent', fig, 'Position', [0.15 0.47 0.81 0.37]);
ax2 = axes('Parent', fig, 'Position', [0.15 0.12 0.81 0.28]);
axs = [ax1 ax2];
yls = {[bottom, 100 + (100 - bottom) * 0.06], [-0.15, 100 * B.fpr_limit * 2.5]};
for i = 1:2
    ax = axs(i);  yl = yls{i};
    hold(ax, 'on');
    if has_band
        a = max(xl(1), lg(B.interval_lo));  b = min(xl(2), lg(B.interval_hi));
        patch('Parent', ax, 'XData', [a b b a]', 'YData', [yl(1) yl(1) yl(2) yl(2)]', 'FaceColor', BAND, 'EdgeColor', 'none');
    end
    a = lg(B.requested_min);  b = lg(B.requested_max);
    patch('Parent', ax, 'XData', [a b b a]', 'YData', [yl(1) yl(1) yl(2) yl(2)]', 'FaceColor', STRIP, 'EdgeColor', 'none');
end
plot(ax2, xl, 100 * B.fpr_limit * [1 1], ':', 'Color', INK2, 'LineWidth', 0.8);
text(xl(2), 100 * B.fpr_limit, sprintf('%g%% (validation limit) ', 100 * B.fpr_limit), 'Parent', ax2, ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'Color', INK2, 'FontName', 'Arial', 'FontSize', 6.5);
for s = 1:2
    m = strcmp(split, SPLITS{s});
    plot(ax1, x(m), rec(m), '-', 'Color', colours{s}, 'LineWidth', 1.3);
    plot(ax2, x(m), fpr(m), '-', 'Color', colours{s}, 'LineWidth', 1.3);
    q = strcmp(fsplit, SPLITS{s}) & freq;
    plot(ax1, fx(q), frec(q), 'o', 'MarkerSize', 3, 'MarkerFaceColor', colours{s}, 'MarkerEdgeColor', 'w');
end
for i = 1:2
    plot(axs(i), lg(B.current) * [1 1], yls{i}, '--', 'Color', INK, 'LineWidth', 0.9);
    if show_mid
        plot(axs(i), lg(B.midpoint) * [1 1], yls{i}, ':', 'Color', NAVY, 'LineWidth', 1.0);
    end
end
left = ~show_mid || B.current < B.midpoint;
if left
    al = {'right', 'left'};  fmt = {'%s ', ' %s'};
else
    al = {'left', 'right'};  fmt = {' %s', '%s '};
end
top = yls{1}(2);
text(lg(B.current), top, sprintf(fmt{1}, sprintf('\\tau = %g', B.current)), 'Parent', ax1, ...
    'HorizontalAlignment', al{1}, 'VerticalAlignment', 'bottom', 'Color', INK, TXT{:});
if show_mid
    text(lg(B.midpoint), top, sprintf(fmt{2}, sprintf('midpoint %.3g', B.midpoint)), 'Parent', ax1, ...
        'HorizontalAlignment', al{2}, 'VerticalAlignment', 'bottom', 'Color', NAVY, TXT{:});
end

[tp, tl] = logodds_ticks(xl);
set(ax1, 'XLim', xl, 'YLim', yls{1}, 'XTick', tp, 'XTickLabel', {});
set(ax2, 'XLim', xl, 'YLim', yls{2}, 'XTick', tp, 'XTickLabel', tl);
ylabel(ax1, 'Recall (%)', TXT{:});
ylabel(ax2, 'FPR (%)', TXT{:});
xlabel(ax2, 'Threshold \tau (log-odds axis), evaluation only', TXT{:});
for i = 1:2
    style_axes(axs(i));
    set(axs(i), 'YGrid', 'on', 'GridColor', GRID);
end
top_legend(fig, {'line', BLUE, 'regular test'; 'line', ORANGE, 'challenge set'; ...
    'patch', BAND, 'validation-optimal interval'; ...
    'patch', STRIP, sprintf('requested %s-%s', tf(B.requested_min), tf(B.requested_max))}, [0.10 0.895 0.88 0.095]);
save_figure(fig, OUT_DIR, 'fig_threshold_evaluation');
