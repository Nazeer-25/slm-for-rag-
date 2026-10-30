%% plot_threshold_sensitivity.m
% Validation recall and false-positive rate (FPR) of the Layer 1 detector over the whole threshold
% range on a log-odds axis (top), with the validation score histogram on the same axis (bottom).
%   green band   validation-optimal interval: every threshold in it has the highest validation
%                recall under FPR <= 2% and the lowest FPR (no validation score lies inside it)
%   grey strip   the requested thresholds 0.10-0.25 (dots: the seven requested values)
%   dashed line  the current threshold 0.175;  dotted line: the midpoint of the green band
% Every threshold inside the green band makes the same decision on every validation chunk, which
% is why recall and FPR are flat between 0.10 and 0.25; the curves bend only outside the band.
%
% Inputs : threshold_sweep_validation.csv, threshold_sensitivity_validation.csv,
%          threshold_band.csv, validation_scores.csv      (written by threshold_sensitivity.py)
% Output : fig_threshold_sensitivity.png (600 dpi) and fig_threshold_sensitivity.pdf
%
% MATLAB Online: upload this file, the helpers (load_csv_columns.m, read_band.m, logodds_ticks.m,
% top_legend.m, new_figure.m, style_axes.m, save_figure.m) and the four CSV files to one folder in
% MATLAB Drive, open that folder and press Run.

DATA_DIR = '.';      % folder that contains the CSV files
OUT_DIR  = '.';      % where the figure is written

B  = read_band(fullfile(DATA_DIR, 'threshold_band.csv'));
SW = load_csv_columns(fullfile(DATA_DIR, 'threshold_sweep_validation.csv'));
GR = load_csv_columns(fullfile(DATA_DIR, 'threshold_sensitivity_validation.csv'));
SC = load_csv_columns(fullfile(DATA_DIR, 'validation_scores.csv'));
col = @(S, name) S.cols{find(strcmp(S.names, name), 1)};
lg  = @(p) log(min(max(p, 1e-12), 1 - 1e-12) ./ (1 - min(max(p, 1e-12), 1 - 1e-12)));
tf  = @(t) regexprep(sprintf('%.3f', t), '(?<=\.\d\d)0$', '');      % 0.1 -> 0.10, 0.125 -> 0.125

BLUE = [42 120 214] / 255;   ORANGE = [235 104 52] / 255;  NAVY = [28 92 171] / 255;
INK = [11 11 11] / 255;      INK2 = [82 81 78] / 255;      GRID = [225 224 217] / 255;
BAND = [223 244 236] / 255;  STRIP = [221 221 219] / 255;  % aqua at 14% and grey at 30% on white
TXT = {'FontName', 'Arial', 'FontSize', 7};

xl = [B.axis_min, B.axis_max];
x = col(SW, 'logit');  rec = 100 * col(SW, 'recall');  fpr = 100 * col(SW, 'fpr');
req = col(GR, 'requested') == 1;
gx = lg(col(GR, 'threshold'));  grec = 100 * col(GR, 'recall');  gfpr = 100 * col(GR, 'fpr');
has_band = isfield(B, 'interval_lo');
show_mid = has_band && abs(lg(B.midpoint) - lg(B.current)) > 0.25;

% histogram of the validation scores in bins of 0.5 log-odds
W = 0.5;
nb = ceil((xl(2) - xl(1)) / W);
lab = col(SC, 'label');
v = min(max(lg(col(SC, 'score')), xl(1)), xl(2) - 1e-9);
k = min(nb, floor((v - xl(1)) / W) + 1);
counts = [accumarray(k(lab == 0), 1, [nb 1]), accumarray(k(lab == 1), 1, [nb 1])];

fig = new_figure(3.45, 3.9);
ax1 = axes('Parent', fig, 'Position', [0.15 0.47 0.81 0.39]);
ax2 = axes('Parent', fig, 'Position', [0.15 0.10 0.81 0.31]);
axs = [ax1 ax2];
yls = {[-4 104], [0.7, max(counts(:)) * 6]};
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

% top: recall and FPR against the threshold
plot(ax1, xl, 100 * B.fpr_limit * [1 1], ':', 'Color', INK2, 'LineWidth', 0.8);
plot(ax1, x, rec, '-', 'Color', ORANGE, 'LineWidth', 1.4);
plot(ax1, x, fpr, '-', 'Color', BLUE, 'LineWidth', 1.4);
plot(ax1, gx(req), grec(req), 'o', 'MarkerSize', 3, 'MarkerFaceColor', ORANGE, 'MarkerEdgeColor', 'w');
plot(ax1, gx(req), gfpr(req), 'o', 'MarkerSize', 3, 'MarkerFaceColor', BLUE, 'MarkerEdgeColor', 'w');
text(xl(2), 100 * B.fpr_limit + 1.5, sprintf('FPR limit %g%% ', 100 * B.fpr_limit), 'Parent', ax1, ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'Color', INK2, 'FontName', 'Arial', 'FontSize', 6.5);

% bottom: validation scores per class
colours = {BLUE, ORANGE};  names = {'benign', 'injected'};
for c = 1:2
    for i = find(counts(:, c)' > 0)
        patch('Parent', ax2, 'XData', (xl(1) + (i - 1) * W + [0 W W 0])', ...
            'YData', [0.7 0.7 counts(i, c) counts(i, c)]', 'FaceColor', colours{c}, 'EdgeColor', 'w', 'LineWidth', 0.25);
    end
    [peak, i] = max(counts(:, c));
    text(min(max(xl(1) + (i - 0.5) * W, xl(1) + 2), xl(2) - 2), peak * 1.7, names{c}, 'Parent', ax2, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', colours{c}, TXT{:});
end

% threshold lines and their labels (the left label is right-aligned and vice versa)
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
text(lg(B.current), 104, sprintf(fmt{1}, sprintf('\\tau = %g', B.current)), 'Parent', ax1, ...
    'HorizontalAlignment', al{1}, 'VerticalAlignment', 'bottom', 'Color', INK, TXT{:});
if show_mid
    text(lg(B.midpoint), 104, sprintf(fmt{2}, sprintf('midpoint %.3g', B.midpoint)), 'Parent', ax1, ...
        'HorizontalAlignment', al{2}, 'VerticalAlignment', 'bottom', 'Color', NAVY, TXT{:});
end

[tp, tl] = logodds_ticks(xl);
set(ax1, 'XLim', xl, 'YLim', yls{1}, 'YTick', 0:25:100, 'XTick', tp, 'XTickLabel', {});
set(ax2, 'XLim', xl, 'YLim', yls{2}, 'YScale', 'log', 'XTick', tp, 'XTickLabel', tl);
ylabel(ax1, 'Validation chunks (%)', TXT{:});
ylabel(ax2, 'Chunks per bin', TXT{:});
xlabel(ax2, 'Threshold \tau, or detector score (log-odds axis)', TXT{:});
for i = 1:2
    style_axes(axs(i));
    set(axs(i), 'YGrid', 'on', 'GridColor', GRID);
end
top_legend(fig, {'line', ORANGE, 'recall (injected chunks flagged)'; 'line', BLUE, 'FPR (benign chunks flagged)'; ...
    'patch', BAND, 'validation-optimal interval'; ...
    'patch', STRIP, sprintf('requested %s-%s', tf(B.requested_min), tf(B.requested_max))}, [0.10 0.905 0.88 0.085]);
save_figure(fig, OUT_DIR, 'fig_threshold_sensitivity');
if has_band
    fprintf('Validation-optimal interval (%.6g, %.6g], midpoint %.4g; selected among the requested: %g\n', ...
        B.interval_lo, B.interval_hi, B.midpoint, B.selected);
end
