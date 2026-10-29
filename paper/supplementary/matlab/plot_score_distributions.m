%% plot_score_distributions.m
% Detector scores of benign and injected chunks as two separate distributions for the regular
% test (left) and the challenge set (right), with the threshold 0.175 as a dashed vertical line.
% Counts are log-scaled so that the few chunks near the threshold remain visible.
%
% Input : scores_regular.csv, scores_challenge.csv  (columns label, score, family; written by
%         score_distributions.py from the saved predictions)
% Output: fig_score_distributions.png (600 dpi) and fig_score_distributions.pdf
%
% MATLAB Online: upload this file, the four helper files and both CSV files to one folder in
% MATLAB Drive, open that folder and press Run.

DATA_DIR  = '.';
OUT_DIR   = '.';
THRESHOLD = 0.175;
BIN_WIDTH = 0.025;          % bin edges align with 0.175

FILES  = {'scores_regular.csv', 'scores_challenge.csv'};
TITLES = {'Regular test', 'Challenge set'};
ROWS   = {'Benign chunks', 'Injected chunks'};
COLOURS = {[42 120 214] / 255, [235 104 52] / 255};      % blue = benign, orange = injected
INK = [11 11 11] / 255;
GRID = [225 224 217] / 255;
TXT = {'FontName', 'Arial', 'FontSize', 7};

edges = 0:BIN_WIDTH:1;
nb = numel(edges) - 1;
pos = [0.15 0.56 0.39 0.36; 0.585 0.56 0.39 0.36; 0.15 0.14 0.39 0.36; 0.585 0.14 0.39 0.36];
fig = new_figure(3.45, 2.9);
axs = zeros(1, 4);
peak = 1;
for c = 1:2
    S = load_csv_columns(fullfile(DATA_DIR, FILES{c}));
    lab = S.cols{find(strcmp(S.names, 'label'), 1)};
    sc = S.cols{find(strcmp(S.names, 'score'), 1)};
    for r = 1:2
        v = sc(lab == r - 1);
        b = min(nb, max(1, floor(v / BIN_WIDTH + 1e-9) + 1));
        counts = accumarray(b(:), 1, [nb 1]);
        peak = max(peak, max(counts));
        k = (r - 1) * 2 + c;
        ax = axes('Parent', fig, 'Position', pos(k, :));
        axs(k) = ax;
        hold(ax, 'on');
        for i = find(counts' > 0)
            patch('Parent', ax, 'XData', (edges(i) + [0 BIN_WIDTH BIN_WIDTH 0])', ...
                'YData', [0.7 0.7 counts(i) counts(i)]', 'FaceColor', COLOURS{r}, 'EdgeColor', 'w', 'LineWidth', 0.3);
        end
        if r == 1
            note = sprintf('%s of %s >= \\tau', commas(sum(v >= THRESHOLD)), commas(numel(v)));
        else
            note = sprintf('%s of %s < \\tau', commas(sum(v < THRESHOLD)), commas(numel(v)));
        end
        text(0.97, 0.93, note, 'Parent', ax, 'Units', 'normalized', 'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'top', 'Color', INK, TXT{:});
        if r == 1
            title(ax, TITLES{c}, 'FontWeight', 'normal', 'Color', INK, TXT{:});
        end
        if c == 1
            ylabel(ax, ROWS{r}, TXT{:});
        end
        if r == 2
            xlabel(ax, 'Detector score s(c)', TXT{:});
        end
    end
end
for k = 1:4
    ax = axs(k);
    set(ax, 'YScale', 'log', 'YLim', [0.7, peak * 2.5], 'XLim', [0 1], ...
        'XTick', [0 THRESHOLD 0.5 1], 'XTickLabel', {'0', '\tau', '0.5', '1'});
    plot(ax, [THRESHOLD THRESHOLD], [0.7, peak * 2.5], '--', 'Color', INK, 'LineWidth', 0.9);
    style_axes(ax);
    set(ax, 'YGrid', 'on', 'GridColor', GRID);
    if k <= 2
        set(ax, 'XTickLabel', {});          % tick labels only under the bottom row
    end
    if mod(k, 2) == 0
        set(ax, 'YTickLabel', {});          % shared y axis: labels only on the left column
    end
end
save_figure(fig, OUT_DIR, 'fig_score_distributions');
