%% make_report_figures.m
% Rebuilds Figures 2-5 of the report from the recorded Layer 1 results
% (run of 29 Oct 2026). No data files or toolboxes are needed.
%
% Steps: open this folder in MATLAB, then press Run (or type
% make_report_figures in the Command Window). PNG (600 dpi) and PDF files
% are written to ./figures_matlab. Figure 1 (the architecture diagram) is a
% drawing, not a chart; reuse the supplied PNG or redraw it in a diagram tool.

OUT_DIR = fullfile('.', 'figures_matlab');

% ---- recorded results ----------------------------------------------------
PARTITION_NAMES = {'Training pool', 'Validation', 'Regular test', 'Challenge'};
PARTITION_SIZE  = [70000 10000 10000 10000];   % chunks, each 50% benign
TRAIN_SAMPLE    = 6000;                        % 3,000 benign + 3,000 injected
THRESHOLD       = 0.175;

CM_REGULAR   = [5000 0; 0 5000];      % rows: actual benign/injected; cols: predicted
CM_CHALLENGE = [5000 0; 191 4809];

FAMILY_NAMES  = {'Multilingual instructions', 'Dependency traps', 'Unicode spoofing', ...
                 'Encoded instructions', 'Few-shot imitation', 'Indirect completion', ...
                 'Polite requests', 'Reasoning lures'};
FAMILY_MISSED = [148 37 6 0 0 0 0 0];
FAMILY_RECALL = [0.7636 0.9391 0.9905 1 1 1 1 1];

% ---- palette (validated for colour-vision deficiency on white) -----------
BLUE   = [42 120 214] / 255;    % regular test
ORANGE = [235 104 52] / 255;    % challenge
INK    = [11 11 11] / 255;
INK2   = [82 81 78] / 255;
GRID   = [225 224 217] / 255;
G_LIGHT = [230 229 223] / 255;
G_MID   = [201 199 191] / 255;
G_DARK  = [143 141 134] / 255;
SEQ = [255 255 255; 205 226 251; 158 197 244; 109 167 236; 57 135 229; 37 106 191; 24 79 149; 13 54 107] / 255;
TXT = {'FontName', 'Arial', 'FontSize', 7};

%% Figure 2 - benchmark partitions
fig = new_figure(3.45, 1.35);
ax = axes('Parent', fig, 'Position', [0.25 0.30 0.72 0.66]);
hold(ax, 'on');
colours = {G_LIGHT, G_MID, BLUE, ORANGE};
notes = {'', '24 seen families; selects epoch and \tau', '24 seen families; 500 new divisions', ...
         '8 withheld families; 500 new divisions'};
for i = 1:4
    y = 5 - i;
    rectangle('Parent', ax, 'Position', [0, y - 0.29, PARTITION_SIZE(i) / 1000, 0.58], ...
        'FaceColor', colours{i}, 'EdgeColor', 'none');
    if i == 1
        rectangle('Parent', ax, 'Position', [0, y - 0.29, TRAIN_SAMPLE / 1000, 0.58], ...
            'FaceColor', G_DARK, 'EdgeColor', 'none');
        text(3, y, '6k', 'Parent', ax, 'Color', 'w', 'HorizontalAlignment', 'center', TXT{:});
        text(38, y, '70,000 chunks; 24 seen families', 'Parent', ax, 'Color', INK, ...
            'HorizontalAlignment', 'center', TXT{:});
    else
        if i >= 3
            label_colour = 'w';
        else
            label_colour = INK;
        end
        text(5, y, '10k', 'Parent', ax, 'Color', label_colour, 'HorizontalAlignment', 'center', TXT{:});
        text(11.2, y, notes{i}, 'Parent', ax, 'Color', INK2, TXT{:});
    end
end
set(ax, 'YTick', 1:4, 'YTickLabel', fliplr(PARTITION_NAMES), 'XLim', [0 70], 'YLim', [0.45 4.45], ...
    'XTick', 0:10:70, 'XTickLabel', {'0', '10k', '20k', '30k', '40k', '50k', '60k', '70k'});
style_axes(ax);
set(ax, 'XGrid', 'on', 'GridColor', GRID, 'Layer', 'bottom');
xlabel(ax, 'Chunks (each partition 50% benign / 50% poisoned)', 'Color', INK2, TXT{:});
save_figure(fig, OUT_DIR, 'fig2_partitions');

%% Figure 3 - confusion matrices
fig = new_figure(3.45, 1.70);
mats = {CM_REGULAR, CM_CHALLENGE};
titles = {'(a) Regular test', '(b) Challenge'};
for p = 1:2
    ax = subplot(1, 2, p, 'Parent', fig);
    hold(ax, 'on');
    M = mats{p};
    for i = 1:2
        for j = 1:2
            v = M(i, j);
            share = v / sum(M(i, :));
            if v == 0
                idx = 1;
            else
                idx = max(2, min(8, round(share * 7) + 1));
            end
            if idx >= 5
                tc = 'w';
            else
                tc = INK;
            end
            edge = 'none';
            if idx == 1
                edge = GRID;
            end
            rectangle('Parent', ax, 'Position', [j - 1 + 0.02, 2 - i + 0.02, 0.96, 0.96], ...
                'FaceColor', SEQ(idx, :), 'EdgeColor', edge);
            text(j - 0.5, 2 - i + 0.62, sprintf('%s', commas(v)), 'Parent', ax, 'Color', tc, ...
                'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'FontName', 'Arial', 'FontSize', 8);
            text(j - 0.5, 2 - i + 0.33, sprintf('%.1f%%', 100 * share), 'Parent', ax, 'Color', tc, ...
                'HorizontalAlignment', 'center', TXT{:});
        end
    end
    axis(ax, 'equal');
    set(ax, 'XLim', [0 2], 'YLim', [0 2], 'XTick', [0.5 1.5], 'XTickLabel', {'Benign', 'Injected'}, ...
        'YTick', [0.5 1.5], 'YTickLabel', {'Injected', 'Benign'}, 'TickLength', [0 0]);
    style_axes(ax);
    set(ax, 'XColor', INK2, 'YColor', INK2);
    title(ax, titles{p}, 'FontWeight', 'normal', 'Color', INK, TXT{:});
    xlabel(ax, 'Predicted', TXT{:});
    if p == 1
        ylabel(ax, 'Actual', TXT{:});
    else
        set(ax, 'YTickLabel', {'', ''});
    end
end
save_figure(fig, OUT_DIR, 'fig3_confusion');

%% Figure 4 - headline metrics with 95% Wilson intervals
metric_names = {'Accuracy', 'Precision', 'Recall (TPR)', 'Specificity (1-FPR)'};
fig = new_figure(3.45, 1.65);
ax = axes('Parent', fig, 'Position', [0.30 0.22 0.62 0.62]);
hold(ax, 'on');
series = {CM_REGULAR, CM_CHALLENGE};
colours = {BLUE, ORANGE};
markers = {'o', 's'};
offsets = [0.14 -0.14];
handles = zeros(1, 2);
for k = 1:2
    M = series{k};
    TN = M(1, 1); FP = M(1, 2); FN = M(2, 1); TP = M(2, 2);
    kk = [TP + TN, TP, TP, TN];
    nn = [sum(M(:)), TP + FP, TP + FN, TN + FP];
    [lo, hi] = wilson_ci(kk, nn);
    v = kk ./ nn;
    for m = 1:4
        y = 5 - m + offsets(k);
        plot(ax, 100 * [lo(m) hi(m)], [y y], '-', 'Color', colours{k}, 'LineWidth', 1.4);
        h = plot(ax, 100 * v(m), y, markers{k}, 'MarkerFaceColor', colours{k}, ...
            'MarkerEdgeColor', 'w', 'MarkerSize', 5, 'LineWidth', 0.8);
        if m == 1
            handles(k) = h;
        end
        if k == 2 && v(m) < 0.995
            text(100 * hi(m) + 0.08, y, sprintf('%.2f%%', 100 * v(m)), 'Parent', ax, 'Color', INK, TXT{:});
        end
    end
end
set(ax, 'YTick', 1:4, 'YTickLabel', fliplr(metric_names), 'XLim', [95 100.25], 'YLim', [0.45 4.55], ...
    'XTick', 95:100, 'XTickLabel', {'95%', '96%', '97%', '98%', '99%', '100%'});
style_axes(ax);
set(ax, 'XGrid', 'on', 'GridColor', GRID);
legend(handles, {'Regular test (seen families)', 'Challenge (withheld families)'}, ...
    'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', TXT{:});
save_figure(fig, OUT_DIR, 'fig4_metrics_ci');

%% Figure 5 - recall per withheld family
[~, order] = sortrows([FAMILY_RECALL(:), (1:numel(FAMILY_RECALL))']);
fig = new_figure(3.45, 2.05);
ax = axes('Parent', fig, 'Position', [0.36 0.18 0.61 0.79]);
hold(ax, 'on');
nf = numel(order);
for r = 1:nf
    i = order(r);
    y = nf + 1 - r;
    if FAMILY_MISSED(i) > 0
        c = ORANGE;
        label = sprintf('%.2f%%  (%d missed)', 100 * FAMILY_RECALL(i), FAMILY_MISSED(i));
        tc = 'w';
    else
        c = G_MID;
        label = '100%  (0 missed)';
        tc = INK;
    end
    rectangle('Parent', ax, 'Position', [0, y - 0.31, 100 * FAMILY_RECALL(i), 0.62], ...
        'FaceColor', c, 'EdgeColor', 'none');
    text(100 * FAMILY_RECALL(i) - 1.5, y, label, 'Parent', ax, 'Color', tc, ...
        'HorizontalAlignment', 'right', TXT{:});
end
set(ax, 'YTick', 1:nf, 'YTickLabel', fliplr(FAMILY_NAMES(order)), 'XLim', [0 100], 'YLim', [0.4 nf + 0.6], ...
    'XTick', 0:25:100, 'XTickLabel', {'0%', '25%', '50%', '75%', '100%'});
style_axes(ax);
set(ax, 'XGrid', 'on', 'GridColor', GRID, 'Layer', 'top');
xlabel(ax, sprintf('Recall on the challenge partition (threshold %.3f)', THRESHOLD), TXT{:});
save_figure(fig, OUT_DIR, 'fig5_family_recall');

fprintf('Done. Figures are in %s\n', OUT_DIR);
