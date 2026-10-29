%% plot_ablation.m
% Four-configuration ablation: attack success (top) and correct answers on injected (middle)
% and clean items (bottom) for no defense, detector only, sanitizer only and the hybrid, on the
% regular test (blue) and the challenge set (orange). Whiskers: 95% Wilson intervals.
%
% Input : ablation_summary.csv  (written by ablation_four_configs.py)
% Output: fig_ablation.png (600 dpi) and fig_ablation.pdf
%
% MATLAB Online: upload this file, the four helper files and the CSV to one folder in MATLAB
% Drive, open that folder and press Run.

DATA_DIR = '.';
OUT_DIR  = '.';

CONFIGS = {'no_defense', 'detector_only', 'sanitizer_only', 'hybrid'};
XLABELS = {'No defense', 'Detector', 'Sanitizer', 'Hybrid'};   % detector only, sanitizer only, both
SPLITS  = {'regular', 'challenge'};
NAMES   = {'Regular test', 'Challenge set'};
COLOURS = {[42 120 214] / 255, [235 104 52] / 255};
METRICS = {'asr', 'asr', {'Attack success', '(injected items)'}; ...
           'correct_attack_rate', 'correct_attack', {'Correct answers', '(injected items)'}; ...
           'correct_clean_rate', 'correct_clean', {'Correct answers', '(clean items)'}};
INK = [11 11 11] / 255;
GRID = [225 224 217] / 255;
TXT = {'FontName', 'Arial', 'FontSize', 7};

S = load_csv_columns(fullfile(DATA_DIR, 'ablation_summary.csv'));
col = @(name) S.cols{find(strcmp(S.names, name), 1)};
split = col('split');
config = col('config');
present = SPLITS(ismember(SPLITS, split));
width = 0.36;
if numel(present) == 1
    width = 0.5;
end
fig = new_figure(3.45, 3.9);
handles = [];
for m = 1:3
    ax = axes('Parent', fig, 'Position', [0.21, 0.09 + (3 - m) * 0.29, 0.76, 0.215]);
    hold(ax, 'on');
    value = col(METRICS{m, 1});
    lo = col([METRICS{m, 2} '_lo']);
    hi = col([METRICS{m, 2} '_hi']);
    for j = 1:numel(present)
        sj = find(strcmp(SPLITS, present{j}));
        for k = 1:numel(CONFIGS)
            row = find(strcmp(split, present{j}) & strcmp(config, CONFIGS{k}), 1);
            if isempty(row)
                continue
            end
            x = k + (j - (numel(present) + 1) / 2) * width;
            v = 100 * value(row);
            h = patch('Parent', ax, 'XData', (x + width * 0.46 * [-1 1 1 -1])', 'YData', [0 0 v v]', ...
                'FaceColor', COLOURS{sj}, 'EdgeColor', 'none');
            if m == 1 && k == 1
                handles(end + 1) = h; %#ok<SAGROW>
            end
            plot(ax, [x x], 100 * [lo(row) hi(row)], '-', 'Color', INK, 'LineWidth', 0.7);
            text(x, min(100 * hi(row), 100) + 2, sprintf('%.0f', v), 'Parent', ax, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', INK, ...
                'FontName', 'Arial', 'FontSize', 6);
        end
    end
    set(ax, 'XLim', [0.4, numel(CONFIGS) + 0.6], 'YLim', [0 118], 'YTick', [0 50 100], ...
        'YTickLabel', {'0%', '50%', '100%'}, 'XTick', 1:numel(CONFIGS));
    if m == 3
        set(ax, 'XTickLabel', XLABELS);
    else
        set(ax, 'XTickLabel', {});
    end
    ylabel(ax, METRICS{m, 3}, TXT{:});
    style_axes(ax);
    set(ax, 'YGrid', 'on', 'GridColor', GRID);
    if m == 1
        legend(ax, handles, NAMES(ismember(SPLITS, present)), 'Location', 'northoutside', ...
            'Orientation', 'horizontal', 'Box', 'off', TXT{:});
    end
end
save_figure(fig, OUT_DIR, 'fig_ablation');
