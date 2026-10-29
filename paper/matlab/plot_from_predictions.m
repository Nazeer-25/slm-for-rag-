%% plot_from_predictions.m
% Recomputes the Layer 1 results from the saved per-chunk predictions and
% draws the figures that need raw scores:
%   A1 score distributions, A2 ROC curves, A3 precision-recall curves,
%   A4 threshold sweep (challenge), A5 recall per withheld family with exact
%   Wilson intervals. It also prints the confusion matrices at THRESHOLD so
%   you can check them against the recorded numbers in the report.
%
% Steps:
%   1) Set RESULTS_DIR to the project's results folder (it contains
%      layer1_predictions.csv and layer1_challenge_predictions.csv).
%   2) Run the script. Columns are detected automatically; if detection
%      fails, the script lists the available columns. Copy the right names
%      into the *_COLUMN settings below and run again.

RESULTS_DIR   = fullfile('..', '..', 'results');
OUT_DIR       = fullfile('.', 'figures_matlab');
THRESHOLD     = 0.175;
LABEL_COLUMN  = '';    % e.g. 'label'           ('' = auto-detect)
SCORE_COLUMN  = '';    % e.g. 'probability'
FAMILY_COLUMN = '';    % e.g. 'attack_family'   (challenge file)

FILES = {fullfile(RESULTS_DIR, 'layer1_predictions.csv'), ...
         fullfile(RESULTS_DIR, 'layer1_challenge_predictions.csv')};
TAGS  = {'Regular test', 'Challenge'};
LABEL_CANDIDATES  = {'label', 'y_true', 'true_label', 'gold', 'target', 'is_injected', 'is_poisoned', ...
                     'injected', 'poisoned', 'class'};
SCORE_CANDIDATES  = {'probability', 'injection_probability', 'p_injection', 'prob_injection', 'score', ...
                     'max_probability', 'prob', 'proba', 'p_inj', 'injection_score'};
FAMILY_CANDIDATES = {'attack_family', 'family', 'attack_type', 'attack', 'technique', 'category'};

BLUE   = [42 120 214] / 255;
ORANGE = [235 104 52] / 255;
INK    = [11 11 11] / 255;
INK2   = [82 81 78] / 255;
GRID   = [225 224 217] / 255;
G_MID  = [201 199 191] / 255;
TXT    = {'FontName', 'Arial', 'FontSize', 7};
COLOURS = {BLUE, ORANGE};

y = cell(1, 2);
s = cell(1, 2);
fam = {};
for i = 1:2
    S = load_csv_columns(FILES{i});
    li = pick_column(S, LABEL_CANDIDATES, LABEL_COLUMN, 'label', 'pred');
    si = pick_column(S, SCORE_CANDIDATES, SCORE_COLUMN, 'score', 'pred_label|predicted_label');
    if isempty(li) || isempty(si)
        error('Set LABEL_COLUMN and SCORE_COLUMN at the top of this script (see the list above).');
    end
    y{i} = to_binary_labels(S.cols{li});
    sc = S.cols{si};
    if iscell(sc)
        sc = str2double(sc);
    end
    s{i} = sc(:);
    if any(isnan(s{i}))
        error('Column "%s" in %s is not numeric.', S.names{si}, FILES{i});
    end
    yhat = s{i} >= THRESHOLD;
    TP = sum(yhat & y{i} == 1);  FN = sum(~yhat & y{i} == 1);
    FP = sum(yhat & y{i} == 0);  TN = sum(~yhat & y{i} == 0);
    fprintf('\n%s (%s): label=%s, score=%s\n', TAGS{i}, FILES{i}, S.names{li}, S.names{si});
    fprintf('  TN %d  FP %d  FN %d  TP %d  |  accuracy %.2f%%  recall %.2f%%  FPR %.3f%%\n', ...
        TN, FP, FN, TP, 100 * (TP + TN) / numel(yhat), 100 * TP / (TP + FN), 100 * FP / (FP + TN));
    if i == 2
        fi = pick_column(S, FAMILY_CANDIDATES, FAMILY_COLUMN, 'attack-family');
        if ~isempty(fi)
            fam = S.cols{fi};
            if ~iscell(fam)
                fam = cellstr(num2str(fam(:)));
            end
            fam = strtrim(fam(:));
        end
    end
end

%% A1 - score distributions (log-scaled counts)
fig = new_figure(3.45, 2.0);
edges = linspace(0, 1, 41);
panel_left = [0.15 0.60];
for i = 1:2
    ax = axes('Parent', fig, 'Position', [panel_left(i) 0.22 0.35 0.64]);
    hold(ax, 'on');
    for cls = 0:1
        v = s{i}(y{i} == cls);
        b = min(numel(edges) - 1, max(1, floor(v * (numel(edges) - 1)) + 1));
        counts = accumarray(b(:), 1, [numel(edges) - 1, 1]);
        if cls == 0
            c = G_MID * 0.75;
        else
            c = COLOURS{i};
        end
        stairs(ax, edges, [counts; counts(end)] + 0.5, '-', 'Color', c, 'LineWidth', 1.1);
    end
    plot(ax, [THRESHOLD THRESHOLD], [0.5 2e4], '-', 'Color', INK, 'LineWidth', 0.8);
    set(ax, 'YScale', 'log', 'XLim', [0 1], 'YLim', [0.5 2e4], 'YTick', [1 10 100 1000 10000], ...
        'YTickLabel', {'1', '10', '100', '1k', '10k'});
    style_axes(ax);
    title(ax, TAGS{i}, 'FontWeight', 'normal', 'Color', INK, TXT{:});
    xlabel(ax, 'Injection score s(c)', TXT{:});
    if i == 1
        ylabel(ax, 'Chunks', TXT{:});
        text(THRESHOLD + 0.03, 6e3, '\tau', 'Parent', ax, 'Color', INK, TXT{:});
    end
end
save_figure(fig, OUT_DIR, 'figA1_score_distributions');

%% A2/A3 - ROC and precision-recall curves
figR = new_figure(3.45, 2.2);
axR = axes('Parent', figR);
hold(axR, 'on');
figP = new_figure(3.45, 2.2);
axP = axes('Parent', figP);
hold(axP, 'on');
hR = zeros(1, 2);
hP = zeros(1, 2);
for i = 1:2
    [fpr, tpr, thr, auc, prec, rec, ap] = roc_curve_points(y{i}, s{i});
    hR(i) = plot(axR, fpr, tpr, '-', 'Color', COLOURS{i}, 'LineWidth', 1.4);
    k = find(thr >= THRESHOLD, 1, 'last');
    plot(axR, fpr(k), tpr(k), 'o', 'MarkerFaceColor', COLOURS{i}, 'MarkerEdgeColor', 'w', 'MarkerSize', 5);
    hP(i) = plot(axP, rec, prec, '-', 'Color', COLOURS{i}, 'LineWidth', 1.4);
    namesR{i} = sprintf('%s (AUC %.4f)', TAGS{i}, auc); %#ok<SAGROW>
    namesP{i} = sprintf('%s (AP %.4f)', TAGS{i}, ap); %#ok<SAGROW>
    fprintf('%s: ROC AUC %.4f, average precision %.4f\n', TAGS{i}, auc, ap);
end
set(axR, 'XLim', [0 1], 'YLim', [0 1.01]);
style_axes(axR);
set(axR, 'XGrid', 'on', 'YGrid', 'on', 'GridColor', GRID);
xlabel(axR, 'False-positive rate', TXT{:});
ylabel(axR, 'True-positive rate', TXT{:});
legend(hR, namesR, 'Location', 'southeast', 'Box', 'off', TXT{:});
save_figure(figR, OUT_DIR, 'figA2_roc');
set(axP, 'XLim', [0 1], 'YLim', [0 1.01]);
style_axes(axP);
set(axP, 'XGrid', 'on', 'YGrid', 'on', 'GridColor', GRID);
xlabel(axP, 'Recall', TXT{:});
ylabel(axP, 'Precision', TXT{:});
legend(hP, namesP, 'Location', 'southwest', 'Box', 'off', TXT{:});
save_figure(figP, OUT_DIR, 'figA3_precision_recall');

%% A4 - threshold sweep on the challenge set
t = linspace(0, 1, 201);
tpr_t = zeros(size(t));
fpr_t = zeros(size(t));
for k = 1:numel(t)
    yhat = s{2} >= t(k);
    tpr_t(k) = sum(yhat & y{2} == 1) / sum(y{2} == 1);
    fpr_t(k) = sum(yhat & y{2} == 0) / sum(y{2} == 0);
end
fig = new_figure(3.45, 1.9);
ax = axes('Parent', fig);
hold(ax, 'on');
h1 = plot(ax, t, 100 * tpr_t, '-', 'Color', ORANGE, 'LineWidth', 1.4);
h2 = plot(ax, t, 100 * fpr_t, '-', 'Color', INK2, 'LineWidth', 1.4);
plot(ax, [THRESHOLD THRESHOLD], [0 100], '-', 'Color', INK, 'LineWidth', 0.8);
set(ax, 'XLim', [0 1], 'YLim', [0 100]);
style_axes(ax);
set(ax, 'YGrid', 'on', 'GridColor', GRID);
xlabel(ax, 'Threshold \tau', TXT{:});
ylabel(ax, 'Rate (%)', TXT{:});
legend([h1 h2], {'Recall (TPR)', 'False-positive rate'}, 'Location', 'east', 'Box', 'off', TXT{:});
save_figure(fig, OUT_DIR, 'figA4_threshold_sweep');

%% A5 - recall per withheld family with exact Wilson intervals
if isempty(fam)
    fprintf('No family column in the challenge file; skipping figure A5.\n');
else
    inj = (y{2} == 1);
    [names, ~, g] = unique(fam(inj));
    n = accumarray(g, 1);
    hit = accumarray(g, double(s{2}(inj) >= THRESHOLD));
    [lo, hi] = wilson_ci(hit, n);
    rec = hit ./ n;
    [~, order] = sortrows([rec, (1:numel(rec))']);
    fprintf('\n%-28s %6s %6s %8s   95%% CI\n', 'Family', 'n', 'missed', 'recall');
    for r = 1:numel(order)
        j = order(r);
        fprintf('%-28s %6d %6d %7.2f%%   [%.2f, %.2f]\n', names{j}, n(j), n(j) - hit(j), 100 * rec(j), ...
            100 * lo(j), 100 * hi(j));
    end
    nf = numel(order);
    fig = new_figure(3.45, 0.35 + 0.22 * nf);
    ax = axes('Parent', fig, 'Position', [0.36 0.18 0.60 0.78]);
    hold(ax, 'on');
    for r = 1:nf
        j = order(r);
        yy = nf + 1 - r;
        if hit(j) < n(j)
            c = ORANGE;
        else
            c = G_MID;
        end
        rectangle('Parent', ax, 'Position', [0, yy - 0.31, 100 * rec(j), 0.62], 'FaceColor', c, 'EdgeColor', 'none');
        plot(ax, 100 * [lo(j) hi(j)], [yy yy], '-', 'Color', INK, 'LineWidth', 0.8);
        text(2, yy, sprintf('%d missed of %d', n(j) - hit(j), n(j)), 'Parent', ax, 'Color', INK, TXT{:});
    end
    set(ax, 'YTick', 1:nf, 'YTickLabel', flipud(names(order)), 'XLim', [0 100], 'YLim', [0.4 nf + 0.6], ...
        'XTick', 0:25:100, 'XTickLabel', {'0%', '25%', '50%', '75%', '100%'});
    style_axes(ax);
    set(ax, 'XGrid', 'on', 'GridColor', GRID, 'Layer', 'top');
    xlabel(ax, 'Recall (bars) with 95% Wilson interval (lines)', TXT{:});
    save_figure(fig, OUT_DIR, 'figA5_family_recall_exact');
end
fprintf('\nDone. Figures are in %s\n', OUT_DIR);
