%% plot_validation_history.m
% Plots the per-epoch validation history saved by the trainer in
% results/layer1_metrics.json (loss curves and validation rates per epoch)
% and marks the selected epoch.
%
% Steps:
%   1) Set METRICS_JSON to the file's location.
%   2) Run. The script searches the JSON for a list of per-epoch records
%      (field names such as validation_history, history or epochs). If it
%      cannot find one, it prints the top-level fields: put the right name in
%      HISTORY_FIELD and run again.

METRICS_JSON  = fullfile('..', '..', 'results', 'layer1_metrics.json');
OUT_DIR       = fullfile('.', 'figures_matlab');
HISTORY_FIELD = '';    % e.g. 'validation_history' ('' = search automatically)

PALETTE = [42 120 214; 235 104 52; 27 175 122; 237 161 0; 232 123 164; 0 131 0; 98 80 214; 227 73 72] / 255;
INK  = [11 11 11] / 255;
GRID = [225 224 217] / 255;
TXT  = {'FontName', 'Arial', 'FontSize', 7};

J = jsondecode(fileread(METRICS_JSON));
H = [];
where = '';
if ~isempty(HISTORY_FIELD)
    [raw, where] = find_field(J, HISTORY_FIELD);
    H = as_struct_array(raw);
else
    keys = {'validation_history', 'val_history', 'history', 'epoch_history', 'epochs', 'epoch_metrics'};
    for k = 1:numel(keys)
        [raw, where] = find_field(J, keys{k});
        H = as_struct_array(raw);
        if ~isempty(H)
            break
        end
    end
end
if isempty(H)
    fprintf('No per-epoch history found in %s. Top-level fields:\n', METRICS_JSON);
    disp(fieldnames(J));
    return
end
fprintf('Using per-epoch records at "%s" (%d epochs)\n', where, numel(H));

f = fieldnames(H);
epoch = (1:numel(H))';
ei = find(strcmpi(f, 'epoch'), 1);
if ~isempty(ei)
    epoch = [H.(f{ei})]';
end
loss_names = {};
loss_vals = {};
rate_names = {};
rate_vals = {};
for k = 1:numel(f)
    if strcmpi(f{k}, 'epoch')
        continue
    end
    vals = {H.(f{k})};
    if all(cellfun(@(x) isnumeric(x) && isscalar(x), vals))
        v = cell2mat(vals);
        v = v(:);
        if ~isempty(regexp(lower(f{k}), 'threshold|tau', 'once'))
            fprintf('  %s per epoch: %s\n', f{k}, mat2str(v', 4));
        elseif ~isempty(strfind(lower(f{k}), 'loss'))
            loss_names{end + 1} = f{k}; %#ok<SAGROW>
            loss_vals{end + 1} = v; %#ok<SAGROW>
        elseif all(v >= 0 & v <= 1)
            rate_names{end + 1} = f{k}; %#ok<SAGROW>
            rate_vals{end + 1} = v; %#ok<SAGROW>
        end
    end
end

selected = [];
for key = {'selected_epoch', 'best_epoch', 'chosen_epoch'}
    [v, w] = find_field(J, key{1});
    if ~isempty(w) && isnumeric(v) && isscalar(v)
        selected = v;
        break
    end
end

fig = new_figure(6.5, 2.3);
groups = {loss_names, rate_names};
values = {loss_vals, rate_vals};
ylabels = {'Loss', 'Validation rate (%)'};
for p = 1:2
    ax = subplot(1, 2, p, 'Parent', fig);
    hold(ax, 'on');
    if isempty(groups{p})
        text(0.5, 0.5, 'no fields of this kind', 'Parent', ax, 'HorizontalAlignment', 'center', TXT{:});
        axis(ax, 'off');
        continue
    end
    scale = 1;
    if p == 2
        scale = 100;
    end
    h = zeros(1, numel(groups{p}));
    for k = 1:numel(groups{p})
        c = PALETTE(1 + mod(k - 1, size(PALETTE, 1)), :);
        h(k) = plot(ax, epoch, scale * values{p}{k}, '-o', 'Color', c, 'MarkerFaceColor', c, ...
            'MarkerEdgeColor', 'w', 'MarkerSize', 4, 'LineWidth', 1.4);
    end
    if ~isempty(selected)
        yl = get(ax, 'YLim');
        plot(ax, [selected selected], yl, '-', 'Color', INK, 'LineWidth', 0.8);
        text(selected, yl(2), ' selected', 'Parent', ax, 'VerticalAlignment', 'top', TXT{:});
    end
    style_axes(ax);
    set(ax, 'XTick', epoch, 'YGrid', 'on', 'GridColor', GRID);
    xlabel(ax, 'Epoch', TXT{:});
    ylabel(ax, ylabels{p}, TXT{:});
    legend(h, strrep(groups{p}, '_', ' '), 'Location', 'best', 'Box', 'off', TXT{:});
end
save_figure(fig, OUT_DIR, 'figA6_validation_history');
