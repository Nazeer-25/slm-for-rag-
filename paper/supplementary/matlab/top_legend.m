function top_legend(fig, items, box)
%TOP_LEGEND  Legend in its own axes above the panels (two columns), so it never covers data.
%   items: n-by-3 cell {style, colour, label}, style 'line' or 'patch'; box: [x y w h] (normalized).
ax = axes('Parent', fig, 'Position', box, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1]);
hold(ax, 'on');
n = size(items, 1);
rows = ceil(n / 2);
for i = 1:n
    c = floor((i - 1) / rows);
    r = mod(i - 1, rows);
    x0 = 0.02 + 0.5 * c;
    y0 = 1 - (r + 0.5) / rows;
    if strcmp(items{i, 1}, 'line')
        plot(ax, x0 + [0 0.07], [y0 y0], '-', 'Color', items{i, 2}, 'LineWidth', 1.4);
    else
        patch('Parent', ax, 'XData', x0 + [0 0.07 0.07 0]', 'YData', y0 + [-0.16 -0.16 0.16 0.16]', ...
            'FaceColor', items{i, 2}, 'EdgeColor', 'none');
    end
    text(x0 + 0.09, y0, items{i, 3}, 'Parent', ax, 'FontName', 'Arial', 'FontSize', 6.5, ...
        'VerticalAlignment', 'middle', 'Color', [11 11 11] / 255);
end
end
