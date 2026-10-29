function save_figure(fig, outdir, name)
%SAVE_FIGURE  Write a figure as a 600-dpi PNG and a vector PDF.
%   Uses exportgraphics (MATLAB R2020a+) when available, otherwise print.
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
png = fullfile(outdir, [name '.png']);
pdf = fullfile(outdir, [name '.pdf']);
if exist('exportgraphics') > 0 %#ok<EXIST>
    exportgraphics(fig, png, 'Resolution', 600);
    exportgraphics(fig, pdf, 'ContentType', 'vector');
else
    print(fig, png, '-dpng', '-r600');
    print(fig, pdf, '-dpdf');
end
fprintf('saved %s\n', png);
end
