function B = read_band(file)
%READ_BAND  The name,value pairs of threshold_band.csv as a struct (an empty value becomes NaN).
%   Fields: current, selected, fpr_limit, requested_min, requested_max, axis_min, axis_max,
%   benign_max, injected_min and, when validation has an optimal interval, interval_lo,
%   interval_hi, midpoint, interval_recall, interval_fpr.
S = load_csv_columns(file);
names = S.cols{1};
vals = S.cols{2};
if iscell(vals)
    vals = str2double(vals);
end
B = struct();
for i = 1:numel(names)
    B.(strtrim(names{i})) = vals(i);
end
end
