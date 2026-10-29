function y = to_binary_labels(col)
%TO_BINARY_LABELS  Map a label column to 0 = benign and 1 = injected.
%   Accepts numbers (0/1), logicals or text such as benign/poisoned,
%   clean/injected or True/False.
if isnumeric(col) || islogical(col)
    y = double(col(:) > 0.5);
    return
end
s = lower(strtrim(col(:)));
pos = {'1', 'true', 'yes', 'poisoned', 'injected', 'injection', 'attack', 'malicious', 'unsafe', 'positive'};
neg = {'0', 'false', 'no', 'benign', 'clean', 'safe', 'negative', 'legitimate'};
y = nan(numel(s), 1);
y(ismember(s, pos)) = 1;
y(ismember(s, neg)) = 0;
if any(isnan(y))
    bad = unique(s(isnan(y)));
    error('Unrecognised label values: %s. Add them to to_binary_labels.m.', ...
        strjoin(bad(1:min(5, numel(bad)))', ', '));
end
end
