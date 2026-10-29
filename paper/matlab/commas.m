function s = commas(n)
%COMMAS  Integer with thousands separators, e.g. 4809 -> '4,809'.
s = sprintf('%d', round(n));
k = numel(s) - 3;
while k > 0 && ~(k == 1 && s(1) == '-')
    s = [s(1:k) ',' s(k+1:end)];
    k = k - 3;
end
end
