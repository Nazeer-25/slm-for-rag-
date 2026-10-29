function [v, where] = find_field(S, key, prefix)
%FIND_FIELD  Depth-first search of a decoded JSON struct for a field name.
%   Returns the first value whose field name equals KEY (case-insensitive)
%   and its dotted path.
v = [];
where = '';
if nargin < 3
    prefix = '';
end
if ~isstruct(S) || isempty(S)
    return
end
f = fieldnames(S);
for i = 1:numel(f)
    if strcmpi(f{i}, key)
        v = S(1).(f{i});
        where = [prefix f{i}];
        return
    end
end
for i = 1:numel(f)
    child = S(1).(f{i});
    if isstruct(child)
        [v, where] = find_field(child, key, [prefix f{i} '.']);
        if ~isempty(where)
            return
        end
    end
end
end
