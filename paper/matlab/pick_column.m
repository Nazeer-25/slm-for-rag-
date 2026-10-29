function idx = pick_column(S, candidates, override, what, exclude)
%PICK_COLUMN  Index of the column to use for a given role.
%   idx = pick_column(S, candidates, override, what, exclude)
%   override : exact column name to use ('' = auto-detect)
%   candidates: preferred names, tried exactly (case-insensitive), then as
%               substrings; names matching the regexp EXCLUDE are skipped.
if nargin < 5
    exclude = '';
end
names = lower(S.names);
if ~isempty(override)
    idx = find(strcmpi(S.names, override), 1);
    if isempty(idx)
        error('Column "%s" not found. Available columns: %s', override, strjoin(S.names, ', '));
    end
    return
end
allowed = true(size(names));
if ~isempty(exclude)
    allowed = cellfun(@isempty, regexp(names, exclude, 'once'));
end
for k = 1:numel(candidates)
    idx = find(strcmp(names, lower(candidates{k})) & allowed, 1);
    if ~isempty(idx)
        return
    end
end
for k = 1:numel(candidates)
    idx = find(~cellfun(@isempty, strfind(names, lower(candidates{k}))) & allowed, 1);
    if ~isempty(idx)
        return
    end
end
idx = [];
fprintf('No %s column found. Available columns: %s\n', what, strjoin(S.names, ', '));
end
