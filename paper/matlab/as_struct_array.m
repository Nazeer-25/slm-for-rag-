function H = as_struct_array(v)
%AS_STRUCT_ARRAY  Turn a decoded JSON list of records into a struct column.
%   jsondecode returns a struct array when all records share the same
%   fields and a cell array otherwise; only the common fields are kept.
H = [];
if isstruct(v)
    H = v(:);
elseif iscell(v) && ~isempty(v) && all(cellfun(@isstruct, v))
    common = fieldnames(v{1});
    for k = 2:numel(v)
        common = intersect(common, fieldnames(v{k}));
    end
    for k = 1:numel(v)
        for c = 1:numel(common)
            H(k, 1).(common{c}) = v{k}.(common{c}); %#ok<AGROW>
        end
    end
end
end
