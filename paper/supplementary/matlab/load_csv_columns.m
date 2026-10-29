function S = load_csv_columns(file)
%LOAD_CSV_COLUMNS  Read a CSV file into S.names (column names) and S.cols.
%   Numeric columns become double column vectors; all other columns become
%   cell arrays of char. Quoted fields may contain commas, quotes ("") and
%   line breaks, as in the chunk text of the predictions files.
if exist(file, 'file') ~= 2
    error('File not found: %s', file);
end
txt = fileread(file);
if ~isempty(txt) && double(txt(1)) == 65279          % UTF-8 byte-order mark
    txt = txt(2:end);
elseif numel(txt) >= 3 && isequal(double(txt(1:3)), [239 187 191])
    txt = txt(4:end);
end
txt = strrep(txt, sprintf('\r\n'), sprintf('\n'));
inq = mod(cumsum(txt == '"'), 2) == 1;                % inside a quoted field
nl = find(txt == sprintf('\n') & ~inq);
if isempty(nl) || nl(end) ~= numel(txt)
    nl = [nl, numel(txt) + 1];
end
starts = [1, nl(1:end-1) + 1];
ends = nl - 1;
isComma = (txt == ',') & ~inq;
rows = cell(numel(starts), 1);
for r = 1:numel(starts)
    a = starts(r);
    b = ends(r);
    if b < a
        continue
    end
    seg = txt(a:b);
    cpos = find(isComma(a:b));
    fs = [1, cpos + 1];
    fe = [cpos - 1, numel(seg)];
    f = cell(1, numel(fs));
    for k = 1:numel(fs)
        v = seg(fs(k):fe(k));
        if numel(v) >= 2 && v(1) == '"' && v(end) == '"'
            v = strrep(v(2:end-1), '""', '"');
        end
        f{k} = v;
    end
    rows{r} = f;
end
rows = rows(~cellfun(@isempty, rows));
if isempty(rows)
    error('Empty CSV file: %s', file);
end
S.names = strtrim(rows{1});
data = rows(2:end);
S.cols = cell(1, numel(S.names));
for i = 1:numel(S.names)
    col = repmat({''}, numel(data), 1);
    for r = 1:numel(data)
        if numel(data{r}) >= i
            col{r} = data{r}{i};
        end
    end
    num = str2double(col);
    filled = ~cellfun(@isempty, strtrim(col));
    if any(filled) && all(~isnan(num(filled)))
        S.cols{i} = num;
    else
        S.cols{i} = col;
    end
end
end
