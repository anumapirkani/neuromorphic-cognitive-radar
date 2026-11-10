function cfg = ncr_set(cfg, path, value)
%NCR_SET  Set a (possibly nested) configuration field given as 'group.field'.
%   cfg = NCR_SET(cfg, 'lrn.bits', 8). Call NCR_DERIVE afterwards (NCR_CONFIG
%   does this automatically for its own overrides).
parts = regexp(path, '\.', 'split');
switch numel(parts)
    case 1
        cfg.(parts{1}) = value;
    case 2
        if ~isfield(cfg, parts{1}) || ~isfield(cfg.(parts{1}), parts{2})
            error('ncr_set:unknown', 'Unknown configuration field "%s".', path);
        end
        cfg.(parts{1}).(parts{2}) = value;
    otherwise
        error('ncr_set:depth', 'Only group.field paths are supported ("%s").', path);
end
end
