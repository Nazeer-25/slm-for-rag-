function [pos, lab] = logodds_ticks(xl)
%LOGODDS_TICKS  Tick positions (log-odds) and labels (threshold values) for a log-odds axis
%   whose limits are xl = [min max].
t = [1e-8 1e-6 1e-4 0.01 0.5 0.99 1-1e-4 1-1e-6 1-1e-8];
names = {'10^{-8}', '10^{-6}', '10^{-4}', '0.01', '0.5', '0.99', '1-10^{-4}', '1-10^{-6}', '1-10^{-8}'};
pos = log(t ./ (1 - t));
keep = pos >= xl(1) & pos <= xl(2);
pos = pos(keep);
lab = names(keep);
end
