function [lo, hi] = wilson_ci(k, n, z)
%WILSON_CI  95% Wilson score interval for k successes out of n trials.
%   [lo, hi] = wilson_ci(k, n) returns proportions in [0, 1]. Works on
%   vectors element-wise. Pass z to change the confidence level.
if nargin < 3
    z = 1.959963984540054;
end
p = k ./ n;
den = 1 + z.^2 ./ n;
centre = (p + z.^2 ./ (2 .* n)) ./ den;
half = z .* sqrt(p .* (1 - p) ./ n + z.^2 ./ (4 .* n.^2)) ./ den;
lo = max(0, centre - half);
hi = min(1, centre + half);
end
