function [fpr, tpr, thr, auc, prec, rec, ap] = roc_curve_points(y, s)
%ROC_CURVE_POINTS  ROC and precision-recall curves without any toolbox.
%   y: 0/1 labels (1 = injected), s: scores (higher = more likely injected).
%   Returns ROC points (fpr, tpr) with thresholds thr, the ROC AUC, the
%   precision-recall points (prec, rec) and the average precision ap.
y = y(:);
s = s(:);
[s_sorted, order] = sort(s, 'descend');
y_sorted = y(order);
tp = cumsum(y_sorted == 1);
fp = cumsum(y_sorted == 0);
last = [find(diff(s_sorted) ~= 0); numel(s_sorted)];   % one point per distinct score
tp = tp(last);
fp = fp(last);
P = sum(y == 1);
Nn = sum(y == 0);
tpr = [0; tp / P];
fpr = [0; fp / Nn];
thr = [Inf; s_sorted(last)];
auc = trapz(fpr, tpr);
prec = tp ./ (tp + fp);
rec = tp / P;
ap = sum(diff([0; rec]) .* prec);
end
