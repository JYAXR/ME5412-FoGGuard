# Chart data

All metric values in CSV/JSON are fractions in [0,1], except seconds, counts and thresholds. HTML formats accuracy and coverage as percentages.

FoG comparison uses matched windows. Direction uses valid known-target windows; rejected predictions count as errors. Accepted-only accuracy excludes rejected predictions. Macro-F1 averages straight/left/right. `support` is per-class window count.

Time recall uses causal raw-time overlap, not the window-classification denominator. The `windows` column in all_metrics.csv identifies the classification grid; it is not the time metric denominator.

methods.csv describes implementation, evaluation_scope.csv gives cohort/pool counts, and figure_selection.csv records the fixed replay display rule. No per-window patient CSV is published. Development values are separately labeled.
