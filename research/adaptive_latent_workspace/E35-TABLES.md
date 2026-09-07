# E35 audited tables

All operational predictions are scored before updates/selection; time and memory are additional costs.

| Training | Regime | Policy | Accuracy | NLL | Latter-half accuracy | Acquired return accuracy (occurrences) | Suffix work/input | Updated chunks | CPU seconds | Peak state KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|e2e_carry|drifting|active_always|88.59%|0.8137|90.71%|74.1% (163)|1.00|100.0%|24.119|28.0|
|e2e_carry|drifting|active_gated|88.68%|0.8476|90.55%|75.0% (163)|1.00|62.3%|22.420|28.0|
|e2e_carry|drifting|bank2_gated|59.29%|1.7012|63.94%|64.5% (101)|4.22|91.9%|29.979|100.0|
|e2e_carry|drifting|bank8_gated|34.92%|2.3258|26.87%|76.7% (47)|9.49|93.8%|41.091|244.0|
|e2e_carry|drifting|bank8_no_select_gated|88.68%|0.8476|90.55%|75.0% (163)|9.70|62.3%|39.380|244.0|
|e2e_carry|expanding|active_always|84.83%|0.9514|90.85%|66.0% (346)|1.00|100.0%|23.787|28.0|
|e2e_carry|expanding|active_gated|85.45%|0.9534|91.55%|67.3% (345)|1.00|71.7%|22.861|28.0|
|e2e_carry|expanding|bank2_gated|48.55%|1.9705|55.13%|57.2% (186)|4.23|94.9%|30.601|100.0|
|e2e_carry|expanding|bank8_gated|39.03%|2.2186|38.68%|75.3% (144)|9.70|92.7%|40.762|244.0|
|e2e_carry|expanding|bank8_no_select_gated|85.45%|0.9534|91.55%|67.3% (345)|9.64|71.7%|39.719|244.0|
|e2e_carry|recurring|active_always|95.51%|0.5312|97.38%|82.8% (266)|1.00|100.0%|24.049|28.0|
|e2e_carry|recurring|active_gated|95.53%|0.6049|97.32%|82.9% (266)|1.00|27.7%|21.631|28.0|
|e2e_carry|recurring|bank2_gated|83.50%|1.0241|94.20%|73.3% (237)|4.10|75.0%|29.881|100.0|
|e2e_carry|recurring|bank8_gated|73.35%|1.3552|78.88%|76.1% (210)|8.68|84.9%|39.020|244.0|
|e2e_carry|recurring|bank8_no_select_gated|95.53%|0.6049|97.32%|82.9% (266)|9.54|27.7%|37.778|244.0|
|e2e_reset|drifting|active_always|87.88%|0.8804|90.19%|69.5% (163)|1.00|100.0%|23.738|28.0|
|e2e_reset|drifting|active_gated|87.98%|0.8944|90.32%|70.4% (163)|1.00|76.8%|22.921|28.0|
|e2e_reset|drifting|bank2_gated|60.40%|1.6556|60.72%|59.6% (105)|4.21|95.3%|30.381|100.0|
|e2e_reset|drifting|bank8_gated|38.11%|2.2475|30.81%|76.0% (48)|9.76|96.9%|41.099|244.0|
|e2e_reset|drifting|bank8_no_select_gated|87.98%|0.8944|90.32%|70.4% (163)|9.69|76.8%|40.400|244.0|
|e2e_reset|expanding|active_always|83.03%|1.0529|89.70%|60.1% (343)|1.00|100.0%|24.033|28.0|
|e2e_reset|expanding|active_gated|83.28%|1.0535|90.05%|61.0% (347)|1.00|86.5%|22.952|28.0|
|e2e_reset|expanding|bank2_gated|43.98%|2.1139|48.52%|49.7% (172)|4.25|98.8%|30.114|100.0|
|e2e_reset|expanding|bank8_gated|34.43%|2.3690|33.31%|74.1% (119)|8.86|97.0%|39.421|244.0|
|e2e_reset|expanding|bank8_no_select_gated|83.28%|1.0535|90.05%|61.0% (347)|9.56|86.5%|40.340|244.0|
|e2e_reset|recurring|active_always|94.91%|0.6084|97.08%|80.0% (266)|1.00|100.0%|23.700|28.0|
|e2e_reset|recurring|active_gated|94.99%|0.6482|97.14%|80.3% (266)|1.00|42.9%|21.444|28.0|
|e2e_reset|recurring|bank2_gated|76.31%|1.2632|84.19%|72.8% (209)|4.17|91.1%|29.717|100.0|
|e2e_reset|recurring|bank8_gated|62.69%|1.6224|61.71%|76.7% (177)|8.87|90.1%|39.672|244.0|
|e2e_reset|recurring|bank8_no_select_gated|94.99%|0.6482|97.14%|80.3% (266)|9.63|42.9%|38.793|244.0|

Return eligibility differs by learner; use the common acquired subset in the paired JSON for controlled return comparisons. State bytes exclude shared base, Python, graphs and outputs. CPU seconds sum all seeds; not isolated accelerator throughput.

| Seed | Training | Regime | Active gated accuracy | Bank8 accuracy | Active gated NLL | Bank8 NLL | Time ratio |
|---|---|---|---:|---:|---:|---:|---:|
|33101|e2e_reset|recurring|95.12%|69.90%|0.6487|1.4338|1.73|
|33102|e2e_reset|recurring|95.48%|49.34%|0.6314|1.9477|1.98|
|33103|e2e_reset|recurring|94.37%|68.84%|0.6644|1.4857|1.84|
|33101|e2e_reset|expanding|82.07%|39.09%|1.0776|2.2677|1.75|
|33102|e2e_reset|expanding|82.98%|35.36%|1.0661|2.3234|1.60|
|33103|e2e_reset|expanding|84.78%|28.83%|1.0169|2.5159|1.80|
|33101|e2e_reset|drifting|88.33%|41.56%|0.8797|2.1519|1.87|
|33102|e2e_reset|drifting|87.49%|35.68%|0.9117|2.3455|1.70|
|33103|e2e_reset|drifting|88.13%|37.07%|0.8917|2.2452|1.82|
|33101|e2e_carry|recurring|95.47%|91.49%|0.6115|0.9040|1.66|
|33102|e2e_carry|recurring|95.58%|65.94%|0.6038|1.5104|1.87|
|33103|e2e_carry|recurring|95.54%|62.61%|0.5993|1.6511|1.89|
|33101|e2e_carry|expanding|85.56%|47.95%|0.9445|2.0267|1.83|
|33102|e2e_carry|expanding|84.29%|28.42%|0.9905|2.4620|1.69|
|33103|e2e_carry|expanding|86.50%|40.71%|0.9252|2.1670|1.83|
|33101|e2e_carry|drifting|89.84%|34.67%|0.8039|2.3798|1.91|
|33102|e2e_carry|drifting|87.34%|33.35%|0.8876|2.3649|1.82|
|33103|e2e_carry|drifting|88.87%|36.74%|0.8514|2.2327|1.77|

| Regime | Specialized delayed table accuracy | Table CPU seconds | Logical key/value bytes |
|---|---:|---:|---:|
|recurring|83.35%|0.008705|256|
|expanding|75.26%|0.008861|256|
|drifting|83.23%|0.010375|256|

Limited recurrence criterion passed: False. This is not full architecture or goal success.
