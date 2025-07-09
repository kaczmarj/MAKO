# Data for MAKO

This directory contains data that can be used to train MAKO models. The table below describes the columns in the file `tcga-brca-labels.csv`.

| Column | Description |
| ------ | ----------- |
| Dataset | Name of the dataset. All rows are 'TCGA'. |
| STUDYID | The ID of the participant. A participant may have multiple slides. |
| SLIDEID | The ID of the whole slide image. |
| ER+/HER2- | If True, this SLIDEID is of a ER+/HER2- specimen. |
| ror_p_score_div100 | The ROR-P score divided by 100. Use this to train ROR-P regression models. |
| rorp | The ROR-P group (low, medium, high). |
| rorp_2class | The binarized ROR-P group (low/medium, high). |
| Basal | The similarity to the Basal subtype. |
| Her2 | The similarity to the Her2-enriched subtype. |
| LumA | The similarity to the LumA subtype. |
| LumB | The similarity to the LumB subtype. |
| PAM50_Subtype_4x | The PAM50 subtype (the max of Basal, Her2, LumA, and LumB). |
| recurrence_event_censored_10_years | Whether the participant had a recurrence event. |
| recurrence_years_censored_10_years | Time of recurrence or time of last followup. |
| race | Race. |
| age | Age in years. |
| stage | TNM stage. |
