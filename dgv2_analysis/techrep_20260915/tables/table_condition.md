# 条件別（NM/BG/CL）の予測性能（自動生成）

汎用 = S_incdino11（`result_incdino_ft_20260806_X2.json` D_per_condition）、歩容特化 = S_sustech6（`presentation_tables_20260726.json` phase1_per_condition / phase2_extra）。
MAE は pt。n=13 の記述統計で並べ替え検定は無し。ρ の大小は「傾向」として書く。

| 予測対象 | 条件 | 汎用 ρ | 汎用 MAE | 歩容特化 ρ | 歩容特化 MAE |
|---|---|---:|---:|---:|---:|
| scratch | NM | 0.698 | 10.26 | 0.868 | 7.41 |
| scratch | BG | 0.879 | 9.30 | 0.835 | 4.48 |
| scratch | CL | 0.665 | 18.29 | 0.956 | 7.66 |
| ft | NM | 0.791 | 3.72 | 0.753 | 4.41 |
| ft | BG | 0.863 | 3.26 | 0.819 | 3.67 |
| ft | CL | 0.511 | 15.55 | 0.896 | 6.26 |

注意: scratch の NM は歩容特化が 0.17 高い。「NM・BG は同じ」ではなく「NM・BG では大差がなく CL で差が開く」と書く（skill §3.3）。
