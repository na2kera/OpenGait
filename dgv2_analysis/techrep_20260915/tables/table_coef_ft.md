# ft 予測の標準化 Ridge 係数（pt / 1SD, 13 fold 平均, 自動生成）

| 指標 | S_inc6 | S_sustech6 |
|---|---:|---:|
| $\log P$（人数） | +1.92 | +3.02 |
| MSD（平均クラス内分散） | +1.98 | -0.67 |
| 1NN | +1.50 | +0.76 |
| kNN ($k=5$) | +2.04 | +2.93 |
| pairFID（平均クラス間距離） | -1.88 | +0.10 |
| FID$_{\mathrm{tt}}$（Train-Test 分布距離） | -3.60 | -5.17 |

S_incdino11（11変数）の ft 係数（絶対値降順）:

| 変数 | 係数 |
|---|---:|
| log_people | +1.91 |
| Inception_kNN | +1.76 |
| Inception_MSD | +1.46 |
| Inception_1NN | +1.40 |
| FID_train_test_Inception | -1.29 |
| DINO_MSD | +1.12 |
| Inception_FID | -1.08 |
| FID_train_test_DINO | -1.02 |
| DINO_FID | -0.86 |
| DINO_kNN | +0.77 |
| DINO_1NN | +0.51 |

注意（0907.md セルフレビュー）: FID_tt が最大絶対値なのは S_inc6 と S_sustech6。S_incdino11 では Inception_kNN / Inception_MSD の方が大きい。
