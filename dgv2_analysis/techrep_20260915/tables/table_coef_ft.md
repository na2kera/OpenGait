# ft 予測の標準化 Ridge 係数（pt / 1SD, 13 fold 平均, 自動生成）

| 指標 | S_inc6 | S_sustech6 |
|---|---:|---:|
| $\log P$（人数） | +2.55 | +3.02 |
| MWCV（平均クラス内分散） | +2.64 | -0.67 |
| 1NN | +1.14 | +0.76 |
| kNN ($k=5$) | +1.68 | +2.93 |
| pairFID（平均クラス間距離） | -0.86 | +0.10 |
| FID$_{\mathrm{tt}}$（Train-Test 分布距離） | -3.08 | -5.17 |

S_incdino11（11変数）の ft 係数（絶対値降順）:

| 変数 | 係数 |
|---|---:|
| log_people | +2.44 |
| Inception_kNN | +1.57 |
| Inception_MSD | +1.51 |
| DINO_MSD | +1.46 |
| FID_train_test_Inception | -1.23 |
| Inception_1NN | +1.16 |
| FID_train_test_DINO | -1.07 |
| Inception_FID | -0.66 |
| DINO_kNN | +0.55 |
| DINO_FID | -0.39 |
| DINO_1NN | +0.26 |

注意（0907.md セルフレビュー）: FID_tt が最大絶対値なのは S_inc6 と S_sustech6。S_incdino11 では Inception_kNN / Inception_MSD（MWCV）の方が大きい。
