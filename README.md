# scuc23

## EDA(1) train.csv, test.csvの概要
- id: 行番号みたいなものなので、説明変数としては不要そう。train.csvとtest.csvに被りもない
- region, year, condition: 欠損なし
- year: 2999以上の値が入っているので、そういうものは、-1000したほうが良さそう
- manufacturer: 全角・半角入り混じっているので名寄せする
- cylinders: otherがある
- fuel: 欠損があるのと、otherもある
- odometer: 負数がある。数十〜30万と幅広い数値
- title_status: 空白がある
- transmission: otherってなんだろ。
- drive: 特筆すべき点なし
- size: ハイフンの違いで別の文字列になっているのがあるため、名寄せする
- type: 空白がある
- paint_color: 特筆すべき点なし
- state: 空白がある
- price: 特筆すべき点なし

## 学習・予測の実行方法
1. git clone https://github.com/3ign0n/scuc23
2. kedro run


## ジャーナル
- 2023/08/19
  - stateをドロップしてみよう（regionと多重共線性あるのでは？）
  - metricsをmaeとmapeの両方にしてみよう
- 2023/08/18
  - priceの対数を試す
  - こんなのドキュメントを見つける(カラムはほぼ一緒だが、欠損値が多くてより厳しいデータ) https://shubh17121996.medium.com/used-car-price-prediction-using-supervised-machine-learning-ea9dace76686 
  - 元ネタのデータセットはおそらくこれ https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/discussion/139771
- 2023/08/11 lightgbm.cvを使ってクロスバリデーションするようにした
- 2023/08/05 pandas profilingの結果を保存するようにした
- 2023/08/04 mlflow導入
- 2023/07/30 kedro+mlflowのチュートリアルを読み始める。[Closed
[Feature Request] Support for Hydra in Kedro](https://github.com/kedro-org/kedro/issues/1303)が、「won't fix」で閉じられてた...hydraの導入は見送ろう
- 2023/07/29 kedroの環境構築に疲れたので、気分転換にCSVの中身をspreadsheetに取り込んでざっと眺める
- 2023/07/28 パイプライン・実験管理・特徴量／ハイパラ管理、色々比較しましたが、kedro+mlflow+hydra+optunaな構成が良さそうってことで、kedroのチュートリアルを読み始める
- 2023/07/27 「全自動」で、とか、「前処理」「学習」「予測」に分けろとか、色々制約がある。これを機に、実験管理もちゃんとやるか。。。と思い立つ
- 2023/07/24 メールで見て、参加を決める
