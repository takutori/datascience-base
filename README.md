# datascience-base

データサイエンスを実施するときの便利な機能をまとめたもの。

baseは最も基盤になるリポジトリであることを示す。


## isntall



### 仮想環境のコピー
toml, lockを新しいルートディレクトリにコピーした上で以下を実行
```
poetry install --no-root --with dev,gpu
```

`--no-root`はこのライブラリをinstallしたい場合のみ外す。
