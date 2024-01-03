スネークマスター、美しく知性体を使ってスネークのゲームに合格しよう


# SnakeAI

[简体中文](README-CN.md) | [English](README-EN.md) | [日本語]

スネークマスター、スネークゲームをパスするためにインテリジェンスを使用する試み
このプロジェクトには、古典的なスネークゲーム、偶発的なトレーニングによって生成されたパスソリューション、ハミルトンリングの歩行（パスすることができるが、いくつかの不正）、および以前のcnnトレーニングの結果、mlpの結果が含まれています。 トレーニングデータも公開され、アップロード用にパッケージ化される。
以下は、私がトレーニングしたモデルである。 

### ファイル構造

```bash
├───main
│   ├───logs
│   ├───trained_models_cnn
│   ├───sound
│   ├───20231217
│   ├───20231219
│   ├───20231220
│   ├───20231222
│   ├───requirements.txt
│   └───scripts
├───utils
│   └───scripts
```

プロジェクトは主にGPUをチェックするスクリプト(gpu.pyとPyTorch.py)、数回分の学習データ(numericというフォルダ)、学習過程のターミナルテキストとデータカーブ(Tensorboardを使用して表示)を含むlogsフォルダで構成されています。requirements.txtはanacondaの設定ファイルです。
check_gpu_status/ PyTorch から GPU を呼び出せるかどうかをチェックします。

## 実行ガイド
本プロジェクトはPythonプログラミング言語に基づいており、主にPyTorch、Pillow（PIL）、NumPyなどの外部コードライブラリを使用して画像処理と深層学習モデルの訓練と推論を行います。プログラムの実行に使用されるPythonのバージョンは3.8.18で、Anacondaを使用してPython環境を設定することを推奨します。以下の設定プロセスは、Windows 11システムでテスト済みです。以下はWindows Terminalの指示です。

### 環境設定

このプロジェクトはPythonプログラミング言語をベースにしており、Pygame、OpenAI Gym、Stable-Baselines3などの外部コードライブラリを使用しています。 Pythonのバージョンは3.8.16で、Pythonの環境設定にはAnacondaを使用することを推奨します。 以下はWindowsのターミナルコマンドです。

```bash
conda環境を作成し、SnakeAIと命名して環境をアクティブ化
conda create -n GallaryAI python=3.8.18
conda activate GallaryAI
```


Windows:
```bash 
公式ウェブサイトに行って対応するバージョンのPyTorchをダウンロード。GPU訓練には完全版のPyTorchの手動インストールが必要
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

プログラムスクリプトを実行してPyTorchがGPUを正常に呼び出せるかテスト
python gpu.py
python PyTorch.py

外部コードライブラリをインストール
pip install -r requirements.txt
```


### トレーニングとテストの実行

以下のコマンドをプロジェクトフォルダから直接実行することで、ゲームをプレイすることができます：

bash
cd [プロジェクトの親フォルダ]/snake-ai/main
python . \``bash cd [プロジェクト親フォルダ]/snake-ai/main python .
```


環境が設定できたら、main/ フォルダにある test_cnn.py を実行してテストします。
```bash
cd "ディレクトリ"
# 畳み込みニューラルネットワークモデルの学習スクリプトを実行します。
python train_cnn.py

# モデルのテストスクリプトを実行します
python test_cnn.py
``

モデルの重みファイルは main/trained_models_cnn/ に保存されます。

モデルを再トレーニングする必要がある場合、train_cnn.pyがあるディレクトリでこのファイルを実行することができます。 テストスクリプトのデフォルトは学習済みモデル、例えばppo_snake_final.zipです。 学習の異なる段階でのAIのパフォーマンスを観察する必要がある場合、テストスクリプトのMODEL_PATH変数を他のモデルファイルのパスに変更することができます。


### カーブの表示

プロジェクトには、トレーニングプロセスの Tensorboard グラフが含まれているため、Tensorboard を使用して詳細なデータを表示できます。 カーブを直接見るには、VSCode Tensorboard プラグインを使用することをお勧めします：

bash
cd "ディレクトリ"
tensorboard --logdir=[親ディレクトリ]∕-bind_all --reload_interval 60
```

このコマンドは、いつでもどこでも簡単に見られるようにポートをパブリックネットワークに公開し、60秒ごとに自動的に画像を更新する。 デフォルトのアドレス`http://localhost:6006/`でTensorboardサービスを開くと、トレーニングプロセスのインタラクティブなグラフをブラウザで見ることができる。
