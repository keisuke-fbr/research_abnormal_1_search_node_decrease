
#必要ライブラリのインストール
from src.data import data_complete
from src.model import result
from src.store import store
from src.store import read
from src.plot import plot_all
from src.plot import plot_predict
from src.plot import data_describe
from tensorflow.keras.callbacks import EarlyStopping



#データの読み込み及び加工、コラムリストの取得
colums_list, data_ex, data_original = data_complete()


#データの確認
print(data_ex)



# ここからモデルの作成に入る

#============================================================================================================================
#ハイパーパラメータの設定

#オートエンコーダの重みの初期化方法
initializer = "glorot_normal"

#モデルの最小化関数のパラメータ
delta = 0.5

#最大エポック数
max_epochs = 10000000

#モデル決定の閾値（再構成誤差のパラメータ）
#この閾値を下回ることで十分な性能を持つモデルとする
error_threshold = 1e+10

# EarlyStoppingの設定
#0.0001を下回ることで学習が完了とする
early_stopping = EarlyStopping(monitor='loss', patience=1, min_delta=1e+1, restore_best_weights=True, mode = "auto")

#============================================================================================================================



#実験の実行及び結果の格納
results_df, traindata_model_df, thresholds = result(data_ex, colums_list,  initializer, error_threshold, max_epochs, early_stopping )


#結果の保存
store(results_df,"model_1")
store(thresholds,"thresholds_1")
store(traindata_model_df, "traindata_model_df_1")


#結果の抜き出し
results_df = read("model_1")
thresholds = read("thresholds_1")
traindata_model_df = read("traindata_model_df_1")


#結果のプロット
plot_all(results_df,thresholds, colums_list, data_original)


plot_predict(traindata_model_df, colums_list, data_ex)


#データの確認
data_describe(traindata_model_df, data_ex)


#モジュールの再リロード
import importlib
import src

importlib.reload(src.model)
importlib.reload(src.plot)

#必要ライブラリのインストール
from src.data import data_complete
from src.model import result
from src.store import store
from src.store import read
from src.plot import plot_all
from src.plot import plot_predict
from src.plot import data_describe






