import matplotlib.pyplot as plt
import pandas as pd

#散布図の作成（特徴量まで含めてすべて）
def plot_all(results_df,thresholds, colums_list, data_ex):

    #異常スコアの出力
    #図の大きさを指定
    fig = plt.figure(figsize=(60,100))
    #はじめのサブプロットの作成（異常スコア）
    ax1 = fig.add_subplot(13,1,1)
    ax1.set_title("autoencoder_score")
    ax1.set_xlabel('time',fontsize=25)
    # 軸の目盛りラベルサイズを変更する
    ax1.tick_params(axis='both', which='major', labelsize=30)

    #正常スコアと異常スコア
    ax1.scatter(results_df["measurement_date"], results_df["anomaly_score"], c='blue', marker='o', edgecolor='k')
    # 各期間の異常スコアの閾値を描画

    # measurement_date の最小値と最大値を取得
    min_date = results_df["measurement_date"].min()
    max_date = results_df["measurement_date"].max()
    for threshold in thresholds:
        # test_start と test_end を datetime 型に変換
        test_start = pd.to_datetime(threshold['test_start'])
        test_end = pd.to_datetime(threshold['test_end'])
        relative_start = (test_start - min_date) / (max_date - min_date)
        relative_end = (test_end - min_date) / (max_date - min_date)
        
        ax1.axhline(y=threshold['threshold'],xmin=relative_start, xmax=relative_end,color='red', linestyle='--', label=f"Threshold (Term {threshold['term']})", linewidth = 10)
    
    # ラベルの設定
    ax1.set_ylabel('Abnormality')


    start_date = pd.to_datetime("2018-06-01")
    end_date = pd.to_datetime("2018-10-01")
    data_ex = data_ex[(data_ex["measurement_date"] >= start_date) & (data_ex["measurement_date"] <= end_date)]
    
    # 元の特徴量の時系列ごとのデータ
    for i, column in enumerate (colums_list):
        ax = fig.add_subplot(13,1,i+2)
        ax.scatter(data_ex["measurement_date"],data_ex[column], color='b')
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.set_xlabel('time')
        ax.set_title(column)
        ax.legend()



#散布図の作成（予測特徴量値と元データの特徴量値）
def plot_predict(traindata_model_df, colums_list, data_ex):

    #異常スコアの出力
    #図の大きさを指定
    fig = plt.figure(figsize=(60,200))
    #はじめのサブプロットの作成（異常スコア）

    # 各期間の異常スコアの閾値を描画

    # measurement_date の最小値と最大値を取得
    min_date = data_ex["measurement_date"].min()
    max_date = data_ex["measurement_date"].max()

    start_date = pd.to_datetime("2016-06-01")
    end_date = pd.to_datetime("2018-09-01")
    data_ex = data_ex[(data_ex["measurement_date"] >= start_date) & (data_ex["measurement_date"] <= end_date)]

    print(traindata_model_df.shape)
    print(data_ex.shape)
    
    # 元の特徴量の時系列ごとのデータ
    for i, column in enumerate (colums_list):
        ax = fig.add_subplot(23,1,2*i+1)
        ax.scatter(data_ex["measurement_date"],data_ex[column], color='b')
        ax.set_xlabel('time')
        ax.set_title(column + ":origin")
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.legend()

        ax = fig.add_subplot(23,1,2*i+2)
        ax.scatter(traindata_model_df["measurement_date"],traindata_model_df[column], color='b')
        ax.set_xlabel('time')
        ax.set_title(column + " : 元データに対する予測値")
        ax.legend()



def data_describe(traindata_model_df, data_ex):

    print("データの構造確認")

    print("元データ")
    print(data_ex.describe())

    print("========================================================================================")
    print("========================================================================================")

    print("再構築データ")
    print(traindata_model_df.describe())


import matplotlib.pyplot as plt

# ユニット数の合計を横軸に、final_lossを縦軸に描写する関数
def plot_final_loss_vs_unit_sum(final_losses_per_units):
    unit_sums = []
    losses = []
    
    # 各 (units_1_3, units_2) に対して合計ユニット数とfinal_lossを取得
    for (units_1_3, units_2), avg_loss in final_losses_per_units.items():
        unit_sum = units_1_3 + units_2  # ユニット数の合計
        unit_sums.append(unit_sum)
        losses.append(avg_loss)
    
    # プロットの作成
    plt.figure(figsize=(8, 6))
    plt.scatter(unit_sums, losses)
    plt.title('Final Loss vs Total Unit Count')
    plt.xlabel('Total Unit Count (units_1_3 + units_2)')
    plt.ylabel('Final Loss')
    plt.grid(True)
    plt.show()

