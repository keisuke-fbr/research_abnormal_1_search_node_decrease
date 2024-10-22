# 必要ライブラリのインストール
import pandas as pd
import numpy as np
from dfply import *
import os
import gc
import joblib
from sklearn import preprocessing


# 生データの読み込みおよびデータの加工
def load_data():
    # キャッシュファイルのパスを設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(current_dir, '../cache/cached_data.joblib')

    # キャッシュファイルが存在すれば、それを読み込む
    if os.path.exists(cache_file):
        print("キャッシュファイルからデータを読み込みます")
        return joblib.load(cache_file)

    # キャッシュファイルがない場合、初回データ読み込みを行う
    print("データファイルからデータを読み込みます")
    
    ecdis_file_path = os.path.join(current_dir, "../../data/ECDIS__FMD-3x00.csv")
    radar_file_path = os.path.join(current_dir, "../../data/Radar__FAR-3xx0.csv")

    # ECDISデータの読み込み
    raw_data_ECDIS = pd.read_csv(ecdis_file_path, header=0, delimiter=',', dtype={"equipment_label_no": "object", "f_shipno": "object"})
    raw_data_ECDIS = raw_data_ECDIS >> mutate(id=X.f_shipno + "-" + X.equipment_label_no,
                                              measurement_ymd=X.measurement_date.str[:10],
                                              measurement_ymd_h=X.measurement_date.str[11:13]) >> mutate(
        measurement_ymd_hms=X.measurement_ymd + " " + X.measurement_ymd_h + ":00:00")

    print("ECDISデータの読み込み完了")
    # Radarデータの読み込み
    raw_data_Radar = pd.read_csv(radar_file_path, header=0, delimiter=',', dtype={"equipment_label_no": "object", "f_shipno": "object"})
    raw_data_Radar = raw_data_Radar >> mutate(id=X.f_shipno + "-" + X.equipment_label_no,
                                              measurement_ymd=X.measurement_date.str[:10],
                                              measurement_ymd_h=X.measurement_date.str[11:13]) >> mutate(
        measurement_ymd_hms=X.measurement_ymd + " " + X.measurement_ymd_h + ":00:00")

    print("Radarデータの読み込み完了")

    print("final_dataの作成開始")

    # 必要なカラムを抽出、および半角処理
    raw_data_ECDIS_select = raw_data_ECDIS[["id","equipment_label_no","f_shipno","measurement_date"
                                            ,"processor_unit_units_hardware_info__serial_number_cpu_bd","monitor1_units__unit","monitor2_units__unit"
                                            ,"measurement_ymd","measurement_ymd_h","measurement_ymd_hms"
                                            ,"monitor1_units_status_main__temp","monitor1_units_status_main__fan1"
                                            ,"monitor1_units_status_main__fan2","monitor1_units_status_main__fan3"
                                            ,"monitor2_units_status_main__temp","monitor2_units_status_main__fan1"
                                            ,"monitor2_units_status_main__fan2","monitor2_units_status_main__fan3","processor_unit_units_status_cpu_board__cpu_fan"
                                            ,"processor_unit_units_status_cpu_board__cpu_bd_temp","processor_unit_units_status_cpu_board__cpu_core_temp"
                                            ,"processor_unit_units_status_cpu_board__gpu_core_temp","processor_unit_units_status_cpu_board__remote1_temp"
                                            ,"processor_unit_units_status_cpu_board__remote2_temp","processor_unit_units_status_cpu_board__cpu_core_vol"
                                            ,"processor_unit_units_status_cpu_board__cpu_bd_vbat","processor_unit_units_status_cpu_board__cpu_bd_p3_3v"
                                            ,"processor_unit_units_status_cpu_board__cpu_bd_p5v","processor_unit_units_status_cpu_board__cpu_bd_p12v"
                                            ,"processor_unit_units_status_cpu_board__cpu_bd_fan1","processor_unit_units_status_cpu_board__cpu_bd_fan2"
                                            ,"processor_unit_units_status_boot_device__wearout_ind"]]


    data_ECDIS=raw_data_ECDIS_select.rename(columns={"processor_unit_units_hardware_info__serial_number_cpu_bd":"processor_unit_units_hardware_info_serial_number_cpu_bd"
                                                     ,"monitor1_units__unit":"monitor1_units_unit"
                                                     ,"monitor2_units__unit":"monitor2_units_unit"
                                                     ,"monitor1_units_status_main__temp":"monitor1_units_status_main_temp"
                                                     ,"monitor1_units_status_main__fan1":"monitor1_units_status_main_fan1"
                                                     ,"monitor1_units_status_main__fan2":"monitor1_units_status_main_fan2"
                                                     ,"monitor1_units_status_main__fan3":"monitor1_units_status_main_fan3"
                                                     ,"monitor2_units_status_main__temp":"monitor2_units_status_main_temp"
                                                     ,"monitor2_units_status_main__fan1":"monitor2_units_status_main_fan1"
                                                     ,"monitor2_units_status_main__fan2":"monitor2_units_status_main_fan2"
                                                     ,"monitor2_units_status_main__fan3":"monitor2_units_status_main_fan3"
                                                     ,"processor_unit_units_status_cpu_board__cpu_fan":"processor_unit_units_status_cpu_board_cpu_fan"
                                                     ,"processor_unit_units_status_cpu_board__cpu_bd_temp":"processor_unit_units_status_cpu_board_cpu_bd_temp"
                                                     ,"processor_unit_units_status_cpu_board__cpu_core_temp":"processor_unit_units_status_cpu_board_cpu_core_temp"
                                                     ,"processor_unit_units_status_cpu_board__gpu_core_temp":"processor_unit_units_status_cpu_board_gpu_core_temp"
                                                     ,"processor_unit_units_status_cpu_board__remote1_temp":"processor_unit_units_status_cpu_board_remote1_temp"
                                                     ,"processor_unit_units_status_cpu_board__remote2_temp":"processor_unit_units_status_cpu_board_remote2_temp"
                                                     ,"processor_unit_units_status_cpu_board__cpu_core_vol":"processor_unit_units_status_cpu_board_cpu_core_vol"
                                                     ,"processor_unit_units_status_cpu_board__cpu_bd_vbat":"processor_unit_units_status_cpu_board_cpu_bd_vbat"
                                                     ,"processor_unit_units_status_cpu_board__cpu_bd_p3_3v":"processor_unit_units_status_cpu_board_cpu_bd_p3_3v"
                                                     ,"processor_unit_units_status_cpu_board__cpu_bd_p5v":"processor_unit_units_status_cpu_board_cpu_bd_p5v"
                                                     ,"processor_unit_units_status_cpu_board__cpu_bd_p12v":"processor_unit_units_status_cpu_board_cpu_bd_p12v"
                                                     ,"processor_unit_units_status_cpu_board__cpu_bd_fan1":"processor_unit_units_status_cpu_board_cpu_bd_fan1"
                                                     ,"processor_unit_units_status_cpu_board__cpu_bd_fan2":"processor_unit_units_status_cpu_board_cpu_bd_fan2"
                                                     ,"processor_unit_units_status_boot_device__wearout_ind":"processor_unit_units_status_boot_device_wearout_ind"
                                                    })


    data_Radar = raw_data_Radar[["id","equipment_label_no","f_shipno","measurement_date",
                                 "processor_unit_units_hardware_info_serial_number_cpu_bd","monitor1_units_unit","monitor2_units_unit",
                                 "measurement_ymd","measurement_ymd_h","measurement_ymd_hms",
                                 "monitor1_units_status_main_temp","monitor1_units_status_main_fan1",
                                "monitor1_units_status_main_fan2","monitor1_units_status_main_fan3"
                                 ,"monitor2_units_status_main_temp"
                                 ,"monitor2_units_status_main_fan1","monitor2_units_status_main_fan2"
                                 ,"monitor2_units_status_main_fan3","processor_unit_units_status_cpu_board_cpu_fan",
                                "processor_unit_units_status_cpu_board_cpu_bd_temp","processor_unit_units_status_cpu_board_cpu_core_temp",
                                "processor_unit_units_status_cpu_board_gpu_core_temp","processor_unit_units_status_cpu_board_remote1_temp",
                                "processor_unit_units_status_cpu_board_remote2_temp","processor_unit_units_status_cpu_board_cpu_core_vol",
                                "processor_unit_units_status_cpu_board_cpu_bd_vbat","processor_unit_units_status_cpu_board_cpu_bd_p3_3v",
                                "processor_unit_units_status_cpu_board_cpu_bd_p5v","processor_unit_units_status_cpu_board_cpu_bd_p12v",
                                "processor_unit_units_status_cpu_board_cpu_bd_fan1","processor_unit_units_status_cpu_board_cpu_bd_fan2",
                                "processor_unit_units_status_storage_device_wearout_ind"]]

    # データの結合
    con=pd.concat([data_Radar, data_ECDIS])
    data_con=con

    # キャッシュの削除
    del raw_data_ECDIS
    del raw_data_Radar
    gc.collect()

    # データの確認
    # なお、「f_shipno」は船舶番号、「equipment_label_no」は装置ラベル番号、「id」は船舶番号と装置ラベル番号を結合したもので装置を特定する。
    print("ECDIS&Radar")
    print("idユニーク数                :",len(data_con["id"].unique()))
    print("f_shipnoユニーク数          :",len(data_con["f_shipno"].unique()))
    print("equipment_label_noユニーク数:",len(data_con["equipment_label_no"].unique()),"\n")

    
    # raw_data.shape
    print("行数: "+str(data_con.shape[0]))
    print("列数: "+str(data_con.shape[1]))
    #データ期間
    print("データ開始日時: "+str(data_con["measurement_date"].min()))
    print("データ終了日時: "+str(data_con["measurement_date"].max()))
    print("f_shipno(ユニーク数): "+str(data_con["f_shipno"].nunique()))
    print("equipment_label_no(ユニーク数): "+str(data_con["equipment_label_no"].nunique()))


    # 日付のフォーマットの変更
    data_con["measurement_ymd_hms"] = pd.to_datetime(data_con["measurement_ymd_hms"], format='%Y-%m-%d %H:%M:%S')

    # データの加工 
    # ・一時間に一個のデータしか用いない（最初のデータ）
    # ・一日取得データ数１５件以上のみ使用

    # 1時間に1データに変更
    data_con["measurement_date"] = pd.to_datetime(data_con["measurement_date"], format='%Y-%m-%d %H:%M:%S')

    data_con["order"] = data_con.groupby(["id", "measurement_ymd", "measurement_ymd_h"])["measurement_date"].rank()

    modified_data = data_con[data_con["order"]==1]


    check_day_cnt = modified_data.groupby(["id", "measurement_ymd"]).agg({"measurement_date":"nunique"}).assign(
        min_measurement_date = modified_data.groupby(['id', "measurement_ymd"]).agg({"measurement_date":"min"}),
        max_measurement_date = modified_data.groupby(['id', "measurement_ymd"]).agg({"measurement_date":"max"})).reset_index()

    check_day_cnt["id_date"] = check_day_cnt["id"]+"-"+check_day_cnt["measurement_ymd"]

    check_day_cnt = check_day_cnt.rename(columns={"measurement_date":"cnt"})

    target_id_date = check_day_cnt[check_day_cnt["cnt"] >= 15]["id_date"]

    # 条件２の適用
    modified_data["tag"] = modified_data["id"]+"-"+modified_data["measurement_ymd"]

    final_data = modified_data[modified_data["tag"].isin(target_id_date)]
    

    print("データをキャッシュに保存します")
    joblib.dump(final_data, cache_file)

    return final_data
    

#実験に使用するインスタンスを抜き出したデータを作成する
def data_model(final_data):
    final_data = final_data

    data_ADP555 = final_data[((final_data["processor_unit_units_hardware_info_serial_number_cpu_bd"].str[1:3])
                          .isin(["16","15","14","13"]))]

    
    
    data_model=data_ADP555[data_ADP555["id"]=="9748019T-325"]
    colums_list=["processor_unit_units_status_cpu_board_cpu_fan","processor_unit_units_status_cpu_board_cpu_bd_fan1"
             ,"processor_unit_units_status_cpu_board_cpu_bd_fan2","processor_unit_units_status_cpu_board_cpu_bd_temp"
             ,"processor_unit_units_status_cpu_board_cpu_core_temp","processor_unit_units_status_cpu_board_gpu_core_temp"
             ,"processor_unit_units_status_cpu_board_cpu_core_vol","processor_unit_units_status_cpu_board_cpu_bd_vbat"
             ,"processor_unit_units_status_cpu_board_cpu_bd_p3_3v","processor_unit_units_status_cpu_board_cpu_bd_p5v"
             ,"processor_unit_units_status_cpu_board_cpu_bd_p12v"]
    
    return data_model, colums_list


#欠損値や標準化などの処理
def data_process(data_model, colums_list):
    #データの標準化（０～１にする）
    scaler = preprocessing.MinMaxScaler()
    data_model[colums_list] = scaler.fit_transform(data_model[colums_list])
    data_model[colums_list] += 0.00001

    #実装に必要なカラムと取得日時のカラムの作成
    data_ex = data_model[colums_list+["measurement_date"]]

    #欠損値処理（欠損値があった場合は落とす）
    data_ex=data_ex.dropna(how='any')
    data_ex=data_ex.reset_index(drop = True)
    data_ex["measurement_date"]=pd.to_datetime(data_ex["measurement_date"], format='%Y/%m/%d %H:%M:%S')

    return data_ex



#ここまでの処理をまとめ一つの関数にする
def data_complete():
    # キャッシュファイルのパスを設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file_ex = os.path.join(current_dir, '../cache/cached_data_ex.joblib')
    cache_file_original = os.path.join(current_dir, '../cache/cached_data_original.joblib')

    # キャッシュファイルが存在すれば、それを読み込む
    if os.path.exists(cache_file_ex) and os.path.exists(cache_file_original):
        print("キャッシュファイルからデータを読み込みます")
        colums_list=["processor_unit_units_status_cpu_board_cpu_fan","processor_unit_units_status_cpu_board_cpu_bd_fan1"
             ,"processor_unit_units_status_cpu_board_cpu_bd_fan2","processor_unit_units_status_cpu_board_cpu_bd_temp"
             ,"processor_unit_units_status_cpu_board_cpu_core_temp","processor_unit_units_status_cpu_board_gpu_core_temp"
             ,"processor_unit_units_status_cpu_board_cpu_core_vol","processor_unit_units_status_cpu_board_cpu_bd_vbat"
             ,"processor_unit_units_status_cpu_board_cpu_bd_p3_3v","processor_unit_units_status_cpu_board_cpu_bd_p5v"
             ,"processor_unit_units_status_cpu_board_cpu_bd_p12v"]
        return colums_list,joblib.load(cache_file_ex), joblib.load(cache_file_original)

    # キャッシュファイルがない場合、初回データ処理を行う
    print("初回データ処理を行います")
    
    #データの読み込み及び加工データの取得
    final_data = load_data()

    #実験に使用するインスタンスのみのデータを抜き取る
    data_original, colums_list = data_model(final_data)

    #標準化や欠損値処理を施す
    data_ex = data_process(data_original,colums_list)

    #originalデータの期間を指定する
    data_original = data_original[data_original["measurement_date"] >= "2017-12-01"]

    #ここまででデータの加工は終了
    #データの保存
    print("データをキャッシュに保存します")
    joblib.dump(data_ex, cache_file_ex)
    joblib.dump(data_original,cache_file_original)

    return colums_list, data_ex, data_original
    