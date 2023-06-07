import numpy as np

for train_size in [10, 15, 20, 25, 30]:
    result_data_path = "data_division_" + str(train_size) + "+100\\bpnn_output_ex1_ddfo_mae_rmse_mape_20_division_" + str(train_size) + "+100_vsg3a2.csv"
    with open(result_data_path, encoding='utf-8') as f:  # 打开文件
        # 读取方式float类型，逗号间隔，跳过首行，
        data = np.loadtxt(f, float, delimiter=',', skiprows=0)
    result_data = np.array([list(row) for row in data])
    print("========================"+str(train_size)+"===================================")
    print(str(train_size)+"mae均值："+str(np.mean(result_data[0])))
    print(str(train_size)+"rmse均值："+str(np.mean(result_data[1])))
    print(str(train_size)+"mape均值："+str(np.mean(result_data[2])))


