import numpy as np
import matplotlib.pyplot as plt

# 初始化四個空的列表
pancreas_list = []
kits_list = []
colon_list = []
lits_list = []
final_pancreas_train = []
final_kits_train = []
final_colon_train = []
final_lits_train = []
final_val = []
final_pancreas_val = []
final_kits_val = []
final_colon_val = []
final_lits_val = []

# 打開並讀取檔案
with open("train_20231013-230005.log", "r") as file:
    next(file)
    for line in file:
        # 檢查每一行是否有關鍵字，並將數值添加到相對應的列表中
        if "pancreas" in line:
            start = line.find("loss:") + 5
            end = line.find(",", start)
            pancreas_list.append(float(line[start:end]))
        elif "kits" in line:
            start = line.find("loss:") + 5
            end = line.find(",", start)
            kits_list.append(float(line[start:end]))
        elif "colon" in line:
            start = line.find("loss:") + 5
            end = line.find(",", start)
            colon_list.append(float(line[start:end]))
        elif "lits" in line:
            start = line.find("loss:") + 5
            end = line.find(",", start)
            lits_list.append(float(line[start:end]))
        elif "Train metrics" in line:
            a = round(np.array(pancreas_list).mean(), 4)
            b = round(np.array(kits_list).mean(), 4)
            c = round(np.array(colon_list).mean(), 4)
            d = round(np.array(lits_list).mean(), 4)
            # print("Total", float(line[line.find(": ") + 1 :]))
            final_pancreas_train.append(1-a)
            final_kits_train.append(1-b)
            final_colon_train.append(1-c)
            final_lits_train.append(1-d)
            pancreas_list = []
            kits_list = []
            colon_list = []
            lits_list = []
        elif "Val metrics" in line:
            a = round(np.array(pancreas_list).mean(), 4)
            b = round(np.array(kits_list).mean(), 4)
            c = round(np.array(colon_list).mean(), 4)
            d = round(np.array(lits_list).mean(), 4)
            final_pancreas_val.append(1-a)
            final_kits_val.append(1-b)
            final_colon_val.append(1-c)
            final_lits_val.append(1-d)
            # print("Total", float(line[line.find(": ") + 1 :]))
            next(file)

plt.figure(figsize=(10,6))
plt.plot(final_kits_train, label='kits train')
plt.plot(final_colon_train, label='colon train')
plt.plot(final_lits_train, label='lits train')
plt.plot(final_pancreas_train, label='pancreas train')
# plt.plot(final_kits_val, label='kits val')
# plt.plot(final_colon_val, label='colon val')
# plt.plot(final_lits_val, label='lits val')
# plt.plot(final_pancreas_val, label='pancreas val')

plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.title('History')
plt.legend()
plt.show()
