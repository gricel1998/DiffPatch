import numpy as np

filename = "pm25_missing.txt"
data = []

with open(filename, "r") as file:
    lines = file.readlines()

# 跳过第一行（标题行）和删除日期和时间
for line in lines[1:]:
    values = line.strip().split(",")[1:]
    # 尝试将每个值转换为浮点数，若转换失败则跳过该行
    try:
        values = [float(value) for value in values]
        data.append(values)
    except ValueError:
        continue

data = np.array(data)
output_filename = "pm25_data.npy"
np.save(output_filename, data)

print(f"数据已保存为 {output_filename} 文件。")