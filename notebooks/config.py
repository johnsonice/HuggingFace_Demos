import os 

a = 'test'
data_folder1= "/media/chengyu/Elements1/HuggingFace"
data_folder2= "../../All_Data/HuggingFace"

if os.path.exists(data_folder1):
    data_folder = data_folder1
elif os.path.exists(data_folder2):
    data_folder = data_folder2