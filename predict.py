import os
import json
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import random

from model import convnext_base as create_model

plot_num=4 #随机选择数目
TEST_IMGES_PATH = r'D:\ProgramCode\PythonFile\galaxy\efficientnetv2\Galaxy_class\Train_images'
Pre_IMGES_PATH = r'D:\ProgramCode\PythonFile\galaxy\ConvNeXt\predict_file'
json_path = r'D:\ProgramCode\PythonFile\galaxy\ConvNeXt\class_indices.json'
model_weight_path = r"D:\ProgramCode\PythonFile\galaxy\ConvNeXt\weights\best_model-29.pth"
TEST_ID_ClASS = r'D:\ProgramCode\PythonFile\galaxy\ConvNeXt\Test_name.csv'
num_classes = 5
img_size = 224
data_transform = transforms.Compose([transforms.CenterCrop(img_size),#中心裁剪图片
                                   transforms.RandomRotation(degrees = 30),
                                   transforms.RandomHorizontalFlip(),#随机水平翻转图片
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

images_messgae = []

test_solutions = pd.read_csv(TEST_ID_ClASS)
test_id_list = test_solutions.iloc[:,0]
test_class_list = test_solutions.iloc[:,1]




# read class_indict
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)


# create model
model = create_model(num_classes=num_classes).to(device)
# load model weights

model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


def plot_img_prodict(path):
    files_id = os.listdir(path)
    imgid_path = random.sample(files_id,plot_num)

    fig = plt.figure(figsize=(16,5))
    for i in range(plot_num):

        img_path_c = Pre_IMGES_PATH + "\\" + imgid_path[i]
        plt.subplot(1,plot_num,i+1)
        assert os.path.exists(img_path_c), "file: '{}' dose not exist.".format(imgid_path[i])
        img = Image.open(img_path_c)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        
        with torch.no_grad():
        # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        
        print_res = "id:{} \n class: {} \n  prob: {:.3}".format(imgid_path[i],
                                                class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())

        plt.title(print_res)
        #print(imgid_path[i])
        '''for i in range(len(predict)):
            print("class: {:10}    prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))'''
        plt.axis(False) 

        
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    #plt.show()
    return fig 



            

        

if __name__ == '__main__':
    plot_img_prodict(Pre_IMGES_PATH)
    