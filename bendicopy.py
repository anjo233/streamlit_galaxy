import streamlit as st
import os
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import time
from model import convnext_base as create_model


import requests
proxies = {
  "http": "http://127.0.0.1:49573",
  "https": "http://127.0.0.1:49573",
}
from streamlit_lottie import st_lottie
st.set_page_config(
    page_title="Galaxy Classification App",
    layout="wide")
st.title("Galaxy Classification")


Pre_IMGES_PATH = r'.\ConvNeXt\predict_file'
json_path = r'.\class_indices.json'
model_weight_path = r".\ConvNeXt\weights\best_model-29.pth"
filePath = r'.\images'
data_transform = transforms.Compose([transforms.CenterCrop(224),#中心裁剪图片
                                   transforms.RandomRotation(degrees = 45),
                                   transforms.RandomHorizontalFlip(),#随机水平翻转图片
                                   transforms.ToTensor()
                        
                                   ])



assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)

try_image = os.listdir(filePath)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes = 5).to(device)
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


@st.cache_data
def load_lottieurl(url:str):
    r = requests.get(url,proxies = proxies)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_data
def plot_img_prodict(path,plot_num):
    files_id = os.listdir(path)
    imgid_path = random.sample(files_id,plot_num)

    fig = plt.figure(figsize=(10,5))
    for i in range(plot_num):

        img_path_c = Pre_IMGES_PATH + "\\" + imgid_path[i]
        plt.subplot(1,plot_num,i+1)
        assert os.path.exists(img_path_c), "file: '{}' dose not exist.".format(imgid_path[i])
        img = Image.open(img_path_c)
        plt.imshow(img)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        predict_cla = predict_img(img)

        print_res = "id:{} \n class: {} \n ".format(imgid_path[i],
                                                class_indict[str(predict_cla)]
                                                )

        plt.title(print_res)
        plt.axis(False) 
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    st.pyplot(fig)



@st.cache_data
def imshow_tensor(selection_img):
    img_y = Image.open(selection_img).convert("RGB")
    st.image(img_y)
    img_tensor = data_transform(img_y)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    return img_y,img_tensor

def predict_img(img):
    with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            
    return predict_cla


#============================================================================ 


with st.spinner('模型加载中。。。'):
    time.sleep(5)


galaxy_url = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_34qRI0i4ti.json")
st_lottie(galaxy_url, speed=0.5,height=100,width=100)


df = pd.read_csv("all_output.csv")
class_name_id = ["雪茄状星系","中间星系","圆形星系","侧向星系","漩涡星系"]
CSV_HEADER = [ "Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
id_str_list = []
for i in range(len(CSV_HEADER)):
    dick = {}
    dick["类别代号"] = CSV_HEADER[i]
    dick["英文名称"] = class_indict[str(i)]
    dick["中文名称"] = class_name_id[i]
    id_str_list.append(dick)

#侧边栏
with st.sidebar:
    owners = st.multiselect(
        " 😽选择类别检索：",
        df['R_class'].unique()  
    )
    img_options = st.selectbox(
        " 选择示例图片",
        try_image,index=1
    )
    "----"
    st.markdown(" **没有图片样本？**")
    st.write(" 在这些网站上找找吧。")
    st.markdown("<a href='https://skyserver.sdss.org/dr18/VisualTools/navi'>SDSS巡天数据库</a>" ,unsafe_allow_html=True)
    st.markdown("<a href='https://www.flickr.com/photos/nasawebbtelescope/albums/with/72177720305127361'>NASA发布JWST图像</a>)",unsafe_allow_html=True)
    st.markdown("<a href='https://esahubble.org/images/'>Hubble图像数据库</a>",unsafe_allow_html=True)
#st.write("当前类别是：{}".format(class_name_id[CSV_HEADER.index(owners)]))
if owners:
    df_zz = df[df['R_class'].isin(owners)]
else:
    owners = None
    df_zz = df

st.markdown("## 预测统计")
st.info("该项目数据集来自SDSS巡天数据集的28793张星系图像，用ConvNeXt网络训练，训练集23038张，测试集5755张", icon="ℹ️")

"""**:blue[下面表格展示在测试集上的预测结果]**"""
col_1, col_2, col_3 = st.tabs(["概况", "数目","详细"])

with col_1:
    col_11, col_12 = st.columns(2)
    with col_11:
        st.write("分类结果，{}数目：{}".format(owners if owners != None else '全部' ,len(df_zz)))
        st.write(df_zz.head())
    with col_12:
        st.write("样本")
        plot_img_prodict(Pre_IMGES_PATH,4)
       
with col_2:
    col_111,col_222 = st.columns((1,3))
    with col_111:
        st.write("类别的数目")
        df_class_count = pd.DataFrame(df.groupby(["R_class"]).count()['id'].sort_values(ascending=False))
        df_class_count.columns = ['num']
        st.write(df_class_count)
    with col_222:
        st.write("类别数目统计")
        st.bar_chart(df_class_count ,width = 200)
with col_3:
    st.dataframe(df_zz)

col_21, col_22 = st.columns(2)
with col_21:
    st.markdown("⭐**说明：** **:blue[类别对应关系]**")
    df_id_str = pd.DataFrame(id_str_list)
    edited_df = st.experimental_data_editor(df_id_str)
with col_22:
    st.markdown('⭐表格中的索引：')
    st.markdown('**id**为图片文件名;')
    st.markdown('**R_class**为图片的真实（标签）类别;')
    st.markdown('**P_class**为模型预测的类别;')
    st.markdown('**Prob**表示评估分数（最大为1表示类别的置信度）;')
    st.markdown('**spot_on**取（0，1）1表示正确，0表示预测错误;')
    st.markdown('**后面的5项表示每个类别对应的评估分数**')
"----"

st.markdown("## :arrow_forward:演示示例")
if img_options =='1.txt':
    img_options = 'NGC2841.png'
selection_img = 'D:\\ProgramCode\\PythonFile\\galaxy\\st_galaxy\\images\\'+img_options
st.write("星系：{}".format(img_options[:-4]))
img_y,img_t = imshow_tensor(selection_img)
exceptimg = st.button("执行")
if exceptimg:
    st.write("分类结果")
    latest = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest.text(f'ing.... {i+1}')
        bar.progress(i+1)
        time.sleep(0.01)
    predict_cla = predict_img(img_t)
    st.title("预测类别{}(class_{})".format(class_indict[str(predict_cla)],predict_cla))
"----"

st.markdown("## :arrow_forward:尝试一下")
st.subheader("选择图片导入")
upload_file = st.file_uploader(
    label = "upload image",type=['jpg','png']
)
if upload_file is not None:
    img,img_tensor = imshow_tensor(upload_file) 
    st.markdown("### 开始预测")
    predict = st.button("start") 
    if predict:
        predict_cla = predict_img(img_tensor)
        st.title("预测类别{}(class_{})".format(class_indict[str(predict_cla)],predict_cla))
else:
    st.stop()


