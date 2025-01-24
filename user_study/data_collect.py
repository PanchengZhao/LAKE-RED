import pandas as pd
import json,sys,os
import shutil


expname=['AB','CI','AdaIN','DCI','LCGNet','TFill','LDM_base','LDM_Repaint','Ours']
expcount={}
for i in expname:
    expcount[i]=[0,0,0]
imgnames = [
'COD_COD10K_COD10K-CAM-4-Amphibian-67-Frog-4784.jpg',
'COD_COD10K_COD10K-CAM-1-Aquatic-15-SeaHorse-1110.jpg',
'COD_COD10K_COD10K-CAM-2-Terrestrial-28-Deer-1762.jpg',
'COD_CAMO_camourflage_00018.jpg',
'COD_CAMO_camourflage_00061.jpg',
'COD_CAMO_camourflage_00064.jpg',
'COD_CAMO_camourflage_00090.jpg',
'COD_CAMO_camourflage_00098.jpg',
'COD_CAMO_camourflage_00122.jpg',
'COD_CAMO_camourflage_00124.jpg',
'COD_CAMO_camourflage_00145.jpg',
'COD_CAMO_camourflage_00160.jpg',
'COD_CAMO_camourflage_00207.jpg',
'COD_CAMO_camourflage_00208.jpg',
'COD_CAMO_camourflage_00422.jpg',
'COD_CAMO_camourflage_00444.jpg',
'COD_CAMO_camourflage_00515.jpg',
'COD_CAMO_camourflage_00688.jpg',
'COD_NC4K_1068.jpg',
'COD_NC4K_1371.jpg']

f=open('./img_seq.json', 'r')
back_dict = json.load(f)
imgnametag=0

df = pd.read_excel('./user_study.xlsx')
for j in range(1,61,3): # for qustion 1
# for j in range(2,62,3): # for qustion 2
# for j in range(3,63,3): # for qustion 3        
    column_data = df.iloc[:, j].tolist()
    for i in column_data:
        expcount[(back_dict[(imgnames[imgnametag][:-4]+'.png')][(i//100)-1])][0]+=1
        expcount[back_dict[(imgnames[imgnametag][:-4]+'.png')][((i%100)//10)-1]][1]+=1
        expcount[back_dict[(imgnames[imgnametag][:-4]+'.png')][((i%10))-1]][2]+=1
    imgnametag+=1
print(expcount)