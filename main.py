import cv2
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import warnings
from numba import jit
import argparse

warnings.filterwarnings("ignore")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="バイラテラルフィルタを用いたモアレ風画像を作成するプログラム")
    parser.add_argument("--fn",required=True)
    parser.add_argument("--T1")
    parser.add_argument("--T2")
    args = parser.parse_args()

    fn=args.fn
    img=cv2.imread(fn,0)
    
    # バイラテラルフィルタ反復回数
    T1=20

    # 反バイラテラルフィルタ反復回数
    T2=40

    alpha=0.001
    beta=0.01
    w=10
    
    bilaterallist=[]
    antibilaterallist=[]
    
    bilaterallist.append(img)
    
    for p in tqdm(range(1,T1+1)):
        bilateral=np.zeros((img.shape[0],img.shape[1]))
        pbar1=tqdm(total=img.shape[0])
        
        
        for j in range(img.shape[0]):
            pbar1.update(1)
            for i in range(img.shape[1]):
                mole=0
                deno=0
                for l in range(j-w,j+w+1):
                    for k in range(i-w,i+w+1):
                        if i-k>511 or i-k<0 or j-l>511 or j-l<0:
                            continue
                        moletmp=np.exp(-1*alpha*((i-k)**2+(j-l)**2)-beta*(bilaterallist[p-1][j][i]-bilaterallist[p-1][l][k])**2)*bilaterallist[p-1][l][k]
                        mole=mole+moletmp
                        denotmp=np.exp(-1*alpha*((i-k)**2+(j-l)**2)-beta*(bilaterallist[p-1][j][i]-bilaterallist[p-1][l][k])**2)
                        deno=deno+denotmp
                bilateral[j][i]=mole/deno
        bilaterallist.append(bilateral)
        pbar1.close()
    
    #print(bilaterallist[T1])
    plt.imsave("test3.jpg",bilaterallist[T1],cmap='Greys_r')

    
    antibilaterallist.append(bilaterallist[T1])

    for q in tqdm(range(1,T2+1)):
        antibilateral=np.zeros((img.shape[0],img.shape[1]))
        pbar2=tqdm(total=img.shape[0])
        
        for j in range(img.shape[0]):
            pbar2.update(1)
            for i in range(img.shape[1]):
                mole=0
                deno=0
                for l in range(j-w,j+w+1):
                    for k in range(i-w,i+w+1):
                        if i-k>511 or i-k<0 or j-l>511 or j-l<0:
                            continue
                        moletmp=np.exp(-1*alpha*((i-k)**2+(j-l)**2)-beta*(antibilaterallist[q-1][j][i]-antibilaterallist[q-1][l][k])**2)*antibilaterallist[q-1][l][k]
                        mole=mole+moletmp
                        denotmp=np.exp(-1*alpha*((i-k)**2+(j-l)**2)-beta*(antibilaterallist[q-1][j][i]-antibilaterallist[q-1][l][k])**2)
                        deno=deno+denotmp
                antibilateral[j][i]=2*antibilaterallist[q-1][j][i]-mole/deno
        antibilaterallist.append(antibilateral)
    plt.imsave("test4.jpg",antibilaterallist[T2],cmap='Greys_r')

    
