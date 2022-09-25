import numpy as np
from NaiveSegmentation.naivesegmentation import naive_seg

from training_eval.losses import jaccard_index

thresholdstd=np.arange(2,5,0.5)
thresholdbright=np.arange(100,200,10)
thresholdcol = np.arange(-50,30,10)
n=100
jacs=np.zeros(len(thresholdbright)*len(thresholdcol)*len(thresholdstd))
index=0
jac_best=0
len_ges=len(jacs)*n
count=0
for stds in thresholdstd:
    for brights in thresholdbright:
        for cols in thresholdcol:
            randints=np.random.randint(0,5108,n)
            jacc=np.zeros(n)
            for j in range(n):
                i=randints[j]
                mask=load_one(i)[1]
                mask_pred=naive_seg(i,stds,brights,cols)
                jacc[j]=jaccard_index(mask,mask_pred)
                count+=1
                print(str(count)+"/"+str(len_ges))
            jac=np.average(jacc)
            if jac>jac_best:
                jac_best=jac
                best_comb=[stds,brights,cols]
                print("New best combination: ("+str(stds)+","+str(brights)+","+str(cols)+")")

print(jac_best)
print(best_comb)