import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage
from training_eval.losses import jaccard_index

#function for computing the three features brightness, color and standard deviation
def feature_computing(image,bool_array,index):
    img_gray = np.array(Image.open("images/"+str(index)+".jpg").convert("L"))
    
    #use a filter to blur small noise
    img_gray_avg = ndimage.uniform_filter(img_gray,size=4,mode="mirror")

    single_segment_col=image[bool_array]

    #compute color differences
    avg_red=np.average(single_segment_col[:,0])
    avg_gr=np.average(single_segment_col[:,1])        
    avg_blue=np.average(single_segment_col[:,2])
    diff=min(avg_gr-avg_red,avg_gr-avg_blue)

    #compute std and brightness
    single_segment=img_gray_avg[bool_array]
    std=np.std(single_segment)
    brightness = np.average(single_segment)

    return std, brightness, diff

def predictor(i):
    im_directory = "images/"
    f_im = im_directory + str(i) + ".jpg"
    im=Image.open(f_im)
    img = np.array(im.convert("RGB"))

    #segmentate the image using felzenszwalb algorithm
    segments = felzenszwalb(img, scale=150, sigma=0.5, min_size=50)

    mask_pred_logreg=np.zeros((256,256))
    mask_pred_qda=np.zeros((256,256))
    mask_pred_lda=np.zeros((256,256))
    mask_pred_svc = np.zeros((256,256))
    mask_pred_rf = np.zeros((256,256))

    #open the pre-trained models
    logregmod = pickle.load(open('logreg_model.sav', 'rb'))
    qdamod = pickle.load(open('qda_model.sav', 'rb'))
    ldamod = pickle.load(open('lda_model.sav', 'rb'))
    svcmod = pickle.load(open('svc.sav','rb'))
    randomforestmod = pickle.load(open('randomforest_model.sav','rb'))

    #for each image segment, predict the label 'is forest' or 'is not forest'
    for j in range(len(np.unique(segments))):
        bool_array = segments==j
        features=np.array(feature_computing(img,bool_array,i)).reshape(1,3)

        label_pred_logreg=logregmod.predict(features)
        label_pred_qda = qdamod.predict(features)
        label_pred_lda= ldamod.predict(features)
        label_pred_svc = svcmod.predict(features)
        label_pred_rf = randomforestmod.predict(features)

        indices=np.where(bool_array!=0)
        mask_pred_logreg[indices]=label_pred_logreg*255
        mask_pred_qda[indices]=label_pred_qda*255
        mask_pred_lda[indices]=label_pred_lda*255
        mask_pred_svc[indices]=label_pred_svc*255
        mask_pred_rf[indices]=label_pred_rf*255
    
    mask_pred_logreg=np.uint8(mask_pred_logreg)
    mask_pred_qda=np.uint8(mask_pred_qda)
    mask_pred_lda=np.uint8(mask_pred_lda)
    mask_pred_svc=np.uint8(mask_pred_svc)
    mask_pred_rf=np.uint8(mask_pred_rf)

    return mask_pred_logreg,mask_pred_qda,mask_pred_lda,mask_pred_svc,mask_pred_rf

def model_analysis():
    #prepare the data set, labels, indices and weights
    data_set=np.load("data.npy")
    labels=np.load("labels.npy")
    test_indices = np.load("test_indices.npy")
    weights=np.load("weights.npy")

    print("Loaded data set")
    m=len(test_indices)

    #train the models
    logreg = LogisticRegression()
    logreg.fit(data_set, labels,weights)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(data_set, labels)

    lda = LinearDiscriminantAnalysis()
    lda.fit(data_set, labels)

    svc = svm.SVC()
    svc.fit(data_set,labels,weights)

    randomforest=RandomForestClassifier()
    randomforest.fit(data_set,labels,weights)

    print("Finished fitting the models")

    #save the models
    pickle.dump(logreg, open('logreg_model.sav', 'wb'))
    pickle.dump(qda, open('qda_model.sav', 'wb'))
    pickle.dump(lda, open('lda_model.sav', 'wb'))
    pickle.dump(randomforest,open('randomforest_model.sav','wb'))
    pickle.dump(svc,open('svc.sav','wb'))


    jacs_logreg=np.zeros(m)
    jacs_qda = np.zeros(m)
    jacs_lda = np.zeros(m)
    jacs_svm=np.zeros(m)
    jacs_rand_forest=np.zeros(m)

    #evaluate the models using jaccard index
    for index in range(m):
        i = test_indices[index]
        preds=predictor(i)
        mask_pred_logreg=preds[0]
        mask_pred_qda=preds[1]
        mask_pred_lda=preds[2]
        mask_pred_svc=preds[3]
        mask_pred_rf=preds[4]

        mask_directory = "masks/"
        f_mask = mask_directory + str(i) + ".jpg"
        mask=Image.open(f_mask)
        mask_data=np.around(np.array(mask.convert("L"))/255)*255
        mask_true=np.uint8(mask_data)

        jacs_logreg[index]=jaccard_index(mask_pred_logreg,mask_true)
        jacs_qda[index]=jaccard_index(mask_pred_qda,mask_true)
        jacs_lda[index]=jaccard_index(mask_pred_lda,mask_true)
        jacs_svm[index]=jaccard_index(mask_pred_svc,mask_true)
        jacs_rand_forest[index]=jaccard_index(mask_pred_rf,mask_true)

        print("Prediction "+ str(index)+"/"+str(m))
    
    avg_jac_logreg=np.average(jacs_logreg)
    avg_jac_qda = np.average(jacs_qda)
    avg_jac_lda = np.average(jacs_lda)
    avg_jac_svm = np.average(jacs_svm)
    avg_jac_rf = np.average(jacs_rand_forest)

    print("Jaccard index of logistic regression: "+str(avg_jac_logreg))
    print("Jaccard index of qda: "+str(avg_jac_qda))
    print("Jaccard index of lda: "+str(avg_jac_lda))
    print("Jaccard index of svm: "+str(avg_jac_svm))
    print("Jaccard index of rf: "+str(avg_jac_rf))