from sklearn.metrics import precision_recall_curve, average_precision_score
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
def compute_auprc(y_pred, y):
    y_pred = y_pred.flatten()
    y = y.flatten()
    precisions, recalls, thresholds = precision_recall_curve(y.astype(int), y_pred)
    auprc = average_precision_score(y.astype(int), y_pred)
    return auprc, precisions, recalls, thresholds
def plot_roc_curve(experiment:None,mean_std:None):

#Alzheimer

    healthy_loss_morphed_array= np.load("./results/"+experiment+"/healthy_loss_morphed.npy")
    healthy_loss_prior_array= np.load("./results/"+experiment+"/healthy_loss_prior.npy")
    alzheimer_loss_morphed_array= np.load("./results/"+experiment+"/alzheimer_loss_morphed.npy")
    alzheimer_loss_prior_array= np.load("./results/"+experiment+"/alzheimer_loss_prior.npy")
    alzheimer_stdlogjacdet_array= np.load("./results/"+experiment+"/alzheimer_stdlogjacdet_array.npy")
    healthy_stdlogjacdet_array= np.load("./results/"+experiment+"/healthy_stdlogjacdet.npy")

    healthy_loss_jacdet_array= np.load("./results/"+experiment+"/healthy_loss_jacdet.npy")
    healthy_loss_logjacdet_array= np.load("./results/"+experiment+"/healthy_loss_logjacdet.npy")
    healthy_norm_loss_jacdet_array= np.load("./results/"+experiment+"/healthy_norm_loss_jacdet_array.npy")
    healthy_norm_loss_logjacdet_array= np.load("./results/"+experiment+"/healthy_norm_loss_logjacdet_array.npy")

    alzheimer_loss_jacdet_array= np.load("./results/"+experiment+"/alzheimer_loss_jacdet.npy")
    alzheimer_loss_logjacdet_array= np.load("./results/"+experiment+"/alzheimer_loss_logjacdet.npy")
    alzheimer_norm_loss_jacdet_array= np.load("./results/"+experiment+"/alzheimer_norm_loss_jacdet_array.npy")
    alzheimer_norm_loss_logjacdet_array= np.load("./results/"+experiment+"/alzheimer_norm_loss_logjacdet_array.npy")
   # healthy_loss_morphed_array= [0.1,0.2,0.6]
    #healthy_loss_prior_array= [0.5,0.8,0.3]
   # alzheimer_loss_morphed_array= [0.4,0.6,0.8]
    #alzheimer_loss_prior_array= [0.123,0.7,0.5]

    y_morphed_healthy=np.zeros(len(healthy_loss_morphed_array))
    y_morphed_alzheimer=np.ones(len(alzheimer_loss_morphed_array))
    y_prior_healthy=np.zeros(len(healthy_loss_prior_array))
    y_prior_alzheimer=np.ones(len(alzheimer_loss_prior_array))

    y_morphed=np.concatenate((y_morphed_healthy,y_morphed_alzheimer))
    y_prior=np.concatenate((y_prior_healthy,y_prior_alzheimer))

    x_morphed=np.concatenate((healthy_loss_morphed_array,alzheimer_loss_morphed_array))
    x_prior=np.concatenate((healthy_loss_prior_array,alzheimer_loss_prior_array))
    x_jacdet=np.concatenate((healthy_stdlogjacdet_array,alzheimer_stdlogjacdet_array))

    x_loss_jacdet=np.concatenate((healthy_loss_jacdet_array,alzheimer_loss_jacdet_array))
    x_loss_logjacdet=np.concatenate((healthy_loss_logjacdet_array,alzheimer_loss_logjacdet_array))
    x_norm_loss_jacdet=np.concatenate((healthy_norm_loss_jacdet_array,alzheimer_norm_loss_jacdet_array))
    x_norm_loss_logjacdet=np.concatenate((healthy_norm_loss_logjacdet_array,alzheimer_norm_loss_logjacdet_array))
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["aqua", "darkorange","black","yellow","red","gray"]

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_morphed,
            name=f"ROC curve for Morphed",
            color=colors[0],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_prior,
            name=f"ROC curve for Prior",
             color=colors[1],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_jacdet,
            name=f"ROC curve for StdLogJacDet",
             color=colors[2],
            ax=ax,
        )
    x_jacdet_norm = (x_jacdet-np.min(x_jacdet))/(np.max(x_jacdet)-np.min(x_jacdet))
    x_morphed_norm = (x_morphed-np.min(x_morphed))/(np.max(x_morphed)-np.min(x_morphed))
    x_prior_norm = (x_prior-np.min(x_prior))/(np.max(x_prior)-np.min(x_prior))

    
    RocCurveDisplay.from_predictions(
            y_prior,
            x_loss_jacdet,
            name=f"ROC curve for Loss*Jacdet",
             color=colors[3],
            ax=ax,
        )

    RocCurveDisplay.from_predictions(
            y_prior,
            x_loss_logjacdet,
            name=f"ROC curve for Loss*LogJacdet",
             color=colors[5],
            ax=ax,
        ),
    
    RocCurveDisplay.from_predictions(
            y_prior,
            x_norm_loss_jacdet,
            name=f"ROC curve for Norm Loss*Jacdet",
             color=colors[3],
            ax=ax,
        )

    RocCurveDisplay.from_predictions(
            y_prior,
            x_norm_loss_logjacdet,
            name=f"ROC curve for Norm Loss*LogJacDet",
             color=colors[5],
            ax=ax,
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Alzheimer vs Healthy")
    plt.legend()
    plt.show()
    plt.savefig("./results/"+experiment+"/roc_curve_ad_vs_healthy_new.png")


# MCI
    healthy_loss_morphed_array= np.load("./results/"+experiment+"/healthy_loss_morphed.npy")
    healthy_loss_prior_array= np.load("./results/"+experiment+"/healthy_loss_prior.npy")
    mci_loss_morphed_array= np.load("./results/"+experiment+"/mci_loss_morphed.npy")
    mci_loss_prior_array= np.load("./results/"+experiment+"/mci_loss_prior.npy")
    mci_stdlogjacdet_array= np.load("./results/"+experiment+"/mci_stdlogjacdet_array.npy")
    healthy_stdlogjacdet_array= np.load("./results/"+experiment+"/healthy_stdlogjacdet.npy")
    mci_meanjacdet_array= np.load("./results/"+experiment+"/mci_meanjacdet_array.npy")
    healthy_meanjacdet_array= np.load("./results/"+experiment+"/healthy_meanjacdet.npy")
   # healthy_loss_morphed_array= [0.1,0.2,0.6]
    #healthy_loss_prior_array= [0.5,0.8,0.3]
   # alzheimer_loss_morphed_array= [0.4,0.6,0.8]
    #alzheimer_loss_prior_array= [0.123,0.7,0.5]

    mci_loss_jacdet_array= np.load("./results/"+experiment+"/mci_loss_jacdet.npy")
    mci_loss_logjacdet_array= np.load("./results/"+experiment+"/mci_loss_logjacdet.npy")
    mci_norm_loss_jacdet_array= np.load("./results/"+experiment+"/mci_norm_loss_jacdet_array.npy")
    mci_norm_loss_logjacdet_array= np.load("./results/"+experiment+"/mci_norm_loss_logjacdet_array.npy")

    y_morphed_healthy=np.zeros(len(healthy_loss_morphed_array))
    y_morphed_mci=np.ones(len(mci_loss_morphed_array))
    y_prior_healthy=np.zeros(len(healthy_loss_prior_array))
    y_prior_mci=np.ones(len(mci_loss_prior_array))

    y_morphed=np.concatenate((y_morphed_healthy,y_morphed_mci))
    y_prior=np.concatenate((y_prior_healthy,y_prior_mci))

    x_morphed=np.concatenate((healthy_loss_morphed_array,mci_loss_morphed_array))
    x_prior=np.concatenate((healthy_loss_prior_array,mci_loss_prior_array))
    x_jacdet=np.concatenate((healthy_stdlogjacdet_array,mci_stdlogjacdet_array))
    x_jacdet_mean=np.concatenate((healthy_meanjacdet_array,mci_meanjacdet_array))

    x_loss_jacdet=np.concatenate((healthy_loss_jacdet_array,mci_loss_jacdet_array))
    x_loss_logjacdet=np.concatenate((healthy_loss_logjacdet_array,mci_loss_logjacdet_array))
    x_norm_loss_jacdet=np.concatenate((healthy_norm_loss_jacdet_array,mci_norm_loss_jacdet_array))
    x_norm_loss_logjacdet=np.concatenate((healthy_norm_loss_logjacdet_array,mci_norm_loss_logjacdet_array))

    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["aqua", "darkorange","black","yellow","red","gray"]

    x_jacdet_norm = (x_jacdet-np.min(x_jacdet))/(np.max(x_jacdet)-np.min(x_jacdet))
    x_morphed_norm = (x_morphed-np.min(x_morphed))/(np.max(x_morphed)-np.min(x_morphed))
    x_mean_norm = (x_jacdet_mean-np.min(x_jacdet_mean))/(np.max(x_jacdet_mean)-np.min(x_jacdet_mean))
    x_prior_norm = (x_prior-np.min(x_prior))/(np.max(x_prior)-np.min(x_prior))

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_morphed_norm,
            name=f"ROC curve for Morphed",
            color=colors[0],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_prior_norm,
            name=f"ROC curve for Prior",
             color=colors[1],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_jacdet_norm,
            name=f"ROC curve for StdLogJacDet",
             color=colors[2],
            ax=ax,
        )


    
    RocCurveDisplay.from_predictions(
            y_prior,
            x_loss_jacdet,
            name=f"ROC curve for Loss*Jacdet",
             color=colors[3],
            ax=ax,
        )

    RocCurveDisplay.from_predictions(
            y_prior,
            x_loss_logjacdet,
            name=f"ROC curve for Loss*LogJacdet",
             color=colors[5],
            ax=ax,
        ),
    
    RocCurveDisplay.from_predictions(
            y_prior,
            x_norm_loss_jacdet,
            name=f"ROC curve for Norm Loss*Jacdet",
             color=colors[3],
            ax=ax,
        )

    RocCurveDisplay.from_predictions(
            y_prior,
            x_norm_loss_logjacdet,
            name=f"ROC curve for Norm Loss*LogJacDet",
             color=colors[5],
            ax=ax,
        )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MCI vs Healthy")
    plt.legend()
    plt.show()
    plt.savefig("./results/"+experiment+"/roc_curve_mci_vs_healthy_new.png")
     

    if mean_std==True:
        fig, ax1 = plt.subplots(figsize=(6, 6))
        print(np.mean(healthy_loss_morphed_array))
        print(np.std(healthy_loss_morphed_array))
        print(np.mean(alzheimer_loss_morphed_array))
        print(np.std(alzheimer_loss_morphed_array))
        print(np.mean(mci_loss_morphed_array))
        print(np.std(mci_loss_morphed_array))
        ax1.errorbar(x=[np.mean(healthy_loss_morphed_array),np.mean(mci_loss_morphed_array),np.mean(alzheimer_loss_morphed_array)],y=["Healthy","MCI","Alzheimer"], xerr=[np.std(healthy_loss_morphed_array),np.std(mci_loss_morphed_array),np.std(alzheimer_loss_morphed_array)], fmt='o')

        ax1.set_title('MSE Loss')
        plt.show()
        plt.savefig("./results/"+experiment+"/mean_std.png")


#plot_roc_curve("b1_from_scratch",True)

def plot_roc_curve_no_deformer(experiment:None,mean_std:None):

#Alzheimer

    healthy_loss_morphed_array= np.load("./results/"+experiment+"/healthy_loss_morphed.npy")
    healthy_loss_prior_array= np.load("./results/"+experiment+"/healthy_loss_prior.npy")
    alzheimer_loss_morphed_array= np.load("./results/"+experiment+"/alzheimer_loss_morphed.npy")
    alzheimer_loss_prior_array= np.load("./results/"+experiment+"/alzheimer_loss_prior.npy")

   # healthy_loss_morphed_array= [0.1,0.2,0.6]
    #healthy_loss_prior_array= [0.5,0.8,0.3]
   # alzheimer_loss_morphed_array= [0.4,0.6,0.8]
    #alzheimer_loss_prior_array= [0.123,0.7,0.5]

    y_morphed_healthy=np.zeros(len(healthy_loss_morphed_array))
    y_morphed_alzheimer=np.ones(len(alzheimer_loss_morphed_array))
    y_prior_healthy=np.zeros(len(healthy_loss_prior_array))
    y_prior_alzheimer=np.ones(len(alzheimer_loss_prior_array))

    y_morphed=np.concatenate((y_morphed_healthy,y_morphed_alzheimer))
    y_prior=np.concatenate((y_prior_healthy,y_prior_alzheimer))

    x_morphed=np.concatenate((healthy_loss_morphed_array,alzheimer_loss_morphed_array))
    x_prior=np.concatenate((healthy_loss_prior_array,alzheimer_loss_prior_array))


    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["aqua", "darkorange","black","yellow","red","gray"]

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_morphed,
            name=f"ROC curve for AE-KL",
            color=colors[0],
            ax=ax,
        )



    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Alzheimer vs Healthy")
    plt.legend()
    plt.show()
    plt.savefig("./results/"+experiment+"/roc_curve_ad_vs_healthy.png")


# MCI
    healthy_loss_morphed_array= np.load("./results/"+experiment+"/healthy_loss_morphed.npy")
    healthy_loss_prior_array= np.load("./results/"+experiment+"/healthy_loss_prior.npy")
    alzheimer_loss_morphed_array= np.load("./results/"+experiment+"/mci_loss_morphed.npy")
    alzheimer_loss_prior_array= np.load("./results/"+experiment+"/mci_loss_prior.npy")

   # healthy_loss_morphed_array= [0.1,0.2,0.6]
    #healthy_loss_prior_array= [0.5,0.8,0.3]
   # alzheimer_loss_morphed_array= [0.4,0.6,0.8]
    #alzheimer_loss_prior_array= [0.123,0.7,0.5]

    y_morphed_healthy=np.zeros(len(healthy_loss_morphed_array))
    y_morphed_alzheimer=np.ones(len(alzheimer_loss_morphed_array))
    y_prior_healthy=np.zeros(len(healthy_loss_prior_array))
    y_prior_alzheimer=np.ones(len(alzheimer_loss_prior_array))

    y_morphed=np.concatenate((y_morphed_healthy,y_morphed_alzheimer))
    y_prior=np.concatenate((y_prior_healthy,y_prior_alzheimer))

    x_morphed=np.concatenate((healthy_loss_morphed_array,alzheimer_loss_morphed_array))
    x_prior=np.concatenate((healthy_loss_prior_array,alzheimer_loss_prior_array))


    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["aqua", "darkorange","black","yellow","red","gray"]

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_morphed,
            name=f"ROC curve for AE-KL",
            color=colors[0],
            ax=ax,
        )


    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MCI vs Healthy")
    plt.legend()
    plt.show()
    plt.savefig("./results/"+experiment+"/roc_curve_mci_vs_healthy.png")
 
    if mean_std==True:
        fig, ax1 = plt.subplots(figsize=(6, 6))
        print(np.mean(healthy_loss_morphed_array))
        print(np.std(healthy_loss_morphed_array))
        print(np.mean(alzheimer_loss_morphed_array))
        print(np.std(alzheimer_loss_morphed_array))
        
        ax1.errorbar(x=[np.mean(healthy_loss_morphed_array),np.mean(alzheimer_loss_morphed_array)],y=["Healthy","Alzheimer"], xerr=[np.std(healthy_loss_morphed_array),np.std(alzheimer_loss_morphed_array)], fmt='o')

        ax1.set_title('variable, asymmetric error')
        plt.show()
        plt.savefig("./results/"+experiment+"/mean_std.png")
def plot_roc_curve_2heads(experiment:None,mean_std:None):

#Alzheimer

    healthy_loss_morphed_array= np.load("./results/"+experiment+"/healthy_loss_morphed.npy")
    healthy_loss_prior_array= np.load("./results/"+experiment+"/healthy_loss_prior.npy")
    alzheimer_loss_morphed_array= np.load("./results/"+experiment+"/alzheimer_loss_morphed.npy")
    alzheimer_loss_prior_array= np.load("./results/"+experiment+"/alzheimer_loss_prior.npy")
    alzheimer_stdlogjacdet_array= np.load("./results/"+experiment+"/alzheimer_stdlogjacdet_array.npy")
    healthy_stdlogjacdet_array= np.load("./results/"+experiment+"/healthy_stdlogjacdet.npy")
    alzheimer_meanjacdet_array= np.load("./results/"+experiment+"/alzheimer_meanjacdet_array.npy")
    healthy_meanjacdet_array= np.load("./results/"+experiment+"/healthy_meanjacdet.npy")

    alzheimer_loss_morphed_array_b1= np.load("./results/"+experiment+"/alzheimer_loss_morphed_b1.npy")
    alzheimer_stdlogjacdet_array_b1= np.load("./results/"+experiment+"/alzheimer_stdlogjacdet_array_b1.npy")
    
    alzheimer_loss_morphed_array_b01= np.load("./results/"+experiment+"/alzheimer_loss_morphed_b01.npy")
    alzheimer_stdlogjacdet_array_b01= np.load("./results/"+experiment+"/alzheimer_stdlogjacdet_array_b01.npy")

    y_morphed_healthy=np.zeros(len(healthy_loss_morphed_array))
    y_morphed_alzheimer=np.ones(len(alzheimer_loss_morphed_array))
    y_prior_healthy=np.zeros(len(healthy_loss_prior_array))
    y_prior_alzheimer=np.ones(len(alzheimer_loss_prior_array))

    y_morphed=np.concatenate((y_morphed_healthy,y_morphed_alzheimer))
    y_prior=np.concatenate((y_prior_healthy,y_prior_alzheimer))

    x_morphed=np.concatenate((healthy_loss_morphed_array,alzheimer_loss_morphed_array))
    x_morphed_b1=np.concatenate((healthy_loss_morphed_array,alzheimer_loss_morphed_array_b1))
    x_morphed_b01=np.concatenate((healthy_loss_morphed_array,alzheimer_loss_morphed_array_b01))
    x_prior=np.concatenate((healthy_loss_prior_array,alzheimer_loss_prior_array))
    x_jacdet=np.concatenate((healthy_stdlogjacdet_array,alzheimer_stdlogjacdet_array))
    x_jacdet_b1=np.concatenate((healthy_stdlogjacdet_array,alzheimer_stdlogjacdet_array_b1))
    x_jacdet_b01=np.concatenate((healthy_stdlogjacdet_array,alzheimer_stdlogjacdet_array_b01))

    x_jacdet_mean=np.concatenate((healthy_meanjacdet_array,alzheimer_meanjacdet_array))

    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["aqua", "darkorange","black","yellow","red","gray"]

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_morphed,
            name=f"ROC curve for Morphed",
            color=colors[0],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_prior,
            name=f"ROC curve for Prior",
             color=colors[1],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_jacdet,
            name=f"ROC curve for StdLogJacDet",
             color=colors[2],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_morphed_b1,
            name=f"ROC curve for Morphed_B1",
             color=colors[1],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_jacdet_b1,
            name=f"ROC curve for StdLogJacDet_B1",
             color=colors[2],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_morphed_b01,
            name=f"ROC curve for Morphed_B01",
             color=colors[1],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_jacdet_b01,
            name=f"ROC curve for StdLogJacDetB_01",
             color=colors[2],
            ax=ax,
        )
    x_jacdet_norm = (x_jacdet-np.min(x_jacdet))/(np.max(x_jacdet)-np.min(x_jacdet))
    x_morphed_norm = (x_morphed-np.min(x_morphed))/(np.max(x_morphed)-np.min(x_morphed))
    x_mean_norm = (x_jacdet_mean-np.min(x_jacdet_mean))/(np.max(x_jacdet_mean)-np.min(x_jacdet_mean))

    
    RocCurveDisplay.from_predictions(
            y_prior,
            x_jacdet_norm*x_morphed_norm,
            name=f"ROC curve for StdLogJacDet*Morphed",
             color=colors[3],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_mean_norm*x_morphed_norm,
            name=f"ROC curve for MeanJacDet*Morphed",
             color=colors[4],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_mean_norm,
            name=f"ROC curve for MeanJacDet",
             color=colors[5],
            ax=ax,
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Alzheimer vs Healthy")
    plt.legend()
    plt.show()
    plt.savefig("./results/"+experiment+"/roc_curve_ad_vs_healthy.png")


# MCI
    healthy_loss_morphed_array= np.load("./results/"+experiment+"/healthy_loss_morphed.npy")
    healthy_loss_prior_array= np.load("./results/"+experiment+"/healthy_loss_prior.npy")
    mci_loss_morphed_array= np.load("./results/"+experiment+"/mci_loss_morphed.npy")
    mci_loss_prior_array= np.load("./results/"+experiment+"/mci_loss_prior.npy")
    mci_stdlogjacdet_array= np.load("./results/"+experiment+"/mci_stdlogjacdet_array.npy")
    healthy_stdlogjacdet_array= np.load("./results/"+experiment+"/healthy_stdlogjacdet.npy")
    mci_meanjacdet_array= np.load("./results/"+experiment+"/mci_meanjacdet_array.npy")
    healthy_meanjacdet_array= np.load("./results/"+experiment+"/healthy_meanjacdet.npy")

    mci_loss_morphed_array_b1= np.load("./results/"+experiment+"/mci_loss_morphed_b1.npy")
    mci_stdlogjacdet_array_b1= np.load("./results/"+experiment+"/mci_stdlogjacdet_b1.npy")
    
    mci_loss_morphed_array_b01= np.load("./results/"+experiment+"/mci_loss_morphed_b01.npy")
    mci_stdlogjacdet_array_b01= np.load("./results/"+experiment+"/mci_stdlogjacdet_b01.npy")
   # healthy_loss_morphed_array= [0.1,0.2,0.6]
    #healthy_loss_prior_array= [0.5,0.8,0.3]
   # alzheimer_loss_morphed_array= [0.4,0.6,0.8]
    #alzheimer_loss_prior_array= [0.123,0.7,0.5]

    y_morphed_healthy=np.zeros(len(healthy_loss_morphed_array))
    y_morphed_mci=np.ones(len(mci_loss_morphed_array))
    y_prior_healthy=np.zeros(len(healthy_loss_prior_array))
    y_prior_mci=np.ones(len(mci_loss_prior_array))

    y_morphed=np.concatenate((y_morphed_healthy,y_morphed_mci))
    y_prior=np.concatenate((y_prior_healthy,y_prior_mci))

    x_morphed=np.concatenate((healthy_loss_morphed_array,mci_loss_morphed_array))
    x_prior=np.concatenate((healthy_loss_prior_array,mci_loss_prior_array))
    x_jacdet=np.concatenate((healthy_stdlogjacdet_array,mci_stdlogjacdet_array))
    x_jacdet_mean=np.concatenate((healthy_meanjacdet_array,mci_meanjacdet_array))

    x_morphed_b1=np.concatenate((healthy_loss_morphed_array,mci_loss_morphed_array_b1))
    x_morphed_b01=np.concatenate((healthy_loss_morphed_array,mci_loss_morphed_array_b01))

    x_jacdet_b1=np.concatenate((healthy_stdlogjacdet_array,mci_stdlogjacdet_array_b1))
    x_jacdet_b01=np.concatenate((healthy_stdlogjacdet_array,mci_stdlogjacdet_array_b01))
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["aqua", "darkorange","black","yellow","red","gray"]

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_morphed,
            name=f"ROC curve for Morphed",
            color=colors[0],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_prior,
            name=f"ROC curve for Prior",
             color=colors[1],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_jacdet,
            name=f"ROC curve for StdLogJacDet",
             color=colors[2],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_morphed_b1,
            name=f"ROC curve for Morphed_B1",
             color=colors[1],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_jacdet_b1,
            name=f"ROC curve for StdLogJacDet_B1",
             color=colors[2],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_morphed_b01,
            name=f"ROC curve for Morphed_B01",
             color=colors[1],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_jacdet_b01,
            name=f"ROC curve for StdLogJacDetB_01",
             color=colors[2],
            ax=ax,
        )
    x_jacdet_norm = (x_jacdet-np.min(x_jacdet))/(np.max(x_jacdet)-np.min(x_jacdet))
    x_morphed_norm = (x_morphed-np.min(x_morphed))/(np.max(x_morphed)-np.min(x_morphed))
    x_mean_norm = (x_jacdet_mean-np.min(x_jacdet_mean))/(np.max(x_jacdet_mean)-np.min(x_jacdet_mean))

    
    RocCurveDisplay.from_predictions(
            y_prior,
            x_jacdet_norm*x_morphed_norm,
            name=f"ROC curve for StdLogJacDet*Morphed",
             color=colors[3],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_mean_norm*x_morphed_norm,
            name=f"ROC curve for MeanJacDet*Morphed",
             color=colors[4],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_prior,
            x_mean_norm,
            name=f"ROC curve for MeanJacDet",
             color=colors[5],
            ax=ax,
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MCI vs Healthy")
    plt.legend()
    plt.show()
    plt.savefig("./results/"+experiment+"/roc_curve_mci_vs_healthy.png")
    
#     fig, ax1 = plt.subplots(figsize=(6, 6))
#     print(np.mean(healthy_loss_morphed_array))
#     print(np.std(healthy_loss_morphed_array))
#     ax1.errorbar(x=np.mean(healthy_loss_morphed_array),y=[1], xerr=np.std(healthy_loss_morphed_array), fmt='o')
#     ax1.set_title('variable, asymmetric error')
#     ax1.set_yscale('log')
#     plt.show()
#     plt.savefig("./results/"+experiment+"/mean_std.png")
#plot_roc_curve("b_0.1_g_0.01_upsample_dc_1e-4_lr_5e-4")    

    if mean_std==True:
        fig, ax1 = plt.subplots(figsize=(6, 6))
        print(np.mean(healthy_loss_morphed_array))
        print(np.std(healthy_loss_morphed_array))
        print(np.mean(alzheimer_loss_morphed_array))
        print(np.std(alzheimer_loss_morphed_array))
        print(np.mean(mci_loss_morphed_array))
        print(np.std(mci_loss_morphed_array))
        ax1.errorbar(x=[np.mean(healthy_loss_morphed_array),np.mean(alzheimer_loss_morphed_array),np.mean(mci_loss_morphed_array)],y=["Healthy","Alzheimer","MCI"], xerr=[np.std(healthy_loss_morphed_array),np.std(alzheimer_loss_morphed_array),np.std(mci_loss_morphed_array)], fmt='o')

        ax1.set_title('variable, asymmetric error')
        plt.show()
        plt.savefig("./results/"+experiment+"/mean_std.png")        
#plot_roc_curve_no_deformer("no_deformer_g_0.01_upsample_dc_1e-4_lr_5e-4",True)         
#plot_roc_curve("b_10_g_0.01_upsample_dc_1e-4_lr_5e-4",True)         
#plot_roc_curve_2heads("b_10_b_1_b01",True)    

def plot_roc_curve_final(experiment:None,mean_std:None):

#Alzheimer

    healthy_loss_morphed_array1= np.load("./results/"+"b_10_g_0.01_upsample_dc_1e-4_lr_5e-4"+"/healthy_loss_morphed.npy")
    healthy_loss_morphed_array2= np.load("./results/"+"ae_kl"+"/healthy_loss_morphed.npy")
    healthy_loss_morphed_array3= np.load("./results/"+"no_deformer_g_0.01_upsample_dc_1e-4_lr_5e-4"+"/healthy_loss_morphed.npy")
    alzheimer_loss_morphed_array1= np.load("./results/"+"b_10_g_0.01_upsample_dc_1e-4_lr_5e-4"+"/mci_loss_morphed.npy")
    alzheimer_loss_morphed_array2= np.load("./results/"+"ae_kl"+"/mci_loss_morphed.npy")
    alzheimer_loss_morphed_array3= np.load("./results/"+"no_deformer_g_0.01_upsample_dc_1e-4_lr_5e-4"+"/mci_loss_morphed.npy")


    #healthy_loss_prior_array= [0.5,0.8,0.3]
   # alzheimer_loss_morphed_array= [0.4,0.6,0.8]
    #alzheimer_loss_prior_array= [0.123,0.7,0.5]

    y_morphed_healthy=np.zeros(len(healthy_loss_morphed_array1))
    y_morphed_alzheimer=np.ones(len(alzheimer_loss_morphed_array1))
    y_prior_healthy=np.zeros(len(healthy_loss_morphed_array1))
    y_prior_alzheimer=np.ones(len(alzheimer_loss_morphed_array1))

    y_morphed=np.concatenate((y_morphed_healthy,y_morphed_alzheimer))
    y_prior=np.concatenate((y_prior_healthy,y_prior_alzheimer))

    x_1=np.concatenate((healthy_loss_morphed_array1,alzheimer_loss_morphed_array1))
    x_2=np.concatenate((healthy_loss_morphed_array2,alzheimer_loss_morphed_array2))
    x_3=np.concatenate((healthy_loss_morphed_array3,alzheimer_loss_morphed_array3))


    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["aqua", "darkorange","black","yellow","red","gray"]

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_1,
            name=f"Morphaeus",
            color=colors[0],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_2,
            name=f"AE-KL",
             color=colors[1],
            ax=ax,
        )
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_3,
            name=f"AE",
             color=colors[2],
            ax=ax,
        )


    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MCI vs Healthy")
    plt.legend()
    plt.show()
    plt.savefig("./results/"+experiment+"/roc_curve_mci_vs_healthy_new.png")
#plot_roc_curve_final("final",False)   
import os 
def plot_roc_curve_final(experiment:None,mean_std:None):

#Alzheimer

        
    alzheimer_loss_jacdet_array=[]
    alzheimer_loss_jacdet_array=np.array(alzheimer_loss_jacdet_array)
    alzheimer_loss_logjacdet_array=[]
    alzheimer_loss_logjacdet_array=np.array(alzheimer_loss_logjacdet_array)
    alzheimer_norm_loss_jacdet_array=[]
    alzheimer_norm_loss_jacdet_array=np.array(alzheimer_norm_loss_jacdet_array)
    alzheimer_norm_loss_logjacdet_array=[]
    alzheimer_norm_loss_logjacdet_array=np.array(alzheimer_norm_loss_logjacdet_array)

    healthy_loss_jacdet_array=[]
    healthy_loss_jacdet_array=np.array(healthy_loss_jacdet_array)
    healthy_loss_logjacdet_array=[]
    healthy_loss_logjacdet_array=np.array(healthy_loss_logjacdet_array)
    healthy_norm_loss_jacdet_array=[]
    healthy_norm_loss_jacdet_array=np.array(healthy_norm_loss_jacdet_array)
    healthy_norm_loss_logjacdet_array=[]
    healthy_norm_loss_logjacdet_array=np.array(healthy_norm_loss_logjacdet_array)

    for root, dirs, files in os.walk("/home/yigit/iml-dl/results/b1_from_scratch"):
        for file in files:
            if file.startswith("alzheimer_loss") and "alzheimer_loss_" not in file:
                loss= np.load("./results/"+"b_10_g_0.01_upsample_dc_1e-4_lr_5e-4"+"/"+file)
                jacdet= np.load("./results/"+"b1_from_scratch"+"/"+"alzheimer_jacdet"+file[file.find("loss")+4:])
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                loss_norm = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
                jacdet_norm = (jacdet-np.min(jacdet))/(np.max(jacdet)-np.min(jacdet))
                log_jac_det_norm=(log_jac_det-np.min(log_jac_det))/(np.max(log_jac_det)-np.min(log_jac_det))

                data1=(loss*np.abs(jacdet)).mean()
                data2=(loss*np.abs(log_jac_det)).mean()
                data3=(loss_norm*jacdet_norm).mean()
                data4=(loss_norm*log_jac_det_norm).mean()
                
                alzheimer_loss_jacdet_array=np.append(alzheimer_loss_jacdet_array,data1)
                alzheimer_loss_logjacdet_array=np.append(alzheimer_loss_logjacdet_array,data2)
                alzheimer_norm_loss_jacdet_array=np.append(alzheimer_norm_loss_jacdet_array,data3)
                alzheimer_norm_loss_logjacdet_array=np.append(alzheimer_norm_loss_logjacdet_array,data4)
            if file.startswith("healthy_loss") and "healthy_loss_" not in file:
                loss= np.load("./results/"+"b_10_g_0.01_upsample_dc_1e-4_lr_5e-4"+"/"+file)
                jacdet= np.load("./results/"+"b1_from_scratch"+"/"+"healthy_jacdet"+file[file.find("loss")+4:])
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                loss_norm = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
                jacdet_norm = (jacdet-np.min(jacdet))/(np.max(jacdet)-np.min(jacdet))
                log_jac_det_norm=(log_jac_det-np.min(log_jac_det))/(np.max(log_jac_det)-np.min(log_jac_det))

                data1=(loss*np.abs(jacdet)).mean()
                data2=(loss*np.abs(log_jac_det)).mean()
                data3=(loss_norm*jacdet_norm).mean()
                data4=(loss_norm*log_jac_det_norm).mean()
                
                healthy_loss_jacdet_array=np.append(healthy_loss_jacdet_array,data1)
                healthy_loss_logjacdet_array=np.append(healthy_loss_logjacdet_array,data2)
                healthy_norm_loss_jacdet_array=np.append(healthy_norm_loss_jacdet_array,data3)
                healthy_norm_loss_logjacdet_array=np.append(healthy_norm_loss_logjacdet_array,data4)

    y_morphed_healthy=np.zeros(len(healthy_loss_jacdet_array))
    y_morphed_alzheimer=np.ones(len(alzheimer_loss_jacdet_array))

    y_morphed=np.concatenate((y_morphed_healthy,y_morphed_alzheimer))



    x_loss_jacdet=np.concatenate((healthy_loss_jacdet_array,alzheimer_loss_jacdet_array))
    x_loss_logjacdet=np.concatenate((healthy_loss_logjacdet_array,alzheimer_loss_logjacdet_array))
    x_norm_loss_jacdet=np.concatenate((healthy_norm_loss_jacdet_array,alzheimer_norm_loss_jacdet_array))
    x_norm_loss_logjacdet=np.concatenate((healthy_norm_loss_logjacdet_array,alzheimer_norm_loss_logjacdet_array))
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["aqua", "darkorange","black","yellow","red","gray"]

    
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_loss_jacdet,
            name=f"ROC curve for Loss*Jacdet",
             color=colors[3],
            ax=ax,
        )

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_loss_logjacdet,
            name=f"ROC curve for Loss*LogJacdet",
             color=colors[5],
            ax=ax,
        ),
    
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_norm_loss_jacdet,
            name=f"ROC curve for Norm Loss*Jacdet",
             color=colors[3],
            ax=ax,
        )

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_norm_loss_logjacdet,
            name=f"ROC curve for Norm Loss*LogJacDet",
             color=colors[5],
            ax=ax,
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Alzheimer vs Healthy")
    plt.legend()
    plt.show()
    plt.savefig("./results/"+"final"+"/roc_curve_ad_vs_healthy_new_new_b1.png")
plot_roc_curve_final("final",False)