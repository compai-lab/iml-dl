import os, json
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torchmetrics.functional import confusion_matrix, accuracy
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def plot_latent_interpolations(model, latent_code, dim_list=[0], num_points=10):
    """
        dim_list: has to be iterable
    """
    x1 = torch.linspace(-4., 4.0, num_points)
    num_points = x1.size(0)
    # z = torch.from_numpy(latent_code)

    outputs = []
    with torch.no_grad():
        for dim in dim_list:
            z = latent_code.repeat(num_points, 1)
            z[:, dim] = x1.contiguous()
            # outputs = torch.sigmoid(self.model.decode(z))
            output = model.decode(z)
            if len(output.size()) == 5: # 3D images
                slice = int(output.size()[4] / 2)
                outputs.append(output[:,:,:,:,slice])
            else:
                outputs.append(output)
    outputs = torch.concat(outputs, 0)
    grid_img = make_grid(outputs.cpu(), nrow=num_points, pad_value=1.0)

    fig = plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    return fig


def plot_latent_interpolations2d(model, latent_code, dim1=0, dim2=1, num_points=5):
    x1 = torch.linspace(-4., 4.0, num_points)
    x2 = torch.linspace(-4., 4.0, num_points)
    z1, z2 = torch.meshgrid([x1, x2])
    num_points = z1.size(0) * z1.size(1)
    z = latent_code.repeat(num_points, 1)
    z[:, dim1] = z1.contiguous().view(1, -1)
    z[:, dim2] = z2.contiguous().view(1, -1)
    # outputs = torch.sigmoid(self.model.decode(z))

    outputs = model.decode(z)
    if len(outputs.size()) == 5:
        outputs_2D = []
        slice = int(outputs.size()[4]/2)
        for idx in range(outputs.size()[0]):
            outputs_2D.append(outputs[idx,:,:,:,slice])

        outputs = torch.concat(outputs_2D,0).view(outputs.size()[0:4])

    grid_img = make_grid(outputs.cpu(), nrow=z1.size(0), pad_value=1.0)
    fig = plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    # interp.save_image(os.path.join(self.image_path, 'reconstruction.png'))
    return fig


def plot_latent_reconstructions(model, dataset, device, num_points=10):
    outputs = []
    #sample_id = range(num_points)
    with torch.no_grad():
        #for data in dataset:
        for sample in range(num_points):
            #img = data[0].to(device)

            data = dataset.dataset.__getitem__(sample)
            img = data[0].to(device).unsqueeze(0)
            rec, _ = model(img)

            if len(img.size()) == 5: # 3D images
                slice = int(img.size()[4] / 2)
                outputs.append(img[sample_id,:,:,:,slice])
                outputs.append(rec[sample_id,:,:,:,slice])
            else:
                outputs.append(img)
                outputs.append(rec)
                #outputs.append(img[sample_id])
                #outputs.append(rec[sample_id])
                #break

        #outputs = torch.concat(outputs, 0)
        tmp = torch.concat(outputs, 0)
        outputs = torch.cat((tmp[::2], tmp[1::2]))
        grid_img = make_grid(outputs.cpu(), nrow=num_points, pad_value=1.0)
        fig = plt.imshow(grid_img.permute(1, 2, 0),cmap='gray')
        #fig = plt.imshow(grid_img.permute(1, 2, 0),cmap='gray')
        plt.axis('off')
    return fig

def plot_conf_mat(pred, labels, dict_classes, binary_label=True):

    #if binary_label:
    #    label_pred = np.zeros((np.shape(pred)[0],1))
    #    label_pred[pred>0.5] = 1
    #else:
    label_pred = np.argmax(pred,axis=1)

    conf_matrix = confusion_matrix(y_true=labels, y_pred=label_pred)

    df = pd.DataFrame()
    df['acc'] = [accuracy_score(labels,label_pred)]
    df['precision'] = [precision_score(labels, label_pred, average='macro')]
    df['recall'] = [recall_score(labels, label_pred, average='macro')]
    df['f1_'] = [f1_score(labels, label_pred, average='macro')]
    df['AUROC'] = [roc_auc_score(labels, pred, multi_class='ovr')]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)

    return fig, df

# def plot_conf_mat(pred, labels, num_classes, dict_classes):
#
#     confmat = confusion_matrix(pred, labels, num_classes=num_classes)
#     confmat = torch.round(confmat.type(torch.FloatTensor)).type(torch.IntTensor).numpy()
#     df_cm = pd.DataFrame(confmat, #/ np.sum(confmat, axis=1)
#                          index=[i for i in dict_classes], columns=[i for i in dict_classes]
#                          )
#     img_cm = plt.figure(figsize=(12, 7))
#     plt.xlabel('Actual', fontsize=10)
#     plt.ylabel('Predicted', fontsize=10)
#     sn.heatmap(df_cm.transpose(), annot=True)
#     return img_cm


def get_feature_names(results_folder, latent_dim):
    """
        Return a list of features names where the attributes dimensions
        are with the attributes names

    """
    feature_names = [f'z{i}' for i in range(latent_dim)]
    file = f'{results_folder}/results_dict.json'
    if os.path.exists(file):
        with open(file, 'r') as f:
            metrics = json.load(f)
        metric_inter = metrics['interpretability']
        keys = list(metric_inter.keys())
        keys.remove('mean')

        dim_list = [metric_inter[K][0] for K in keys]
        for dim, K in zip(dim_list, keys):
            feature_names[dim] = f'z{dim} ({K})'
    return feature_names

"""
From m-pax_lib Kleine et al.,

"""

class AttributionLatentY:
    """Computes and visualizes the attribution of the latent representation into the prediction.
    The object contains two methods:
        - attribution
        - visualization
    Whereby the visualization method calls the attribution method.

    """

    def __init__(self, dataloader, labels_name, encoder, head, index, results_folder, device):
        """Called upon initialization. Selects label names based on dataset name.

        Parameters
        ----------
        dataloader : torch.utils.data.Dataloader
            Provides an iterable dataloader over the given dataset.
        labels_name : str
            Name of labels_names.
        encoder : src.models.tcvae_conv.betaTCVAE_Conv or src.models.tcvae_resnet.betaTCVAE_ResNet
            beta-TCVAE trained encoder, encoding the (disentangled) latent representations.
        head : src.models.head_mlp.MLP
            Head for downstream task prediction.
        index : int
            Index of image for attribution visualization.
        results_folder : str
            Folder of the encoder model.
        """
        self.dataloader = dataloader
        self.encoder = encoder
        self.head = head
        self.index = index
        self.device = device
        self.labels_name = list(np.sort(labels_name))
        self.feature_names = get_feature_names(results_folder, self.head.latent_dim)
        self.blocked_features = self.head.blocked_latent_features

        """
        if dataset == "MNISTDataModule":
            self.labels_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            self.labels_name = ['NOR','MINF','DCM','HCM','RV']
"""
    def attribution(self):
        """Computes expected gradient based attribution for the latent representation
        of the image selected via "index". Is called within the visualization method.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Returns attribution map, l  atent representation, and respective label.
        """
        with torch.no_grad():
            images, labels, _, _ = next(iter(self.dataloader))
            rand_img_dist, _, _,_ = next(iter(self.dataloader))

            images = images[self.index :]
            labels = labels[self.index :]

            encoding = self.encoder.encode(images.to(self.device))
            encoding_dist = self.encoder.encode(rand_img_dist.to(self.device))

        model_name = self.head.dl_config['model']['module_name'].split('.')[-1]
        if model_name == 'beta_vae_higgings':

            exp = shap.GradientExplainer(self.head, data=encoding[1]['z_mu'])
            loc_encoding = encoding_dist[1]['z_mu']

        elif len(encoding_dist)>1:
            try:
                exp = shap.GradientExplainer(self.head, data=encoding_dist[2].loc)
            except:
                exp = shap.GradientExplainer(self.head, data=encoding_dist[0])
                loc_encoding = encoding_dist[0]
            else:
                loc_encoding = encoding[2].loc
        else:
            exp = shap.GradientExplainer(self.head, data=encoding_dist.loc)
            loc_encoding = encoding.loc

        attributions_gs = exp.shap_values(loc_encoding)
        return attributions_gs, loc_encoding, labels

    def visualization(self):
        """Computes and saves graphics for the via "index" selected representation into "output_dir".
        Also calls attribution computation.
        """

        attributions_gs, encoding, labels = self.attribution()

        encoding = encoding[:,:self.blocked_features[0]]
        for i in range(len(attributions_gs)):
            attributions_gs[i] = attributions_gs[i][:, :self.blocked_features[0]]

        fig_global = plt.figure(figsize=(9, 5), dpi=200)
        shap.summary_plot(
            attributions_gs,
            encoding,
            plot_type="bar",
            feature_names=self.feature_names,
            color=plt.cm.tab10,
            class_names=self.labels_name,
            show=False,
            max_display=16
        )

        fig_local = plt.figure(figsize=(5, 4), dpi=200)
        """
        shap.multioutput_decision_plot(
            np.zeros((1, len(self.labels_name))).tolist()[0],
            attributions_gs,
            highlight=labels[0],
            legend_labels=self.labels_name,
            legend_location="lower right",
            show=False,
            feature_names=self.feature_names,
            auto_size_plot=False,
            row_index=0,
            link="logit",
        )

        plt.tight_layout()
        """
        return fig_global, fig_local
