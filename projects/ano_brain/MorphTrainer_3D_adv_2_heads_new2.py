from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
from optim.losses.NCC2 import *
import matplotlib.pyplot as plt
import copy
from optim.losses.monai_perceptual_loss import PerceptualLoss2
from optim.losses import PerceptualLoss,MedicalNetPerceptualSimilarity
from GenerativeModels.generative.losses import PatchAdversarialLoss
from monai.networks.layers import Act
from GenerativeModels.generative.networks.nets import PatchDiscriminator
from dl_utils.visualization import plot_warped_grid
import torchvision.transforms as transforms
from model_zoo.deformer_3D import *
class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        self.ncc_loss = NCC2()
        self.deform_R = DisplacementRegularizer('gradient-l2')
        self.beta_max = training_params['beta'] if 'beta' in training_params.keys() else 3
        self.delta = training_params['delta'] if 'delta' in training_params.keys() else 1
        self.gamma = training_params['gamma'] if 'gamma' in training_params.keys() else 1

        self.max_iter = training_params['max_iter'] if 'max_iter' in training_params.keys() else 500
      #  self.criterion_PL = PerceptualLoss(device=device)

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """

        if model_state is not None:
            pretrained_dict=model_state
          #  pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'deformer' not in k}
            self.model.load_state_dict(pretrained_dict,strict=False)

        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        self.early_stop = False
        

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_adv_loss,batch_disc_loss,batch_loss, batch_loss_pl,batch_loss_after_deformation, batch_loss_pl_after_deformation,batch_loss_reg, batch_loss_deform, count_images = 0.0,0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0, 0
            self.beta = np.clip(self.beta_max * (epoch / self.max_iter), 1e-3, self.beta_max)
            for data in self.train_ds:
               
                images = data[0].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h,d= images.shape

                count_images += b
                # Forward Pass
                reconstruction, result_dict = self.model(transformed_images, registration=False)

                global_prior = result_dict['x_prior']
                reversed_img = result_dict['x_reversed']
                deformation = result_dict['deformation']

                # Losses
               # loss_rec = self.criterion_rec(global_prior, images, result_dict)
                loss_rec_after_deformation = self.criterion_rec(reconstruction, images, result_dict)
                reg_deform = self.deform_R(deformation)
                loss_deform = self.ncc_loss(images, reconstruction) if torch.equal(reconstruction, reversed_img) \
                    else (self.ncc_loss(images, reconstruction) + self.ncc_loss(global_prior, reversed_img))/2
                    
                loss = self.delta * loss_deform + self.beta * reg_deform 


                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = False
                for param in self.model.deformer.parameters():
                    param.requires_grad = True
                    
                self.optimizer.zero_grad()
                # Backward Pass
                loss.backward()
                self.optimizer.step()

                #batch_loss += loss_rec.item() * images.size(0)
                batch_loss_after_deformation += loss_rec_after_deformation.item() * images.size(0)
                batch_loss_reg += reg_deform.item() * images.size(0)
                batch_loss_deform += loss_deform.item() * images.size(0)

            torch.cuda.empty_cache()
            #epoch_loss = (batch_loss) / count_images if count_images > 0 else batch_loss
            epoch_loss_after_deformation =( batch_loss_after_deformation) / count_images if count_images > 0 else batch_loss
            epoch_loss_reg = (batch_loss_reg) / count_images if count_images > 0 else batch_loss_reg
            epoch_loss_deformation = (batch_loss_deform) / count_images if count_images > 0 else batch_loss_deform

    

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss_after_deformation, end_time - start_time, count_images))
            #wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_After_Deformation": epoch_loss_after_deformation, '_step_': epoch})
            wandb.log({"Train/Loss_Reg_": epoch_loss_reg, '_step_': epoch})
            wandb.log({"Train/Loss_Deformation_": epoch_loss_deformation, '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(),
                        'optimizer_weights': self.optimizer.state_dict(),
                        'epoch': epoch}, self.client_path + '/latest_model.pt')
            
            img = transformed_images[0].cpu().detach().numpy()
            # print(np.min(img), np.max(img))
            rec_ = reconstruction[0].cpu().detach().numpy()
            gl_prior = global_prior[0].cpu().detach().numpy()
           # deff=rec_
            deff = deformation[0,:,:,:,:].detach().cpu().numpy()
            # print(f'rec: {np.min(rec)}, {np.max(rec)}')
            elements = [img,gl_prior, rec_,np.abs(gl_prior-img), np.abs(rec_ - img),deff]
            v_maxs = [1, 1, 1,0.5,0.5,0.5]
            diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            diffp.set_size_inches(len(elements) * 4, 3 * 4)
            for i in range(len(elements)):
                for axis in range(3):
                    if i!=len(elements)-1:
                        axarr[axis, i].axis('off')
                        v_max = v_maxs[i]
                        c_map = 'gray' if v_max == 1 else 'plasma'
                        # print(elements[i].shape)
                        if axis == 0:
                            el = np.squeeze(elements[i])[int(w / 2), :, :]
                        elif axis == 1:
                            el = np.squeeze(elements[i])[:, int(w / 2), :]
                        else:
                            el = np.squeeze(elements[i])[:, :, int(w / 2)]

                        axarr[axis, i].imshow(np.squeeze(el).T, vmin=0, vmax=v_max, cmap=c_map, origin='lower')
                    else:
                        
                        if axis == 0:
                            temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, int(w / 2),:,:],axes=(1,2)), np.rot90(elements[i][np.newaxis,1, int(w / 2),:,:],axes=(1,2))), 0)
                            plot_warped_grid(ax=axarr[axis, i],disp=temp) # .rot90(axes=(2,3)
                        elif axis == 1:
                            temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                            plot_warped_grid(ax=axarr[axis, i],disp=temp)
                        else:
                            temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                            plot_warped_grid(ax=axarr[axis, i],disp=temp)
            wandb.log({'Train/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.model.state_dict(), self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        torch.cuda.empty_cache()
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_after_deformation': 0,
            task + '_loss_pl': 0,
        }
        loss_mse_array=[]
        loss_mse_array=np.array(loss_mse_array)
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w,d = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                x_rec, rec_dict = self.test_model(x)
               # rec_dict = self.test_model(x)
                x_ = rec_dict['x_prior']
                deformation = rec_dict['deformation']
                loss_rec = self.criterion_rec(x_, x, rec_dict)

                loss_mse = self.criterion_rec(x_rec, x)
          
             #   loss_pl = self.criterion_PL(x_, x)
                loss_mse_array= np.append(loss_mse_array,loss_mse.detach().cpu().numpy())
                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_after_deformation'] += loss_mse.item() * x.size(0)
             #   metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)
               # print("Val " +str(loss_pl))
               # print("val " +str(metrics[task + '_loss_pl']))
               # print(str(test_total))
                if test_total<=20:
                    img = x.detach().cpu()[0].numpy()
                    # print(np.min(img), np.max(img))
                    rec_ = x_rec.detach().cpu()[0].numpy()
                    gl_prior = x_.detach().cpu()[0].numpy()
                    # deff=rec_
                    deff = deformation[0,:,:,:,:].detach().cpu().numpy()
                    # print(f'rec: {np.min(rec)}, {np.max(rec)}')
                    elements = [img,gl_prior, rec_, np.abs(gl_prior-img),np.abs(rec_ - img),deff]
                    v_maxs = [1, 1, 1,0.5,0.5,0.5]
                    diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                    diffp.set_size_inches(len(elements) * 4, 3 * 4)
                    for i in range(len(elements)):
                        for axis in range(3):
                            if i!=len(elements)-1:
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'gray' if v_max == 1 else 'plasma'
                                # print(elements[i].shape)
                                if axis == 0:
                                    el = np.squeeze(elements[i])[int(w / 2), :, :]
                                elif axis == 1:
                                    el = np.squeeze(elements[i])[:, int(w / 2), :]
                                else:
                                    el = np.squeeze(elements[i])[:, :, int(w / 2)]

                                axarr[axis, i].imshow(np.squeeze(el).T, vmin=0, vmax=v_max, cmap=c_map, origin='lower')
                            else:
                                
                                if axis == 0:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, int(w / 2),:,:],axes=(1,2)), np.rot90(elements[i][np.newaxis,1, int(w / 2),:,:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp) # .rot90(axes=(2,3)
                                elif axis == 1:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
                                else:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
                    wandb.log({task + '/Example_': [
                            wandb.Image(diffp, caption="Iteration_" + str(epoch)+"_"+str(test_total))]})
        if task=='Test':
            fig, ax = plt.subplots()
            ax.hist(loss_mse_array, bins=len(loss_mse_array)) 
#            wandb.log({"Test_histogram": fig})
        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        wandb.log({'beta': self.beta, '_step_': epoch})

        epoch_val_loss = (metrics[task + '_loss_rec'])/ test_total
        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                self.best_weights = copy.deepcopy(model_weights)
                self.best_opt_weights = copy.deepcopy(opt_weights)
                torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights,
                            'epoch': epoch}, self.client_path + '/best_model.pt')
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)
