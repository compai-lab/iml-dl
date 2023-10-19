from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
from optim.losses.NCC2 import *
import matplotlib.pyplot as plt
import copy
from optim.losses import PerceptualLoss
from dl_utils.visualization import plot_warped_grid
import matplotlib
matplotlib.use("Agg")
import torch
class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
       
        self.deform_R = DisplacementRegularizer2D('gradient-l2')
        self.beta_max = training_params['beta'] if 'beta' in training_params.keys() else 3
        self.delta = training_params['delta'] if 'delta' in training_params.keys() else 1
        self.max_iter = training_params['max_iter'] if 'max_iter' in training_params.keys() else 500
        self.criterion_PL = PerceptualLoss(device=device)
        self.ncc_window_size=training_params['ncc_window_size'] if 'ncc_window_size' in training_params.keys() else 9
       # self.ncc_loss = NCC(win=self.ncc_window_size)
        self.ncc_loss = NCC2()
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
            self.model.load_state_dict(model_state)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        epoch_losses = []
        epoch_losses_pl = []
        epoch_losses_after_deformation = []
        epoch_losses_pl_after_deformation = []
        epoch_losses_reg = []
        epoch_losses_deformation = []
        self.early_stop = False
        wandb.init()
        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, batch_loss_pl,batch_loss_after_deformation, batch_loss_pl_after_deformation,batch_loss_reg, batch_loss_deform, count_images = 1.0,1.0,1.0, 1.0, 1.0, 1.0, 0
            self.beta = np.clip(self.beta_max * (epoch / self.max_iter), 1e-3, self.beta_max)

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h = images.shape

                count_images += b
                # Forward Pass
                reconstruction, result_dict = self.model(transformed_images, registration=False)
                global_prior = result_dict['x_prior']
                reversed_img = torch.from_numpy(result_dict['x_reversed']).to(self.device)
                deformation = torch.from_numpy(result_dict['deformation']).to(self.device)
                reconstruction=torch.from_numpy(reconstruction).to(self.device)
                # Losses
                loss_rec = self.criterion_rec(global_prior, images, result_dict)
                loss_pl = self.criterion_PL(global_prior, images).mean()
                loss_rec_after_deformation = self.criterion_rec(reconstruction, images, result_dict)
                loss_pl_after_deformation = self.criterion_PL(reconstruction, images).mean()
                reg_deform = self.deform_R(deformation)
                loss_deform = self.ncc_loss(images.to(self.device), reconstruction) if torch.equal(reconstruction, reversed_img) \
                    else (self.ncc_loss(images, reconstruction) + self.ncc_loss(global_prior, reversed_img))/2
                loss = loss_rec + self.alfa * loss_pl
                if epoch > 20:
                    loss = loss_rec + self.alfa * loss_pl + self.delta * loss_deform + self.beta * reg_deform

                self.optimizer.zero_grad()
                # Backward Pass
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)  # to avoid nan loss
                self.optimizer.step()

                batch_loss += loss_rec.item() * images.size(0)
                batch_loss_pl += loss_pl.item() * images.size(0)
                batch_loss_after_deformation += loss_rec_after_deformation.item() * images.size(0)
                batch_loss_pl_after_deformation += loss_pl_after_deformation.item() * images.size(0)
                batch_loss_reg += reg_deform.item() * images.size(0)
                batch_loss_deform += loss_deform.item() * images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_loss_pl = batch_loss_pl / count_images if count_images > 0 else batch_loss_pl
            epoch_loss_after_deformation = batch_loss_after_deformation / count_images if count_images > 0 else batch_loss
            epoch_loss_pl_after_deformation = batch_loss_pl_after_deformation / count_images if count_images > 0 else batch_loss_pl
            epoch_loss_reg = batch_loss_reg / count_images if count_images > 0 else batch_loss_reg
            epoch_loss_deformation = batch_loss_deform / count_images if count_images > 0 else batch_loss_deform

            epoch_losses.append(epoch_loss)
            epoch_losses_pl.append(epoch_loss_pl)
            epoch_losses_after_deformation.append(epoch_loss_after_deformation)
            epoch_losses_pl_after_deformation.append(epoch_loss_pl_after_deformation)
            epoch_losses_reg.append(epoch_loss_reg)
            epoch_losses_deformation.append(epoch_loss_deformation)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_PL_": epoch_loss_pl, '_step_': epoch})
            wandb.log({"Train/Loss_After_Deformation": epoch_loss_after_deformation, '_step_': epoch})
            wandb.log({"Train/Loss_PL_After_Deformation": epoch_loss_pl_after_deformation, '_step_': epoch})
            
            wandb.log({"Train/Loss_Reg_": epoch_loss_reg, '_step_': epoch})
            wandb.log({"Train/Loss_Deformation_": epoch_loss_deformation, '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(),
                        'optimizer_weights': self.optimizer.state_dict(),
                        'epoch': epoch}, self.client_path + '/latest_model.pt')
            global_prior = global_prior[:,:,:,:].detach().cpu()[0].numpy()
            # global_prior[0, 0], global_prior[0, 1] = 0, 1
            rec = reconstruction[:,:,:,:].detach().cpu()[0].numpy()
            # rec[0, 0], rec[0, 1] = 0, 1
            img = transformed_images[:,:,:,:].detach().cpu()[0].numpy()
            deff = deformation[:,:,:,:].detach().cpu()[0].numpy()
            # img[0, 0], img[0, 1] = 0, 1
            # grid_image = np.hstack([img, global_prior, rec])
            elements = [img,global_prior, rec, np.abs(global_prior - img),np.abs(rec-img),deff]
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
                        el = np.squeeze(elements[i])[axis, :, :]

                        axarr[axis, i].imshow(np.squeeze(el).T, vmin=0, vmax=v_max, cmap=c_map, origin='lower')
                    else:
                        plot_warped_grid(ax=axarr[axis, i],disp=np.squeeze(deff[axis*2:axis*2+2,:,:]))
            wandb.log({'Train' + '/Example_': [
                    wandb.Image(diffp, caption="Iteration_" + str(epoch))]})    


            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights

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
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                x_rec, rec_dict = self.test_model(x)
                x_ = rec_dict['x_prior']
                deformation =torch.from_numpy( rec_dict['deformation']).to(self.device)
                x_rec=torch.from_numpy(x_rec).to(self.device)
                loss_rec = self.criterion_rec(x_, x, rec_dict)
                loss_mse = self.criterion_MSE(x_, x)
                loss_mse += self.criterion_MSE(x_rec, x)
                loss_mse /= 2.0
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

        for i in range(1):
            global_prior = x_.detach().cpu()[i].numpy()
            # global_prior[0, 0], global_prior[0, 1] = 0, 1
            rec = x_rec.detach().cpu()[i].numpy()
            # rec[0, 0], rec[0, 1] = 0, 1
            img = x.detach().cpu()[i].numpy()
            # img[0, 0], img[0, 1] = 0, 1
            # grid_image = np.hstack([img, global_prior, rec])
            deff = deformation[:,:,:,:].detach().cpu()[0].numpy()
            # img[0, 0], img[0, 1] = 0, 1
            # grid_image = np.hstack([img, global_prior, rec])
            elements = [img,global_prior, rec, np.abs(global_prior - img),np.abs(rec-img),deff]
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
                        el = np.squeeze(elements[i])[axis, :, :]

                        axarr[axis, i].imshow(np.squeeze(el).T, vmin=0, vmax=v_max, cmap=c_map, origin='lower')
                    else:
                        plot_warped_grid(ax=axarr[axis, i],disp=np.squeeze(deff[axis*2:axis*2+2,:,:]))
        
            wandb.log({task + '/Example_': [
                    wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        wandb.log({'beta': self.beta, '_step_': epoch})

        epoch_val_loss = metrics[task + '_loss_mse'] / test_total
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
