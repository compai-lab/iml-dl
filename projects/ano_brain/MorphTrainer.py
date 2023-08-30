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
            batch_loss, batch_loss_pl, batch_loss_reg, batch_loss_deform, count_images = 1.0, 1.0, 1.0, 1.0, 0
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
                reversed_img = result_dict['x_reversed']
                deformation = result_dict['deformation']
                # Losses
                loss_rec = self.criterion_rec(global_prior, images, result_dict)
                loss_pl = self.criterion_PL(global_prior, images)
                reg_deform = self.deform_R(deformation)
                loss_deform = self.ncc_loss(images, reconstruction) if torch.equal(reconstruction, reversed_img) \
                    else (self.ncc_loss(images, reconstruction) + self.ncc_loss(global_prior, reversed_img))/2
                loss = loss_rec + self.alfa * loss_pl
                if epoch > 10:
                    loss = loss_rec + self.alfa * loss_pl + self.delta * loss_deform + self.beta * reg_deform

                self.optimizer.zero_grad()
                # Backward Pass
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)  # to avoid nan loss
                self.optimizer.step()

                batch_loss += loss_rec.item() * images.size(0)
                batch_loss_pl += loss_pl.item() * images.size(0)
                batch_loss_reg += reg_deform.item() * images.size(0)
                batch_loss_deform += loss_deform.item() * images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_loss_pl = batch_loss_pl / count_images if count_images > 0 else batch_loss_pl
            epoch_loss_reg = batch_loss_reg / count_images if count_images > 0 else batch_loss_reg
            epoch_loss_deformation = batch_loss_deform / count_images if count_images > 0 else batch_loss_deform

            epoch_losses.append(epoch_loss)
            epoch_losses_pl.append(epoch_loss_pl)
            epoch_losses_reg.append(epoch_loss_reg)
            epoch_losses_deformation.append(epoch_loss_deformation)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_PL_": epoch_loss_pl, '_step_': epoch})
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
            elements = [img, global_prior, rec, np.abs(global_prior-img), np.abs(rec-img),deff]
            diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            diffp.set_size_inches(len(elements) * 4, 4)
            for i in range(len(axarr)):
                axarr[i].axis('off')
                if i!=len(axarr)-1:
                    v_max = 1 if i < np.floor(((len(elements)-1) / 2)) + 1 else 0.5
                    c_map = 'gray' if i < np.floor(((len(elements)-1) / 2)) + 1 else 'inferno'               
                    axarr[i].imshow(np.squeeze(elements[i].transpose(1, 2, 0)), vmin=0, vmax=v_max, cmap=c_map)
                else:
                    plot_warped_grid(ax=axarr[i],disp=deff)
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
                deformation = rec_dict['deformation']
                loss_rec = self.criterion_rec(x_, x, rec_dict)
                loss_mse = self.criterion_MSE(x_, x)
                loss_mse += self.criterion_MSE(x_rec, x)
                loss_mse /= 2.0
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

                global_prior = x_.detach().cpu()[0].numpy()
                # global_prior[0, 0], global_prior[0, 1] = 0, 1
                rec = x_rec.detach().cpu()[0].numpy()
                # rec[0, 0], rec[0, 1] = 0, 1
                img = x.detach().cpu()[0].numpy()
                # img[0, 0], img[0, 1] = 0, 1
                # grid_image = np.hstack([img, global_prior, rec])
                deff = deformation.detach().cpu()[0].numpy()
                # img[0, 0], img[0, 1] = 0, 1
                # grid_image = np.hstack([img, global_prior, rec])
                elements = [img, global_prior, rec, np.abs(global_prior-img), np.abs(rec-img),deff]
                diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                diffp.set_size_inches(len(elements) * 4, 4)
                for i in range(len(axarr)):
                    axarr[i].axis('off')
                    if i!=len(axarr)-1:
                        v_max = 1 if i < np.floor(((len(elements)-1) / 2)) + 1 else 0.5
                        c_map = 'gray' if i < np.floor(((len(elements)-1) / 2)) + 1 else 'inferno'               
                        axarr[i].imshow(np.squeeze(elements[i].transpose(1, 2, 0)), vmin=0, vmax=v_max, cmap=c_map)
                    else:
                        plot_warped_grid(ax=axarr[i],disp=deff)
            
                wandb.log({task + '/Example_': [
                        wandb.Image(diffp, caption="Iteration_" + str(epoch))]})
                if task=='Val':
                    break

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

