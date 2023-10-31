from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
import matplotlib.pyplot as plt
import copy
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from net_utils.nets.discriminators import *
from optim.losses.monai_patch_adversarial_loss import PatchAdversarialLoss
from optim.losses.kl import KL_loss
from optim.losses.monai_perceptual_loss import PerceptualLoss2 as PL2


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)

        self.discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
        self.discriminator.to(device)
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.loss_perceptual = PL2(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).to(device)

        self.adv_weight = 0.005 # 0.01
        self.alfa = 0.05
        self.kl_weight = 1e-6
        lr = training_params['optimizer_params']['lr']
        self.optimizer_g = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(), lr=lr)
        self.autoencoder_warm_up_n_epochs = 5


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
            self.model.load_state_dict(model_state)  # load weights
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        epoch_losses = []
        epoch_losses_pl = []
        epoch_losses_g = []
        epoch_losses_d = []
        self.early_stop = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, batch_loss_pl, batch_loss_g, batch_loss_d, count_images = 0.0, 0.0, 0.0, 0.0, 0
            # for epoch in range(n_epochs):
            self.model.train()
            self.discriminator.train()

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                transformed_images = self.transform(copy.deepcopy(images)) if self.transform is not None else images
                b, c, w, h, d = images.shape
                count_images += b

                # Forward Pass
                self.optimizer_g.zero_grad(set_to_none=True)
                reconstruction, z_mu, z_sigma = self.model(transformed_images)
                kl_loss = KL_loss(z_mu, z_sigma)
                loss_rec = self.criterion_rec(reconstruction, images, {'z': z_mu})

                loss_pl = self.loss_perceptual(reconstruction.float(), images.float())
                loss_g = loss_rec + self.kl_weight * kl_loss + self.alfa * loss_pl

                # Generator part
                if epoch > self.autoencoder_warm_up_n_epochs:
                    logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += self.adv_weight * generator_loss

                loss_g.backward()
                self.optimizer_g.step()

                ## Discriminator part
                if epoch > self.autoencoder_warm_up_n_epochs:
                    self.optimizer_d.zero_grad(set_to_none=True)
                    logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = self.discriminator(images.contiguous().detach())[-1]
                    loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    loss_d = self.adv_weight * discriminator_loss

                    loss_d.backward()
                    self.optimizer_d.step()

                    batch_loss += loss_rec.item() * images.size(0)
                    batch_loss_pl += loss_pl.item() * images.size(0)
                    batch_loss_g += loss_g.item() * images.size(0)
                    batch_loss_d += loss_d.item() * images.size(0)
            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_loss_pl = batch_loss_pl / count_images if count_images > 0 else batch_loss_pl
            epoch_loss_g = batch_loss_g / count_images if count_images > 0 else batch_loss_g
            epoch_loss_d = batch_loss_d / count_images if count_images > 0 else batch_loss_d

            epoch_losses.append(epoch_loss)
            epoch_losses_pl.append(epoch_loss_pl)
            epoch_losses_g.append(epoch_loss_g)
            epoch_losses_d.append(epoch_loss_d)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_PL_": epoch_loss_pl, '_step_': epoch})
            wandb.log({"Train/Loss_G_": epoch_loss_g, '_step_': epoch})
            wandb.log({"Train/Loss_D_": epoch_loss_d, '_step_': epoch})


            # Save latest model
            torch.save(
                {'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                    , 'epoch': epoch}, self.client_path + '/latest_model.pt')

            img = transformed_images[0].cpu().detach().numpy()
            # print(np.min(img), np.max(img))
            rec_ = reconstruction[0].cpu().detach().numpy()
            # print(f'rec: {np.min(rec)}, {np.max(rec)}')
            elements = [img, rec_, np.abs(rec_ - img)]
            v_maxs = [1, 1, 0.5]
            diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            diffp.set_size_inches(len(elements) * 4, 3 * 4)
            for i in range(len(axarr)):
                for axis in range(3):
                    axarr[axis, i].axis('off')
                    v_max = v_maxs[i]
                    c_map = 'gray' if v_max == 1 else 'plasma'
                    # print(elements[i].shape)
                    if axis == 0:
                        el = np.squeeze(elements[i])[int(w / 2), :, :]
                    elif axis == 1:
                        el = np.squeeze(elements[i])[:, int(h / 2), :]
                    else:
                        el = np.squeeze(elements[i])[:, :, int(d / 2)]

                    axarr[axis, i].imshow(np.squeeze(el).T, vmin=0, vmax=v_max, cmap=c_map, origin='lower')

            wandb.log({'Train/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        del self.discriminator
        del self.loss_perceptual
        torch.cuda.empty_cache()

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
                b, c, h, w, d = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                x_, z_mu, z_sigma = self.test_model(x)

                loss_rec = self.criterion_rec(x_, x, {'z': z_mu})
                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.loss_perceptual(x_.float(), x.float())

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

        img = x.detach().cpu()[0].numpy()
        rec_ = x_.detach().cpu()[0].numpy()

        elements = [img, rec_, np.abs(rec_ - img)]
        v_maxs = [1, 1, 0.5]
        diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        diffp.set_size_inches(len(elements) * 4, 3 * 4)
        for i in range(len(axarr)):
            for axis in range(3):
                axarr[axis, i].axis('off')
                v_max = v_maxs[i]
                c_map = 'gray' if v_max == 1 else 'plasma'
                # print(elements[i].shape)
                if axis == 0:
                    el = np.squeeze(elements[i])[int(h / 2), :, :]
                elif axis == 1:
                    el = np.squeeze(elements[i])[:, int(w / 2), :]
                else:
                    el = np.squeeze(elements[i])[:, :, int(d / 2)]

                axarr[axis, i].imshow(np.squeeze(el).T, vmin=0, vmax=v_max, cmap=c_map, origin='lower')

        wandb.log({task + '/Example_': [
            wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '_loss_rec'] / test_total
        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                           self.client_path + '/best_model.pt')
                self.best_weights = copy.deepcopy(model_weights)
                self.best_opt_weights = copy.deepcopy(opt_weights)
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)
