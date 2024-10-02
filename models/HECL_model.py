import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
import re
from collections import OrderedDict
from .base_model import BaseModel
import util.util as util
from . import networks

class HECL(BaseModel):
    def name(self):
        return 'HECL'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        torch.backends.cudnn.benchmark = True

        self.isTrain = opt.isTrain
        self.traverse = (not self.isTrain) and opt.traverse

        self.compare_to_trained_outputs = (not self.isTrain) and opt.compare_to_trained_outputs
        if self.compare_to_trained_outputs:
            self.compare_to_trained_class = opt.compare_to_trained_class
            self.trained_class_jump = opt.trained_class_jump

        self.deploy = (not self.isTrain) and opt.deploy
        if not self.isTrain and opt.random_seed != -1:
            torch.manual_seed(opt.random_seed)
            torch.cuda.manual_seed_all(opt.random_seed)
            np.random.seed(opt.random_seed)

        self.nb = opt.batchSize
        self.size = opt.fineSize
        self.ngf = opt.ngf
        self.ngf_global = self.ngf

        self.numClasses = opt.numClasses
        self.use_moving_avg = not opt.no_moving_avg

        self.no_cond_noise = opt.no_cond_noise
        style_dim = opt.gen_dim_per_style * self.numClasses
        self.duplicate = opt.gen_dim_per_style

        self.cond_length = style_dim

        if not self.isTrain:
            self.debug_mode = opt.debug_mode
        else:
            self.debug_mode = False

        # Generators
        self.netG = self.parallelize(networks.define_Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsample,
                                    id_enc_norm=opt.id_enc_norm, gpu_ids=self.gpu_ids, padding_type='reflect', style_dim=style_dim,
                                    init_type='kaiming', conv_weight_norm=opt.conv_weight_norm,
                                    decoder_norm=opt.decoder_norm, activation=opt.activation,
                                    adaptive_blocks=opt.n_adaptive_blocks, normalize_mlp=opt.normalize_mlp,
                                    modulated_conv=opt.use_modulated_conv))

        if self.isTrain and self.use_moving_avg:
            self.G_moving_avg = networks.define_Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsample,
                                                id_enc_norm=opt.id_enc_norm, gpu_ids=self.gpu_ids, padding_type='reflect', style_dim=style_dim,
                                                init_type='kaiming', conv_weight_norm=opt.conv_weight_norm,
                                                decoder_norm=opt.decoder_norm, activation=opt.activation,
                                                adaptive_blocks=opt.n_adaptive_blocks, normalize_mlp=opt.normalize_mlp,
                                                modulated_conv=opt.use_modulated_conv)
            self.G_moving_avg.train(False)
            self.requires_grad(self.G_moving_avg, flag=False)
            self.accumulate(self.G_moving_avg, self.netG, decay=0)

        # Discriminator network
        if self.isTrain:
            self.netD = self.parallelize(networks.define_D(opt.output_nc, opt.ndf, n_layers=opt.n_layers_D,
                                         numClasses=self.numClasses, gpu_ids=self.gpu_ids,
                                         init_type='kaiming'))

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if (not self.isTrain) or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if (not self.isTrain) or (self.isTrain and opt.continue_train) else opt.load_pretrain
            if self.isTrain:
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                if self.use_moving_avg:
                    self.load_network(self.G_moving_avg, 'G_moving_avg', opt.which_epoch, pretrained_path)
            elif self.use_moving_avg:
                self.load_network(self.netG, 'G_moving_avg', opt.which_epoch, pretrained_path)
            else:
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)


        # set loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = self.parallelize(networks.SelectiveClassesNonSatGANLoss())
            self.R1_reg = networks.R1_reg()
            self.age_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.identity_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.struct_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.texture_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.criterionCycle = self.parallelize(networks.FeatureConsistency())
            self.criterionRec = self.parallelize(networks.FeatureConsistency())
            self.criterionPixel = self.parallelize(networks.FeatureConsistency())
            self.criterionSem = self.parallelize(networks.FeatureConsistency())
            self.contrastive_loss = ContrastiveLoss(temperature=0.5).cuda()

            # initialize optimizers
            self.old_lr = opt.lr

            # set optimizer G
            paramsG = []
            params_dict_G = dict(self.netG.named_parameters())

            for key, value in params_dict_G.items():
                decay_cond = ('decoder.mlp' in key)
                if opt.decay_adain_affine_layers:
                    decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                if decay_cond:
                    paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                else:
                    paramsG += [{'params':[value],'lr':opt.lr}]

            self.optimizer_G = torch.optim.Adam(paramsG, lr=opt.lr, betas=(opt.beta1, opt.beta2))

            # set optimizer D
            paramsD = list(self.netD.parameters()) 
            self.optimizer_D = torch.optim.Adam(paramsD, lr=opt.lr, betas=(opt.beta1, opt.beta2))


    def parallelize(self, model):
        # parallelize a network
        if self.isTrain and len(self.gpu_ids) > 0:
            return networks._CustomDataParallel(model)
        else:
            return model


    def requires_grad(self, model, flag=True):
        # freeze network weights
        for p in model.parameters():
            p.requires_grad = flag


    def accumulate(self, model1, model2, decay=0.999):
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        model1_parallel = isinstance(model1, nn.DataParallel)
        model2_parallel = isinstance(model2, nn.DataParallel)

        for k in params1.keys():
            if model2_parallel and not model1_parallel:
                k2 = 'module.' + k
            elif model1_parallel and not model2_parallel:
                k2 = re.sub('module.', '', k)
            else:
                k2 = k
            params1[k].data.mul_(decay).add_(params2[k2].data, alpha=1 - decay)

    
    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty


    def set_inputs(self, data, mode='train'):

        if mode == 'train':
            real_A = data['A']
            real_B = data['B']

            self.class_A = data['A_class']
            self.class_B = data['B_class']

            self.reals = torch.cat((real_A, real_B), 0)

            if len(self.gpu_ids) > 0:
                self.reals = self.reals.cuda()


        else:
            inputs = data['Imgs']
            if inputs.dim() > 4:
                inputs = inputs.squeeze(0)

            self.class_A = data['Classes']
            if self.class_A.dim() > 1:
                self.class_A = self.class_A.squeeze(0)

            if torch.is_tensor(data['Valid']):
                self.valid = data['Valid'].bool()
            else:
                self.valid = torch.ones(1, dtype=torch.bool)

            if self.valid.dim() > 1:
                self.valid = self.valid.squeeze(0)

            if isinstance(data['Paths'][0], tuple):
                self.image_paths = [path[0] for path in data['Paths']]
            else:
                self.image_paths = data['Paths']

            self.isEmpty = False if any(self.valid) else True
            if not self.isEmpty:
                available_idx = torch.arange(len(self.class_A))
                select_idx = torch.masked_select(available_idx, self.valid).long()
                inputs = torch.index_select(inputs, 0, select_idx)

                self.class_A = torch.index_select(self.class_A, 0, select_idx)
                self.image_paths = [val for i, val in enumerate(self.image_paths) if self.valid[i] == 1]

            self.reals = inputs

            if len(self.gpu_ids) > 0:
                self.reals = self.reals.cuda()


    def get_conditions(self, mode='train'):
        if mode == 'train':
            nb = self.reals.shape[0] // 2
        elif self.traverse or self.deploy:
            if self.traverse and self.compare_to_trained_outputs:
                nb = 2
            else:
                nb = self.numClasses
        else:
            nb = self.numValid

        condG_A_gen = self.Tensor(nb, self.cond_length)
        condG_B_gen = self.Tensor(nb, self.cond_length)
        condG_A_orig = self.Tensor(nb, self.cond_length)
        condG_B_orig = self.Tensor(nb, self.cond_length)


        if self.no_cond_noise:
            noise_sigma = 0
        else:
            noise_sigma = 0.2

        for i in range(nb):
            condG_A_gen[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
            condG_A_gen[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1
            if not (self.traverse or self.deploy):
                condG_B_gen[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_gen[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                condG_A_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_A_orig[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                condG_B_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_orig[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1

        if mode == 'train':
            self.gen_conditions =  torch.cat((condG_A_gen, condG_B_gen), 0) 
            self.cyc_conditions = torch.cat((condG_B_gen, condG_A_gen), 0)
            self.orig_conditions = torch.cat((condG_A_orig, condG_B_orig),0)

        else:
            self.gen_conditions = condG_A_gen
            if not (self.traverse or self.deploy):
                self.cyc_conditions = condG_B_gen
                self.orig_conditions = condG_A_orig

    def update_Generator(self, infer=False):
        # Generator optimization setp
        self.optimizer_G.zero_grad()
        self.get_conditions()

        rec_images, gen_images, cyc_images, orig_id_features, \
        orig_structure_feat, orig_texture_feat, orig_sem_feat, orig_age_features, fake_id_features, fake_struct_features, fake_sem_feat, fake_age_features = \
        self.netG(self.reals, self.gen_conditions, self.cyc_conditions, self.orig_conditions)

        disc_out = self.netD(gen_images)

        # self-reconstruction loss
        if self.opt.lambda_rec > 0:
            loss_G_Rec = self.criterionRec(rec_images, self.reals) * self.opt.lambda_rec
        else:
            loss_G_Rec = torch.zeros(1).cuda()

        # cycle loss
        if self.opt.lambda_cyc > 0:
            loss_G_Cycle = self.criterionCycle(cyc_images, self.reals) * self.opt.lambda_cyc
        else:
            loss_G_Cycle = torch.zeros(1).cuda()

        # identity feature loss
        loss_G_identity = self.identity_reconst_criterion(fake_id_features, orig_id_features) * self.opt.lambda_id
        
        # age feature loss
        loss_G_age = self.age_reconst_criterion(fake_age_features, self.gen_conditions) * self.opt.lambda_age

        # orig age feature loss
        loss_G_age += self.age_reconst_criterion(orig_age_features, self.orig_conditions) * self.opt.lambda_age

        # adversarial loss
        target_classes = torch.cat((self.class_B,self.class_A),0)
        loss_G_Adv = self.criterionGAN(disc_out, target_classes, True, is_gen=True)

        # semantic loss
        loss_G_semantic = self.criterionSem(orig_sem_feat, fake_sem_feat) * self.opt.lambda_sem
        
        # overall loss
        loss_G = (loss_G_Adv + loss_G_Rec + loss_G_Cycle + \
        loss_G_identity + loss_G_age + loss_G_semantic).mean()

        # loss_G_Adv : Adversarial Loss
        # loss_G_Rec : Self reconstruction loss
        # loss_G_cycle : Cycle consistency loss
        # loss_G_identity : ID loss
        # loss_G_age : Age loss
        # loss_G_semantic : Semantic loss

        loss_G.backward()
        self.optimizer_G.step()

        # update exponential moving average
        if self.use_moving_avg:
            self.accumulate(self.G_moving_avg, self.netG)

        # generate images for visdom
        if infer:
            if self.use_moving_avg:
                with torch.no_grad():
                    orig_id_features_out, orig_struct, orig_text, orig_semantic, _ = self.G_moving_avg.encode(self.reals)
                    # within domain decode
                    if self.opt.lambda_rec > 0:
                        rec_images_out = self.G_moving_avg.decode(orig_struct, orig_text, orig_semantic, self.orig_conditions)

                    # cross domain decode
                    gen_images_out = self.G_moving_avg.decode(orig_struct, orig_text, orig_semantic, self.gen_conditions)
                    # encode generated
                    fake_id_features_out, fake_struct, fake_text, fake_semantic, _ = self.G_moving_avg.encode(gen_images)
                    # decode generated
                    if self.opt.lambda_cyc > 0:
                        cyc_images_out = self.G_moving_avg.decode(fake_struct, fake_text, fake_semantic, self.cyc_conditions)
            else:
                gen_images_out = gen_images
                if self.opt.lambda_rec > 0:
                    rec_images_out = rec_images
                if self.opt.lambda_cyc > 0:
                    cyc_images_out = cyc_images

        loss_dict = {'loss_G_gen_Adv': loss_G_Adv.mean(), 'loss_G_Cycle': loss_G_Cycle.mean(),
                     'loss_G_Rec': loss_G_Rec.mean(), 'loss_G_identity_reconst': loss_G_identity.mean(),
                     'loss_G_age_reconst': loss_G_age.mean(), 'loss_G_semantic': loss_G_semantic.mean()
                     }

        return [loss_dict,
                None if not infer else self.reals,
                None if not infer else gen_images_out,
                None if not infer else rec_images_out,
                None if not infer else cyc_images_out]

    def update_Discriminator(self):

        self.optimizer_D.zero_grad()
        self.get_conditions()

        _, gen_images, _, _, _, _, _, _, _, _, _, _ = self.netG(self.reals, self.gen_conditions, None, None, disc_pass=True)

        # fake discriminator pass
        fake_disc_in = gen_images.detach()
        fake_disc_out = self.netD(fake_disc_in)

        # real discriminator pass
        real_disc_in = self.reals
        real_disc_in.requires_grad_()

        real_disc_out = self.netD(real_disc_in)

        # Fake GAN loss
        fake_target_classes = torch.cat((self.class_B,self.class_A),0)
        real_target_classes = torch.cat((self.class_A,self.class_B),0)

        loss_D_fake = self.criterionGAN(fake_disc_out, fake_target_classes, False, is_gen=False)

        # Real GAN loss
        loss_D_real = self.criterionGAN(real_disc_out, real_target_classes, True, is_gen=False)

        # R1 regularization
        loss_D_reg = self.R1_reg(real_disc_out, real_disc_in)

        # Contrastive loss
        loss_D_contrastive = (self.contrastive_loss(real_disc_out, real_target_classes.cuda()) 
                              + self.contrastive_loss(fake_disc_out, fake_target_classes.cuda())) * self.opt.lambda_contrastive

        loss_D = (loss_D_fake + loss_D_real + loss_D_reg + loss_D_contrastive).mean()
        loss_D.backward()
        self.optimizer_D.step()

        return {'loss_D_real': loss_D_real.mean(), 'loss_D_fake': loss_D_fake.mean(), 
                'loss_D_reg': loss_D_reg.mean(), 'loss_D_contrastive': loss_D_contrastive.mean()}


    def inference(self, data):
        self.set_inputs(data, mode='test')
        if self.isEmpty:
            return

        self.numValid = self.valid.sum().item()
        sz = self.reals.size()
        self.fake_B = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
        self.cyc_A = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])

        with torch.no_grad():
            if self.traverse or self.deploy:
                if self.traverse and self.compare_to_trained_outputs:
                    start = self.compare_to_trained_class - self.trained_class_jump
                    end = start + (self.trained_class_jump * 2) * 2
                    self.class_B = torch.arange(start, end, step=self.trained_class_jump*2, dtype=self.class_A.dtype)
                else:
                    self.class_B = torch.arange(self.numClasses, dtype=self.class_A.dtype)

                self.get_conditions(mode='test')

                self.fake_B = self.netG.infer(self.reals, self.gen_conditions, traverse=self.traverse, deploy=self.deploy, interp_step=self.opt.interp_step)
            else:
                for i in range(self.numClasses):
                    self.class_B = self.Tensor(self.numValid).long().fill_(i)
                    self.get_conditions(mode='test')

                    if self.isTrain:
                        self.fake_B[i, :, :, :, :] = self.G_moving_avg.infer(self.reals, self.gen_conditions)
                    else:
                        self.fake_B[i, :, :, :, :] = self.netG.infer(self.reals, self.gen_conditions)

                    cyc_input = self.fake_B[i, :, :, :, :]

                    if self.isTrain:
                        self.cyc_A[i, :, :, :, :] = self.G_moving_avg.infer(cyc_input, self.cyc_conditions)
                    else:
                        self.cyc_A[i, :, :, :, :] = self.netG.infer(cyc_input, self.cyc_conditions)

            visuals = self.get_visuals()

        return visuals


    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.use_moving_avg:
            self.save_network(self.G_moving_avg, 'G_moving_avg', which_epoch, self.gpu_ids)


    def update_learning_rate(self):
        lr = self.old_lr * self.opt.decay_gamma
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            mult = param_group.get('mult', 1.0)
            param_group['lr'] = lr * mult
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


    def get_visuals(self):
        return_dicts = [OrderedDict() for i in range(self.numValid)]

        real_A = util.tensor2im(self.reals.data)
        fake_B_tex = util.tensor2im(self.fake_B.data)

        if self.debug_mode:
            rec_A_tex = util.tensor2im(self.cyc_A.data[:,:,:,:,:])

        if self.numValid == 1:
            real_A = np.expand_dims(real_A, axis=0)

        for i in range(self.numValid):
            # get the original image and the results for the current samples
            curr_real_A = real_A[i, :, :, :]
            real_A_img = curr_real_A[:, :, :3]

            # start with age progression/regression images
            if self.traverse or self.deploy:
                curr_fake_B_tex = fake_B_tex
                orig_dict = OrderedDict([('original_img', real_A_img)])
            else:
                curr_fake_B_tex = fake_B_tex[:, i, :, :, :]
                orig_dict = OrderedDict([('original_img_class_' + str(self.class_A[i].item()), real_A_img)])

            return_dicts[i].update(orig_dict)

            if self.traverse:
                out_classes = curr_fake_B_tex.shape[0]
            else:
                out_classes = self.numClasses

            for j in range(out_classes):
                fake_res_tex = curr_fake_B_tex[j, :, :, :3]
                fake_dict_tex = OrderedDict([('transformed_class_' + str(j), fake_res_tex)])
                return_dicts[i].update(fake_dict_tex)

            if not (self.traverse or self.deploy):
                if self.debug_mode:
                    curr_rec_A_tex = rec_A_tex[:, i, :, :, :]
                    orig_dict = OrderedDict([('original_img2', real_A_img)])
                    return_dicts[i].update(orig_dict)
                    for j in range(self.numClasses):
                        rec_res_tex = curr_rec_A_tex[j, :, :, :3]
                        rec_dict_tex = OrderedDict([('reconstructed_from_class_' + str(j), rec_res_tex)])
                        return_dicts[i].update(rec_dict_tex)

        return return_dicts

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.size(0)
        feature_dim = features.size(1)
        features = features.view(batch_size, feature_dim, -1).mean(dim=2)

        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # Contrastive loss
        logits = similarity_matrix
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(features.device))
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)
        
        # Mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        loss = -mean_log_prob_pos.mean()
        return loss


class InferenceModel(HECL):
    def forward(self, data):
        return self.inference(data)
