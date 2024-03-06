import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import functools
from torch.autograd import Function
from math import sqrt
import math

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def get_norm_layer(norm_type='instance'):
    if norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'pixel':
        norm_layer = PixelNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_Generator(input_nc, output_nc, ngf, n_downsample_global=2,
             id_enc_norm='pixel', gpu_ids=[], padding_type='reflect',
             style_dim=50, init_type='gaussian',
             conv_weight_norm=False, decoder_norm='pixel', activation='lrelu',
             adaptive_blocks=4, normalize_mlp=False, modulated_conv=False):

    id_enc_norm = get_norm_layer(norm_type=id_enc_norm)

    netG = Disentangled_Generator(input_nc, output_nc, ngf, n_downsampling=n_downsample_global,
                     id_enc_norm=id_enc_norm, padding_type=padding_type, style_dim=style_dim,
                     conv_weight_norm=conv_weight_norm, decoder_norm=decoder_norm,
                     actvn=activation, adaptive_blocks=adaptive_blocks,
                     normalize_mlp=normalize_mlp, modulated_conv=modulated_conv)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])

    netG.apply(weights_init(init_type))

    return netG

def define_D(input_nc, ndf, n_layers=6, numClasses=2, gpu_ids=[],
             init_type='gaussian'):

    netD = StyleGANDiscriminator(input_nc, ndf=ndf, n_layers=n_layers, numClasses=numClasses)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])

    netD.apply(weights_init(init_type))

    return netD

class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            print(name)
            return getattr(self.module, name)


class FeatureConsistency(nn.Module):
    def __init__(self):
        super(FeatureConsistency, self).__init__()

    def __call__(self,input,target):
        return torch.mean(torch.abs(input - target))


class R1_reg(nn.Module):
    def __init__(self, lambda_r1=10.0):
        super(R1_reg, self).__init__()
        self.lambda_r1 = lambda_r1

    def __call__(self, d_out, d_in):
        """Compute gradient penalty: (L2_norm(dy/dx))**2."""
        b = d_in.shape[0]
        dydx = torch.autograd.grad(outputs=d_out.mean(),
                                   inputs=d_in,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx_sq = dydx.pow(2)
        assert (dydx_sq.size() == d_in.size())
        r1_reg = dydx_sq.sum() / b

        return r1_reg * self.lambda_r1


class SelectiveClassesNonSatGANLoss(nn.Module):
    def __init__(self):
        super(SelectiveClassesNonSatGANLoss, self).__init__()
        self.sofplus = nn.Softplus()

    def __call__(self, input, target_classes, target_is_real, is_gen=False):
        bSize = input.shape[0]
        b_ind = torch.arange(bSize).long()
        relevant_inputs = input[b_ind, target_classes, :, :]
        if target_is_real:
            loss = self.sofplus(-relevant_inputs).mean()
        else:
            loss = self.sofplus(relevant_inputs).mean()

        return loss

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)

class ModulatedConv2d(nn.Module):
    def __init__(self, fin, fout, kernel_size, padding_type='reflect', upsample=False, downsample=False, latent_dim=256, normalize_mlp=False):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = fin
        self.out_channels = fout
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample
        padding_size = kernel_size // 2
        if kernel_size == 1:
            self.demudulate = False
        else:
            self.demudulate = True

        self.weight = nn.Parameter(torch.Tensor(fout, fin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1, fout, 1, 1))
        self.conv = F.conv2d

        if normalize_mlp:
            self.mlp_class_std = nn.Sequential(EqualLinear(latent_dim, fin), PixelNorm())
        else:
            self.mlp_class_std = EqualLinear(latent_dim, fin)

        self.blur = Blur(fout)

        if padding_type == 'reflect':
            self.padding = nn.ReflectionPad2d(padding_size)
        else:
            self.padding = nn.ZeroPad2d(padding_size)

        if self.upsample:
            self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        if self.downsample:
            self.downsampler = nn.AvgPool2d(2)

        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, input, latent):
        fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        weight = self.weight * sqrt(2 / fan_in)
        weight = weight.view(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        s = 1 + self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
        weight = s * weight
        if self.demudulate:
            d = torch.rsqrt((weight ** 2).sum(4).sum(3).sum(2) + 1e-5).view(-1, self.out_channels, 1, 1, 1)
            weight = (d * weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        else:
            weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = self.upsampler(input)

        if self.downsample:
            input = self.blur(input)

        b,_,h,w = input.shape
        input = input.view(1,-1,h,w)
        input = self.padding(input)
        out = self.conv(input, weight, groups=b).view(b, self.out_channels, h, w) + self.bias

        if self.downsample:
            out = self.downsampler(out)

        if self.upsample:
            out = self.blur(out)

        return out

class Modulated_1D(nn.Module):
    def __init__(self, in_channel, latent_dim, normalize_mlp=False):
        super().__init__()

        if normalize_mlp:
            self.mlp_class_std = nn.Sequential(EqualLinear(latent_dim, in_channel), PixelNorm())
        else:
            self.mlp_class_std = EqualLinear(latent_dim, in_channel)

    def forward(self, x, latent):

        s = 1 + self.mlp_class_std(latent)
        x = x * s
        d = torch.rsqrt((x ** 2).sum(1)+1e-5).view(-1,1)
        x = x * d
        return x
        
class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class StandardConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, input):
        return self.conv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction.apply

class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, fc_dim, n_fc, weight_norm=False, activation='relu', normalize_mlp=False):
        super(MLP, self).__init__()

        linear = EqualLinear if weight_norm else nn.Linear
        actvn = nn.LeakyReLU(0.2, True) if activation == 'lrelu' else nn.ReLU(True)

        layers = []

        if normalize_mlp:
            layers.append(PixelNorm())
        layers.extend(self._make_layer(input_dim, fc_dim, linear, actvn, normalize_mlp))

        for _ in range(n_fc - 2):
            layers.extend(self._make_layer(fc_dim, fc_dim, linear, actvn, normalize_mlp))
        layers.append(linear(fc_dim, out_dim))

        if normalize_mlp:
            layers.append(PixelNorm())
        self.model = nn.Sequential(*layers)

    def _make_layer(self, in_features, out_features, linear, actvn, normalize_mlp):
        layer = [linear(in_features, out_features), actvn]
        if normalize_mlp:
            layer.append(PixelNorm())
        return layer

    def forward(self, input):
        return self.model(input)

class StyledConvBlock(nn.Module):
    def __init__(self, fin, fout, latent_dim=256, padding='reflect', upsample=False, downsample=False,
                 actvn='lrelu', use_pixel_norm=False, normalize_affine_output=False, modulated_conv=False):
        super(StyledConvBlock, self).__init__()
        self.use_pixel_norm = use_pixel_norm
        self.upsample = upsample
        self.downsample = downsample
        self.modulated_conv = modulated_conv
        self.padding_type = padding
        self.actvn_type = actvn

        self.actvn_gain = sqrt(2) if modulated_conv else 1.0
        activation = nn.ReLU(True) if actvn == 'relu' else nn.LeakyReLU(0.2, True)

        conv2d = ModulatedConv2d if modulated_conv else EqualConv2d
        padding_layer = nn.ReflectionPad2d if padding == 'reflect' else nn.ZeroPad2d

        self.conv0 = self._make_layer(fin, fout, latent_dim, normalize_affine_output, conv2d, padding_layer, activation)
        self.conv1 = self._make_layer(fout, fout, latent_dim, normalize_affine_output, conv2d, padding_layer, activation, is_second=True)

    def _make_layer(self, fin, fout, latent_dim, normalize_affine_output, conv2d, padding_layer, activation, is_second=False):
        if self.modulated_conv:
            return conv2d(fin, fout, kernel_size=3, padding_type=self.padding_type, upsample=self.upsample and not is_second,
                          downsample=self.downsample and is_second, latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            layers = []
            if self.upsample and not is_second:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(padding_layer(1))
            layers.append(conv2d(fin, fout, kernel_size=3))
            if self.downsample and is_second:
                layers.append(Blur(fout))
                layers.append(nn.AvgPool2d(2))
            return nn.Sequential(*layers)

    def forward(self, input, latent=None):
        out = self.conv0(input, latent) if self.modulated_conv else self.conv0(input)
        out = self._apply_activation(out)

        out = self.conv1(out, latent) if self.modulated_conv else self.conv1(out)
        out = self._apply_activation(out)

        return out

    def _apply_activation(self, out):
        activation = nn.ReLU(True) if self.actvn_type == 'relu' else nn.LeakyReLU(0.2, True)
        out = activation(out) * self.actvn_gain
        if self.use_pixel_norm:
            out = PixelNorm()(out)
        return out

class DisentangledEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=7,
                 norm_layer=PixelNorm, padding_type='reflect',
                 conv_weight_norm=False, activation_type='relu'):
        assert n_blocks >= 0
        super(DisentangledEncoder, self).__init__()

        # Define parameters
        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        conv2d = EqualConv2d if conv_weight_norm else nn.Conv2d
        activation = nn.LeakyReLU(0.2, True) if activation_type == 'lrelu' else nn.ReLU(True)

        encoder1_layers = [
            padding_layer(3), 
            conv2d(input_nc, ngf, kernel_size=7, padding=0), 
            norm_layer(ngf), 
            activation
        ]

        for i in range(n_downsampling):
            mult = 2 ** i
            encoder1_layers.extend([
                padding_layer(1),
                conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                norm_layer(ngf * mult * 2), 
                activation
            ])
        self.encoder1 = nn.Sequential(*encoder1_layers)

        mult = 2 ** n_downsampling

        encoder2_layers = [
            ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)
            for _ in range(n_blocks - 1)
        ]
        self.encoder2 = nn.Sequential(*encoder2_layers)

        id_layer = [
            ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, conv_weight_norm=conv_weight_norm),
            padding_layer(1),
            conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
            norm_layer(ngf * mult * 2), 
            activation
        ]
        self.id_layer = nn.Sequential(*id_layer)

        self.structure_layer = ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)

        text_layers = [
            padding_layer(1), 
            conv2d(256, 512, kernel_size=3, stride=2, padding=0), 
            norm_layer(ngf * mult * 2), 
            activation,
            padding_layer(1), 
            conv2d(512, 1024, kernel_size=3, stride=2, padding=0), 
            norm_layer(ngf * mult * 2), 
            activation,
            nn.AdaptiveAvgPool2d(1),
            conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        ]
        self.text_layer = nn.Sequential(*text_layers)

        # layer for morphlogical features
        morphological_layers = [
            padding_layer(1),
            conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=0),
            norm_layer(ngf * mult),
            activation,
            nn.AdaptiveAvgPool2d(1),
            conv2d(ngf * mult, 256, kernel_size=1, stride=1, padding=0)
        ]
        self.morphological_layer = nn.Sequential(*morphological_layers)

    def forward(self, input):
        feat1 = self.encoder1(input)
        structure = self.structure_layer(feat1)
        feat2 = self.encoder2(feat1)
        id_feat = self.id_layer(feat2)
        texture = self.text_layer(feat2)
        morph = self.morphological_layer(feat2)

        return id_feat, structure, texture, morph

class Disentangled_AgeEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=4, style_dim=100, padding_type='reflect', conv_weight_norm=False, actvn='lrelu'):
        super(Disentangled_AgeEncoder, self).__init__()

        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        conv2d = EqualConv2d if conv_weight_norm else nn.Conv2d
        activation = nn.LeakyReLU(0.2, True) if actvn == 'lrelu' else nn.ReLU(True)

        # Basic feature extraction
        self.basic_encoder = nn.Sequential(
            padding_layer(3),
            conv2d(input_nc, ngf, kernel_size=7, padding=0),
            activation,
            *self._make_downsampling_layers(ngf, n_downsampling, conv2d, padding_layer, activation)
        )

        # Fine-Grained Age-Related Features
        self.fine_age_features = nn.Sequential(
            conv2d(ngf * 2**n_downsampling, ngf * 2**(n_downsampling + 1), kernel_size=3, padding=1),
            activation,
            conv2d(ngf * 2**(n_downsampling + 1), style_dim, kernel_size=3, padding=1),
            activation,
            nn.AdaptiveAvgPool2d(1)
        )

        # Global Age-Related Features
        self.global_age_features = nn.Sequential(
            conv2d(ngf * 2**n_downsampling, ngf * 2**(n_downsampling + 2), kernel_size=5, padding=2),
            activation,
            nn.AdaptiveAvgPool2d(2),
            conv2d(ngf * 2**(n_downsampling + 2), style_dim, kernel_size=1),
            activation
        )

    def _make_downsampling_layers(self, ngf, n_downsampling, conv2d, padding_layer, activation):
        layers = []
        for i in range(n_downsampling):
            mult = 2**i
            layers.extend([
                padding_layer(1),
                conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                activation
            ])
        return layers

    def forward(self, input):
        basic_features = self.basic_encoder(input)
        fine_features = self.fine_age_features(basic_features).view(basic_features.size(0), -1)
        global_features = self.global_age_features(basic_features).view(basic_features.size(0), -1)
        combined_features = torch.cat((fine_features, global_features), dim=1)
        return combined_features


class Disentangled_StyledDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, style_dim=50, latent_dim=256, n_downsampling=2,
                 padding_type='reflect', actvn='lrelu', use_tanh=True, use_pixel_norm=False,
                 normalize_mlp=False, modulated_conv=False):
        super(Disentangled_StyledDecoder, self).__init__()

        mult = 2**n_downsampling # mult = 4
        last_upconv_out_layers = ngf * mult // 4 # last_upconv_out_layers = 64


        self.StyledConvBlock_0 = StyledConvBlock(256, 256, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)
        
        self.StyledConvBlock_1 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_2 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_3 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)
        
        self.StyledConvBlock_up0 = StyledConvBlock(256, 128, latent_dim=latent_dim,
                                                   padding=padding_type, upsample=True, actvn=actvn,
                                                   use_pixel_norm=use_pixel_norm,
                                                   normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)
        
        self.StyledConvBlock_up1 = StyledConvBlock(128, 64, latent_dim=latent_dim,
                                                   padding=padding_type, upsample=True, actvn=actvn,
                                                   use_pixel_norm=use_pixel_norm,
                                                   normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)

        self.conv_img = nn.Sequential(EqualConv2d(last_upconv_out_layers, output_nc, 1), nn.Tanh())
        self.mlp = MLP(style_dim, latent_dim, 256, 8, weight_norm=True, activation=actvn, normalize_mlp=normalize_mlp)

        self.s_transform = ModulatedConv2d(256, 256, kernel_size=3, padding_type=padding_type, upsample=False,
                                           latent_dim=256, normalize_mlp=normalize_mlp)

        self.t_transform = Modulated_1D(256,256,normalize_mlp=normalize_mlp)

    def forward(self, struct_feat, text_feat, morph_feat, target_age=None, traverse=False, deploy=False, interp_step=0.5):

        if target_age is not None:
            if traverse:
                alphas = torch.arange(1,0,step=-interp_step).view(-1,1).cuda()
                interps = len(alphas)
                orig_class_num = target_age.shape[0]
                output_classes = interps * (orig_class_num - 1) + 1
                temp_latent = self.mlp(target_age)
                latent = temp_latent.new_zeros((output_classes, temp_latent.shape[1]))
            else:
                latent = self.mlp(target_age)
        else:
            latent = None

        if traverse:
            struct_feat = struct_feat.repeat(output_classes, 1,1,1)
            text_feat = text_feat.repeat(output_classes, 1,1,1)
            morph_feat = morph_feat.repeat(output_classes, 1,1,1)
            for i in range(orig_class_num-1):
                latent[interps*i:interps*(i+1), :] = alphas * temp_latent[i,:] + (1 - alphas) * temp_latent[i+1,:]
            latent[-1,:] = temp_latent[-1,:]
        elif deploy:
            output_classes = target_age.shape[0]
            struct_feat = struct_feat.repeat(output_classes, 1,1,1)
            text_feat = text_feat.repeat(output_classes, 1,1,1)
            morph_feat = morph_feat.repeat(output_classes, 1,1,1)

        if target_age is not None:
            B, C, W, H = struct_feat.size()         
            new_struct = self.s_transform(struct_feat,latent)
            text_feat = text_feat.contiguous().reshape(B,C)
            new_text = self.t_transform(text_feat,latent)
        else:
            B, C, W, H = struct_feat.size()
            new_struct = struct_feat
            new_text = text_feat.contiguous().reshape(B,C)

        combined_feat = new_struct + morph_feat

        out = self.StyledConvBlock_0(combined_feat, new_text)
        out = self.StyledConvBlock_1(out, new_text)
        out = self.StyledConvBlock_2(out, new_text)
        out = self.StyledConvBlock_3(out, new_text)
        out = self.StyledConvBlock_up0(out, new_text)
        out = self.StyledConvBlock_up1(out, new_text)
        out = self.conv_img(out)

        return out

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise

# Disentangled generator with disentagled id-encoder and disentangled face aging
class Disentangled_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, style_dim=50, n_downsampling=2,
                 n_blocks=4, adaptive_blocks=4, id_enc_norm=PixelNorm,
                 padding_type='reflect', conv_weight_norm=False,
                 decoder_norm='pixel', actvn='lrelu', normalize_mlp=False,
                 modulated_conv=False):
        super(Disentangled_Generator, self).__init__()
        self.id_encoder = DisentangledEncoder(input_nc, ngf, n_downsampling, n_blocks, id_enc_norm,
                                          padding_type, conv_weight_norm=conv_weight_norm,
                                          activation_type='relu')
        self.age_encoder = Disentangled_AgeEncoder(input_nc, ngf=ngf, n_downsampling=4, style_dim=100,
                                      padding_type=padding_type, actvn=actvn,
                                      conv_weight_norm=conv_weight_norm)

        use_pixel_norm = decoder_norm == 'pixel'
        self.decoder = Disentangled_StyledDecoder(output_nc, ngf=ngf, style_dim=style_dim,
                                     n_downsampling=n_downsampling, actvn=actvn,
                                     use_pixel_norm=use_pixel_norm,
                                     normalize_mlp=normalize_mlp,
                                     modulated_conv=modulated_conv)

    def encode(self, input):
        if torch.is_tensor(input):
            id_features, struct_features, text_features, morph_features = self.id_encoder(input)
            age_features = self.age_encoder(input)
            return id_features, struct_features, text_features, morph_features, age_features
        else:
            return None, None, None, None, None

    def decode(self, struct_features, text_features, morph_features, target_age_features=None, traverse=False, deploy=False, interp_step=0.5):
        if torch.is_tensor(struct_features):
            return self.decoder(struct_features, text_features, morph_features, target_age=target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step)
        else:
            return None

    # parallel forward
    def forward(self, input, target_age_code, cyc_age_code, source_age_code, disc_pass=False):
        orig_id_features, orig_structure_feat, orig_texture_feat, orig_morph_feat = self.id_encoder(input)
        orig_age_features = self.age_encoder(input)
        if disc_pass:
            rec_out = None

        else:
            rec_out = self.decode(orig_structure_feat, orig_texture_feat, orig_morph_feat, target_age_features = source_age_code)

        gen_out = self.decode(orig_structure_feat, orig_texture_feat, orig_morph_feat, target_age_features = target_age_code)

        if disc_pass:
            fake_id_features = None
            fake_structure_feat = None
            fake_morph_feat = None
            fake_age_features = None
            cyc_out = None
        else:
            fake_id_features, fake_structure_feat, fake_texture_feat, fake_morph_feat = self.id_encoder(gen_out)
            fake_age_features = self.age_encoder(gen_out)
            cyc_out = self.decode(fake_structure_feat, fake_texture_feat, fake_morph_feat, target_age_features = cyc_age_code)
        return rec_out, gen_out, cyc_out, orig_id_features, orig_structure_feat, orig_texture_feat, orig_morph_feat, orig_age_features, \
            fake_id_features, fake_structure_feat, fake_morph_feat, fake_age_features

    def infer(self, input, target_age_features, traverse=False, deploy=False, interp_step=0.5):
        id_features, structure_feat, texture_feat, morph_feat = self.id_encoder(input)
        out = self.decode(structure_feat, texture_feat, morph_feat, target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step)
        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True),
                 conv_weight_norm=False, use_pixel_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation,
                                                conv_weight_norm, use_pixel_norm)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, conv_weight_norm, use_pixel_norm):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        self.use_pixel_norm = use_pixel_norm
        if self.use_pixel_norm:
            self.pixel_norm = PixelNorm()

        conv_block += [conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class StyleGANDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, numClasses=2, padding_type='reflect'):
        super(StyleGANDiscriminator, self).__init__()
        self.n_layers = n_layers
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        activation = nn.LeakyReLU(0.2,True)

        sequence = [padding_layer(0), EqualConv2d(input_nc, ndf, kernel_size=1), activation]

        nf = ndf
        for n in range(n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [StyledConvBlock(nf_prev, nf, downsample=True, actvn=activation)]

        self.model = nn.Sequential(*sequence)

        output_nc = numClasses
        self.gan_head = nn.Sequential(padding_layer(1), EqualConv2d(nf+1, nf, kernel_size=3), activation,
                                      EqualConv2d(nf, output_nc, kernel_size=4), activation)

    def minibatch_stdev(self, input):
        out_std = torch.sqrt(input.var(0, unbiased=False) + 1e-8)
        mean_std = out_std.mean()
        mean_std = mean_std.expand(input.size(0), 1, input.size(2), input.size(3))
        out = torch.cat((input, mean_std), 1)
        return out

    def forward(self, input):
        features = self.model(input)
        gan_out = self.gan_head(self.minibatch_stdev(features))
        return gan_out


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)