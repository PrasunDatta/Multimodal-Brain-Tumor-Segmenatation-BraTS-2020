"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
import torch
import torch.nn as nn
from siren_pytorch import Sine
import torch.nn.functional as F
from config import config
class DownSampling(nn.Module):
    # 3x3x3 convolution and 1 padding as default
    def __init__(self, inChans, outChans, stride=2, kernel_size=3, padding=1, dropout_rate=None):
        super(DownSampling, self).__init__()
        
        self.dropout_flag = False
        self.conv1 = nn.Conv3d(in_channels=inChans, 
                     out_channels=outChans, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding,
                     bias=False)
        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(dropout_rate,inplace=True)
            
    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_flag:
            out = self.dropout(out)
        return out
    
class EncoderBlock(nn.Module):
    '''
    Encoder block; Green
    '''
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu", normalizaiton="group_normalization"):
        super(EncoderBlock, self).__init__()
        
        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        # elif activation == "elu":
        #     self.actv1 = nn.ELU(inplace=True)
        #     self.actv2 = nn.ELU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        
        
    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)
        
        out += residual
        
        return out


class AttentionBlock(nn.Module):
    '''
    stride = 2; please set --attention parameter to 1 when activate this block. the result name should include "att2".
    To fit in the structure of the V-net: F_l = F_int
    '''
    def __init__(self, F_g, F_l, F_int, scale_factor=2, mode="trilinear"):
        super(AttentionBlock, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  # reduce num_channels
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=scale_factor, stride=scale_factor, padding=0, bias=True),  # downsize
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, visualize=False):
        '''

        :param g: gate signal from coarser scale
        :param x: the output of the l-th layer in the encoder
        :param visualize: enable this when plotting attention matrix
        :return:
        '''
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        relu = self.relu(g1 + x1)
        sig = self.psi(relu)
        alpha = nn.functional.interpolate(sig, scale_factor=self.scale_factor, mode=self.mode)

        if visualize:
            return alpha
        else:
            return x * alpha


class LinearUpSampling(nn.Module):
    '''
    Trilinear interpolate to upsampling
    '''
    def __init__(self, inChans, outChans, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        # self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
    
    def forward(self, x, skipx=None):
        out = self.conv1(x)  # reduce num_channels
        out = nn.functional.interpolate(out, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        # up-size * 2

        if skipx is not None:
            out += skipx

        return out
    
class DecoderBlock(nn.Module):
    '''
    Decoder block
    '''
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu", normalizaiton="group_normalization"):
        super(DecoderBlock, self).__init__()
        
        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        # elif activation == "elu":
        #     self.actv1 = nn.ELU(inplace=True)
        #     self.actv2 = nn.ELU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=outChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        
        
    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)
        
        out += residual
        
        return out
    
class OutputTransition(nn.Module):
    '''
    Decoder output layer 
    output the prediction of segmentation result
    '''
    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.actv1 = torch.sigmoid
        
    def forward(self, x):
        return self.actv1(self.conv1(x))

class VDResampling(nn.Module):
    '''
    Variational Auto-Encoder Resampling block
    '''
    def __init__(self, inChans=256, outChans=256, dense_features=(10, 12, 8), stride=2, kernel_size=3, padding=1,
                 activation="relu", normalizaiton="group_normalization"):
        super(VDResampling, self).__init__()
        
        midChans = int(inChans / 2)
        self.dense_features = dense_features
        if normalizaiton == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dense1 = nn.Linear(in_features=16*dense_features[0]*dense_features[1]*dense_features[2], out_features=256)
        self.dense2 = nn.Linear(in_features=128, out_features=128*dense_features[0]*dense_features[1]*dense_features[2])
        self.up0 = LinearUpSampling(128, outChans)
        
    def forward(self, x):
        out = self.gn1(x)
        out = self.actv1(out)
        out = self.conv1(out)   # 16*10*12*8
        out = out.view(-1, self.num_flat_features(out))  # flatten
        out_vd = self.dense1(out)
        distr = out_vd 
        out = VDraw(out_vd)  # 128
        out = self.dense2(out)
        out = self.actv2(out)
        out = out.view((-1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))  # flat to conv
        # out = out.view((1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))
        out = self.up0(out)  # include conv1 and upsize 256*20*24*16
        
        return out, distr
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features

def VDraw(x):
    # Generate a Gaussian distribution with the given mean(128-d) and std(128-d)
    return torch.distributions.Normal(x[:, :128], x[:, 128:]).sample()

class VDecoderBlock(nn.Module):
    '''
    Variational Decoder block
    '''
    def __init__(self, inChans, outChans, activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VDecoderBlock, self).__init__()

        self.up0 = LinearUpSampling(inChans, outChans, mode=mode)
        self.block = DecoderBlock(outChans, outChans, activation=activation, normalizaiton=normalizaiton)
    
    def forward(self, x):
        out = self.up0(x)
        out = self.block(out)

        return out

class VAE(nn.Module):
    '''
    Variational Auto-Encoder : to group the features extracted by Encoder
    '''
    def __init__(self, inChans=256, outChans=4, dense_features=(10, 12, 8),
                 activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VAE, self).__init__()

        self.vd_resample = VDResampling(inChans=inChans, outChans=inChans, dense_features=dense_features)
        self.vd_block2 = VDecoderBlock(inChans, inChans//2)
        self.vd_block1 = VDecoderBlock(inChans//2, inChans//4)
        self.vd_block0 = VDecoderBlock(inChans//4, inChans//8)
        self.vd_end = nn.Conv3d(inChans//8, outChans, kernel_size=1)
        
    def forward(self, x):
        out, distr = self.vd_resample(x)
        out = self.vd_block2(out)
        out = self.vd_block1(out)
        out = self.vd_block0(out)
        out = self.vd_end(out)

        return out, distr

#Intermediate_sequential
class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs
# Transformer 
class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))

from torch import Tensor
import torch.nn.functional as F

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)
# Transformer Segment Ends Here.
#Position Encoding 
class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 7680, 512)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings
# Position encoding ends here.

#After transformer 2 enblocks 
class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm3d(512 // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512 // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class AttentionVNet(nn.Module):
    def __init__(self, config):
        super(AttentionVNet, self).__init__()
        
        self.config = config
        # some critical parameters
        self.inChans = config["input_shape"][1]
        self.input_shape = config["input_shape"]
        self.seg_outChans = 3
        self.activation = config["activation"]
        self.normalizaiton = config["normalizaiton"]
        self.mode = config["mode"]
        #Transformer Layer Addition
        self.img_dim = config["img_dim"]
        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.patch_dim = config["patch_dim"]
        self.num_channels = config["num_channels"]
        self.dropout_rate = config["dropout_rate"]
        self.attn_dropout_rate = config["attn_dropout_rate"]
        self.num_layers = config["num_layers"]
        self.hidden_dim = config["hidden_dim"]

        self.num_patches = int((self.img_dim // self.patch_dim) ** 3)
        self.seq_length = self.num_patches
        # self.flatten_dim = 128 * num_channels
        self.positional_encoding_type = config["positional_encoding_type"] 
        # self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if self.positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif self.positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            self.embedding_dim,
            self.num_layers,
            self.num_heads,
            self.hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)

        # if self.conv_patch_representation:

        self.conv_x = nn.Conv3d(
                256,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        # self.Unet = Unet(in_channels=4, base_channels=16, num_classes=4)
        # self.gn1 = nn.GroupNorm(num_groups=8, num_channels= 256)
        self.bn = nn.BatchNorm3d(256)
        self.relu = nn.ReLU(inplace=True)
        
        # Encoder Blocks
        self.in_conv0 = DownSampling(inChans=self.inChans, outChans=32, stride=1, dropout_rate=0.2)
        self.en_block0 = EncoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block1_1 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block2_1 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_1 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_2 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_3 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)

        # Decoder Blocks
        self.de_up2 = LinearUpSampling(256, 128, mode=self.mode)
        self.de_block2 = DecoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up1 = LinearUpSampling(128, 64, mode=self.mode)
        self.de_block1 = DecoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up0 = LinearUpSampling(64, 32, mode=self.mode)
        self.de_block0 = DecoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_end = OutputTransition(32, self.seg_outChans)

        # Attention Blocks
        self.ag_2 = AttentionBlock(256, 128, 128)  # forward(g, x)
        self.ag_1 = AttentionBlock(128, 64, 64)
        self.ag_0 = AttentionBlock(64, 32, 32)

        # Variational Auto-Encoder
        if self.config["VAE_enable"]:
            self.dense_features = (self.input_shape[2]//16, self.input_shape[3]//16, self.input_shape[4]//16)
            self.vae = VAE(256, outChans=self.inChans - 3, dense_features=self.dense_features)   ### 7 ——> 4

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 2)
    def decode(self,x_tr,intmd_x, intmd_layers=[1, 2, 3, 4]):

        assert intmd_layers is not None, "pass the intermediate layers for MLA"
        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()
        return encoder_outputs[all_keys[0]]

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.input_shape[2] / self.patch_dim),
            int(self.input_shape[3] / self.patch_dim),
            int(self.input_shape[4] / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

    def forward(self, x):
        out_init = self.in_conv0(x)  # (7, 128, 192, 160)
        out_en0 = self.en_block0(out_init)
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0))) 
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))
        out_en3 = self.en_block3_3(self.en_block3_2(self.en_block3_1(self.en_block3_0(self.en_down3(out_en2)))))

        transformer_input = out_en3
        print(f"Transformer_input: {transformer_input.shape}")
        x_tr = self.bn(transformer_input) #x = transformer_input
        x_tr = self.relu(x_tr)
        x_tr = self.conv_x(x_tr)
        x_tr = x_tr.permute(0, 2, 3, 4, 1).contiguous()
        x_tr = x_tr.view(x_tr.size(0), -1, self.embedding_dim)
        print(f"Before position embedding: {x_tr.shape}")
        x_tr = self.position_encoding(x_tr)
        x_tr = self.pe_dropout(x_tr)

        # apply transformer
        x_tr, intmd_x = self.transformer(x_tr)
        x_tr = self.pre_head_ln(x_tr)

 

        x8 = self.decode(x_tr,intmd_x,intmd_layers=[1, 2, 3, 4])

        x8 = self._reshape_output(x8)
        print(f"End_Trans_After_rehape: {x8.shape}")
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8)
        print(f"After Enblocks_Transformer Output: {x8.shape}")
        
        out_de2 = self.de_block2(self.de_up2(out_en3, self.ag_2(x8, out_en2)))
        out_de1 = self.de_block1(self.de_up1(out_de2, self.ag_1(out_de2, out_en1)))
        out_de0 = self.de_block0(self.de_up0(out_de1, self.ag_0(out_de1, out_en0)))

        out_end = self.de_end(out_de0)
        
        if self.config["VAE_enable"]:
            out_vae, out_distr = self.vae(x8)
            out_final = torch.cat((out_end, out_vae), 1)
            return out_final, out_distr
        
        return out_end

class AttentionVNetForVisual(AttentionVNet):
    def __init__(self, config):
        super(AttentionVNetForVisual, self).__init__(config)

    def forward(self, x):
        out_init = self.in_conv0(x)  # (7, 128, 192, 160)
        out_en0 = self.en_block0(out_init)  # 32 * 128^3
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0)))  # 64 * 64^3
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))  # 128 * 32^3
        out_en3 = self.en_block3_3(
            self.en_block3_2(
                self.en_block3_1(
                    self.en_block3_0(
                        self.en_down3(out_en2)))))  # 256 * 16^3

        out_de2 = self.de_block2(self.de_up2(out_en3, self.ag_2(out_en3, out_en2)))  # 128 * 32^3
        out_de1 = self.de_block1(self.de_up1(out_de2, self.ag_1(out_de2, out_en1)))  # 64 * 64^3
        att_matrix = self.ag_0(out_de1, out_en0, True)  # 32 * 128^3
        return att_matrix

if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 192, 160), device=cuda0)
        # _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        model = AttentionVNet(config=config)
        model.cuda()
        y, distr = model(x)
        print(y.shape)




