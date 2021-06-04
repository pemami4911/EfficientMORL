import torch
import torch.nn as nn
import math
from sacred import Ingredient
from lib.dual_gru import GRUCell, DualGRU
from lib.utils import init_weights, _softplus_to_std, mvn, std_mvn
from lib.utils import gmm_negativeloglikelihood, gaussian_negativeloglikelihood
from lib.geco import GECO
import numpy as np

net = Ingredient('Net')

@net.config
def cfg():
    input_size = [3,64,64] # [C, H, W]
    z_size = 64
    K = 4
    stochastic_layers = 3  # L
    refinement_iters = 3  # I
    log_scale = math.log(0.1)  # log base e
    geco_warm_start = 1000
    image_decoder = 'iodine'  # iodine (CLEVR6) small (tetris/sprites)
    image_likelihood = 'Gaussian'  # Gaussian, GMM
    use_DualGRU = True
    bottom_up_prior = False  # if False, uses reverse prior
    reverse_prior_plusplus = True  # if True, use reverse prior++


class SlotAttention(nn.Module):
    @net.capture
    def __init__(self, K, z_size, input_size, batch_size):
        super(SlotAttention, self).__init__()
        self.K = K
        self.z_size = z_size
        self.scale = z_size ** -0.5
        self.eps = 1e-8
        self.attention_iters = 3
        self.C = input_size[0]
        self.H = input_size[1]
        self.W = input_size[2]

        self.positional_embedding = SlotAttention.create_positional_embedding(self.H, self.W, batch_size)
        self.pos_embed_projection = nn.Linear(4, 64)
        self.encoder_pt_1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True)
        )
        self.encoder_pt_2 = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64,64),
            nn.ReLU(True),
            nn.Linear(64,64)
        )

        self.norm_slots = nn.LayerNorm(self.z_size)
        self.norm_pre_ff = nn.LayerNorm(self.z_size)

        self.to_q = nn.Linear(self.z_size, self.z_size, bias = False)
        self.to_k = nn.Linear(64, self.z_size, bias = False)
        self.to_v = nn.Linear(64, self.z_size, bias = False)

        self.mlp = nn.Sequential(
                nn.Linear(self.z_size, 128),
                nn.ReLU(True),
                nn.Linear(128, self.z_size)
        )
        self.gru = nn.GRU(self.z_size, self.z_size)
        self.init_slots = nn.Parameter(torch.cat([torch.zeros(1,self.z_size), torch.ones(1,self.z_size)],1))
        
        self.image_decoder = ImageDecoder(z_size=z_size, batch_size=batch_size)
        init_weights(self.image_decoder, 'xavier')



    @staticmethod
    def create_positional_embedding(h, w, batch_size):
        dist_right = torch.linspace(1, 0, w).view(w,1,1).repeat(1,h,1)  # [w,h,1]
        dist_left = torch.linspace(0, 1, w).view(w,1,1).repeat(1,h,1)
        dist_top = torch.linspace(0, 1, h).view(1,h,1).repeat(w,1,1)
        dist_bottom = torch.linspace(1, 0, h).view(1,h,1).repeat(w,1,1)
        return torch.cat([dist_right, dist_left, dist_top, dist_bottom],2).unsqueeze(0).repeat(batch_size,1,1,1)

    
    def forward(self, inputs):
        """
        inputs (if not an image) is [b, n, d1], else [N,C,H,W]
        slots ~ \mathcal{N}(\lambda_0). FloatTensor of shape [b, k, d2]

        output is [b,k,d2]
        """
        x_orig = inputs

        pos_embed = self.positional_embedding.to(inputs.device)
        inputs = self.encoder_pt_1(inputs)
        inputs = inputs.permute(0,2,3,1).contiguous()  # [N,H,W,64]
        pos_embed = self.pos_embed_projection(pos_embed)
        inputs += pos_embed  # [N,H,W,64]
        inputs = inputs.view(pos_embed.shape[0], -1, 64)  # [N,64,H*W]
        inputs = self.encoder_pt_2(inputs)  # [N,H*W,64]
        b, n, d = inputs.shape

        k, v = self.to_k(inputs), self.to_v(inputs)
        
        slots = self.init_slots.repeat(b * self.K, 1)  # [b,k,2*z_size]
        loc, sp = slots.chunk(2, dim=1)
        loc = loc.contiguous()
        sp = sp.contiguous()
        init_slot_distribution = mvn(loc, sp)
        slots = init_slot_distribution.rsample()
        slots = slots.view(b, self.K, self.z_size)
       
        for i in range(self.attention_iters):
            slots_prev = slots  # [N, K, z_size]
            
            slots = self.norm_slots(slots_prev)
            q = self.to_q(slots)            

            #dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            q *= self.scale
            dots = torch.einsum('bid,bjd->bij', q, k)
            
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)
            
            self.gru.flatten_parameters()
            slots, _ = self.gru(
                    updates.reshape(1,-1,self.z_size),  # [1, N*K, self.z_size]
                    slots_prev.reshape(1,-1,self.z_size))  # [1, N*K, self.z_size]
            
            slots = slots.reshape(b, -1, self.z_size)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        # decode
        slots = slots.view(-1, self.z_size)
        # [N*K, M, C, H, W], [N*K, M, 1, H, W]
        x_loc, mask_logits = self.image_decoder(slots)
        slots = slots.view(-1, self.K, self.z_size)
        mask_logits = mask_logits.reshape(-1, self.K, 1, self.H, self.W)
        x_loc = x_loc.view(-1, self.K, self.C, self.H, self.W)
        mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)
    
        x_orig = (x_orig + 1) / 2.
        mse = ((x_orig - (x_loc * mask_logprobs.exp()).sum(1)) ** 2).sum()
        mse = mse / b

        outs = {
            'total_loss': mse,
            'x_means': x_loc,
            'masks': mask_logprobs,
            'slots': slots
        }
        return outs


class ImageDecoder(nn.Module):
    """
    Decodes the individual Gaussian image componenets
    into RGB and mask
    """
    @net.capture
    def __init__(self, input_size, z_size, image_decoder, K, batch_size):
        super(ImageDecoder, self).__init__()
        self.h, self.w = input_size[1], input_size[2]
        output_size = 4
        small_grid_size = 6
        # Strides (2,2) with padding --- goes down to (8,8). From Slot Attention paper
        if image_decoder == 'big':
            self.decode = nn.Sequential(
                nn.ConvTranspose2d(z_size, 64, 5, 2, 2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 5, 2, 2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 5, 2, 2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 5, 2, 2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 5, 1, 2, output_padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, output_size, 3, 1, 1, output_padding=0)
            )
            self.z_grid_shape = (small_grid_size, small_grid_size)
            self.positional_embedding = SlotAttention.create_positional_embedding(small_grid_size, small_grid_size, K * batch_size)
        elif image_decoder == 'iodine':
            self.decode = nn.Sequential(
                nn.Conv2d(z_size, 64, 3, 1),
                nn.ELU(True),
                nn.Conv2d(64, 64, 3, 1),
                nn.ELU(True),
                nn.Conv2d(64, 64, 3, 1),
                nn.ELU(True),
                nn.Conv2d(64, 64, 3, 1),
                nn.ELU(True),
                nn.Conv2d(64, output_size, 3, 1)
            )
            self.z_grid_shape = (self.h + 10, self.w + 10)
            self.positional_embedding = SlotAttention.create_positional_embedding(
                    self.z_grid_shape[0], self.z_grid_shape[1], K * batch_size)

        elif image_decoder == 'small':
            self.decode = nn.Sequential(
                nn.Conv2d(z_size, 32, 5, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 5, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 5, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, output_size, 3, 1, 1)
            )
            self.z_grid_shape = (self.h + 6, self.w + 6)
            self.positional_embedding = SlotAttention.create_positional_embedding(
                    self.z_grid_shape[0], self.z_grid_shape[1], K * batch_size)
        self.pos_embed_projection = nn.Linear(4, z_size)


    def forward(self, z):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, self.z_grid_shape[0], self.z_grid_shape[1])
        
        pos_embed = self.positional_embedding.to(z.device)  # [N,H,W,4]
        pos_embed = self.pos_embed_projection(pos_embed)  # [N,H,W,64]
        pos_embed = pos_embed.permute(0,3,1,2).contiguous()

        z_b = z_b + pos_embed
        out = self.decode(z_b) # [batch_size * K, output_size, h, w]
        return torch.sigmoid(out[:,:3]), out[:,3]


class IndependentPrior(nn.Module):
    @net.capture
    def __init__(self, z_size, K):
        super(IndependentPrior, self).__init__()
        self.z_size = z_size
        self.K = K
        self.z_linear = nn.Sequential(
            nn.Linear(self.z_size, 128),
            nn.ELU(True))
        self.z_mu = nn.Linear(128, self.z_size)
        self.z_softplus = nn.Linear(128, self.z_size)

        init_weights(self.z_linear, 'xavier')
        init_weights(self.z_mu, 'xavier')
        init_weights(self.z_softplus, 'xavier')


    def forward(self, slots):
        """
        slots is [N,K,D]
        """
        slots = self.z_linear( slots )  # [N,K,D]
        loc_z = self.z_mu( slots )
        sp_z = self.z_softplus( slots )
        return loc_z, sp_z

class RefinementNetwork(nn.Module):
    """
    EM refinement
    """
    @net.capture
    def __init__(self, z_size):
        super(RefinementNetwork, self).__init__()
        
        self.recurrence = nn.GRU(z_size, z_size)
        self.encoding  = nn.Sequential(
            nn.Linear(4 * z_size, 128),
            nn.ELU(True),
            nn.Linear(128, z_size)
        )
        self.loc = nn.Linear(z_size, z_size)
        self.softplus = nn.Linear(z_size, z_size)

        init_weights(self.loc, 'xavier')
        init_weights(self.softplus, 'xavier')
        init_weights(self.encoding, 'xavier')

        self.loc_LN = nn.LayerNorm((z_size,), elementwise_affine=False)
        self.softplus_LN = nn.LayerNorm((z_size,), elementwise_affine=False)
        

    def forward(self, loss, lamda, hidden_state, eval_mode):
        """
        Args: 
            loss: [N] scalar outputs provided to torch.autograd.grad  
            lamda: [N*K, 2 * z_size] current posterior parameters
        Returns:
            lamda_next: [N*K, 2 * z_size], the updated posterior parameters
            hidden_state: next recurrent hidden state
        """
        d_lamda = torch.autograd.grad(loss, lamda, create_graph=not eval_mode, \
            retain_graph=not eval_mode, only_inputs=True)

        d_loc, d_sp = d_lamda[0].chunk(2, 1)
        d_loc, d_sp = d_loc.contiguous(), d_sp.contiguous()

        d_loc = self.loc_LN(d_loc).detach()
        d_sp = self.softplus_LN(d_sp).detach()

        x = self.encoding( torch.cat([lamda, d_loc, d_sp], 1))
        x = x.unsqueeze(0)
        self.recurrence.flatten_parameters()
        x, hidden_state = self.recurrence(x, hidden_state)
        x = x.squeeze(0)
        return torch.cat([self.loc(x), self.softplus(x)], 1), hidden_state


class HVAENetworks(nn.Module):
    @net.capture
    def __init__(self, K, z_size, input_size, stochastic_layers, use_DualGRU, batch_size):
        super(HVAENetworks, self).__init__()
        self.K = K
        self.z_size = z_size
        self.batch_size = batch_size
        self.num_stochastic_layers = stochastic_layers
        self.scale = z_size ** -0.5
        self.eps = 1e-8
        self.C, self.H, self.W = input_size
        self.use_DualGRU = use_DualGRU

        h = input_size[1]
        w = input_size[2]
        self.positional_embedding = SlotAttention.create_positional_embedding(h,w, batch_size)
        self.pos_embed_projection = nn.Linear(4, 64)
        self.encoder_pt_1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True)
        )
        self.encoder_pt_2 = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64,64),
            nn.ReLU(True),
            nn.Linear(64,64)
        )
        
        self.norm_slots = nn.LayerNorm(self.z_size)
        self.norm_mu_pre_ff = nn.LayerNorm(self.z_size)
        self.norm_softplus_pre_ff = nn.LayerNorm(self.z_size)
         
        self.to_q = nn.Linear(self.z_size, self.z_size, bias = False)
        self.to_k = nn.Linear(64, self.z_size, bias = False)
        self.to_v = nn.Linear(64, self.z_size, bias = False)

        if self.use_DualGRU:
            self.gru = DualGRU(GRUCell, self.z_size, self.z_size)
        else:
            self.gru = nn.GRU(2*self.z_size, 2*self.z_size)
            self.project_update = nn.Linear(self.z_size, 2*self.z_size)
            init_weights(self.project_update, 'xavier')

        self.mlp_mu = nn.Sequential(
                nn.Linear(self.z_size, self.z_size*2),
                nn.ReLU(True),
                nn.Linear(self.z_size*2, self.z_size)
        )
        self.mlp_softplus = nn.Sequential(
                nn.Linear(self.z_size, 2 * self.z_size),
                nn.ReLU(True),
                nn.Linear(self.z_size * 2, self.z_size)
        )

        self.image_decoder = ImageDecoder(z_size=z_size, batch_size=batch_size)
        self.init_posterior = nn.Parameter(torch.cat([torch.zeros(1,self.z_size), torch.ones(1,self.z_size)],1))
        self.indep_prior = IndependentPrior()

        init_weights(self.encoder_pt_1, 'xavier')
        init_weights(self.encoder_pt_2, 'xavier')
        init_weights(self.to_q, 'xavier')
        init_weights(self.to_k, 'xavier')
        init_weights(self.to_v, 'xavier')             
        init_weights(self.mlp_mu, 'xavier')
        init_weights(self.mlp_softplus, 'xavier')
        init_weights(self.image_decoder, 'xavier')        
        

    def forward(self, x, debug):

        pos_embed = self.positional_embedding.to(x.device)
        x = self.encoder_pt_1(x)
        x = x.permute(0,2,3,1).contiguous()  # [N,H,W,64]
        pos_embed = self.pos_embed_projection(pos_embed)
        x += pos_embed  # [N,H,W,64]
        x = x.view(pos_embed.shape[0], -1, 64)  # [N,64,H*W]
        inputs = self.encoder_pt_2(x)  # [N,H*W,64]

        x_locs, masks, posteriors = [], [], []
        all_samples = {}

        lamda = self.init_posterior.repeat(self.batch_size * self.K, 1)  # [N*K,2*z_size]
        # For L = 0 case...
        loc, sp = lamda.chunk(2, dim=1)
        loc = loc.contiguous()
        sp = sp.contiguous()
        init_posterior = mvn(loc, sp)
        slots = init_posterior.rsample()
        loc_shape = loc.shape
        slots = slots.view(-1, self.K, self.z_size)  # [N, K, D]
        
        slots_mu = loc
        slots_softplus = sp

        k, v = self.to_k(inputs), self.to_v(inputs)
        
        for layer in range(self.num_stochastic_layers):  
            # scaled dot-product attention
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            q *= self.scale

            dots = torch.einsum('bid,bjd->bij', q, k)
            # dots is [N, K, HW]
            attn = dots.softmax(dim=1) + self.eps
            all_samples[f'attn_{layer}'] = attn.view(self.batch_size, self.K, 1, self.H, self.W).detach()
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            if self.use_DualGRU:
                updates_recurrent = torch.cat([updates, updates],2)
                slots_recurrent = torch.cat([slots_mu, slots_softplus],1)
                slots_recurrent, _ = self.gru(
                    updates_recurrent.reshape(1,-1,2*self.z_size),
                    slots_recurrent.reshape(-1,2*self.z_size))
                slots_mu, slots_softplus = slots_recurrent[0].chunk(2,1)
            else:
                updates_recurrent = self.project_update(updates)
                slots_recurrent = torch.cat([slots_mu, slots_softplus],1)
                slots_recurrent, _ = self.gru(
                    updates_recurrent.reshape(1,-1,2*self.z_size),
                    slots_recurrent.reshape(1,-1,2*self.z_size))
                slots_mu, slots_softplus = slots_recurrent[0].chunk(2,1)                
                
            slots_mu = slots_mu + self.mlp_mu(self.norm_mu_pre_ff(slots_mu))
            slots_softplus = slots_softplus + self.mlp_softplus(self.norm_softplus_pre_ff(slots_softplus))
            
            # necessary for autodiff in refinement steps
            lamda = torch.cat([slots_mu, slots_softplus], 1)  # [N*K, 2*z_size]
            slots_mu, slots_softplus = lamda.chunk(2,1)

            posterior_z = mvn(slots_mu, slots_softplus)
            slots = posterior_z.rsample()

            posteriors += [posterior_z]
            #all_samples[f'posterior_z_{layer}'] = slots.view(-1, self.K, self.z_size)
            
            if debug:
                # decode
                slots_ = slots.view(-1, self.z_size)
                # [N*K, M, C, H, W], [N*K, M, 1, H, W]
                x_loc, mask_logits = self.image_decoder(slots_)
                slots_ = slots_.view(-1, self.K, self.z_size)
                mask_logits = mask_logits.view(-1, self.K, 1, self.H, self.W)
                mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)                
                x_loc = x_loc.view(-1, self.K, self.C, self.H, self.W)
                all_samples[f'means_{layer}'] = x_loc
                all_samples[f'masks_{layer}'] = mask_logprobs
                
            if layer == self.num_stochastic_layers-1:
                continue

            slots = slots.view(-1, self.K, self.z_size)

        # decode
        slots = slots.view(-1, self.z_size)
        # [N*K, M, C, H, W], [N*K, M, 1, H, W]
        x_loc, mask_logits = self.image_decoder(slots)
        slots = slots.view(-1, self.K, self.z_size)
        mask_logits = mask_logits.view(-1, self.K, 1, self.H, self.W)
        x_loc = x_loc.view(-1, self.K, self.C, self.H, self.W)
        return x_loc, mask_logits, posteriors, all_samples, lamda


class EfficientMORL(nn.Module):
    @net.capture
    def __init__(self, K, z_size, input_size, batch_size, stochastic_layers,
                 log_scale, image_likelihood, geco_warm_start, refinement_iters,
                 bottom_up_prior, reverse_prior_plusplus, use_geco=False):
        super(EfficientMORL, self).__init__()
        self.K = K
        self.input_size = input_size
        self.stochastic_layers = stochastic_layers
        self.image_likelihood = image_likelihood
        self.batch_size = batch_size
        self.gmm_log_scale = torch.FloatTensor([log_scale])
        self.refinement_iters = refinement_iters
        self.bottom_up_prior = bottom_up_prior 
        self.reverse_prior_plusplus = reverse_prior_plusplus
        if self.reverse_prior_plusplus:
            assert not self.bottom_up_prior  # must be false
        self.z_size = z_size
        
        self.hvae_networks = HVAENetworks(batch_size=batch_size)
        self.refinenet = RefinementNetwork()

        self.h_0 = torch.zeros(1, self.batch_size * self.K, self.z_size)

        self.use_geco = use_geco
        self.geco_warm_start = geco_warm_start
        self.geco_C_ema = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.geco_beta = nn.Parameter(torch.tensor(0.55), requires_grad=False)
            

    def two_stage_inference(self, x, geco, global_step, kl_beta,
                            get_posterior=False, debug=False):
        total_loss = 0.
        final_nll = 0.
        final_kl = 0.
        level_nll = []
        deltas = []
        C, H, W = self.input_size
    
        all_auxiliary = {}
    
        x_orig = x.clone()
        x_orig = (x_orig + 1) / 2.  # to (0,1)
        
        # x_loc are the RGB components
        # mask_logits are the unnormalized masks
        # posteriors is an array of the L Gaussian intermediate posteriors
        # auxiliary_outs are for visualization
        # posterior_lamda are the layer L Gaussian parameters [mu, sigma]
        x_loc, mask_logits, posteriors, auxiliary_outs, posterior_lamda = self.hvae_networks(x, debug)
        mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)
        all_auxiliary = {**all_auxiliary, **auxiliary_outs}
        
        if self.refinement_iters > 0:
            h = self.h_0.to(x.device)

        for refinement_iter in range(self.refinement_iters+1):
            
            # Recompute x_loc and mask_logits with the updated posterior_lamda
            if refinement_iter > 0:
                # update posterior_lamda
                delta_posterior_lamda, h = self.refinenet(loss, posterior_lamda, h, not self.training)
                posterior_lamda = posterior_lamda + delta_posterior_lamda
                deltas += [torch.mean(torch.norm(delta_posterior_lamda, dim=1)).detach()]
                if refinement_iter == self.refinement_iters:
                    deltas = torch.stack(deltas)

                # decode
                loc, sp = posterior_lamda.chunk(2,1)
                posterior = mvn(loc, sp)

                slots = posterior.rsample()

                slots = slots.view(-1, self.z_size)
                # [N*K, M, C, H, W], [N*K, M, 1, H, W]
                x_loc, mask_logits = self.hvae_networks.image_decoder(slots)
                slots = slots.view(-1, self.K, self.z_size)
                mask_logits = mask_logits.view(-1, self.K, 1, H, W)
                x_loc = x_loc.view(-1, self.K, C, H, W)
                mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)

            # image likelihood for computing NLL
            if self.image_likelihood == 'GMM':
                log_var = (2 * self.gmm_log_scale).view(1,1,1,1,1).repeat(1, self.K,1,1,1).to(x_orig.device)
                nll = gmm_negativeloglikelihood(x_orig, x_loc, log_var, mask_logprobs)
            elif self.image_likelihood == 'Gaussian':
                log_var = (2 * self.gmm_log_scale).view(1,1,1,1).to(x_orig.device)
                nll = gaussian_negativeloglikelihood(x_orig, torch.sum(x_loc * mask_logprobs.exp(), dim=1), log_var)

            # Hierarchical prior computation
            if refinement_iter == 0:
                # top-down kl
                kl_div = torch.zeros(self.batch_size).to(x.device)
                
                # If using the reversed prior
                if not self.bottom_up_prior:

                    for layer in list(range(self.stochastic_layers))[::-1]:
                        # top layer is standard Gaussian
                        if layer == self.stochastic_layers-1:
                            prior_z = std_mvn(shape=[self.batch_size * self.K, self.z_size], device=x.device)
                        else:
                            # z^l+1 ~ q(z^{l+1} | z^l, x)
                            z = posteriors[layer+1].rsample()
                            loc_z, sp_z = self.hvae_networks.indep_prior(z.view(-1, self.K, self.z_size))
                            loc_z = loc_z.view(self.batch_size * self.K, -1)
                            sp_z = sp_z.view(self.batch_size * self.K, -1)
                            # p(z^l | z^l+1)
                            prior_z = mvn(loc_z, sp_z)
                            
                        kl = torch.distributions.kl.kl_divergence(posteriors[layer], prior_z)
                        kl= kl.view(self.batch_size, self.K).sum(1)
                        kl_div += kl
                else:
                    for layer in range(self.stochastic_layers):
                        if layer == 0:
                            prior_z = std_mvn(shape=[self.batch_size * self.K, self.z_size], device=x.device)
                        else:
                            z = posteriors[layer-1].rsample()
                            loc_z, sp_z = self.hvae_networks.indep_prior(z.view(-1, self.K, self.z_size))
                            loc_z = loc_z.view(self.batch_size * self.K, -1)
                            sp_z = sp_z.view(self.batch_size * self.K, -1)
                            prior_z = mvn(loc_z, sp_z)

                        kl = torch.distributions.kl.kl_divergence(posteriors[layer], prior_z)
                        kl = kl.view(self.batch_size, self.K).sum(1)
                        kl_div += kl

            # Refinement step KL
            else:
                if not self.reverse_prior_plusplus or self.stochastic_layers == 0:
                    prior_z = std_mvn(shape=[self.batch_size * self.K, self.z_size], device=x.device)
                # else, prior_z = p(z^1 | z^2) when self.reverse_prior_plusplus is True

                # posterior is q(z; \lambda^{(L,i)})
                kl = torch.distributions.kl.kl_divergence(posterior, prior_z)
                kl = kl.view(self.batch_size, self.K).sum(1)
                kl_div = kl

            final_kl = torch.mean(kl_div)
            final_nll = torch.mean(nll)
        
            all_auxiliary[f'means_{(self.stochastic_layers-1+refinement_iter)}'] = x_loc
            all_auxiliary[f'masks_{(self.stochastic_layers-1+refinement_iter)}'] = mask_logprobs

            
            if kl_beta == 0. or self.geco_warm_start > global_step or geco is None:
                loss = torch.mean(nll + kl_beta * kl_div)
            else:
                loss = kl_beta * torch.mean(kl_div) - geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll))
            
            ## N.b. this is the opposite of IODINE; places more weight on the earlier losses than the later ones 
            # from refinement
            total_loss += (((self.refinement_iters+2 - refinement_iter+1) / (self.refinement_iters+1)) * loss)
       
        if get_posterior:
            if self.refinement_iters == 0:
                return posteriors[-1]
            else:
                return posterior
    
        return all_auxiliary, total_loss, final_nll, final_kl, deltas


    def forward(self, x, geco, global_step, kl_beta, debug=False):
        """
        x: [batch_size, C, H, W]
        """
        auxiliary_outs, total_loss, nll, kl, deltas = self.two_stage_inference(
                                             x, geco, global_step,
                                             kl_beta=kl_beta, debug=debug)
            
        outs = {
            'total_loss': total_loss,
            'nll': nll,
            'kl': kl,
        }
        if len(deltas) > 0:
            outs['deltas'] = deltas

        return {**outs, **auxiliary_outs}
