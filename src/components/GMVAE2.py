from curses import noecho
from turtle import forward
from sympy import im
import torch
from torch import nn, optim
from torch.nn import functional as F
import math
import numpy as np


class GMVAE2(nn.Module):
    def __init__(self, K, input_dim, x_dim, w_dim, device):
        super(GMVAE2, self).__init__()
        self.K = K
        self.input_dim = input_dim
        self.x_dim = x_dim
        self.w_dim = w_dim
        self.hidden_dim = input_dim
        self.device = device

        # Q_xw
        self.fc_mean_x = nn.Linear(self.hidden_dim, self.x_dim)
        self.fc_var_x = nn.Linear(self.hidden_dim, self.x_dim)
        self.fc_mean_w = nn.Linear(self.hidden_dim, self.w_dim)
        self.fc_var_w = nn.Linear(self.hidden_dim, self.w_dim)

        # infer_qc_wz
        self.softmax_qz = nn.Linear((self.w_dim+self.x_dim),self.K)
        # infer_qc_wz
        self.softmax_qz = nn.Linear((self.w_dim+self.x_dim),self.K)
  
        # Px_wz
        self.fc_x_means = nn.ModuleList()
        self.fc_x_vars = nn.ModuleList()
        self.x_mean_list = list()
        self.x_var_list = list()
        for i in range(self.K):
            self.fc_x_means.append(nn.Linear(self.w_dim, self.x_dim))
            self.fc_x_vars.append(nn.Linear(self.w_dim, self.x_dim))



    def Q_xw(self, y):

        h = y
        mean_x = self.fc_mean_x(h)
        # var_x = torch.exp(self.fc_var_x(h))
        var_x=torch.relu(self.fc_var_x(h))+1e-6
        mean_w = self.fc_mean_w(h)
        var_w = torch.relu(self.fc_var_w(h))+1e-6
       

        return mean_x, var_x, mean_w, var_w

    def Px_wz(self, w):

        self.x_mean_list = []
        self.x_var_list = []


        for i, l in enumerate(self.fc_x_means):
            self.x_mean_list.append(l(w))
        for i, l in enumerate(self.fc_x_vars):
            a = l(w)
            self.x_var_list.append(torch.relu(l(w))+1e-6)

        return self.x_mean_list, self.x_var_list


 

    def reparameterize(self, mu, var, dim1, dim2):
        eps = torch.randn(dim1, dim2).to(self.device)
        return mu + eps*torch.sqrt(var)

    def KL_recons_loss(self, x_samples, y_recons_mean, y_recons_var):
        loss = torch.zeros(1, requires_grad=True, device = self.device)
        logvar = torch.log(y_recons_var)
        loss = torch.sum(torch.sum(-logvar - torch.pow(x_samples - y_recons_mean, 2)/(2*torch.pow(y_recons_var, 2)), 1), 0)
        #loss = 0.5 * torch.sum(torch.sum(var + torch.pow(mean, 2) - 1  - logvar, 1), 0)

        return loss / (x_samples.size()[0]*x_samples.size()[1])

    def KL_gaussian_loss(self, mean, var):
        loss = torch.zeros(1, requires_grad=True, device = self.device)
        logvar = torch.log(var)
        loss = 0.5 * torch.sum(torch.sum(var + torch.pow(mean, 2) - 1  - logvar, 1), 0)

        return loss / (mean.size()[0]*mean.size()[1])

    def KL_uniform_loss(self, qz):
        loss = torch.zeros(1, requires_grad=True, device = self.device)
        for k in range(self.K):
            loss = loss + torch.sum(qz[:,k] * (torch.log(self.K * qz[:,k] + 1e-10)),0)

        return loss / (qz.size()[0]*qz.size()[1])

    def KL_conditional_loss(self, qz, mean_x, var_x, x_mean_list, x_var_list):
        # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
        x_mean_stack = torch.stack(x_mean_list)
        x_var_stack = torch.stack(x_var_list)
        K, bs, num_sample = x_mean_stack.size()

        loss = torch.zeros(1, requires_grad=True, device = self.device)
        for i in range(num_sample):
            x_mean_2 = x_mean_stack[:,:,i].view(bs, K)
            x_mean_1 = mean_x[:,i].view(bs, -1).repeat(1, K)
            x_var_2 = x_var_stack[:,:,i].view(bs, K)
            x_var_1 = var_x[:,i].view(bs, -1).repeat(1, K)
            # 化简可得， x_var_2是方差的平方，
            KL_batch = 0.5 * (torch.log(x_var_2) - torch.log(x_var_1) - 1 + (x_var_1 + torch.pow(x_mean_1 - x_mean_2, 2))/x_var_2)
            weighted_KL = torch.sum(KL_batch*qz, 1)
            loss = loss + torch.sum(weighted_KL,0)/weighted_KL.size()[0]

        return loss / num_sample

    def Qz_xw(self, x_sample,w_sample):
        qz=self.softmax_qz(torch.cat((x_sample,w_sample),-1))
        qz=torch.softmax(qz,dim=-1)
        return qz
    

    def loss_function(self, y):
        mean_x, var_x, mean_w, var_w = self.Q_xw(y)
        w_sample = self.reparameterize(mu = mean_w, var = var_w, dim1 = mean_w.size()[0], dim2 = mean_w.size()[1])
        x_sample = self.reparameterize(mu = mean_x, var = var_x, dim1 = mean_x.size()[0], dim2 = mean_x.size()[1])
       
        x_mean_list, x_var_list = self.Px_wz(w_sample)
        qz = self.Qz_xw(x_sample,w_sample)


   
        #recons_loss = self.KL_recons_loss(x_sample, y_recons_mean, y_recons_var)
        reg_w_loss = self.KL_gaussian_loss(mean_w, var_w)
        reg_z_loss = self.KL_uniform_loss(qz)
        reg_cond_loss = self.KL_conditional_loss(qz, mean_x, var_x, x_mean_list, x_var_list)
        total_loss =  reg_w_loss + reg_z_loss + reg_cond_loss
        loss={"reg_w_loss":reg_w_loss,"reg_z_loss":reg_z_loss,"reg_cond_loss":reg_cond_loss,"total_loss":total_loss}

        return x_sample,loss
  
    def forward(self,y):
        mean_x, var_x, _, _ = self.Q_xw(y)
        x_sample = self.reparameterize(mu = mean_x, var = var_x, dim1 = mean_x.size()[0], dim2 = mean_x.size()[1])
        return x_sample


class InferenceNet(nn.Module):
    def __init__(self, v_dim, h_dim, w_dim, n_classes):
        super(InferenceNet, self).__init__()

        self.h_dim = h_dim
        self.v_dim = v_dim
        self.w_dim = w_dim
        self.n_classes = n_classes
        self.hidden_dim=32

        # Q(h|v)
        self.Qh_v_mean = torch.nn.Sequential(
            nn.Linear(v_dim, self.hidden_dim), 
            nn.ReLU(),
    
            nn.Linear(self.hidden_dim, h_dim)
        )

        # output is logstd / 2.
        self.Qh_v_var = torch.nn.Sequential(
            nn.Linear(v_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, h_dim)
        )

        # Q(w|v)
        self.Qw_v_mean = torch.nn.Sequential(
            nn.Linear(v_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, w_dim)
        )

        # output is logstd / 2.
        self.Qw_v_var = torch.nn.Sequential(
            nn.Linear(v_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, w_dim)
        )

        # P(c|w, h)
        self.Qc_wh = torch.nn.Sequential(
            nn.Linear(w_dim + h_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            nn.Linear(self.hidden_dim, n_classes),
            nn.Softmax(dim = 1)
        )
    
    def infer_h(self, v, n_particle = 1):
        v = v.view(v.shape[0], -1)
        h_mean = self.Qh_v_mean(v)
        h_var = self.Qh_v_var(v)
        h_sample = self.sample(h_mean, h_var, n_particle)
        return h_mean, h_var, h_sample

    def infer_w(self, v, n_particle = 1):
        v = v.view(v.shape[0], -1)
        w_mean = self.Qw_v_mean(v)
        w_var = self.Qw_v_var(v)
        w_sample = self.sample(w_mean, w_var, n_particle)
        return w_mean, w_var, w_sample
    
    def sample(self, mean, logstd, n_particle = 1):
        # mean, logstd: [bs, sample_dim]
        # eps = torch.randn_like(mean.expand(n_particle, -1, -1))
        eps = torch.randn_like(mean)
        # eps [n_particle, bs, sample_dim]
        sample = mean + eps * (logstd * 2).exp()
        # sample [n_particle, bs, sample_dim]
        return sample
    
    def forward(self, X):
        h, *_ = self.infer_h(X)
        w, *_ = self.infer_w(X)
        return h, w    

class GenerationNet(nn.Module):
    def __init__(self, v_dim, h_dim, w_dim, n_classes):
        super(GenerationNet, self).__init__()
        self.h_dim = h_dim
        self.v_dim = v_dim
        self.w_dim = w_dim
        self.n_classes = n_classes
        self.hidden_dim=32
        # P(h|w,c) c = 1,2,3,...,K
        self.Ph_wc_mean_list = list()
        for i in range(self.n_classes):
            Ph_wc_mean = nn.Sequential(
                nn.Linear(w_dim, self.hidden_dim), 
                nn.ReLU(),
            
                nn.Linear(self.hidden_dim, h_dim)
            )
            self.Ph_wc_mean_list.append(Ph_wc_mean)
        self.Ph_wc_mean_list = nn.ModuleList(self.Ph_wc_mean_list)
        self.Ph_wc_var_list = list()
        for i in range(self.n_classes):
            Ph_wc_var = nn.Sequential(
                nn.Linear(w_dim, self.hidden_dim), 
                nn.ReLU(),
                # nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.ReLU(), 
                nn.Linear(self.hidden_dim, h_dim)
            )
            self.Ph_wc_var_list.append(Ph_wc_var)
        self.Ph_wc_var_list = nn.ModuleList(self.Ph_wc_var_list)
        
   

        self.Pc_wh = nn.Sequential(
            nn.Linear(w_dim + h_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            nn.Linear(self.hidden_dim, n_classes), 
            nn.Softmax(dim = -1)
        )

        # P(v|h)
        self.Pv_h_mean = nn.Sequential(
            nn.Linear(h_dim, self.hidden_dim), 
            nn.ReLU(),
          
            nn.Linear(self.hidden_dim, v_dim)
        )
        self.Pv_h_var = nn.Sequential(
            nn.Linear(h_dim, self.hidden_dim), 
            nn.ReLU(),
        
            nn.Linear(self.hidden_dim, v_dim)
        )


    def infer_c(self, w_sample, h_sample):
        # w, h : [M, bs, w_dim or h_dim]
        concat = torch.cat([w_sample, h_sample], axis = -1)
        prob_c = self.Pc_wh(concat)
        return prob_c
    
    def sample(self, mean, logstd, n_particle = 1):
      
        eps = torch.randn_like(mean)

        sample = mean + eps * (logstd * 2).exp()
    
        return sample
 
    
    def gen_h(self, w, c):     

        h_mean=None
        h_var=None
        outshape=list(w.shape)
        outshape[-1]=self.h_dim
        w=w.reshape(-1,w.shape[-1])
        c=c.reshape(-1,c.shape[-1])
        for i in range(self.n_classes):
     
            weightc=c[:,i].reshape(-1,1)
            if h_mean==None:
                h_mean = weightc*self.Ph_wc_mean_list[i](w)
                h_var = weightc*self.Ph_wc_var_list[i](w)
            else:
                h_mean += weightc*self.Ph_wc_mean_list[i](w)
                h_var += weightc*self.Ph_wc_var_list[i](w)
          
        
        h_mean=h_mean.reshape(outshape)
        h_var=h_var.reshape(outshape)
        return h_mean, h_var
    
    def gen_v(self, h):
        v_mean = self.Pv_h_mean(h)
        v_var = self.Pv_h_var(h)
        return v_mean, v_var
    
    def forward_h(self,w,c):
        h_mean, h_var = self.gen_h(w, c)
        h=self.sample(h_mean,h_var)
        return h
        
    
    def forward(self, w, c):
        h, _ = self.gen_h(w, c)
        v, _ = self.gen_v(h)
        return v

class GMVAE(nn.Module):
    def __init__(self, v_dim, h_dim, w_dim, n_classes):
        super(GMVAE, self).__init__()
        self.v_dim = v_dim
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.n_classes = n_classes
        self.Q = InferenceNet(v_dim, h_dim, w_dim, n_classes)
        self.P = GenerationNet(v_dim, h_dim, w_dim, n_classes)

    def ELBO(self, X, M = 1):
        h_mean, h_logstd, h_sample = self.Q.infer_h(X, n_particle = M)  # h_sample: [M, batch_size, h_dim]
        w_mean, w_logstd, w_sample = self.Q.infer_w(X, n_particle = M)  # logstd = log(sigma) / 2.0 w_sample: [M, bs, w_dim]
    
        prob_c = self.P.infer_c(w_sample, h_sample) # [M, bs, n_classes]
      
        recon_loss = self.recon_loss(h_sample, X)
        kl_loss_c = self.kl_c_loss(prob_c)
        kl_loss_w = self.kl_w_loss(w_mean, w_logstd)
        kl_loss_h = self.kl_h_loss(h_mean, h_logstd, w_sample, prob_c)

        loss = recon_loss + kl_loss_c + kl_loss_h + kl_loss_w
        loss = torch.mean(loss)
        return loss,h_sample

    def ELBO_rl(self, X, M = 1):
        h_mean, h_logstd, h_sample = self.Q.infer_h(X, n_particle = M)  # h_sample: [M, batch_size, h_dim]
        w_mean, w_logstd, w_sample = self.Q.infer_w(X, n_particle = M)  # logstd = log(sigma) / 2.0 w_sample: [M, bs, w_dim]

        prob_c = self.P.infer_c(w_sample, h_sample) # [M, bs, n_classes]

        kl_loss_c = self.kl_c_loss(prob_c)
        kl_loss_w = self.kl_w_loss(w_mean, w_logstd)
        kl_loss_h = self.kl_h_loss(h_mean, h_logstd, w_sample, prob_c)
     
        loss =  kl_loss_c + kl_loss_h + kl_loss_w
        loss = torch.mean(loss)
        loss_set={"reg_w_loss":kl_loss_w,"reg_z_loss":kl_loss_c,"reg_cond_loss":kl_loss_h,"total_loss":loss,"prob_c":prob_c}
        return h_sample,loss_set

    def recon_loss(self, h_sample, X, type = 'bernoulli'):
        if type == 'gaussian':
            x_mean, x_logstd = self.P.gen_v(h_sample)
            loss = (X - x_mean) ** 2 / (2. * (2 * x_logstd).exp()) + np.log(2. * np.pi) / 2. + x_logstd
            return loss
        elif type == 'bernoulli':
            recon_x, _ = self.P.gen_v(h_sample)
            X_view = X.reshape(X.shape[0], -1).expand_as(recon_x)
            loss = F.binary_cross_entropy_with_logits(input=recon_x, target=X_view, reduction='none')
            return torch.mean(torch.sum(loss, axis = -1), axis = 0)
    
    def kl_w_loss(self, w_mean, w_logstd):
        kl = -w_logstd + ((w_logstd * 2).exp() + torch.pow(w_mean, 2) - 1.) / 2.
        kl = kl.sum(dim=-1)
        return kl
    
    def kl_c_loss(self, c_prob):
        # c_prob [M, bs, num_classes]
        kl = c_prob * (torch.log(c_prob + 1e-10) + torch.log(torch.Tensor([self.n_classes])).to('cuda'))
        kl = torch.mean(torch.sum(kl, axis = -1), axis = 0)
        return kl 

    def kl_h_loss(self, q_h_v_mean, q_h_v_logstd, w_sample, c_prob):
      
        w_sample=w_sample.expand(1, -1, -1)
        M, bs, _ = w_sample.shape
        c = torch.eye(self.n_classes).expand(M, bs, -1, -1).to('cuda') # [M, bs, n_classes, n_classes]
        w_sample = w_sample.unsqueeze(2).expand(-1,-1,self.n_classes,-1)
        # c = c.cuda()

        h_wc_mean, h_wc_logstd = self.P.gen_h(w_sample, c) # [M, bs, n_classes, h_dim]
        q_h_v_mean = q_h_v_mean.unsqueeze(0).unsqueeze(2).expand_as(h_wc_mean)
        q_h_v_logstd = q_h_v_logstd.unsqueeze(0).unsqueeze(2).expand_as(h_wc_logstd)

        kl = ((q_h_v_logstd - h_wc_logstd) * 2).exp() - 1.0 - q_h_v_logstd * 2 + h_wc_logstd * 2
        kl += torch.pow((q_h_v_mean - h_wc_mean), 2) / (h_wc_logstd * 2).exp()
        kl = torch.sum(kl, axis = -1) * 0.5 # [M, bs, n_classes]

        kl = torch.sum(kl * c_prob, axis = -1)

        return torch.mean(kl, axis = 0)


    def forward(self, X,M=1):
        _, _, h_sample = self.Q.infer_h(X, n_particle = M)  # h_sample: [M, batch_size, h_dim]
        _, _, w_sample = self.Q.infer_w(X, n_particle = M)  # logstd = log(sigma) / 2.0 w_sample: [M, bs, w_dim]
        prob_c = self.P.infer_c(w_sample, h_sample) # [M, bs, n_classes]
        output={'h':h_sample,'w':w_sample,'c':prob_c}
        return output
    
    def forward_loss(self, X,M=1):
        h_mean, h_logstd, h_sample = self.Q.infer_h(X, n_particle = M)  # h_sample: [M, batch_size, h_dim]
        w_mean, w_logstd, w_sample = self.Q.infer_w(X, n_particle = M)  # logstd = log(sigma) / 2.0 w_sample: [M, bs, w_dim]

        prob_c = self.P.infer_c(w_sample, h_sample) # [M, bs, n_classes]
      
        kl_loss_c = self.kl_c_loss(prob_c)
        kl_loss_w = self.kl_w_loss(w_mean, w_logstd)
        kl_loss_h = self.kl_h_loss(h_mean, h_logstd, w_sample, prob_c)
     
        loss =  kl_loss_c + kl_loss_h + kl_loss_w
        loss = torch.mean(loss)
        loss_set={"reg_w_loss":kl_loss_w,"reg_z_loss":kl_loss_c,"reg_cond_loss":kl_loss_h,"total_loss":loss,"w":w_sample,"c":prob_c}
        return h_sample,loss_set
    
    def forward_H(self, X):
        _, _, h_sample = self.Q.infer_h(X, n_particle = 1)
        return h_sample
   
    def loss_function(self,X):
        return self.ELBO_rl(X)
        

    


if __name__=="__main__":
  print("test")
  model=GMVAE(v_dim=10,h_dim=4,w_dim=5,n_classes=3)
  y=torch.rand(3,10)
  print(model.loss_function(y))
  

