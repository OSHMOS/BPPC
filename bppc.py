import torch
import numpy as np
import torch.nn as nn
from grid_sample1d.op import GridSample1d

############# 17 keypoints #######
# 0 - pelvis
# 1 - rhip, 2 - rknee, 3 - rankle
# 4 - lhip, 5 - lknee, 6 - lankle,
# 7 - belly, 8 - neck, 9 - upper neck, 10 - head,
# 11 - lshol, 12 - lelbow, 13 - lwrist,
# 14 - rshol, 15 - relbow, 16 - rwrist

torch.cuda.empty_cache()

def ohkm(loss, topk): # loss.size() = tensor(1, 77, 17)(b=1, f, k)
  # topk *= 2
  loss = loss[0]
  ohkm_loss = 0.
  for i in range(loss.size()[0]):
      sub_loss = loss[i]
      topk_val, topk_idx = torch.topk(
          sub_loss, k=topk, dim=0, sorted=False
      )
      tmp_loss = torch.gather(sub_loss, 0, topk_idx)
      ohkm_loss += torch.sum(tmp_loss) / topk
      # ohkm_loss /= loss.size()[0]
  return ohkm_loss

grid_sample1d = GridSample1d(padding_mode=False, align_corners=True)

# Load 3D standard motion data
# sm_3d = np.load('data/sm_3D/sm_3D_golf_label.npz')['reconstruction'] # golf
sm_3d = np.load('data/sm_3D/sm_3D_baseball_label.npz')['reconstruction'] # baseball
# sm_3d = np.load('data/sm_3D/sm_3D_base_by_bppc_left.npz')['sm_3d'] # bppc 1
# sm_3d = np.load('data/sm_3D/sm_3D_base_by_bppc_right.npz')['sm_3d'] # bppc 2
sm_3d = sm_3d.astype('float32')
sm_3d = torch.tensor(sm_3d).unsqueeze(0).cuda()
# sm_3d = torch.tensor(sm_3d).cuda() # bppc 1, 2

# Add an additional dimension for homogenous coordinates
sm_3d = torch.cat((sm_3d, torch.ones(sm_3d.shape[0],sm_3d.shape[1],sm_3d.shape[2],1 ).cuda()), dim=3)
# 1, 16, 17, 4
class BPPC(nn.Module):
  def __init__(self, handed_option, lambda_ohkm, lambda_reg, lambda_vel, lambda_accel, pred, c_scores):
    super().__init__()
    if handed_option == 'right':
      # reverse
      sm_3d[:,:,:,0] = - sm_3d[:,:,:,0]

      for i in [1, 2, 3]:
        tmp = sm_3d[:,:,i,:].clone()
        sm_3d[:,:,i,:] =  sm_3d[:,:,i+3,:].clone()
        sm_3d[:,:,i+3,:] =  tmp

      for i in [14, 15, 16]:
        tmp = sm_3d[:,:,i,:].clone()
        sm_3d[:,:,i,:] =  sm_3d[:,:,i-3,:].clone()
        sm_3d[:,:,i-3,:] =  tmp

    self.pred = pred.clone().detach()
    self.c_scores = torch.from_numpy(c_scores).cuda() # 1x16x17 = #batch x #frame x #keypoints

    self.lambda_ohkm = lambda_ohkm
    self.lambda_reg = lambda_reg
    self.lambda_vel = lambda_vel
    self.lambda_accel = lambda_accel
    
    # Learnable keypoints
    self.params_kp = nn.Parameter(pred.clone().detach())

    # Learnable offset for the standard motion
    self.params_sm_offset = nn.Parameter(sm_3d.clone().detach()[:,0:1,:,:]*0)

    # Learnable time parameters
    params_time = torch.zeros((pred.shape[0],2)).cuda()
    params_time[:,0] = -1.0
    params_time[:,1] = 1.0
    self.params_time = nn.Parameter(params_time)

    # Learnable projection matrix
    params_projection = torch.zeros((2, 4)).cuda()
    params_projection[0, 0] = 0.2
    params_projection[0, 3] = 0.4
    params_projection[1, 1] = 0.2
    params_projection[1, 3] = 0.2
    self.params_projection = nn.Parameter(params_projection)

    # Loss function
    self.criterion_mse = torch.nn.MSELoss(reduction='mean')
    
    # Optimizers (adamw -> weight_decay=0)
    self.bppc_optimizer_kp = torch.optim.AdamW([self.params_kp], lr=0.001, weight_decay=0, amsgrad=True) # cfg.TRAIN.LR = 0.001
    self.bppc_optimizer = torch.optim.AdamW([self.params_projection, self.params_time, self.params_sm_offset], lr=0.001, weight_decay=0, amsgrad=True) # cfg.TRAIN.LR = 0.001
  
  def forward(self):
    return
  
  def training_step(self, batch=None): # alignment
    # Forward pass for training
    if len(self.params_kp.shape)==4:
      b,f,k,x = self.params_kp.size()
      params_kp_sk = self.params_kp.reshape(b,f,-1)
    else:
      b,f,k = self.params_kp.size()
      params_kp_sk = self.params_kp
      k = k//2
      x = 2

    grid_sm = nn.functional.interpolate(self.params_time.unsqueeze(1).unsqueeze(1), size=(1, sm_3d.shape[1]), mode='bilinear', align_corners=True).squeeze(1).squeeze(1)
    params_kp_sk = grid_sample1d(params_kp_sk.permute(0,2,1).contiguous(), grid_sm.contiguous())
    params_kp_sk = params_kp_sk.permute(0,2,1)
    params_kp_sk = params_kp_sk.reshape(b,sm_3d.shape[1],k,x)

    sm_kp = torch.matmul(self.params_projection, sm_3d.unsqueeze(4)+ self.params_sm_offset.unsqueeze(4)).squeeze(4) # sm in test viewpoint

    c_scores = grid_sample1d(self.c_scores.permute(0,2,1).contiguous(), grid_sm.contiguous())
    c_scores = c_scores.permute(0,2,1)

    loss = self.criterion_mse(params_kp_sk, sm_kp)
    f_loss = c_scores.unsqueeze(3) * (params_kp_sk - sm_kp)**2
    loss += self.lambda_ohkm * ohkm(f_loss, 17)
    loss += self.lambda_reg * self.criterion_mse(self.params_sm_offset, self.params_sm_offset.detach()*0)

    return loss

  # bppc_optimizer로 self.params_kp optimize
  def optimize(self, iters, batch=None):
    for i in range(iters):
      torch.cuda.empty_cache()
      loss = self.training_step()
      self.bppc_optimizer.zero_grad()
      loss.backward(retain_graph=False)
      self.bppc_optimizer.step()

  def training_step_kp(self, batch=None):
    if len(self.params_kp.shape) == 4:
      b, f, k, x = self.params_kp.size()
      params_kp_sk = self.params_kp.reshape(b, f, -1)
    else:
      b, f, k = self.params_kp.size()
      params_kp_sk = self.params_kp
      k = k//2
      x = 2

    hrnet = self.pred.detach()
    params_kp_sk_ori = params_kp_sk
    sm_kp = torch.matmul(self.params_projection, sm_3d.unsqueeze(4)+ self.params_sm_offset.unsqueeze(4)).squeeze(4) # standard motion kpts
    sm_kp = sm_kp[:,:,:,0:2:1]
    
    grid = nn.functional.interpolate(self.params_time.unsqueeze(1).unsqueeze(1), size=(1, sm_3d.shape[1]), mode='bilinear', align_corners=True).squeeze(1).squeeze(1)
    params_kp_sk = grid_sample1d(params_kp_sk.permute(0, 2, 1).contiguous(), grid.contiguous())
    params_kp_sk = params_kp_sk.permute(0, 2, 1)
    params_kp_sk = params_kp_sk.reshape(b, sm_3d.shape[1], k, x)

    loss = 0.0  * self.criterion_mse(params_kp_sk, sm_kp)
    loss += self.lambda_vel * self.criterion_mse(params_kp_sk[:,1:,:,:]-params_kp_sk[:,:-1,:,:], sm_kp[:,1:,:,:]-sm_kp[:,:-1,:,:]) # velocity # Eq. 6 loss term 1
    loss += self.lambda_accel * self.criterion_mse(params_kp_sk[:,2:,:,:]-params_kp_sk[:,:-2,:,:], sm_kp[:,2:,:,:]-sm_kp[:,:-2,:,:]) # accel # Eq. 6 loss term 2

    params_kp_sk_ori = params_kp_sk_ori.reshape(b, params_kp_sk_ori.shape[1], k, 2)
    # hrnet = hrnet.reshape(b, params_kp_sk_ori.shape[1], k, x)
    hrnet = hrnet.reshape(b, params_kp_sk_ori.shape[1], k, 2)

    f_loss = self.c_scores.unsqueeze(3) * (params_kp_sk_ori - hrnet) ** 2
    loss += self.lambda_ohkm * ohkm(f_loss, 17) # Eq. 6 loss term 3

    # hrnet = hrnet.reshape(b, params_kp_sk_ori.shape[1], k, x)
    hrnet = hrnet.reshape(b, params_kp_sk_ori.shape[1], k * 2 )

    self.bppc_kpts = params_kp_sk_ori.reshape(b, params_kp_sk_ori.shape[1], k * 2) # bppc kpts

    sm_kp_vis_params = torch.tensor([[-1.0,1.0]]).cuda().unsqueeze(1).unsqueeze(1)
    
    self.params_time = nn.Parameter(torch.clamp(self.params_time, min=-1, max=1))

    start_time = int(torch.round((self.params_time[0][0]+1)*hrnet.shape[1]/2).detach().cpu().numpy())
    end_time = int(torch.round((self.params_time[0][1]+1)*hrnet.shape[1]/2).detach().cpu().numpy())-1

    grid = nn.functional.interpolate(sm_kp_vis_params, size=(1, end_time-start_time+1), mode='bilinear', align_corners=True).squeeze(1).squeeze(1)
    sm_kp = sm_kp.reshape(b, sm_kp.shape[1], -1)
    sm_kp = grid_sample1d(sm_kp.permute(0, 2, 1).contiguous(), grid.contiguous())
    sm_kp = sm_kp.permute(0, 2, 1)
    
    self.sm_kpts = torch.zeros_like(hrnet.reshape(b, params_kp_sk_ori.shape[1], -1))
    self.sm_kpts[:,start_time:end_time+1,:] = sm_kp

    return loss


  # bppc_optimizer로 self.params_kp optimize
  def optimize_kp(self, iters, batch=None):
    for i in range(iters):
      torch.cuda.empty_cache()
      loss = self.training_step_kp()
      self.bppc_optimizer_kp.zero_grad()
      loss.backward(retain_graph=False)
      self.bppc_optimizer_kp.step()
