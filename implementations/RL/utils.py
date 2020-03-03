import nibabel as nib
import numpy as np
import os
import shutil
import tensorboardX
import torch

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn= nn*s
        pp += nn
    return pp

def save_checkpoint(state,is_best,save_path,filename = 'checkpoint.pth.tar'):
    torch.save(state,os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path, 'model_best.pth.tar'))

def read_nii(file):
    f = nib.load(file)
    return f.get_fdata(), f.affine

def save_nii(data, filename, affine=np.eye(4)):
    nib.save(nib.Nifti1Image(data, affine=affine), filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

class Logger(object):

    def __init__(self, model_name,header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter("./runs/"+model_name.split("/")[-1].split(".h5")[0])

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']
        
        for col in self.header[1:]:
            self.writer.add_scalar(phase+"/"+col,float(values[col]),int(epoch))


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

#def calculate_accuracy(outputs, targets):
#    batch_size = targets.size(0)
#
#    _, pred = outputs.topk(1, 1, True)
#    pred = pred.t()
#    correct = pred.eq(targets.view(1, -1))
#    n_correct_elems = correct.float().sum().data[0]
#
#    return n_correct_elems / batch_size

def calculate_accuracy(outputs, targets, threshold=0.5):
    return dice_coefficient(outputs, targets, threshold)

def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8):
    batch_size = targets.size(0)
    y_pred = outputs[:,0,:,:,:]
    y_truth = targets[:,0,:,:,:]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    intersection = torch.sum(torch.mul(y_pred, y_truth)) + eps/2
    union = torch.sum(y_pred) + torch.sum(y_truth) + eps
    dice = 2 * intersection / union 
    
    return dice / batch_size

def load_old_model(model, optimizer, saved_model_path):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path)
    epoch = checkpoint["epoch"]
    # model.load_state_dict(checkpoint["state_dict"])
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.vae.load_state_dict(checkpoint["vae"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return model, epoch, optimizer 

def normalize_data(data, mean, std):
    pass    

# Simple replay buffer
class ReplayBuffer(object):
	def __init__(self):
		self.storage = []

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size=100):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind:
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
