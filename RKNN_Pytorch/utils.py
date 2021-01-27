import torch.nn as nn

import collections
import torch

import collections
import torch
import numpy as np
import matplotlib.pyplot as plt


def paras_summary(input_size, model):
	# input_size=[c, h, w]
	def register_hook(module):
		def hook(module, input, output):
			class_name = str(module.__class__).split('.')[-1].split("'")[0]
			module_idx = len(summary)

			m_key = '%s-%i' % (class_name, module_idx + 1)
			summary[m_key] = collections.OrderedDict()
			summary[m_key]['input_shape'] = list(input[0].size())
			summary[m_key]['input_shape'][0] = -1
			summary[m_key]['output_shape'] = list(output.size())
			summary[m_key]['output_shape'][0] = -1

			params = 0
			if hasattr(module, 'weight'):
				params += torch.prod(torch.LongTensor(list(module.weight.size())))
				if module.weight.requires_grad:
					summary[m_key]['trainable'] = True
				else:
					summary[m_key]['trainable'] = False
			if hasattr(module, 'bias'):
				params += torch.prod(torch.LongTensor(list(module.bias.size())))
			summary[m_key]['nb_params'] = params

		if not isinstance(module, nn.Sequential) and \
				not isinstance(module, nn.ModuleList) and \
				not (module == model):
			hooks.append(module.register_forward_hook(hook))

	# check if there are multiple inputs to the network
	if isinstance(input_size[0], (list, tuple)):
		x = [torch.rand(1, *in_size) for in_size in input_size]
	else:
		x = torch.rand(1, *input_size)

	# create properties
	summary = collections.OrderedDict()
	hooks = []
	# register hook
	model.apply(register_hook)
	# make a forward pass
	model(x)
	# remove these hooks
	for h in hooks:
		h.remove()

	return summary


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./ckpt/best_loss.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: a
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func("EarlyStopping counter: %d out of %d"%(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func('Validation loss decreased (%.3f --> %.3f).  Saving model ...'%(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def draw_result_pic(history, add_name="ft"):
	history = np.array(history)
	plt.plot(history[:, 0:2])
	plt.legend(['Tr Loss', 'Val Loss'])
	plt.xlabel('Epoch Number')
	plt.ylabel('Loss')
	# plt.ylim(a, b)
	plt.savefig("./loss_curve_"+str(add_name)+".png")
	# plt.show()
	plt.clf()

	plt.plot(history[:, 2:4])
	plt.legend(['Tr Accuracy', 'Val Accuracy'])
	plt.xlabel('Epoch Number')
	plt.ylabel('Accuracy')
	plt.ylim(0, 1)
	plt.savefig("./accuracy_curve_"+str(add_name)+".png")
	# plt.show()
