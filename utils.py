import os
import pdb
import math
import random
import shutil
import numpy as np

import torch


class PadSequence:
	def __call__(self, batch):
		batch_idx = []
		batch_len = []
		batch_label = []
		batch_p = []
		for b in batch:
			batch_len.append(len(b[0]))
			if isinstance(b[0], list):
				batch_idx += b[0]
			else:
				batch_idx.append(b[0])
			batch_label.append(b[1])
			batch_p.append(b[2])

		lens = [len(x) for x in batch_idx]
		padded_sents = torch.zeros(len(batch_idx), max(lens)).long()
		for i, sent in enumerate(batch_idx):
			padded_sents[i, :lens[i]] = torch.LongTensor(sent)

		return padded_sents, torch.tensor(batch_label).long(), batch_len, lens, batch_p, torch.stack([torch.tensor(x[-1]) for x in batch]).float()


class SadPadSequence:
	def __call__(self, batch):
		lens = [len(x[0]) for x in batch]
		padded_sents = torch.zeros(len(batch), max(lens)).long()
		for i, x in enumerate(batch):
			padded_sents[i, :lens[i]] = torch.LongTensor(x[0])

		return padded_sents, torch.tensor([x[1] for x in batch]).long(), torch.stack([x[2] for x in batch]), lens, [x[-2] for x in batch], torch.tensor([x[-1] for x in batch]).long()
		# return padded_sents, torch.tensor([x[1] for x in batch]).long(), lens, [x[-1] for x in batch]


class AdjustLR(object):
	def __init__(self, optimizer, init_lr, sleep_epochs=3, half=5, verbose=0):
		super(AdjustLR, self).__init__()
		self.optimizer = optimizer
		self.sleep_epochs = sleep_epochs
		self.half = half
		self.init_lr = init_lr
		self.verbose = verbose

	def step(self, epoch):
		if epoch >= self.sleep_epochs:
			for idx, param_group in enumerate(self.optimizer.param_groups):
				new_lr = self.init_lr[idx] * \
					math.pow(0.5, (epoch-self.sleep_epochs+1)/float(self.half))
				param_group['lr'] = new_lr
			if self.verbose:
				print('>>> reduce learning rate <<<')


class DoublePadSequence:
	def __call__(self, batch):
		batch_comment_idx = []
		comment_num = []
		batch_label = []
		batch_fake_label = []
		batch_p = []
		for b in batch:
			comment_num.append(len(b[0]))
			batch_comment_idx += b[0]
			batch_label.append(b[2])
			batch_p.append(b[-3])
			batch_fake_label.append(b[-2])

		comment_lens = [len(x) for x in batch_comment_idx]
		padded_comment_sents = torch.zeros(
			len(batch_comment_idx), max(comment_lens)).long()
		for i, sent in enumerate(batch_comment_idx):
			padded_comment_sents[i, :comment_lens[i]] = torch.LongTensor(sent)

		text_lens = [len(x[1]) for x in batch]
		padded_text_sents = torch.zeros(len(batch), max(text_lens)).long()
		for i, x in enumerate(batch):
			padded_text_sents[i, :text_lens[i]] = torch.LongTensor(x[1])

		return padded_text_sents, padded_comment_sents, torch.tensor(batch_label).long(), text_lens, comment_lens, comment_num, torch.stack([x[3] for x in batch]), batch_p, torch.tensor(batch_fake_label).long(), [torch.from_numpy(x[-1]).float() for x in batch] 


class DeanPadSequence:
	def __call__(self, batch):
		batch_comment_idx = []
		comment_num = []
		batch_label = []
		batch_fake_label = []
		batch_p = []
		for b in batch:
			comment_num.append(len(b[0]))
			batch_comment_idx += b[0]
			batch_label.append(b[2])
			batch_p.append(b[3])

		comment_lens = [len(x) for x in batch_comment_idx]
		padded_comment_sents = torch.zeros(
			len(batch_comment_idx), max(comment_lens)).long()
		for i, sent in enumerate(batch_comment_idx):
			padded_comment_sents[i, :comment_lens[i]] = torch.LongTensor(sent)

		text_lens = [len(x[1]) for x in batch]
		padded_text_sents = torch.zeros(len(batch), max(text_lens)).long()
		for i, x in enumerate(batch):
			padded_text_sents[i, :text_lens[i]] = torch.LongTensor(x[1])

		return padded_text_sents, padded_comment_sents, torch.tensor(batch_label).long(), text_lens, comment_lens, comment_num, batch_p, torch.stack([torch.from_numpy(x[5]) for x in batch]).float(), torch.stack([torch.from_numpy(x[6]) for x in batch]).float(), torch.cat([torch.from_numpy(x[7]) for x in batch], 0).float()



def init_log_dir(opt):
	if os.path.exists(os.path.join('./save', opt.name)):
		shutil.rmtree(os.path.join('./save', opt.name))

	os.mkdir(os.path.join('./save', opt.name))
	with open(os.path.join('./save', opt.name, 'options.txt'), "a") as f:
		for k, v in vars(opt).items():
			f.write('{} -> {}\n'.format(k, v))
			print('{} -> {}\n'.format(k, v))

	os.mkdir(os.path.join('./save', opt.name, 'check'))
	os.mkdir(os.path.join('./save', opt.name, 'imgs'))
	os.mkdir(os.path.join('./save', opt.name, 'tb'))


def get_dataset(options):
	if options.dataset == 'reddit':
		from data.reddit import RedditLoader
		dataset_train = RedditLoader(options, mode='train')
		dataset_val = RedditLoader(options, mode='val')
		dataset_test = RedditLoader(options, mode='test')
	elif options.dataset == 'reddit_emb':
		from data.reddit import RedditEmbLoader
		dataset_train = RedditEmbLoader(options, mode='train')
		dataset_val = RedditEmbLoader(options, mode='val')
		dataset_test = RedditEmbLoader(options, mode='test')
	elif options.dataset == 'reddit_text':
		from data.reddit import RedditTextLoader
		dataset_train = RedditTextLoader(options, mode='train')
		dataset_val = RedditTextLoader(options, mode='val')
		dataset_test = RedditTextLoader(options, mode='test')
	elif options.dataset == 'twitter_comment':
		from data.twitter import TwitterCommentLoader
		dataset_train = TwitterCommentLoader(options, mode='train')
		dataset_val = TwitterCommentLoader(options, mode='val')
		dataset_test = TwitterCommentLoader(options, mode='test')
	elif options.dataset == 'twitter_content':
		from data.twitter import TwitterContentLoader
		dataset_train = TwitterContentLoader(options, mode='train')
		dataset_val = TwitterContentLoader(options, mode='val')
		dataset_test = TwitterContentLoader(options, mode='test')
	elif options.dataset == 'twitter_double':
		from data.twitter import TwitterDoubleLoader
		dataset_train = TwitterDoubleLoader(options, mode='train')
		dataset_val = TwitterDoubleLoader(options, mode='val')
		dataset_test = TwitterDoubleLoader(options, mode='test')
	elif options.dataset == 'twitter_attribute':
		from data.twitter import TwitterAttributeLoader
		dataset_train = TwitterAttributeLoader(options, mode='train')
		dataset_val = TwitterAttributeLoader(options, mode='val')
		dataset_test = TwitterAttributeLoader(options, mode='test')
	elif options.dataset == 'twitter_fauxbuster':
		from data.twitter import TwitterFauxBusterLoader
		dataset_train = TwitterFauxBusterLoader(options, mode='train')
		dataset_val = TwitterFauxBusterLoader(options, mode='val')
		dataset_test = TwitterFauxBusterLoader(options, mode='test')
	elif options.dataset == 'twitter_image':
		from data.twitter import TwitterImagerLoader
		dataset_train = TwitterImagerLoader(options, mode='train')
		dataset_val = TwitterImagerLoader(options, mode='val')
		dataset_test = TwitterImagerLoader(options, mode='test')
	elif options.dataset == 'twitter_zlatkova':
		from data.twitter import TwitterZlatkovaLoader
		dataset_train = TwitterZlatkovaLoader(options, mode='train')
		dataset_val = TwitterZlatkovaLoader(options, mode='val')
		dataset_test = TwitterZlatkovaLoader(options, mode='test')
	elif options.dataset == 'twitter_dean':
		from data.twitter import TwitterDeanLoader
		dataset_train = TwitterDeanLoader(options, mode='train')
		dataset_val = TwitterDeanLoader(options, mode='val')
		dataset_test = TwitterDeanLoader(options, mode='test')
	elif options.dataset == 'sad':
		from data.sad import SadLoader
		dataset_train = SadLoader(options, mode='train')
		dataset_val = SadLoader(options, mode='val')
		dataset_test = SadLoader(options, mode='test')
	elif options.dataset == 'sem':
		from data.ei import SemLoader
		dataset_train = SemLoader(options, mode='train')
		dataset_val = SemLoader(options, mode='val')
		dataset_test = SemLoader(options, mode='test')
	elif options.dataset == 'aifn':
		from data.twitter import TwitterAIFNLoader
		dataset_train = TwitterAIFNLoader(options, mode='train')
		dataset_val = TwitterAIFNLoader(options, mode='val')
		dataset_test = TwitterAIFNLoader(options, mode='test')
	elif options.dataset == 'ps':
		from data.ps import PSLoader
		dataset_train = PSLoader(options, mode='train')
		dataset_val = PSLoader(options, mode='val')
		dataset_test = PSLoader(options, mode='test')
	elif options.dataset == 'ps_sad':
		from data.ps import TextLoader
		dataset_train = TextLoader(options, mode='train')
		dataset_val = TextLoader(options, mode='val')
		dataset_test = TextLoader(options, mode='test')
	elif options.dataset == 'global':
		from data.twitter import TwitterGlobalLoader
		dataset_train = TwitterGlobalLoader(options, mode='train')
		dataset_val = TwitterGlobalLoader(options, mode='val')
		dataset_test = TwitterGlobalLoader(options, mode='test')
	else:
		raise

	return (dataset_train, dataset_val, dataset_test)


def get_model(options):
	if options.encode == '1dconv':
		from model.encode import Temporal1DConv
		encode = Temporal1DConv(options)
	elif options.encode == 'gru':
		from model.encode import GRU
		encode = GRU(options)
	elif options.encode == 'bgru':
		from model.encode import BiGRU
		encode = BiGRU(options)
	elif options.encode == 'abgru':
		from model.encode import AttenBiGRU
		encode = AttenBiGRU(options)
	elif options.encode == 'abgru_resnet':
		from model.encode import AttenBiGRU_ResNet
		encode = AttenBiGRU_ResNet(options)
	elif options.encode == 'abgru_resnet_ex':
		from model.encode import AttenBiGRU_ResNet_Ex
		encode = AttenBiGRU_ResNet_Ex(options)
	elif options.encode == 'double':
		from model.encode import DoubleEmbed
		encode = DoubleEmbed(options)
	elif options.encode == 'ae':
		from model.autoencoder import StackedAutoEncoder
		encode = StackedAutoEncoder(options)
	elif options.encode == 'fc':
		from model.autoencoder import FullConnect
		encode = FullConnect(options)
	elif options.encode == 'resnet':
		from model.encode import ResNet18
		encode = ResNet18(options)
	elif options.encode == 'zlatkova':
		from model.encode import Zlatkova
		encode = Zlatkova(options)
	elif options.encode == 'han':
		from model.encode import HANEmbed
		encode = HANEmbed(options)
	elif options.encode == 'dean':
		from model.encode import DeanEmbed
		encode = DeanEmbed(options)
	elif options.encode == 'eann':
		from model.encode import EANNEmbed
		encode = EANNEmbed(options)
	elif options.encode == 'aifn':
		from model.encode import AIFNEmbed
		encode = AIFNEmbed(options)
	elif options.encode == 'res':
		from model.encode import ResEmbed
		encode = ResEmbed(options)
	elif options.encode == 'mvae':
		from model.encode import MVAEEmbed
		encode = MVAEEmbed(options)
	elif options.encode == 'global':
		from model.encode import GlobalEmbed
		encode = GlobalEmbed(options)
	else:
		raise

	if options.middle == 'pass':
		from model.encode import Pass
		middle = Pass(options)
	elif options.middle == 'atten_only':
		from model.middle import AttenOnly
		middle = AttenOnly(options)
	elif options.middle == 'wf':
		from model.middle import WeightFusion
		middle = WeightFusion(options)
	elif options.middle == 'cm':
		from model.middle import CrossMinus
		middle = CrossMinus(options)
	elif options.middle == 'defend':
		from model.middle import BaseDefend
		middle = BaseDefend(options)
	elif options.middle == 'defend_loss':
		from model.middle import LossDefend
		middle = LossDefend(options)
	elif options.middle == 'han':
		from model.middle import BaseHAN
		middle = BaseHAN(options)
	elif options.middle == 'hpa':
		from model.middle import BaseHPA
		middle = BaseHPA(options)
	elif options.middle == 'eann':
		from model.middle import BaseEANN
		middle = BaseEANN(options)
	elif options.middle == 'mvae':
		from model.middle import BaseMVAE
		middle = BaseMVAE(options)
	else:
		raise

	if options.decode == 'pass':
		from model.encode import Pass
		decode = Pass(options)
	elif options.decode == 'base':
		from model.decode import Base
		decode = Base(options)
	elif options.decode == 'base_double':
		from model.decode import BaseDouble
		decode = BaseDouble(options)
	elif options.decode == 'base_mvae':
		from model.decode import BaseMVAE
		decode = BaseMVAE(options)
	elif options.decode == 'avgp':
		from model.decode import AveragePool
		decode = AveragePool(options)
	elif options.decode == 'attp':
		from model.decode import AttentionPool
		decode = AttentionPool(options)
	else:
		raise

	return encode, middle, decode


def update_pretrain(opt, models):
	if opt.pretrain == '':
		return models

	elif opt.pretrain == 'sad':
		encode = models[0]
		best_pth = torch.load(os.path.join(
			'./save/abgru_sad/check/best.pth.tar'), map_location='cpu')
		encode.load_state_dict(best_pth['encode'])

		return (encode, models[1], models[2])
	else:
		raise

def Most_Common(lst):
	from collections import Counter
	
	data = Counter(lst)
	return data.most_common(1)[0][0]
