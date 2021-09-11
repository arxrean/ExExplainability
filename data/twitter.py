import os
import re
import pdb
import glob
import json
import nltk
import random
import string
import numpy as np
from tqdm import tqdm
from PIL import Image
import scipy.sparse as sp
from dateutil import parser
import textacy.preprocessing
import matplotlib.pyplot as plt
from collections import Counter
from pycontractions import Contractions
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def summary(root='./dataset/new'):
    trues = glob.glob(os.path.join(root, 'true', '*', '*.json'))
    fakes = glob.glob(os.path.join(root, 'fake', '*', '*.json'))
    print('true samples:{}'.format(len(trues)))
    print('fake samples:{}'.format(len(fakes)))
    print('fake 00 samples:{}'.format(
        len([x for x in fakes if x.split('_')[-1] == '100'])))
    print('fake 10 samples:{}'.format(
        len([x for x in fakes if x.split('_')[-1] == '110'])))
    print('fake 01 samples:{}'.format(
        len([x for x in fakes if x.split('_')[-1] == '101'])))
    print('fake 11 samples:{}'.format(
        len([x for x in fakes if x.split('_')[-1] == '111'])))

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    true_comments = sorted(
        [len(os.listdir(os.path.join(x, 'comments'))) for x in trues])
    plt.bar(range(len(true_comments)), true_comments)
    plt.title('True Sample - Comment')
    plt.subplot(122)
    fake_comments = sorted(
        [len(os.listdir(os.path.join(x, 'comments'))) for x in fakes])
    plt.bar(range(len(fake_comments)), fake_comments)
    plt.title('Fake Sample - Comment')
    plt.savefig('./repo/imgs/sample_comment.png')
    plt.close()

    print('true comments:{}'.format(
        np.sum([len(os.listdir(os.path.join(x, 'comments'))) for x in trues])))
    print('fake comments:{}'.format(
        np.sum([len(os.listdir(os.path.join(x, 'comments'))) for x in fakes])))
    print('fake 00 samples:{}'.format(
        np.sum([len(os.listdir(os.path.join(x, 'comments'))) for x in fakes if x.split('_')[-1] == '100'])))
    print('fake 10 samples:{}'.format(
        np.sum([len(os.listdir(os.path.join(x, 'comments'))) for x in fakes if x.split('_')[-1] == '110'])))
    print('fake 01 samples:{}'.format(
        np.sum([len(os.listdir(os.path.join(x, 'comments'))) for x in fakes if x.split('_')[-1] == '101'])))
    print('fake 11 samples:{}'.format(
        np.sum([len(os.listdir(os.path.join(x, 'comments'))) for x in fakes if x.split('_')[-1] == '111'])))

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    true_comments = sorted(
        [len(os.listdir(os.path.join(x, 'comments'))) for x in trues])
    plt.bar(range(len(true_comments)), true_comments)
    plt.title('True Sample - Comment')
    plt.subplot(122)
    fake_comments = sorted(
        [len(os.listdir(os.path.join(x, 'comments'))) for x in fakes])
    plt.bar(range(len(fake_comments)), fake_comments)
    plt.title('Fake Sample - Comment')
    plt.savefig('./repo/imgs/sample_comment.png')
    plt.close()


def preprocess(root='./dataset/new'):
    cont = Contractions('./repo/GoogleNews-vectors-negative300.bin.gz')
    # data = glob.glob(os.path.join(root, 'true', '*')) + glob.glob(os.path.join(root, 'fake', '*'))
    data = glob.glob('./dataset/SemEval2018/data/*/*.json')

    # for x in data:
    # 	tweets = glob.glob(os.path.join(x, '*.json')) + \
    # 		glob.glob(os.path.join(x, 'comments', '*.json'))
    # 	tweets = [x for x in tweets if 'emotion' not in x]
    # 	for i, t in enumerate(tweets):
    for t in data:
        with open(t, 'r') as f:
            tweet = json.load(f)

        if 'token' in tweet.keys():
            continue

        print(t)
        full_text = tweet['full_text']
        if '\'' in full_text:
            full_text = list(cont.expand_texts(
                [full_text], precise=True))[0]
        # tagged_sent = nltk.pos_tag(nltk.word_tokenize(full_text))
        full_text = full_text.lower()
        full_text = textacy.preprocessing.replace.replace_emails(
            full_text, 'EMAIL')
        full_text = textacy.preprocessing.replace.replace_urls(
            full_text, 'URL')
        full_text = textacy.preprocessing.replace.replace_numbers(
            full_text, 'NUMBER')
        full_text = textacy.preprocessing.replace.replace_emojis(
            full_text, 'EMOJI')
        full_text = full_text.replace('\n', '')
        full_text = full_text.replace('\t', '')
        # if i == 0:
        # 	full_text = full_text.replace('_URL_', '')

        full_text = textacy.preprocessing.remove.remove_punctuation(
            full_text)
        tokens = word_tokenize(full_text)
        tweet['token'] = tokens

        with open(t, 'w') as f:
            json.dump(tweet, f)


def getEmotion(root='./dataset/new'):
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
    authenticator = IAMAuthenticator(
        "846DnKMXusUjV_McT7jQhdtK7EC1UJHYArEhpD-d9hJ0")
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2020-02-24',
        authenticator=authenticator
    )

    data = glob.glob(os.path.join(root, 'fake', '*')) + \
        glob.glob(os.path.join(root, 'true', '*'))
    cnt = [0, 0]
    for file in tqdm(data):
        tweets = glob.glob(os.path.join(file, '*_body.json')) + \
            glob.glob(os.path.join(file, 'comments', '*_body.json'))
        tweets = [x for x in tweets if 'emotion' not in x]
        for t in tweets:
            cnt[0] += 1
            if os.path.exists(t.replace('_body.json', '_emotion.json')):
                continue
            with open(t, 'r') as f:
                text = json.load(f)['full_text']

            assert text != ''
            try:
                response = natural_language_understanding.analyze(
                    text=text, features=Features(emotion=EmotionOptions()))
            except Exception as e:
                cnt[1] += 1

            out_t = t.replace('_body.json', '_emotion.json')
            with open(out_t, 'w') as f:
                json.dump(response.result, f)

    print(cnt)


def getAttitude(root='./dataset/new'):
    from nltk.corpus import stopwords
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob

    def preprocessText(text):
        text = re.sub('[(]?https?://(?:[-\w.\/\%])+[)]?', '', text)
        text = re.sub('pic.twitter.com/([-\w.\/\%])+', '', text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [t.lower() for t in tokens]
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if not t in stop_words]
        cleanedText = ' '.join(tokens)

        return cleanedText

    data = glob.glob(os.path.join(root, 'fake', '*')) + \
        glob.glob(os.path.join(root, 'true', '*'))
    for file in tqdm(data):
        tweets = glob.glob(os.path.join(file, '*_body.json')) + \
            glob.glob(os.path.join(file, 'comments', '*_body.json'))
        tweets = [x for x in tweets if 'emotion' not in x]
        for t in tweets:
            with open(t, 'r') as f:
                text = json.load(f)['full_text']

            text = preprocessText(text)
            sid = SentimentIntensityAnalyzer()
            polarity_scores = sid.polarity_scores(text)
            blob_pattern = TextBlob(text)
            blob_pattern = blob_pattern.sentiment
            np.savez(t.replace('_body.json', '_attitude.npz'), sid=polarity_scores, blob={
                     'polarity': blob_pattern.polarity, 'subjectivity': blob_pattern.subjectivity})


def construct_aj_matrix(root='./dataset/new'):
    fake_data = glob.glob(os.path.join(root, 'fake', '*')) + \
        glob.glob(os.path.join(root, 'true', '*'))
    for item in fake_data:
        comments = sorted(glob.glob(os.path.join(item, 'comments', '*.json')))
        comment_ids = []
        for c in comments:
            tweet = json.load(open(c))
            comment_ids.append(tweet['id_str'])

        graph = np.zeros((len(comment_ids), len(comment_ids)))
        for c in comments:
            tweet = json.load(open(c))
            id = tweet['id_str']
            reply_id = tweet['in_reply_to_status_id_str']
            if id in comment_ids and reply_id in comment_ids:
                graph[comment_ids.index(id), comment_ids.index(reply_id)] = 1

        # if '1129439204285587458' not in item and '1129439204285587458' not in item:
        # 	assert np.sum(graph) > 0
        if np.sum(graph) == 0:
            print(item)
        np.savez(os.path.join(item, 'comment_graph.npz'),
                 graph=graph, ids=comment_ids)


def normalize_aj_matrix(root='./dataset/new'):
    fake_data = glob.glob(os.path.join(root, 'fake', '*')) + \
        glob.glob(os.path.join(root, 'true', '*'))

    for item in tqdm(fake_data):
        print(item)
        graph = np.load(os.path.join(item, 'comment_graph.npz'))
        graph, ids = graph['graph'], list(graph['ids'])
        tmp = np.zeros((graph.shape[0]+1, graph.shape[0]+1))
        tmp[1:, 1:] = graph
        tmp[1:, 0] = 1
        tmp[0, 1:] = 1
        ngraph = normalize_adj(tmp + sp.eye(tmp.shape[0]))
        np.savez(os.path.join(item, 'comment_graph.npz'),
                 graph=graph, ids=ids, ngraph=ngraph)


def random_walk(root='./dataset/new', m=100, k=10, mode='feedback'):
    def get_emotion_by_path(path):
        if os.path.exists(path):
            return max(list(json.load(open(path))['emotion']['document']['emotion'].values()))
        else:
            return 0.25

    def get_attitude_by_path(path):
        path = path.replace('.json', '.npz')
        if not os.path.exists(path):
            return 2

        item = np.load(path, allow_pickle=True)
        sid, blob = item['sid'].item(), item['blob'].item()
        if blob['polarity'] <= -0.8:
            return 0
        origin_path = path.replace('_attitude.npz', '.json')
        text = json.load(open(origin_path))
        if 'fake' in text['token'] or 'lie' in text['token'] or 'false alarm' in text['full_text'] or 'not true' in text['full_text']:
            return 0
        if text['retweet_count'] > 0:
            return 1

        return 2

    def get_feedback_by_path(path):
        path = path.replace('_feedback', '_body')
        text = json.load(open(path))

        return text['favorite_count']

    data = glob.glob(os.path.join(root, 'fake', '*')) + \
        glob.glob(os.path.join(root, 'true', '*'))

    if mode == 'emotion':
        get_mode_by_path = get_emotion_by_path
    elif mode == 'attitude':
        get_mode_by_path = get_attitude_by_path
    elif mode == 'feedback':
        get_mode_by_path = get_feedback_by_path
    else:
        raise

    for x in data:
        print(x)
        if os.path.exists(os.path.join(x, x.split('/')[-1].split('_')[0]+'_rm_{}.npy'.format(mode) if '_' in x else x.split('/')[-1]+'_rm_{}.npy'.format(mode))):
            continue
        body = glob.glob(os.path.join(x, '*_body.json'))[0]
        comments = sorted(
            glob.glob(os.path.join(x, 'comments', '*_body.json')))
        graph = np.load(os.path.join(x, 'comment_graph.npz'))
        graph, ids = graph['graph'], list(graph['ids'])
        random_res = np.zeros((m, k))
        for i in range(m):
            random_res[i][0] = get_mode_by_path(
                body.replace('_body', '_{}'.format(mode)))
            first_comment = random.sample(comments, 1)[0]
            first_comment_emotion = get_mode_by_path(
                first_comment.replace('_body', '_{}'.format(mode)))
            random_res[i][1] = first_comment_emotion
            comment_id = first_comment.split('/')[-1].replace('_body.json', '')
            for j in range(2, k):
                pool = graph[:, ids.index(comment_id)]
                if np.sum(pool) == 0:
                    break
                comment_id = ids[random.sample(
                    list(np.where(pool == 1)[0]), 1)[0]]
                comment_path = '/'.join(first_comment.split('/')
                                        [:-1])+'/{}_{}.json'.format(comment_id, mode)
                random_res[i][j] = get_mode_by_path(comment_path)

        np.save(os.path.join(x, x.split('/')[-1].split('_')[0]+'_rm_{}.npy'.format(
            mode) if '_' in x else x.split('/')[-1]+'_rm_{}.npy'.format(mode)), random_res)


def get_tf_idf_vector(root='./dataset/new'):
    data = glob.glob(os.path.join(root, 'fake', '*')) + \
        glob.glob(os.path.join(root, 'true', '*'))

    embeds = []
    for x in data:
        body = glob.glob(os.path.join(x, '*_body.json'))[0]
        text = json.load(open(body))['full_text']
        embeds.append(text)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(embeds)
    for x, p in zip(X, data):
        body = glob.glob(os.path.join(p, '*_body.json'))[0]
        body = body.replace('_body.json', '_tfidf.npy')
        np.save(body, x.todense())


def get_img_transform(opt, mode):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if mode == 'train':
        return transforms.Compose([
            # transforms.RandomResizedCrop(
            # 	(opt.crop_size, opt.crop_size)),
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.RandomCrop((opt.crop_size, opt.crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((opt.crop_size, opt.crop_size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_img(item_id, path=False):
    if path:
        if os.path.exists(item_id.replace('_body.json', '.jpeg')):
            return item_id.replace('_body.json', '.jpeg')
        if os.path.exists(item_id.replace('_body.json', '.jpg')):
            return item_id.replace('_body.json', '.jpg')
        if os.path.exists(item_id.replace('_body.json', '.png')):
            return item_id.replace('_body.json', '.png')
    else:
        if os.path.exists(item_id.replace('_body.json', '.jpeg')):
            return Image.open(item_id.replace('_body.json', '.jpeg')).convert('RGB')
        if os.path.exists(item_id.replace('_body.json', '.jpg')):
            return Image.open(item_id.replace('_body.json', '.jpg')).convert('RGB')
        if os.path.exists(item_id.replace('_body.json', '.png')):
            return Image.open(item_id.replace('_body.json', '.png')).convert('RGB')


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx = sp.csr_matrix(mx)
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    return np.array(mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo().todense())


def print_test_faux_samples(root='./dataset/new'):
    data = sorted(glob.glob(os.path.join(root, 'true', '*', '*_body.json'))) + \
        sorted(glob.glob(os.path.join(root, 'fake', '*', '*_body.json')))

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    test_faux = [x for x in test if '_' in x.split('/')[-2]]


class TwitterAttributeLoader(Dataset):
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode

        data = sorted(glob.glob(os.path.join(opt.twitter_root, 'true', '*', '*_body.json'))) + \
            sorted(glob.glob(os.path.join(
                opt.twitter_root, 'fake', '*', '*_body.json')))

        train, test = train_test_split(data, test_size=0.2, random_state=42)
        self.data = test if mode == 'test' else train

    # feat loader
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.opt.attribute == 'emotion':
            item = np.load(item.replace('_body.json', '_rm_emotion.npy'))
        elif self.opt.attribute == 'attitude':
            item = np.load(item.replace('_body.json', '_rm_attitude.npy'))
        elif self.opt.attribute == 'feedback':
            item = np.load(item.replace('_body.json', '_rm_feedback.npy'))
            item = (item - np.min(item))/(np.max(item)-np.min(item))

        return item, self.data[idx]

    def __len__(self):
        return len(self.data)


class TwitterFauxBusterLoader(Dataset):
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode

        data = sorted(glob.glob(os.path.join(opt.twitter_root, 'true', '*', '*_body.json'))) + \
            sorted(glob.glob(os.path.join(
                opt.twitter_root, 'fake', '*', '*_body.json')))

        train, test = train_test_split(data, test_size=0.2, random_state=42)
        train, val = train_test_split(train, test_size=0.2, random_state=42)

        if mode == 'test':
            self.data = test
        elif mode == 'val':
            self.data = val
        else:
            self.data = train

    # feat loader
    def __getitem__(self, idx):
        item = self.data[idx]
        emotion = np.load(item.replace('_body.json', '_emotion_feat.npy'))
        attitude = np.load(item.replace('_body.json', '_attitude_feat.npy'))
        feedback = np.load(item.replace('_body.json', '_feedback_feat.npy'))

        doc1 = np.load(item.replace('_body.json', '_docvec_0.npy'))
        doc2 = np.load(item.replace('_body.json', '_docvec_1.npy'))
        doc3 = np.load(item.replace('_body.json', '_docvec_2.npy'))

        feat = np.concatenate(
            [emotion, attitude, feedback, doc1, doc2, doc3], 0)
        return feat, 1 if '/fake/' in self.data[idx] else 0, self.data[idx]

    def __len__(self):
        return len(self.data)

    def k_fold(self, k, fold=10, mode='train'):
        # pdb.set_trace()
        np.random.seed(self.opt.seed)
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        indices = np.array_split(indices, fold)
        if mode == 'train':
            self.data = [self.data[x] for x in np.concatenate(
                [indices[i] for i in range(fold) if i != k])]
        else:
            self.data = [self.data[x] for x in indices[k]]

        return self
