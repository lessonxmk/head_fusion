import glob
import os
import pickle
import random
import time
import math
import logging
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import librosa
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(111111)
# setup_seed(123456)
# setup_seed(0)
# setup_seed(999999)
setup_seed(987654)

attention_head = 2
attention_hidden = 392

import features
import model
import data_loader
import CapsNet

learning_rate = 0.001
Epochs = 50
BATCH_SIZE = 32

T_stride = 2
T_overlop = T_stride / 2
overlapTime = {
    'neutral': 1,
    'happy': 1,
    'sad': 1,
    'angry': 1,
}
FEATURES_TO_USE = 'mfcc'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
featuresExist = True
impro_or_script = 'impro'
featuresFileName = 'features_{}_{}.pkl'.format(FEATURES_TO_USE, impro_or_script)
toSaveFeatures = True
WAV_PATH = "E:/Test/IEMOCAP/"
RATE = 16000
MODEL_NAME = 'MyModel_2'
MODEL_PATH = 'models/{}_{}.pth'.format(MODEL_NAME, FEATURES_TO_USE)

dict = {
    'neutral': torch.Tensor([0]),
    'happy': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'angry': torch.Tensor([3]),
}
label_num = {
    'neutral': 0,
    'happy': 0,
    'sad': 0,
    'angry': 0,
}


def process_data(path, t=2, train_overlap=1, val_overlap=1.6, RATE=16000):
    path = path.rstrip('/')
    wav_files = glob.glob(path + '/*.wav')
    meta_dict = {}
    val_dict = {}
    LABEL_DICT1 = {
        '01': 'neutral',
        # '02': 'frustration',
        # '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        # '06': 'fearful',
        '07': 'happy',  # excitement->happy
        # '08': 'surprised'
    }

    label_num = {
        'neutral': 0,
        'happy': 0,
        'sad': 0,
        'angry': 0,
    }

    n = len(wav_files)
    train_files = []
    valid_files = []
    train_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))
    valid_indices = list(set(range(n)) - set(train_indices))
    for i in train_indices:
        train_files.append(wav_files[i])
    for i in valid_indices:
        valid_files.append(wav_files[i])

    print("constructing meta dictionary for {}...".format(path))
    for i, wav_file in enumerate(tqdm(train_files)):
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            continue
        label = LABEL_DICT1[label]

        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            continue

        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(label)
            assert t - train_overlap > 0
            index += int((t - train_overlap) * RATE / overlapTime[label])
            label_num[label] += 1
        X1 = np.array(X1)
        meta_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    print("building X, y...")
    train_X = []
    train_y = []
    for k in meta_dict:
        train_X.append(meta_dict[k]['X'])
        train_y += meta_dict[k]['y']
    train_X = np.row_stack(train_X)
    train_y = np.array(train_y)
    assert len(train_X) == len(train_y), "X length and y length must match! X shape: {}, y length: {}".format(
        train_X.shape, train_y.shape)

    if (val_overlap >= t):
        val_overlap = t / 2
    for i, wav_file in enumerate(tqdm(valid_files)):
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            continue
        label = LABEL_DICT1[label]
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            continue
        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(label)
            index += int((t - val_overlap) * RATE)

        X1 = np.array(X1)
        val_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    return train_X, train_y, val_dict


def process_features(X, u=255):
    X = torch.from_numpy(X)
    max = X.max()
    X = X / max
    X = X.float()
    X = torch.sign(X) * (torch.log(1 + u * torch.abs(X)) / torch.log(torch.Tensor([1 + u])))
    X = X.numpy()
    return X


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件

    log_name = 'train.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)

    if (featuresExist == True):
        with open(featuresFileName, 'rb')as f:
            features = pickle.load(f)
        train_X_features = features['train_X']
        train_y = features['train_y']
        valid_features_dict = features['val_dict']
    else:
        logging.info("creating meta dict...")
        train_X, train_y, val_dict = process_data(WAV_PATH, t=T_stride, train_overlap=T_overlop)
        print(train_X.shape)
        print(len(val_dict))

        print("getting features")
        logging.info('getting features')
        feature_extractor = features.FeatureExtractor(rate=RATE)
        train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_X)
        valid_features_dict = {}
        for _, i in enumerate(val_dict):
            X1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict[i]['X'])
            valid_features_dict[i] = {
                'X': X1,
                'y': val_dict[i]['y']
            }
        if (toSaveFeatures == True):
            features = {'train_X': train_X_features, 'train_y': train_y,
                        'val_dict': valid_features_dict}
            with open(featuresFileName, 'wb') as f:
                pickle.dump(features, f)

    # logging.info("µ-law expansion")
    # train_X_features = process_features(train_X_features)
    # valid_X_features = process_features(valid_X_features)

    for i in train_y:
        label_num[i] += 1
    weight = torch.Tensor([(sum(label_num.values()) - label_num['neutral']) / sum(label_num.values()),
                           (sum(label_num.values()) - label_num['happy']) / sum(label_num.values()),
                           (sum(label_num.values()) - label_num['sad']) / sum(label_num.values()),
                           (sum(label_num.values()) - label_num['angry']) / sum(label_num.values())]).cuda()

    train_data = data_loader.DataSet(train_X_features, train_y)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = model.MACNN(attention_head, attention_hidden)
    # model=model.ACNN()
    # model = model.PaseACNN()
    # model = model.VLSS_CNN()
    # model = model.CapNetCNN(0.6)
    # model = model.OctConvACNN()

    if torch.cuda.is_available():
        model = model.cuda()

    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=1e-6)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logging.info("training...")
    maxWA = 0
    maxUA = 0
    totalrunningTime = 0
    for i in range(Epochs):
        #
        startTime = time.clock()
        tq = tqdm(total=len(train_y))
        model.train()
        print_loss = 0
        for _, data in enumerate(train_loader):
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out = model(x.unsqueeze(1))
            # out = model(x)
            loss = criterion(out, y.squeeze(1))
            print_loss += loss.data.item() * BATCH_SIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.update(BATCH_SIZE)
        tq.close()
        print('epoch: {}, loss: {:.4}'.format(i, print_loss / len(train_X_features)))
        logging.info('epoch: {}, loss: {:.4}'.format(i, print_loss))
        if (i > 0 and i % 10 == 0):
            learning_rate = learning_rate / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        # validation
        endTime = time.clock()
        totalrunningTime += endTime - startTime
        print(totalrunningTime)
        model.eval()
        UA = [0, 0, 0, 0]
        num_correct = 0
        class_total = [0, 0, 0, 0]
        matrix = np.mat(np.zeros((4, 4)), dtype=int)
        for _, i in enumerate(valid_features_dict):
            x, y = valid_features_dict[i]['X'], valid_features_dict[i]['y']
            x = torch.from_numpy(x).float()
            y = dict[y[0]].long()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            if (x.size(0) == 1):
                x = torch.cat((x, x), 0)
            out = model(x.unsqueeze(1))
            # out = model(x)
            pred = torch.Tensor([0, 0, 0, 0]).cuda()
            for j in range(out.size(0)):
                pred += out[j]
            pred = pred / out.size(0)
            pred = torch.max(pred, 0)[1]
            if (pred == y):
                num_correct += 1
            matrix[int(y), int(pred)] += 1

        for i in range(4):
            for j in range(4):
                class_total[i] += matrix[i, j]
            UA[i] = round(matrix[i, i] / class_total[i], 3)
        WA = num_correct / len(valid_features_dict)
        if (maxWA < WA):
            maxWA = WA
            torch.save(model.state_dict(), MODEL_PATH)
        if (maxUA < sum(UA) / 4):
            maxUA = sum(UA) / 4
        print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        logging.info('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        print(matrix)
        logging.info(matrix)
