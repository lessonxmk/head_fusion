# from pydub import AudioSegment
# import glob
# from tqdm import tqdm
#
# ogg_list = glob.glob('e:/test/esc-50/*/*.ogg')
# for path in tqdm(ogg_list):
#     audio = AudioSegment.from_ogg(path)
#     audio = audio.set_frame_rate(16000)
#     audio.export(path + '.wav', format='wav')

NOISE_TYPE = {
    0: '101 - Dog',
    1: '102 - Rooster',
    2: '103 - Pig',
    3: '104 - Cow',
    4: '105 - Frog',
    5: '106 - Cat',
    6: '107 - Hen',
    7: '108 - Insects',
    8: '109 - Sheep',
    9: '110 - Crow',
    10: '201 - Rain',
    11: '202 - Sea waves',
    12: '203 - Crackling fire',
    13: '204 - Crickets',
    14: '205 - Chirping birds',
    15: '206 - Water drops',
    16: '207 - Wind',
    17: '208 - Pouring water',
    18: '209 - Toilet flush',
    19: '210 - Thunderstorm',
    20: '301 - Crying baby',
    21: '302 - Sneezing',
    22: '303 - Clapping',
    23: '304 - Breathing',
    24: '305 - Coughing',
    25: '306 - Footsteps',
    26: '307 - Laughing',
    27: '308 - Brushing teeth',
    28: '309 - Snoring',
    29: '310 - Drinking - sipping',
    30: '401 - Door knock',
    31: '402 - Mouse click',
    32: '403 - Keyboard typing',
    33: '404 - Door - wood creaks',
    34: '405 - Can opening',
    35: '406 - Washing machine',
    36: '407 - Vacuum cleaner',
    37: '408 - Clock alarm',
    38: '409 - Clock tick',
    39: '410 - Glass breaking',
    40: '501 - Helicopter',
    41: '502 - Chainsaw',
    42: '503 - Siren',
    43: '504 - Car horn',
    44: '505 - Engine',
    45: '506 - Train',
    46: '507 - Church bells',
    47: '508 - Airplane',
    48: '509 - Fireworks',
    49: '510 - Hand saw',
}
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


SEED = [111111, 123456, 0, 999999, 987654]
BEGIN_TYPE = 26
attention_head = 4
attention_hidden = 32
NOISE_OFFSET = 0
Amplitude_Factor = 0.1
import features
import model as MODEL
import data_loader
import CapsNet

# learning_rate = 0.001
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
WAV_PATH = "../IEMOCAP/"
RATE = 16000

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


def process_data(path, t=2, train_overlap=1, val_overlap=1.6, RATE=16000, noise_type=0, offset=0):
    path = path.rstrip('/')
    wav_files = glob.glob(path + '/*.wav')
    noise_files = glob.glob('../ESC-50/' + NOISE_TYPE[noise_type] + '/*.wav')
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

        noise, _ = librosa.load(noise_files[int(np.random.choice(len(noise_files), 1))], sr=RATE)
        noise = noise * Amplitude_Factor
        noise = np.concatenate((noise[int(offset * RATE):], noise[:int(offset * RATE)]))
        while (len(noise) < len(wav_data)):
            noise = np.concatenate((noise, noise))
        wav_data = wav_data + noise[:len(wav_data)]

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


def process_noise_valid(path, t=2, train_overlap=1, val_overlap=1.6, RATE=16000, noise_type=0, offset=0):
    path = path.rstrip('/')
    wav_files = glob.glob(path + '/*.wav')
    noise_files = glob.glob('../ESC-50/' + NOISE_TYPE[noise_type] + '/*.wav')
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

    n = len(wav_files)
    train_files = []
    valid_files = []
    train_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))
    valid_indices = list(set(range(n)) - set(train_indices))
    for i in train_indices:
        train_files.append(wav_files[i])
    for i in valid_indices:
        valid_files.append(wav_files[i])

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

        noise, _ = librosa.load(noise_files[int(np.random.choice(len(noise_files), 1))], sr=RATE)
        noise = noise * Amplitude_Factor
        noise = np.concatenate((noise[int(offset * RATE):], noise[:int(offset * RATE)]))
        while (len(noise) < len(wav_data)):
            noise = np.concatenate((noise, noise))
        wav_data = wav_data + noise[:len(wav_data)]

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
    return val_dict


if __name__ == '__main__':
    for noise_type in range(len(NOISE_TYPE)):
        if (noise_type < BEGIN_TYPE):
            continue

        for seed in SEED:
            MODEL_NAME = 'MACNN_Noise_{}_offset{}_seed{}_AF{}'.format(NOISE_TYPE[noise_type], NOISE_OFFSET, seed,
                                                                      Amplitude_Factor)
            MODEL_PATH = 'models/{}_{}.pth'.format(MODEL_NAME, FEATURES_TO_USE)
            learning_rate = 0.001
            setup_seed(seed)
            feature_extractor = features.FeatureExtractor(rate=RATE)
            train_X, train_y, val_dict = process_data(WAV_PATH, t=T_stride, train_overlap=T_overlop,
                                                      noise_type=noise_type, offset=NOISE_OFFSET)
            train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_X)
            valid_features_dict = {}
            for _, i in enumerate(val_dict):
                X1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict[i]['X'])
                valid_features_dict[i] = {
                    'X': X1,
                    'y': val_dict[i]['y']
                }

            train_data = data_loader.DataSet(train_X_features, train_y)
            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

            model = MODEL.MACNN(attention_head, attention_hidden)

            if torch.cuda.is_available():
                model = model.cuda()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                   weight_decay=1e-6)

            maxWA = 0
            maxUA = 0
            for epoch in range(Epochs):
                #
                # startTime = time.clock()
                #tq = tqdm(total=len(train_y))
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
                    #tq.update(BATCH_SIZE)
                #tq.close()
                if (epoch > 0 and epoch % 10 == 0):
                    learning_rate = learning_rate / 10
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                # validation
                # endTime = time.clock()
                # totalrunningTime += endTime - startTime
                # print(totalrunningTime)
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
                print('noise:{},epoch:{}'.format(NOISE_TYPE[noise_type], epoch))
                # print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
                # print(matrix)

            setup_seed(seed)
            valid_noise_dict = process_noise_valid(WAV_PATH, t=T_stride, train_overlap=T_overlop,
                                                   noise_type=noise_type, offset=NOISE_OFFSET)
            valid_noise_features_dict = {}
            for _, i in enumerate(valid_noise_dict):
                X1 = feature_extractor.get_features(FEATURES_TO_USE, valid_noise_dict[i]['X'])
                valid_noise_features_dict[i] = {
                    'X': X1,
                    'y': valid_noise_dict[i]['y']
                }
            model = MODEL.MACNN(attention_head, attention_hidden)
            model.load_state_dict(torch.load(MODEL_PATH))
            model=model.cuda()
            model.eval()
            noiseWA = 0
            noiseUA = [0,0,0,0]
            num_noise_correct=0
            class_noise_total = [0, 0, 0, 0]
            matrix_noise = np.mat(np.zeros((4, 4)), dtype=int)
            for _, i in enumerate(valid_noise_features_dict):
                x, y = valid_noise_features_dict[i]['X'], valid_noise_features_dict[i]['y']
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
                    num_noise_correct += 1
                matrix_noise[int(y), int(pred)] += 1
            for i in range(4):
                for j in range(4):
                    class_noise_total[i] += matrix_noise[i, j]
                noiseUA[i] = round(matrix_noise[i, i] / class_noise_total[i], 3)
            noiseWA = num_noise_correct / len(valid_noise_features_dict)



            with open('noise_base_26_2.csv', 'a') as f:
                f.write('{},{},{},{},{},{}\n'.format(NOISE_TYPE[noise_type], seed, maxWA, maxUA,noiseWA,sum(noiseUA)/4))
