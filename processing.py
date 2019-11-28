import os
import librosa
import re
import numpy as np
import pandas as pd
from sklearn import preprocessing

#sampling rate = 44100
#hop_length = 20ms
#window_length = 40ms

def load_wavs(wav_dir, sr):
    wavs = list()
    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr, mono=True)
        wavs.append(wav)
    return wavs

def load_wavs_dict(wav_dir, sr):
    wavs = dict()
    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr,  mono=True)
        wavs[file_path] = wav
    return wavs

def make_spectrogram(wav_dir, mbe_dir, sr, nfft):
    wavs = load_wavs_dict(wav_dir, sr)
    for wav_label in wavs:
        file_label = re.findall("([^\/]*)\.wav$", wav_label)
        file_path = os.path.join(mbe_dir, file_label[0] + ".npy")
        np.save(file_path,librosa.feature.melspectrogram(wavs[wav_label], n_fft = nfft, hop_length = nfft//2, win_length = nfft, sr=sr, n_mels=40).T)
    print("wav convert to mbe is compleated\n")

def load_anns(ann_dir):
    anns = list()
    for file in os.listdir(ann_dir):
        file_path = os.path.join(ann_dir, file)
        ann = pd.read_table(file_path)
        anns.append(ann)
    return anns

def load_anns_dict(ann_dir):
    anns = dict()
    for file in os.listdir(ann_dir):
        file_path = os.path.join(ann_dir, file)
        ann = pd.read_table(file_path, header = None)
        anns[ann[6][0]] = ann
    return anns

def make_anns(ann_dir, label_dir, mbe_dir, classes, sr, nfft):
    class_to_number = dict()
    for i, label in enumerate(classes):
        class_to_number[label] = i
    print(class_to_number)
    anns = load_anns_dict(ann_dir)
    for ann_label in anns:
        ann = anns[ann_label]
        file_path = os.path.join(label_dir, ann_label + ".npy")
        mbe_time = len(np.load(os.path.join(mbe_dir,ann_label) + ".npy"))
        label_array = np.zeros((mbe_time, len(classes)))
        for i in range(len(anns[ann_label])):
            begin_frame = int(ann[2][i]*nfft / (sr*2))
            end_frame = int(ann[3][i]*nfft / (sr*2))
            class_number = class_to_number[ann[4][i]]
            label_array[begin_frame:end_frame, class_number] = 1
        np.save(file_path, label_array)
    print("label making is done\n")

def load_desc_file(_desc_file):
    _desc_dict = dict()

    __class_labels ={
        "brakes squeaking": 0,
        "car": 1,
        "children": 2,
        "large vehicle": 3,
        "people speaking": 4,
        "people walking": 5
    }

    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        name = name.split(".")[0]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[2]), float(words[3]), __class_labels[words[-1]]])
    return _desc_dict

def make_validation_data_1(mbe_dir, label_dir, validation_dir, fold):
    train_file = os.path.join(validation_dir, 'street_fold{}_train.txt'.format(fold))
    evaluate_file = os.path.join(validation_dir, 'street_fold{}_evaluate.txt'.format(fold))
    train_dict = load_desc_file(train_file)
    val_dict = load_desc_file(evaluate_file)

    X_train, Y_train, X_val, Y_val = None, None, None, None

    for key in train_dict.keys():
        print("train:{}".format(key))
        tmp_mbe_file = os.path.join(mbe_dir, '{}.npy'.format(key))
        tmp_mbe = np.load(tmp_mbe_file)
        tmp_label_file = os.path.join(label_dir, '{}.npy'.format(key))
        tmp_label = np.load(tmp_label_file)
        if X_train is None:
            X_train, Y_train = tmp_mbe, tmp_label
        else:
            X_train, Y_train = np.concatenate((X_train, tmp_mbe), 0), np.concatenate((Y_train, tmp_label), 0)

    for key in val_dict.keys():
        print("test:{}".format(key))
        tmp_mbe_file = os.path.join(mbe_dir, '{}.npy'.format(key))
        tmp_mbe = np.load(tmp_mbe_file)
        tmp_label_file = os.path.join(label_dir, '{}.npy'.format(key))
        tmp_label = np.load(tmp_label_file)
        if X_val is None:
            X_val, Y_val = tmp_mbe, tmp_label
        else:
            X_val, Y_val = np.concatenate((X_val, tmp_mbe), 0), np.concatenate((Y_val, tmp_label), 0)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, Y_train, X_val, Y_val

def make_validation_data(mbe_dir, label_dir, in_dim, out_dim):
    X = np.empty([0, in_dim])
    Y = np.empty([0, out_dim])
    for file in os.listdir(mbe_dir):
        file_label = re.findall("(.*)\.npy$", file)
        x_dir = os.path.join(mbe_dir, file_label[0] + ".npy")
        y_dir = os.path.join(label_dir, file_label[0] + ".npy")
        x = np.load(x_dir)
        y = np.load(y_dir)
        X = np.concatenate([X, x], 0)
        Y = np.concatenate([Y, y], 0)
    return X, Y

def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
            data = data.reshape((data.shape[0] // subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
            data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
            data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1], data.shape[2]))
    return data

if __name__ == "__main__":
    wav_dir = "audio/street"
    mbe_dir = "mbe/street"
    ann_dir = "meta/street"
    label_dir = "label/street"
    classes = ["brakes squeaking", "car", "children", "large vehicle", "people speaking","people walking"]
    make_spectrogram(wav_dir, mbe_dir, 44100, 2048)
    make_anns(ann_dir, label_dir, mbe_dir, classes, 44100, 2048)
