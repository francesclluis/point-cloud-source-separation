import open3d as o3d
import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
import librosa

from . import point_transforms as ptransforms


class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, opt, max_sample=-1, split='train'):
        # params
        self.num_frames = opt.num_frames
        self.stride_frames = opt.stride_frames
        self.frameRate = opt.frameRate
        self.voxel_size = opt.voxel_size
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.audSec = 1. * self.audLen / self.audRate
        self.binary_mask = opt.binary_mask
        self.rgbs_feature = opt.rgbs_feature

        # STFT params
        self.log_freq = opt.log_freq
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.HS = opt.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        # initialize point transform
        self._init_ptransform()

        self.num_channels = opt.num_channels

        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                self.list_sample.append(row)
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise('Error list_sample!')

        if self.split == 'train':
            self.list_sample *= opt.dup_trainset
            random.shuffle(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    def _init_ptransform(self):
        point_transform_list = []
        color_transform_list = []

        if self.split == 'train':
            point_transform_list.append(ptransforms.RandomRotation(axis=np.array([0, 1, 0])))
            point_transform_list.append(ptransforms.RandomTranslation(0.4))
            point_transform_list.append(ptransforms.RandomScale(0.5, 2))
            point_transform_list.append(ptransforms.RandomShear())
            color_transform_list.append(ptransforms.RandomGaussianNoise())
            color_transform_list.append(ptransforms.RandomValue())
            color_transform_list.append(ptransforms.RandomSaturation())
        else:
            pass #apply no transformation in evaluation mode

        self.point_transform = transforms.Compose(point_transform_list)
        self.color_transform = transforms.Compose(color_transform_list)

    def _create_coords(self, points):
        coords = []
        for xyz in points:
            coords.append(np.floor(xyz/self.voxel_size))
        return coords

    def _load_frames(self, paths):
        points = []
        rgbs = []
        for path in paths:
            xyz, color = self._load_frame(path)
            points.append(xyz)
            rgbs.append(color)
        points = self.point_transform(points)
        if self.rgbs_feature:
            rgbs = self.color_transform(rgbs)
        coords = self._create_coords(points)
        return points, coords, rgbs

    def _load_frame(self, path):
        pcd = o3d.io.read_point_cloud(path)
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        return points, colors

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            audio_raw, rate = torchaudio.load(path, channels_first=False)
            audio_raw = audio_raw.numpy().astype(np.float32)

            # convert to mono
            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            else:
                audio_raw = audio_raw[:, 0]
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)

        return audio_raw, rate

    def _load_audio(self, path, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        len_raw = audio_raw.shape[0]
        center = np.random.randint(self.audLen//2 + 1, len_raw - self.audLen//2)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]

        # mix
        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)

        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)

        # to tensor
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)
