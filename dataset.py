# dataset.py
# TODO가 아닌 부분도 얼마든지 수정 가능합니다.
# 단, 수정 금지라고 쓰여있는 항목에 대해서는 수정하지 말아주세요. (불가피하게 수정이 필요할 경우 메일로 미리 문의)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns


# !!! 수정 금지 !!!
VAR_LIST = ['ACC_x', 'ACC_y', 'ACC_z', #가속도
            'GRVT_x', 'GRVT_y', 'GRVT_z', #중력가속도
            'MAG_x', 'MAG_y', 'MAG_z', #자기장
            'ORT_a_cos', 'ORT_a_sin', # 방향 azimuth
            'ORT_p_cos', 'ORT_p_sin', # 방향 pitch
            'ORT_r_cos', 'ORT_r_sin'] # 방향 roll

# !!! 수정 금지 !!!
VAR_DICT = {var: i for i, var in enumerate(VAR_LIST)}




class SensorDataset(Dataset):
    def __init__(self, in_columns, mode='train', data_dir='data', data_fname='da23_sensor_data.npz'):
        self.in_columns = in_columns  # 인스턴스 변수로 in_columns를 저장
        self.mode = mode  # 인스턴스 변수로 mode를 저장
        data_fname = data_fname.replace('.npz', f'({mode}).npz')
        fname = os.path.join(data_dir, data_fname)
        if not os.path.exists(fname):
            raise FileNotFoundError(f'{fname} does not exist')
        
        var_idx = [VAR_DICT[var] for var in in_columns]

        data = np.load(fname)
        if mode in ['train', 'valid', 'test']:
            self.x = torch.from_numpy(data['X'][:, var_idx]).float()
            self.y = torch.from_numpy(data['Y']).float()
        else:
            raise ValueError(f'Invalid mode {mode}')
        

        # 평균과 표준편차 계산
        self.mean = torch.mean(self.x, dim=[0, 2], keepdim=True)
        self.std = torch.std(self.x, dim=[0, 2], keepdim=True)

        #del data
        # 노이즈 추가 후 데이터 증강
        self.add_noise()
        #self.augment_data()
        self.augment_class1_data()

    def augment_data(self, time_shift_max=10):
        # 시간적 변형을 위한 데이터 증강 로직
        augmented_x = []
        for x in self.x:
            time_shift = np.random.randint(-time_shift_max, time_shift_max)
            x_shifted = torch.roll(x, shifts=time_shift, dims=1)
            augmented_x.append(x_shifted)
        self.x = torch.stack(augmented_x)

   

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 데이터 가져오기 및 정규화
        x = self.x[idx]
        x_normalized = (x - self.mean) / (self.std + 1e-6)

        # 잡음 제거 적용 (예시 로직)
        x_denoised = self.denoise(x_normalized)

        # 필요한 경우 데이터 형태 변환
        x_denoised = x_denoised.squeeze(0)

        return x_denoised, self.y[idx]


    def denoise(self, x):

            ##의미있는부분 XX -> 제가 잘못짠건데 이거 지우면 안돌아가지더라고요...##
        return x

    def augment_data(self, time_shift_max=10, noise_level=0.02):
        """
        데이터에 시간적 변형과 노이즈를 추가하는 함수
        time_shift_max: 최대 시간 변형 길이
        noise_level: 추가되는 노이즈의 수준
        """
        #augmented_x = []
        #for i in range(len(self.x)):
        #    x = self.x[i]

            # 시간적 변형
        #    time_shift = np.random.randint(-time_shift_max, time_shift_max)
        #    x_shifted = torch.roll(x, shifts=time_shift, dims=1)

            # 노이즈 추가
        #    noise = torch.randn_like(x) * noise_level
        #    x_noisy = x_shifted + noise

        #    augmented_x.append(x_noisy)

        #self.x = torch.stack(augmented_x)
        pass


    def plot_sample_with_outliers(self, idx, k=2):
        """
        주어진 샘플에 대해 시계열 데이터와 이상치를 시각화합니다.
        idx: 샘플 인덱스
        k: 이상치 탐지를 위한 표준 편차의 배수
        """
        sample_x = self.x[idx].T.numpy()
        mean = np.mean(sample_x, axis=0)
        std = np.std(sample_x, axis=0)

        outliers = np.abs(sample_x - mean) > k * std
        plt.figure(figsize=(12, 6))
        for i, column in enumerate(self.in_columns):
            plt.subplot(len(self.in_columns), 1, i + 1)
            plt.plot(sample_x[:, i], label=f'{column}')
            plt.scatter(np.where(outliers[:, i])[0], sample_x[outliers[:, i], i], c='red')  # 이상치 강조
            plt.legend()

        sample_y = 'pos' if self.y[idx].numpy() else 'neg'
        plt.suptitle(f'{self.mode} data : {idx}-th sample : y={sample_y}')
        plt.show()

    def remove_outliers(self, k=2):
        """
        모든 데이터 샘플에서 이상치를 0으로 대체합니다.
        k: 이상치 탐지를 위한 표준 편차의 배수
        """
        for i in range(len(self.x)):
            sample_x = self.x[i]
            mean = torch.mean(sample_x, dim=1, keepdim=True)
            std = torch.std(sample_x, dim=1, keepdim=True)

            outliers = torch.abs(sample_x - mean) > k * std
            sample_x[outliers] = 0  # 이상치를 0으로 대체

            self.x[i] = sample_x
    
    def add_noise(self, noise_level=0.02):
        """
        모든 데이터 샘플에 무작위 노이즈를 추가하는 함수
        noise_level: 추가되는 노이즈의 수준
        """
        # x 데이터에 노이즈 추가
        for i in range(len(self.x)):
            noise = torch.randn_like(self.x[i]) * noise_level
            self.x[i] += noise
    
    def augment_class1_data(self, time_shift_max=10, noise_level=0.02, class_value=1):
        """
        class1 데이터에 대해 시간적 변형과 노이즈를 추가하는 함수
        time_shift_max: 최대 시간 변형 길이
        noise_level: 추가되는 노이즈의 수준
        class_value: class1에 해당하는 값 (예: 1)
        """
        augmented_x = []
        for i in range(len(self.x)):
            if self.y[i] == class_value:
                x = self.x[i]

                # 시간적 변형
                time_shift = np.random.randint(-time_shift_max, time_shift_max)
                x_shifted = torch.roll(x, shifts=time_shift, dims=1)

                # 노이즈 추가
                noise = torch.randn_like(x) * noise_level
                x_noisy = x_shifted + noise

                augmented_x.append(x_noisy)
            else:
                augmented_x.append(self.x[i])

        self.x = torch.stack(augmented_x)
    
    def reduce_class0_data(self, reduction_factor=0.5):
        """
        클래스 0의 데이터를 감소시키는 함수
        reduction_factor: 감소시킬 비율 (0 ~ 1 사이)
        """
        class0_indices = [i for i in range(len(self.y)) if self.y[i] == 0]
        reduced_indices = np.random.choice(class0_indices, int(len(class0_indices) * reduction_factor), replace=False)

        self.x = torch.cat([self.x[i].unsqueeze(0) for i in range(len(self.x)) if i not in reduced_indices])
        self.y = torch.cat([self.y[i].unsqueeze(0) for i in range(len(self.y)) if i not in reduced_indices])

    
    def find_missing_values(self):
        """
        데이터셋에서 결측치를 찾는 함수입니다.
        """
        missing_values = torch.isnan(self.x)  # 결측치가 있는 위치를 True로 표시
        return missing_values 
    
    def sensor_to_img(self, idx):
        """
        센서 데이터를 이미지로 변환
        """
        data = self.x[idx].numpy()
        plt.figure()
        for i, column in enumerate(self.in_columns):
            plt.plot(data[:, i], label=column)
        plt.legend()
        plt.title(f'Sensor Data to Image for index {idx}')
        #plt.close()  # 화면에 표시하지 않고 닫음
        return plt













if __name__ == "__main__":

    in_columns = ['ACC_x', 'ACC_y', 'ACC_z', 
                  'ORT_a_cos', 'ORT_a_sin', 
                  'ORT_p_cos', 'ORT_p_sin', 
                  'ORT_r_cos', 'ORT_r_sin']
    mode = 'valid'
    data_dir = 'data'
    data_fname = 'da23_sensor_data.npz'

    dataset = SensorDataset(in_columns=in_columns,
                            mode=mode,
                            data_dir=data_dir,
                            data_fname=data_fname)
    
    print(f'dataset length : {len(dataset)}')
    print(f'x.shape : {dataset.x.shape}')
    print(f'y.shape : {dataset.y.shape}')
    missing_values = dataset.find_missing_values()
    print(f'Missing values in dataset: \n{missing_values}')

    # 결측치의 위치와 개수를 확인하려면:
    missing_indices = torch.where(missing_values)
    print(f'Number of missing values: {len(missing_indices[0])}')

    import matplotlib.pyplot as plt
    sample_idx = 0
    sample_x = dataset.x[sample_idx].T.numpy()
    sample_x = (sample_x - sample_x.mean(axis=0)) / sample_x.std(axis=0)  # normalize
    sample_y = 'pos' if dataset.y[sample_idx].numpy() else 'neg'
    plt.plot(sample_x)
    plt.title(f'{mode} data : {sample_idx}-th sample : y={sample_y}')
    plt.legend(in_columns)
    plt.show()

    sample_idx = 0
    fig = dataset.sensor_to_img(sample_idx)
    plt.show()  # 그래프를 화면에 표시
    
