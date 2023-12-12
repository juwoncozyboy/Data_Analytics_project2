# test.py
# !!! 전부 수정 금지 !!!

# 검증은 현재 지정된 것과 동일하게 valid 모드에서 진행됩니다.
# 테스트는 향후 test 모드에서 진행됩니다. (데이터 미제공)


import os
from omegaconf import OmegaConf

import numpy as np
import torch
import dataset as module_data
import model as module_arch
import metric as module_metric
from utils import MetricTracker

MODE = 'valid'  # !!! 수정 금지 !!!

def get_test_y(mode):
    """
    Do not fix this function
    """
    if mode == 'valid':
        fname = 'data/da23_sensor_data(valid).npz'
    elif mode == 'test':
        fname = 'data/da23_sensor_data(test).npz'
    y = np.load(fname)[f'Y']

    return y


def main(config):
    
    # 데이터 불러오기
    test_dataset = getattr(module_data, config.dataset.type)(mode=MODE,
                                                             **config.dataset.args)
    # 데이터로더 설정하기
    test_dataloader = getattr(torch.utils.data, config.dataloader.type)(test_dataset, 
                                                                        batch_size=128,
                                                                        shuffle=False,
                                                                        num_workers=4,)

    ###################################################################################################################
    # !!! 수정 금지 !!! : inspect test data
    if abs(test_dataset.y.numpy() - get_test_y(MODE)).sum() > 1e-3:
        raise ValueError('your test data is wrong!')
    ###################################################################################################################

    # 모델 불러오기
    input_size = test_dataset.__getitem__(0)[0].size(0)
    seq_len = test_dataset.__getitem__(0)[0].size(1)
    model = getattr(module_arch, config.model.type)(input_size=input_size, 
                                                    seq_len=seq_len,
                                                    **config.model.args[config.model.type])
    model.load_state_dict(torch.load(os.path.join(config.test.load_dir, config.test.load_fname)))
    model.eval()

    # GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델을 GPU 메모리에 올리기
    model = model.to(device)

    # metric 설정
    metrics = [getattr(module_metric, met) for met in config.test.metrics]
    metric_tracker_test = MetricTracker(*config.test.metrics)

    # 테스트 진행
    for i, (x, y) in enumerate(test_dataloader):
        x, y = x.to(device), y.to(device)
        pred_y = model(x)
        output = (torch.sigmoid(pred_y) > 0.5).squeeze()
        target = y.bool().squeeze()
        for met in metrics:
            metric_tracker_test.update(met.__name__, *met(output, target))
        
    print('\nTEST ' + ',  '.join([f'{k.upper()}: {v:.4f}' for k, v in metric_tracker_test.result().items()]))

    

if __name__ == '__main__':

    cfg = OmegaConf.load(os.path.join('src', f'config.yaml'))
    
    main(cfg)
