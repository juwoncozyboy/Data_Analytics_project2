# train.py
# 각종 주석은 이해를 돕기 위한 것으로, 삭제하고 프로젝트를 진행하셔도 무방합니다.
# TODO가 아닌 부분도 얼마든지 삭제, 추가, 수정 가능합니다.
# 단, 수정 금지라고 쓰여있는 항목에 대해서는 수정하지 말아주세요. (불가피하게 수정이 필요해질 경우 메일로 문의)

import os
import datetime
from omegaconf import OmegaConf
import logging

import torch
import dataset as module_data
import model as module_arch
import metric as module_metric
from utils import fix_seed, MetricTracker

import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter("runs/")




def main(config, logger, tb_logger=None):

    # 시드 고정
    fix_seed(config['seed'])

    # 데이터 불러오기
    train_dataset = getattr(module_data, config.dataset.type)(mode='train', 
                                                              **config.dataset.args)
    # 데이터로더 설정하기
    train_dataloader = getattr(torch.utils.data, config.dataloader.type)(train_dataset, 
                                                                         **config.dataloader.args)

    # 모델 선언하기
    input_size = train_dataset.__getitem__(0)[0].size(0)
    seq_len = train_dataset.__getitem__(0)[0].size(1)
    model = getattr(module_arch, config.model.type)(input_size=input_size, 
                                                    seq_len=seq_len, 
                                                    **config.model.args)

    # resume이 참일 경우, 해당 경로에서 모델 불러와서 학습 진행
    if config.train.resume:
        model.load_state_dict(torch.load(config.train.resume_path))

    # MPS 지원 여부 확인
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 모델을 GPU 메모리에 올리기
    model = model.to(device)

    # loss 및 metric 설정
    criterion = getattr(torch.nn, config.loss)().to(device)
    metrics = [getattr(module_metric, met) for met in config.metrics]
    metric_tracker_train = MetricTracker('loss', *config.metrics)
        
    # optimizer 및 scheduler 설정
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim, config.optimizer.type)(trainable_params, 
                                                            **config.optimizer.args)
    if config.lr_scheduler.use:
        lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.type)(optimizer, 
                                                                                   **config.lr_scheduler.args)

    # 학습 진행
    print('Start training...')
    for epoch in range(1, config.train.epochs+1):
        
        # train loop
        model.train()
        metric_tracker_train.reset()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # loss 및 metric 결과 저장
            metric_tracker_train.update('loss', loss.item())
            output = (torch.sigmoid(output) > 0.5).squeeze()
            target = target.bool().squeeze()
            for met in metrics:
                metric_tracker_train.update(met.__name__, *met(output, target))

        # scheduler 업데이트
        if config.lr_scheduler.use:
            lr_scheduler.step()
        
        
        # tensorboard에 loss, metric, lr 기록
        if tb_logger is not None:
            for key, value in metric_tracker_train.result().items():
                tb_logger.add_scalar(f'train/{key}', value, epoch)
            tb_logger.add_scalar('train/loss', metric_tracker_train.avg('loss'), epoch)
            tb_logger.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)
        

        # 현재 epoch 결과를 print_period마다 출력
        print_result = (epoch % config.train.print_period) == 0
        if print_result:
            logger.info(f'[Epoch {epoch:02d}/{config.train.epochs:02d}] ' +
                        f'TRAIN ' +
                        ',  '.join([f'{k.upper()}: {v:.4f}' for k, v in metric_tracker_train.result().items()]))
    

        # TODO: 최종 제출할 모델을 골라, teamX_model.pt로 이름을 수동 변경해주시면 됩니다.
        if epoch % config.train.save_period == 0:
            print('-' * 10)
            save_file_path = os.path.join(config.train.save_dir,   
                                          #f'checkpoints/{config.train.save_model_name}-e{epoch}.pt')
                                          #f'checkpoints/{config.train.save_model_name}.pt')
                                           f'checkpoints/teamF_model.pt')
            
            # 만일 이미 같은 이름의 파일이 존재한다면, 덮어쓸 것인지 물어봄
            if os.path.exists(save_file_path):
                overwrite = input(f'{save_file_path} already exists. overwrite? [y/N]')
                if overwrite != 'y':
                    print(f'skip save model: {save_file_path}')
                    print('-' * 10)
                    continue
            torch.save(model.state_dict(), save_file_path)
            config.train.save_model_list.append(save_file_path)  # 저장할 경우, 해당 모델의 명칭을 config에 리스트로 추가
            print(f'save model: {save_file_path}')
            print('-' * 10)

    print('=' * 20)
    print('... done!')

if __name__ == "__main__":

    # config 불러오기
    cfg = OmegaConf.load(os.path.join('src', f'config.yaml'))
    # 사용되지 않는 config 정보 삭제 (model.args 부분)
    cfg.model.args = cfg.model.args[cfg.model.type]
    # 저장한 모델의 경로를 저장할 리스트 생성
    cfg.train.save_model_list = []

    # 각종 디렉토리 생성
    os.makedirs(os.path.join(cfg.train.save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(cfg.train.save_dir, 'logs'), exist_ok=True)

    # logger 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'{cfg.train.save_dir}/logs/{time}.log')
    logger.addHandler(file_handler)

    # tensorboard 설정
    if cfg.train.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f'{cfg.train.save_dir}/tensorboard/{time}')

    # main 함수 실행
    main(config=cfg, logger=logger, tb_logger=writer if cfg.train.tensorboard else None)

    # 사용한 config 정보를 logs/ 디렉토리에 재저장
    with open(f'{cfg.train.save_dir}/logs/{time}.yaml', 'w') as f:
        OmegaConf.save(cfg, f)