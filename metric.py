# metric.py
# 원하는 metric 함수를 추가하면 됩니다.

# 배치 사이즈에 따라 계산되는 값이 달라질 수 있으므로, 
# 지난 번 프로젝트와 달리 배치 단위로 계산하지 않고 전체 데이터에 대해 계산하도록 만들기 위해
# (분자, 분모) 형태로 반환하도록 수정하고 이를 MetricTracker.update()에 반영하였습니다.
# dataloader의 배치 사이즈를 바꾸어도 성능이 그대로 나오는지 확인해보세요.


import torch

def acc(output, target):
    with torch.no_grad():
        assert len(output) == len(target)
        correct = 0
        correct += torch.sum(output == target).item()
    return correct, len(target)
    

def precision(output, target):
    with torch.no_grad():
        assert len(output) == len(target)
        tp = torch.sum(output & target)
        fp = torch.sum(output & ~target)
        return tp, tp + fp


    

def recall(output, target):
    with torch.no_grad():
        assert len(output) == len(target)
        tp = torch.sum(output & target)
        fn = torch.sum(~output & target)
        return tp, tp + fn
    

def f1(output, target):
    with torch.no_grad():
        assert len(output) == len(target)
        tp = torch.sum(output & target)
        fp = torch.sum(output & ~target)
        fn = torch.sum(~output & target)
        return tp, (tp + (fp + fn) / 2)


def specificity(output, target):
    with torch.no_grad():
        assert len(output) == len(target)
        tn = torch.sum(~output & ~target)
        fp = torch.sum(output & ~target)
        return tn, tn + fp
    

def fnr(output, target):
    with torch.no_grad():
        assert len(output) == len(target)
        tp = torch.sum(output & target)
        fn = torch.sum(~output & target)
        return fn, tp + fn

    

def fpr(output, target):
    with torch.no_grad():
        assert len(output) == len(target)
        tn = torch.sum(~output & ~target)
        fp = torch.sum(output & ~target)
        return fp, tn + fp
    

if __name__ == '__main__':
    import torch
    
    output = torch.tensor([10, 4, -2, 3, -1], dtype=torch.float32)
    target = torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)
    output = torch.sigmoid(output) > 0.5
    target = target.bool()

    print(f'Accuracy  : {acc(output, target)[0] * 100:.2f} %, '
          f'{acc(output, target)[1]} samples')
    print(f'Precision : {precision(output, target)[0] * 100:.2f} %, '
          f'{precision(output, target)[1]} samples')
    print(f'Recall    : {recall(output, target)[0] * 100:.2f} %, '
          f'{recall(output, target)[1]} samples')
    print(f'F1        : {f1(output, target)[0] * 100:.2f} %, '
          f'{f1(output, target)[1]} samples')
