# model.py
# 베이스라인 코드로 제공되는 CNN과 LSTM 모델 모두 마지막 출력이 logit의 형태이며,
# 이를 sigmoid 함수를 통과시켜 확률값으로 변환하는 과정이 모델에서 이루어지지 않고,
# 손실 함수 중 BCEWithLogitsLoss를 사용하여 이 과정이 손실 함수 내부에서 이루어지도록 설정되어 있습니다.
# (BCEWithLogitsLoss = sigmoid + BCELoss)
# 따라서, 모델을 추가 및 수정할 때에도 마지막 출력값이 logit의 형태가 되도록 설계해야 합니다.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self):
        """
        Forward pass logic
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)



class CNN(BaseModel):
    def __init__(self, input_size=15, seq_len=200, output_size=1, 
                 cnn_hidden_list=[16, 32], fc_hidden_list=[], dropout_p=0, batch_norm=False):
        super(CNN, self).__init__()

        def conv1d(inp, oup, kernel, stride, pad, batch_norm, dropout_p):
            layers = [
                nn.Conv1d(inp, oup, kernel, stride, pad, bias=not batch_norm),
                nn.ReLU(inplace=True)
            ]
            if batch_norm:
                layers.insert(1, nn.BatchNorm1d(oup))
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            return nn.Sequential(*layers)

        self.cnn_layers = nn.Sequential()
        cnn_output_len = seq_len
        cnn_hidden_list = [input_size] + cnn_hidden_list

        for i, (inp, oup) in enumerate(zip(cnn_hidden_list[:-1], cnn_hidden_list[1:])):
            self.cnn_layers.add_module(f'conv1d{i}', conv1d(inp, oup, kernel=3, stride=1, pad=1, 
                                                           batch_norm=batch_norm, dropout_p=dropout_p))
            cnn_output_len = (cnn_output_len + 2 - 3) // 1 + 1  # (n + 2p - k) / s + 1
            self.cnn_layers.add_module(f'pool{i}', nn.MaxPool1d(kernel_size=2, stride=2))
            cnn_output_len = (cnn_output_len - 2) // 2 + 1  # (n - k) / s + 1

        fc_input_size = cnn_hidden_list[-1] * cnn_output_len
        fc_hidden_list = [fc_input_size] + fc_hidden_list + [output_size]
        self.fc_layers = nn.Sequential()
        
        for i, (inp, oup) in enumerate(zip(fc_hidden_list[:-1], fc_hidden_list[1:])):
            self.fc_layers.add_module(f'fc{i}', nn.Linear(inp, oup))
            if i < len(fc_hidden_list) - 2:
                self.fc_layers.add_module(f'fc{i}_relu', nn.ReLU())
                if dropout_p:
                    self.fc_layers.add_module(f'fc{i}_dropout', nn.Dropout(dropout_p))

        self.init_weights()

    def forward(self, x):
        # input x: (batch_size, input_size, seq_len)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class LSTM(BaseModel):
    ''' Initialize the model '''
    def __init__(self, input_size=15, seq_len=200, output_size=1, 
                 lstm_hidden_dim=64, lstm_n_layer=2, bidirectional=True,
                 fc_hidden_list=[], dropout_p=0, batch_norm=False):
        '''
        input_size: 입력 데이터의 feature 수
        seq_len: 입력 데이터의 시퀀스 길이
        output_size: 출력 데이터의 차원 (=1)
        lstm_hidden_dim: LSTM layer의 hidden 차원
        lstm_n_layer: LSTM layer의 layer 수
        fc_hidden_list: FC layer의 hidden 차원 리스트 ([]일 경우, 1차원으로 요약하는 layer 하나만 적용)
        dropout_p: dropout 확률 (0~1, 0이면 dropout을 사용하지 않음)
        batch_norm: batch normalization 사용 여부
        '''
        super(LSTM, self).__init__()

        self.lstm_layers = nn.LSTM(input_size=input_size, 
                                   hidden_size=lstm_hidden_dim, 
                                   num_layers=lstm_n_layer, 
                                   batch_first=True, 
                                   bidirectional=bidirectional, 
                                   dropout=dropout_p)
        
        self.fc_layers = nn.Sequential()
        fc_hidden_list = [lstm_hidden_dim * (2 if bidirectional else 1)] + fc_hidden_list + [output_size]
        for i, (inp, oup) in enumerate(zip(fc_hidden_list[:-1], fc_hidden_list[1:])):
            self.fc_layers.add_module(f'fc{i}', nn.Linear(inp, oup))
            if i < len(fc_hidden_list) - 2:
                self.fc_layers.add_module(f'fc{i}_relu', nn.ReLU())
                if dropout_p:
                    self.fc_layers.add_module(f'fc{i}_dropout', nn.Dropout(dropout_p))

        self.init_weights()


    def forward(self, x):
        # input x: (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, input_size)
        x, _ = self.lstm_layers(x)  # (batch_size, seq_len, lstm_hidden_dim)
        x = x[:, -1, :]  # (batch_size, lstm_hidden_dim)
        x = self.fc_layers(x)

        return x  # (batch_size, output_size)
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

class CRNN(BaseModel):
    ''' Initialize the model '''
    def __init__(self, input_size=15, seq_len=200, output_size=1, 
                 cnn_hidden_list=[16, 32, 64, 128],  # CNN 레이어 수 증가
                 lstm_hidden_dim=64, lstm_n_layer=2, 
                 bidirectional=True, fc_hidden_list=[], dropout_p=0, batch_norm=False):
        '''
        input_size: 입력 데이터의 feature 수
        seq_len: 입력 데이터의 시퀀스 길이
        output_size: 출력 데이터의 차원 (=1)
        cnn_hidden_list: CNN layer의 hidden 차원 리스트
        lstm_hidden_dim: LSTM layer의 hidden 차원
        lstm_n_layer: LSTM layer의 layer 수
        fc_hidden_list: FC layer의 hidden 차원 리스트 ([]일 경우, 1차원으로 요약하는 layer 하나만 적용)
        dropout_p: dropout 확률 (0~1, 0이면 dropout을 사용하지 않음)
        batch_norm: batch normalization 사용 여부
        '''
        super(CRNN, self).__init__()

        self.cnn = CNN(input_size=input_size, seq_len=seq_len, output_size=output_size, 
                       cnn_hidden_list=cnn_hidden_list, fc_hidden_list=[], 
                       dropout_p=dropout_p, batch_norm=batch_norm)

        lstm_input_size = cnn_hidden_list[-1]
        self.lstm = LSTM(input_size=lstm_input_size, seq_len=seq_len, output_size=output_size, 
                         lstm_hidden_dim=lstm_hidden_dim, lstm_n_layer=lstm_n_layer, 
                         bidirectional=bidirectional, fc_hidden_list=[], 
                         dropout_p=dropout_p, batch_norm=batch_norm)

        self.fc_layers = nn.Sequential()
        fc_hidden_list = [lstm_hidden_dim * (2 if bidirectional else 1)] + fc_hidden_list + [output_size]
        for i, (inp, oup) in enumerate(zip(fc_hidden_list[:-1], fc_hidden_list[1:])):
            self.fc_layers.add_module(f'fc{i}', nn.Linear(inp, oup))
            if i < len(fc_hidden_list) - 2:
                self.fc_layers.add_module(f'fc{i}_relu', nn.ReLU())
                if dropout_p:
                    self.fc_layers.add_module(f'fc{i}_dropout', nn.Dropout(dropout_p))

        self.init_weights()

    def forward(self, x):
        # CNN 레이어 적용
        x = self.cnn.cnn_layers(x)
        
        # LSTM 레이어 적용
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        x, _ = self.lstm.lstm_layers(x)
        
        # Flatten
        x = x[:, -1, :]  # (batch_size, lstm_hidden_dim)
        
        # FC 레이어 적용
        x = self.fc_layers(x)

        return x  # (batch_size, output_size)

    def init_weights(self):
        self.cnn.init_weights()
        self.lstm.init_weights()

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters:
            input_dim (int): Number of channels of input tensor.
            hidden_dim (int): Number of channels of hidden state.
            kernel_size (int or tuple): Size of the convolutional kernel.
            bias (bool): Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ResBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(BaseModel):
    def __init__(self, input_size, seq_len, output_size, num_blocks):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential()
        for _ in range(num_blocks):
            self.layers.add_module('block', ResBlock(64))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




if __name__ == '__main__':
    import torch
    
    batch_size = 7
    input_size = 6
    seq_len = 20

    model5 = CNN(input_size=input_size,
                 seq_len=seq_len,
                 cnn_hidden_list=[4,8],
                 fc_hidden_list=[128, 64],
                 dropout_p=0.2, 
                 batch_norm=True)

    model9 = LSTM(input_size=input_size,
                 seq_len=seq_len,
                 lstm_hidden_dim=16,
                 lstm_n_layer=2,
                 bidirectional=True,
                 fc_hidden_list=[128, 64],
                 dropout_p=0.2, 
                 batch_norm=True)
    
    model0 = CRNN(input_size=input_size,
                 seq_len=seq_len,
                 cnn_hidden_list=[4, 8],
                 lstm_hidden_dim=16,
                 lstm_n_layer=2,
                 bidirectional=True,
                 fc_hidden_list=[128, 64],
                 dropout_p=0.2, 
                 batch_norm=True)
    
    model = ResNet(input_size=input_size, 
                          seq_len=seq_len, 
                          output_size=1, 
                          num_blocks=3)
                 
    
    print(model)

    # (batch_size, input_size, seq_len) -> (batch_size, 1) 가 잘 되는지 확인
    x = torch.randn(batch_size, input_size, seq_len)
    y = model(x)
    print(f'\nx: {x.shape} => y: {y.shape}')