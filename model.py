import torch

# standart convolution layer with BN & Activation layers
class std_CONV_Act_BN(torch.nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size = 3, 
                    stride = 1, padding = 1, groups=1, activation = None):
        super(std_CONV_Act_BN, self).__init__()
        layers = [torch.nn.Conv2d(in_channels = inp_channels, out_channels = out_channels, 
                      kernel_size  = kernel_size, stride = stride, padding = padding, 
                      groups = groups,bias = False),]
        layers.append(torch.nn.BatchNorm2d(num_features = out_channels))
        if activation != None:
            layers.append(activation(inplace=True))
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        result = self.layers(x)
        return result 

# deep-wise separable convolution with BN & Activation layers
class dws_CONV_Act_BN(torch.nn.Module):
    def __init__(self, inp_channels, out_channels, stride):
        super(dws_CONV_Act_BN, self).__init__()
        block = std_CONV_Act_BN
        self.layers = torch.nn.Sequential(
            # deep-wise conv
            block(inp_channels = inp_channels, out_channels = inp_channels, 
                kernel_size = 3, stride = stride, padding = 1, groups = inp_channels,
                activation = torch.nn.ReLU),
            #point-wise conv
            block(inp_channels = inp_channels, out_channels = out_channels, 
                kernel_size  = 1, stride = 1, padding = 0, groups = 1, activation = torch.nn.ReLU)
        )
            
    def forward(self, x):
        result = self.layers(x)
        return result

class bottleneck(torch.nn.Module):
    def __init__(self, inp_channels, out_channels, stride, expansion):
        super(bottleneck, self).__init__()
        block = std_CONV_Act_BN
        self.active_res_conns = (inp_channels == out_channels)&(stride == 1) 
        self.layers = torch.nn.Sequential(
            # point-wise convolution
            block(inp_channels = inp_channels, out_channels = inp_channels*expansion, 
                kernel_size = 1, stride = 1, padding = 0, groups = 1, activation = torch.nn.ReLU6),
            # deep-wise convolution
            block(inp_channels = inp_channels*expansion, out_channels = inp_channels*expansion, 
                kernel_size = 3, stride = stride, padding = 1, groups = inp_channels, 
                                 activation = torch.nn.ReLU6),
            # linear point-wise convolution
            block(inp_channels = inp_channels*expansion, out_channels = out_channels, 
                kernel_size  = 1, stride = 1, padding = 0, groups = 1)
        )
        
    def forward(self,x):
        result = self.layers(x)
        if self.active_res_conns:
            result = x + result
        return result

class Hard_Sigmoid(torch.nn.Module):
    def __init__(self, inplace = True):
        super(Hard_Sigmoid, self).__init__()
        self.inplace = inplace
        
    def forward(self, x):
    # piece-wise linear hard  analog of torch.nn.Sigmoid layer:
    # https://arxiv.org/pdf/1905.02244.pdf pp 5.2.1
        return torch.nn.functional.relu6(x+3, self.inplace) / 6

class Hard_Swish(torch.nn.Module):
    def __init__(self, inplace = True):
        super(Hard_Swish, self).__init__()
        self.inplace = inplace
        
    def forward(self, x):
        return x * torch.nn.functional.relu6(x+3, self.inplace) / 6
    
class SE_module(torch.nn.Module):
    def __init__(self, num_features, decrease_ratio = 4):
        super(SE_module, self).__init__()
        self.Pooling_Layer = torch.nn.AdaptiveAvgPool2d(1)
        self.FC_Sequence = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features//4),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(num_features//4, num_features),
            # replace Sigmoid with custom Hard_Sigmoid
            Hard_Sigmoid()
            # torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        #(batch_size,channels,1,1)->(batch_size, channels)
        res = self.Pooling_Layer(x).reshape(batch_size, channels)       
        # back to 4-dimensional format (batch_size, channels, 1, 1)
        res = self.FC_Sequence(res).reshape(batch_size, channels, 1, 1)
        # expand for add
        res = x * res.expand_as(x)
        return res
        
class mobileV3_bottleneck(torch.nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, stride, expanded_channels, 
                 activation, SE_mode):
        super(mobileV3_bottleneck, self).__init__()
        
        self.active_res_conns = (inp_channels == out_channels)&(stride == 1)
        self.activation = torch.nn.ReLU6 if activation == 'HS' else Hard_Swish
        self.padding    = (kernel_size - 1) // 2
        
        self.block    = std_CONV_Act_BN
        self.block_SE = SE_module
        
        # point-wise convolution
        self.PW_before_SE = self.block(inp_channels = inp_channels, out_channels = expanded_channels, 
                kernel_size = 1, stride = 1, padding = 0, groups = 1, activation = torch.nn.ReLU6)
        
        # deep-wise convolution
        layers = [torch.nn.Conv2d(in_channels = expanded_channels, out_channels = expanded_channels, 
                      kernel_size  = kernel_size, stride = stride, padding = self.padding, 
                      groups = expanded_channels, bias = False),
                  torch.nn.BatchNorm2d(num_features = expanded_channels),]
        if SE_mode:
            layers.append(self.block_SE(expanded_channels))
        layers.append(self.activation(inplace=True))
        self.DWC_w_SE = torch.nn.Sequential(*layers)
        
        # linear point-wise convolution
        self.PW_after_SE = self.block(inp_channels = expanded_channels, out_channels = out_channels, 
                kernel_size  = 1, stride = 1, padding = 0, groups = 1)
        
    def forward(self,x):
        result = self.PW_before_SE(x)
        result = self.DWC_w_SE(result)
        result = self.PW_after_SE(result)
        if self.active_res_conns:
            result = result + x
        return result
    
class MobileNet(torch.nn.Module):
    def __init__(self, n_class=100, input_size=224, version = 'v1'):
        super(MobileNet, self).__init__()
        assert version in ['v1','v2','v3']
        block_std        = std_CONV_Act_BN
        block_dwc        = dws_CONV_Act_BN
        block_bottleneck = bottleneck
        block_mobile_v3  = mobileV3_bottleneck
        
        
        if version == 'v1':          
            # https://arxiv.org/pdf/1704.04861.pdf
            conv_settings = [
                [  32,  64,1],[  64, 128,2],[ 128, 128,1],
                [ 128, 256,2],[ 256, 256,1],[ 256, 512,2],
                [ 512, 512,1],[ 512,1024,2],[1024,1024,1],
            ]
            layers = [block_std(3, 32, 3, 2),]
            for i, (inp,out,stride) in enumerate(conv_settings): 
                if i != 6:
                    layers.append(block_dwc(inp,out,stride))
                else:
                    for j in range(5):
                        layers.append(block_dwc(inp,out,stride))
            
            self.convolutional = torch.nn.Sequential(*layers)
            out_channels = 1024
        
        elif version == 'v2':
            # https://arxiv.org/pdf/1801.04381.pdf
            conv_settings = [
                [1, 16,  1, 1],[6, 24, 2, 2],[6, 32 , 3, 2],[6, 64,  4, 2],
                [6, 96,  3, 1],[6,160, 3, 2],[6, 320, 1, 1],
            ]
            layers = [block_std(3, 32, 3, 2),]
            inp_channels = 32
            for (expansion, out_channels, repeat_num, stride) in conv_settings:
                for j in range(repeat_num):
                    _stride = stride if j == 0 else 1
                    layers.append(block_bottleneck(inp_channels, out_channels, _stride, expansion))
                    inp_channels = out_channels
                    
            layers.append(block_std(320, 1280, 3, 1))
            self.convolutional = torch.nn.Sequential(*layers)
            out_channels = 1280
            
        elif version == 'v3':
            # architecture - https://arxiv.org/pdf/1905.02244.pdf
            # SE-Module    - https://arxiv.org/pdf/1709.01507.pdf
            layers = []
            conv_settings = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],[3, 64,  24,  False, 'RE', 2],[3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],[5, 120, 40,  True,  'RE', 1],[5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],[3, 200, 80,  False, 'HS', 1],[3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],[3, 480, 112, True,  'HS', 1],[3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],[5, 960, 160, True,  'HS', 1],[5, 960, 160, True,  'HS', 1],
            ]
            inp_channels = 16
            layers = [block_std(3, inp_channels, 3, 2, activation = Hard_Swish),]
            
            for (kernel_size, expanded_channels, out_channels, SE, activation, stride) in conv_settings:
                layers.append(block_mobile_v3(inp_channels, out_channels, kernel_size, stride, 
                                              expanded_channels, activation, SE))
                inp_channels = out_channels
            self.convolutional = torch.nn.Sequential(*layers)
        else:
            raise NotImplementedError
            
        self.avg_pooling = torch.nn.AvgPool2d(kernel_size = 7)
        #self.classifier  = torch.nn.Linear(out_channels, n_class)
        self.classifier  = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(out_channels, n_class),
        )
            
    def forward(self, x):
        x = self.convolutional(x)
        x = self.avg_pooling(x)
        x = x.flatten(start_dim = 1) # x = x.view(-1,1024)
        x = self.classifier(x)
        return x     
    
    
if __name__ == '__main__':
    
    model_v1 = MobileNet(n_class=10, input_size=224, version = 'v1')
    model_v2 = MobileNet(n_class=10, input_size=224, version = 'v2')
    model_v3 = MobileNet(n_class=10, input_size=224, version = 'v3')

    x = torch.randn(1,3,224,224)

    rez1 = model_v1(x)
    rez2 = model_v2(x)
    rez3 = model_v3(x)

    rez1.argmax(),rez2.argmax(),rez3.argmax()