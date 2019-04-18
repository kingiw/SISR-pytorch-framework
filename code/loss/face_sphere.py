import torch
from torch import nn

class Sphere20a(nn.Module):
    def __init__(self):
        super(Sphere20a, self).__init__()
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3, 64, 3, 2, 1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512 * 7 * 6, 512)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0), -1)
        x = self.fc5(x)

        return x

class Sphere20aFeatureExtractor(nn.Module):
    def __init__(self,
                 model_path,
                 use_input_norm=False,
                 n_GPUs=1
                ):
        super(Sphere20aFeatureExtractor, self).__init__()
        model = Sphere20a()

        self.device = torch.device('cuda') if n_GPUs > 0 else torch.device('cpu')
        model.to(self.device)

        from collections import OrderedDict
        state_dict_remove_fc = OrderedDict()
        for x, y in torch.load(model_path).items():
            if x[:3] != 'fc6':
                state_dict_remove_fc[x] = y
        model.load_state_dict(state_dict_remove_fc)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        if n_GPUs == 1:
            model = model.cuda()
        elif n_GPUs > 1:
            model = nn.DataParallel(model)
        # No need to BP to variable
        self.features = model
        for k, v in self.features.named_parameters():
            v.requires_grad = False
        self.features.eval()

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features.forward(x)
        return output

class FaceSphereLoss(nn.Module):
    def __init__(self, n_GPUs):
        super(FaceSphereLoss, self).__init__()
        self.features = Sphere20aFeatureExtractor(
            model_path = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/pretrained/sphere20a_20171020.pth",
            n_GPUs = n_GPUs
        )
        self.criterion = nn.CosineEmbeddingLoss()
        self.device = torch.device('cpu') if n_GPUs == 0 else torch.device('cuda')
    
    def forward(self, x, y):
        """
        x, y should be on the same device as the Feature Extractor
        """
        x_fea, y_fea = self.features(x), self.features(y)
        batch = x_fea.shape[0]
        flags = torch.ones(batch).to(self.device)
        # print(x_fea.is_cuda, y_fea.is_cuda, flags.is_cuda)
        return self.criterion(x_fea, y_fea, flags)
        