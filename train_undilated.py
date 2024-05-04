import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

class UndilatedASSP(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(UndilatedASSP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=1,
                              padding=0,
                              dilation=1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=1,
                              bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=1,
                              bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=1,
                              bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              dilation=1,
                              bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.convf = nn.Conv2d(in_channels=out_channels * 5, 
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              dilation=1,
                              bias=False)
        self.bnf = nn.BatchNorm2d(out_channels)
        self.adapool = nn.AdaptiveAvgPool2d(1)

        self.final_non_assp_conv = nn.Conv2d(in_channels=256, 
                                       out_channels=num_classes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        
        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)

        x5 = nn.functional.interpolate(x5, size = tuple(x4.shape[-2:]), mode='bilinear')
        x = torch.cat((x1,x2,x3,x4,x5), dim=1)
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)
        return x

class UndilatedDeepLabv3(nn.Module):
    def __init__(self, in_channels, out_channels_, num_classes):
        super(UndilatedDeepLabv3, self).__init__()
        self.resnet = models.resnet50()
        self.assp = UndilatedASSP(in_channels, 
                                out_channels_,
                                num_classes)
        self.final_conv = nn.Conv2d(in_channels=256, 
                                    out_channels=num_classes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        
    def forward(self,x):
        _, _, h, w = x.shape
        x = self.resnet(x)
        x = self.assp(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear')
        return x

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

def train():
    model = UndilatedDeepLabv3(in_channels=1024, out_channels_=256, num_classes=12)
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.optim.Adam(model.parameters(), lr=0.001)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")