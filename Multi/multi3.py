import torch
import torch.nn as nn
from dataset import fusionDataset
from torch.utils.data import Dataset, DataLoader
# 비디오 스트림 처리를 위한 CNN 모델
class AudioStream(nn.Module):
    def __init__(self):
        super(VideoStream, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# 오디오 스트림 처리를 위한 CNN 모델
class VideoStream(nn.Module):
    def __init__(self):
        super(AudioStream, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 128)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class TwoStreamNetwork(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamNetwork, self).__init__()
        self.video_stream = VideoStream()
        self.audio_stream = AudioStream()
        # 두 스트림에서 나온 특징을 연결하는 레이어를 정의합니다.
        self.concat_layer = nn.Linear(256, 128)
        # 최종 분류기를 정의합니다.
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, audio_input, video_input):
        # 비디오 스트림과 오디오 스트림을 각각 처리합니다.
        video_features = self.video_stream(video_input)
        audio_features = self.audio_stream(audio_input)
        
        # 비디오 및 오디오 특징을 연결합니다.
        combined_features = torch.cat((video_features, audio_features), dim=1)
        
        # 연결된 특징을 처리하고 최종 분류를 수행합니다.
        output = self.concat_layer(combined_features)
        output = self.classifier(output)
        
        return output

feature_path = '../Video/data/video'
audio_path = '../Audio/data/wav'
train_dataset = fusionDataset(feature_path,audio_path)
test_dataset = fusionDataset(feature_path,audio_path,mode='test')

batch_size = 32
train_dataloader = fusionDataset(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = fusionDataset(test_dataset, batch_size=batch_size, shuffle=False)
model = TwoStreamNetwork(7)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

total_step = len(train_dataloader)
num_epochs = 150

for epoch in range(num_epochs):
    for i, (audio, video, label) in enumerate(train_dataloader):
        outputs = model(audio, video)
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}')

# 모델 평가
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataset:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')