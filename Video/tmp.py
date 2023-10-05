import os
import torch
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLPClassifier2(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128):
        super(MLPClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

import torch.nn as nn

class MLPClassifier3(nn.Module):
    def __init__(self):
        super(MLPClassifier3, self).__init__()

        # 3x96x96 입력을 1D 벡터로 평탄화
        input_size = 3 * 96 * 96

        # MLP 아키텍처 정의
        self.layers = nn.Sequential(
            nn.Linear(input_size, 4096),   # 첫 번째 은닉층
            nn.ReLU(),
            nn.Linear(4096, 2048),   # 첫 번째 은닉층
            nn.ReLU(),
            nn.Linear(2048, 1024),   # 첫 번째 은닉층
            nn.ReLU(),
            nn.Linear(1024, 512),   # 첫 번째 은닉층
            nn.ReLU(),
            nn.Linear(512, 256),         # 두 번째 은닉층
            nn.ReLU(),
            nn.Linear(256, 128),         # 세 번째 은닉층
            nn.ReLU(),
            nn.Linear(128, 16)           # 출력층 (16개의 클래스)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 입력을 1D 벡터로 평탄화
        return self.layers(x)

def save_img(result, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(result['Epoch'], result['Loss'], label='Loss')
    plt.plot(result['Epoch'], result['Accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss and Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path,'loss_accuracy_plot.png'))
    print("Graph saved.")
tmp = []
class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data_list = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if int(file.split('-')[2]) ==1:
                    continue
                file_path = os.path.join(root, file)
                self.data_list.append(file_path)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        dash = self.data_list[idx]
        data = np.load(dash)
        label = int(dash.split('-')[2]) - 2
        return torch.tensor(data, dtype=torch.float32).squeeze().reshape(2800), label

# 입력 특성 크기, 클래스 개수, 은닉층 크기 설정
input_size = 2800  # [3, 400]의 입력 특성
num_classes = 7
hidden_size = 128

# MLP 모델 생성
# model = MLPClassifier2(input_size, num_classes, hidden_size)
# model = MLPClassifier2(input_size, num_classes, hidden_size)
model = MLPClassifier(input_size, num_classes, hidden_size)

batch_size = 32
dataset = CustomDataset(folder_path='./data/features/')  # 데이터셋 폴더 경로 설정
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# dataset = make_dataloader.Dataset96(folder_path='../../Aicity/9696shape/A1_feature/')  # 데이터셋 폴더 경로 설정
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()  # 분류 문제이므로 CrossEntropyLoss 사용
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []
accuracy_list = []
best_accuracy = 0.0

# # 모델에 입력 데이터 전달하여 예측 수행
num_epochs = 30
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_data in dataloader:
        optimizer.zero_grad()
        
        input_data = batch_data[0]
        labels = batch_data[1]
        
        # Forward 패스
        outputs = model(input_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_samples
    
    epoch_list.append(epoch)
    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)
    
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_model_state = model.state_dict()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        

save_path = f"./checkpoint/classifier/{datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')}/"
os.makedirs(save_path, exist_ok=True)
if best_model_state is not None:
    torch.save(best_model_state, os.path.join(save_path,'best_model.pth'))
    
results = pd.DataFrame({'Epoch': epoch_list, 'Loss': loss_list, 'Accuracy': accuracy_list})
save_img(results, save_path)

# model.load_state_dict(torch.load('./checkpoint/classifier/2023_09_12 04_30_24/best_model.pth'))
# model.eval()

# correct = 0
# incoreect = 0

# # path = '../../Aicity/400shape/A1_feature_400_test/user_id_84935/'
# # lists = sorted(os.listdir(path),key= lambda x: int(x.split('_')[-2]))
# result = []
# # with torch.no_grad():
# #     for file_name in lists:
        
# #         output_path = os.path.join(path, file_name)
# #         if file_name.split('_')[0] == 'Dashboard':
# #             data = []
            
# #             rear = file_name.replace('Dashboard', 'Rear_view')
# #             right = file_name.replace('Dashboard', 'Right_side_window')
            
# #             dash = np.load(os.path.join(path, file_name))
# #             rear = np.load(os.path.join(path, rear))
# #             right = np.load(os.path.join(path, right))
            
# #             data.extend([dash, rear, right])
# #             concatenated_data = np.concatenate(data, axis=0)

# #             input_data = torch.tensor(concatenated_data, dtype=torch.float32).squeeze().reshape(1200)
# #             output = model(input_data)
# #             value, indices = torch.max(output, dim=0)
# #             # if value  < 15:
# #             #     output = 0
# #             # else:
# #             #     output = int(indices)
# #             output = int(indices)
# #             result.append(str(output))
# # text = " ".join(result)

# # with open("result/26233.txt",'w') as f:
# #     f.write(text)

# # with torch.no_grad():
# #     for batch_data in dataloader:
# #         input_data = batch_data[0]
# #         labels = batch_data[1]
        
# #         outputs = model(input_data)
# #         outputs = torch.argmax(outputs, dim=1)
        
# #         for a,b in zip(outputs, labels):
# #             if a == b:
# #                 correct += 1
# #             else:
# #                 incoreect +=1

# # print(correct / (correct + incoreect))

# path = '../../Aicity/400shape/A1_feature_400_test/'
# lists = os.listdir(path)

# with torch.no_grad():
#     for outputs_path in lists:
#         correct = 0
#         incoreect = 0
#         correct_num = 0.0
#         result = []
#         output_path = os.path.join(path, outputs_path)
#         for name in os.listdir(output_path):
#             if name.split('_')[0] == 'Dashboard':
#                 data = []
                
#             #     rear = name.replace('Dashboard', 'Rear_view')
#             #     right = name.replace('Dashboard', 'Right_side_window')
                
#             #     dash = np.load(os.path.join(output_path, name))
#             #     rear = np.load(os.path.join(output_path, rear))
#             #     right = np.load(os.path.join(output_path, right))
                
#             #     data.extend([dash, rear, right])
#             #     concatenated_data = np.concatenate(data, axis=0)
#             #     input_data = torch.tensor(concatenated_data, dtype=torch.float32).squeeze().reshape(1200)
#             #     outputs = int(torch.argmax(model(input_data)))
                
#             #     result.append(outputs)
#         #         data = []
#                 rear = name.replace('Dashboard', 'Rear_view')
#                 right = name.replace('Dashboard', 'Right_side_window')
#                 label = int(name.split('_')[-2])
                
#                 dash = np.load(os.path.join(output_path, name))
#                 rear = np.load(os.path.join(output_path, rear))
#                 right = np.load(os.path.join(output_path, right))
                
#                 data.extend([dash, rear, right])
#                 concatenated_data = np.concatenate(data, axis=0)
#                 input_data = torch.tensor(concatenated_data, dtype=torch.float32).squeeze().reshape(1200)
#                 outputs = model(input_data)
#                 value, indices = torch.max(outputs, dim=0)
                
#                 if indices == label:
#                     correct +=1
#                     correct_num += value
#                 else:
#                     incoreect +=1
#         print(f"{outputs_path}: {correct / (correct + incoreect)}")
#         print(correct_num / correct)
#         # print(result)