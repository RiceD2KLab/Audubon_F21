import torch  # version =  1.10.0 
from torch.utils.data import DataLoader
import torchvision.datasets as dsets 
import torchvision.transforms as transforms

batch_size = 100
# MNIST dataset.
train_dataset = dsets.MNIST(root='./pymnist', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./pymnist', train=False, transform=transforms.ToTensor(), download=True)
# load_data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

input_size = 784
hidden_size = 500
num_classes = 10

# LeNet5 Model
class Neural_net(nn.Module):
    def __init__(self, input_num, hidden_size, output_num):
        super(Neural_net, self).__init__()
        self.layers1 = nn.Linear(input_num, hidden_size)
        self.layers2 = nn.Linear(hidden_size, output_num)

    def forward(self, x):
        out = self.layers1(x)
        out = torch.relu(out)
        out = self.layers2(out)
        return out
net = Neural_net(input_size, hidden_size, num_classes)

# training
learning_rate = 1e-1
num_epoches = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
for epoch in range(num_epoches):
    print("current epoch = {}".format(epoch))
    for i, (images,labels) in enumerate(train_loader):
        images = torch.tensor(images.view(-1, 28*28))
        labels = torch.tensor(labels)

        outputs = net(images)
        loss = criterion(outputs, labels)  # calculate loss
        optimizer.zero_grad()  # clear net state before backward
        loss.backward()       
        optimizer.step()   # update parameters

        if i%100 == 0:
            print("current loss = %.5f" %loss.item())
print("finished training")

# prediction
total = 0
correct = 0

# Define Your Label(Categories Name)
Label = ['0','1','2','3','4','5','6','7','8','9']
confusion = ConfusionMatrix(num_classes=10, labels=Label)
for images, labels in test_loader:
    images = torch.tensor(images.view(-1, 28*28))
    labels = torch.tensor(labels)
    outputs = net(images)

    _,predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()

    # Update Confusion Matrix with Predicted Label and True Label
    confusion.update(predicts, labels)
print("Accuracy = %.2f" %(100*correct/total))

# Confusion Matrix Visualizatoin
confusion.plot()
confusion.summary()
