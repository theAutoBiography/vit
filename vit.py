# Import libraries
import torch
import torchvision
import torch.utils.data as dataloader
import torch.nn as nn

# Transformation for PIL to tensor format
transformation_operation = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Import dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformation_operation)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation_operation)

# Define variables
num_classes = 10 # Helps define the final number of nodes in the ViT
batch_size = 64
num_channels = 1
image_size = 28
patch_size = 7
num_patches = (image_size // patch_size) ** 2
embedding_dimensions = 64
attention_heads = 4
transformer_blocks = 4
mlp_hidden_nodes = 128
learning_rate = 0.001
epochs = 5

# Define dataset batches
train_loader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Part 1: Patch embedding
class PatchEmbedding(nn.Module):
  def __init__(self):
    super().__init__()
    self.patch_embed = nn.Conv2d(num_channels, embedding_dimensions, kernel_size=patch_size, stride=patch_size)
  
  def forward(self, x):
    # patch embedding 
    x = self.patch_embed(x)
    # flattening
    x = x.flatten(2)
    x = x.transpose(1,2) # Final dimensionality will be (64,16,64)

    return x

# Part 2: Transformer encoder
class TransformerEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(embedding_dimensions)
    self.layer_norm2 = nn.LayerNorm(embedding_dimensions)
    self.multihead_attention = nn.MultiheadAttention(embedding_dimensions, attention_heads, batch_first = True)
    self.mlp = nn.Sequential(
        nn.Linear(embedding_dimensions, mlp_hidden_nodes),
        nn.GELU(),
        nn.Linear(mlp_hidden_nodes, embedding_dimensions)
    )
  
  def forward(self, x):
    residual1 = x
    x = self.layer_norm1(x)
    x = self.multihead_attention(x,x,x)[0]
    x = residual1 + x

    residual2 = x
    x = self.layer_norm2(x)
    x = self.mlp(x)
    x = residual2 + x

    return x

# Part 3: MLP_head
class MLP_head(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_norm = nn.LayerNorm(embedding_dimensions)
    self.mlp_head = nn.Linear(embedding_dimensions, num_classes)

  def forward(self, x):
    x = self.layer_norm(x)
    x = self.mlp_head(x)
    return x

# Vision Transformer Class
class VisionTransformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.patch_embedding = PatchEmbedding()
    self.cls_token = nn.Parameter(torch.randn(1,num_channels,embedding_dimensions))
    self.position_embedding = nn.Parameter(torch.randn(1,1+num_patches,embedding_dimensions))
    self.transformer_blocks = nn.Sequential(*[TransformerEncoder() for _ in range(transformer_blocks)])
    self.mlp_head = MLP_head()

  def forward(self,x):
    x = self.patch_embedding(x)
    B = x.size(0)
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim = 1)
    x = x + self.position_embedding
    x = self.transformer_blocks(x)
    x = x[:, 0]
    x = self.mlp_head(x)
    return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

from torch.nn.modules import loss
# Training loop

for epoch in range(epochs):
  model.train()
  total_loss = 0
  correct_epoch = 0
  total_epoch = 0
  print(f"\nEpoch {epoch + 1}")
  for batch_idx, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    accuracy = 100.0 * correct / labels.size(0)
    correct_epoch += correct
    total_epoch += labels.size(0)
    if batch_idx % 100 == 0:
      print(f"Batch {batch_idx + 1:3d}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%")
  
  epoch_acc = 100.0 * correct_epoch / total_epoch
  print(f"==> Epoch {epoch+1} Summary: Total Loss = {total_loss:.4f}, Accuracy = {epoch_acc:.2f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
  for images, labels in val_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    preds = outputs.argmax(dim=1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)

test_accuracy = 100.0 * correct / total
print(f"\n==> Val Accuracy: {test_accuracy:.2f}%")

import matplotlib.pyplot as plt

# show 10 predictions from the first test batch
model.eval()
images, labels = next(iter(val_loader))
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
  outputs = model(images)
  preds = outputs.argmax(dim=1)

# Move to CPU for plotting
images = images.cpu()
preds = preds.cpu()
labels = labels.cpu()

# Plot first 10 images
plt.figure(figsize=(12,4))
for i in range(10):
  plt.subplot(2,5,i+1)
  plt.imshow(images[i].squeeze(), cmap='gray')
  plt.title(f"Pred: {preds[i].item()}, Label: {labels[i].item()}")
  plt.axis('off')
plt.show()