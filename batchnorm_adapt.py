import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils import *

seed_everything(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model1 = BasicModel().to(device)
model2 = resnet18(weights=True).to(device)

# Copying resnet running mean and var, just to observe if anything has changed after AdaBN
# Otherwise, it will just show mean of 0s and 1s and it will be confusing
model1.bn1.running_mean = model2.bn1.running_mean
model1.bn1.running_var = model2.bn1.running_var
model1.bn2.running_mean = model2.layer1[0].bn1.running_mean
model1.bn2.running_var = model2.layer1[0].bn1.running_var

# Creating a fake dataset and a test dataloader
dataset = ImageGeneratorDataset(2)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = model1

# Printing the model to find out the layer name variables
# print(model)

# Here comes the deal
model.train()
bn_stats = compute_bn_stats(model, dataloader)

# Access layer-wise mean and variance statistics and print
for layer_name, stats in bn_stats.items():
    print(f"Layer: {layer_name}")
    print(f"Mean: {stats['mean']}")
    print(f"Variance: {stats['var']}")

# Finally done
replace_bn_stats(model, bn_stats)

# Now all I need to do is to continue to run the test set through the model and get outputs