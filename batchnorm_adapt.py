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

# FIXED: Creating a more realistic dataset with more samples
# Changed from 2 samples to 50 for better statistics
dataset = ImageGeneratorDataset(50)  # FIXED: Increased sample count
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # FIXED: Increased batch size

model = model1

# Printing the model to find out the layer name variables
print("Model architecture:")
print(model)

print("\nBatchNorm layers in the model:")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        print(f"  {name}: {module.num_features} features")

# Store original statistics for comparison
print("\nOriginal BatchNorm statistics:")
original_stats = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        original_stats[name] = {
            'mean': module.running_mean.clone(),
            'var': module.running_var.clone()
        }
        print(f"{name} - Mean: {module.running_mean[:5]}")
        print(f"{name} - Var:  {module.running_var[:5]}")

# CRITICAL FIX: Here comes the deal - model.eval() instead of model.train()!
print(f"\n{'=' * 60}")
print("APPLYING ADAPTIVE BATCH NORMALIZATION (AdaBN)")
print(f"{'=' * 60}")
model.eval()  # FIXED: Changed from model.train() to model.eval()

bn_stats = compute_bn_stats(model, dataloader)

# Access layer-wise mean and variance statistics and print
print(f"\nComputed target domain statistics:")
for layer_name, stats in bn_stats.items():
    print(f"\nLayer: {layer_name}")
    print(f"  Mean shape: {stats['mean'].shape}")
    print(f"  Mean values (first 5): {stats['mean'][:5]}")
    print(f"  Variance shape: {stats['var'].shape}")
    print(f"  Variance values (first 5): {stats['var'][:5]}")

    # Verify the shapes are correct (should be (64,) not scalar)
    assert len(stats['mean'].shape) == 1, f"Mean should be 1D tensor, got {stats['mean'].shape}"
    assert len(stats['var'].shape) == 1, f"Variance should be 1D tensor, got {stats['var'].shape}"
    print(f"  ✅ Shape verification passed!")

# Finally done
print(f"\nReplacing BatchNorm statistics...")
replace_bn_stats(model, bn_stats)

print(f"\nVerifying statistics were actually updated:")
# Compare before and after
for name, module in model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d) and name in original_stats:
        old_mean = original_stats[name]['mean']
        old_var = original_stats[name]['var']
        new_mean = module.running_mean
        new_var = module.running_var

        mean_diff = torch.abs(new_mean - old_mean).mean().item()
        var_diff = torch.abs(new_var - old_var).mean().item()

        print(f"\n{name}:")
        print(f"  Mean changed by: {mean_diff:.6f}")
        print(f"  Var changed by:  {var_diff:.6f}")
        print(f"  Updated: {'✅ YES' if mean_diff > 1e-6 else '❌ NO'}")

# Now all I need to do is to continue to run the test set through the model and get outputs
print(f"\nTesting the adapted model with new data:")
test_dataset = ImageGeneratorDataset(10)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        data = data.to(device)
        output = model(data)
        print(f"Batch {i + 1}: Output shape {output.shape}, Mean: {output.mean().item():.4f}")

print(f"\n{'=' * 60}")
print("AdaBN SUCCESSFULLY APPLIED!")
print(f"{'=' * 60}")