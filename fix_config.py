with open('config/config.py', 'r') as f:
    content = f.read()

content = content.replace("accelerator: str = 'gpu'", "accelerator: str = 'cpu'")
content = content.replace("precision: str = '16-mixed'", "precision: str = '32'")
content = content.replace("persistent_workers: bool = True", "persistent_workers: bool = False")
content = content.replace("num_workers: int = 4  # Reduced workers to save memory", "num_workers: int = 0  # 0 for CPU training")

with open('config/config.py', 'w') as f:
    f.write(content)

print('Config updated for CPU training!')
print('Changes made:')
print('  accelerator: gpu -> cpu')
print('  precision: 16-mixed -> 32')
print('  persistent_workers: True -> False')
print('  num_workers: 4 -> 0')