# 定义可视化函数
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tabpfn.encoders import ModelWithAttention

def visualize_attention_weights(attention_weights_list):
    for step, weights in enumerate(attention_weights_list):
        plt.figure(figsize=(10, 6))
        sns.heatmap(weights.detach().cpu().numpy(), cmap='viridis')
        plt.title(f'Attention Weights - Step {step + 1}')
        plt.xlabel('Feature Index')
        plt.ylabel('Sample Index')
        plt.show()

# 定义测试函数
def test_attention_encoder():
    input_dim = 128
    emsize = 64
    output_dim = 10

    # 实例化 ModelWithAttention
    attention_encoder = ModelWithAttention(
        input_dim=input_dim,
        emsize=emsize,
        output_dim=output_dim,
        n_steps=3,
        gamma=1.3,
        epsilon=1e-15,
    )

    # 生成测试输入
    x = torch.randn(32, input_dim) * 0.1  # 大部分特征较小
    x[:, 0] = 10  # 第一个特征值显著大

    # 前向传递
    output, M_loss, attention_weights_list = attention_encoder(x)
    
    # 打印结果
    print("Output shape:", output.shape)
    print("Output with all ones input:", output)
    print("M_loss:", M_loss)
    
    # 打印注意力权重
    for step, weights in enumerate(attention_weights_list):
        print(f"Step {step + 1} Attention Weights Shape: {weights.shape}")
        print(f"Step {step + 1} Attention Weights: {weights}")
    
    # 可视化注意力权重
    visualize_attention_weights(attention_weights_list)

# 调用测试函数
if __name__ == "__main__":
    test_attention_encoder()