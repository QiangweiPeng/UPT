import torch
import numpy as np

def convert_mixed_array_to_2d(arr):
    """
    将 dtype=object 的 array (包含 tensor / numpy / nan) 转换为纯二维 numpy 数组。
    对于 nan 行，会用 np.nan 填充。
    """
    # 把每个元素都转成 numpy 或 nan
    processed = []
    for a in arr:
        if isinstance(a, torch.Tensor):
            processed.append(a.detach().cpu().numpy())
        elif isinstance(a, np.ndarray):
            processed.append(a)
        elif a is None or (isinstance(a, float) and np.isnan(a)):
            processed.append(np.nan)
        else:
            raise TypeError(f"Unsupported element type: {type(a)}")
    
    # 找出有效的非 nan 元素
    valid = [a for a in processed if isinstance(a, np.ndarray)]
    if not valid:
        raise ValueError("No valid tensor/array elements found.")
    
    # 检查维度一致性
    shapes = {a.shape for a in valid}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent tensor shapes found: {shapes}")
    
    d = valid[0].shape[0]  # 向量维度
    result = np.full((len(processed), d), np.nan, dtype=float)
    
    # 填充结果
    for i, a in enumerate(processed):
        if isinstance(a, np.ndarray):
            result[i] = a
    
    return result