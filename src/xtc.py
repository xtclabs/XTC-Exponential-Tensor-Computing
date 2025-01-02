import numpy as np

class XTC:
    def __init__(self, shape, data=None):
        self.shape = shape
        self.data = np.array(data) if data is not None else np.zeros(shape)

    def create_tensor(self, shape, data=None):
        """Create a tensor with specified shape and optional data"""
        self.shape = shape
        self.data = np.array(data) if data is not None else np.zeros(shape)
        return self
    
    def matmul(self, tensor_a, tensor_b):
        """Matrix multiplication of two tensors"""
        return np.matmul(tensor_a.data, tensor_b.data)

# Instantiate XTC class for operations
xtc = XTC(shape=(3, 3))
tensor1 = xtc.create_tensor((3, 3), data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor2 = xtc.create_tensor((3, 3), data=[[9, 8, 7], [6, 5, 4], [3, 2, 1]])

# Matrix multiplication example
result = xtc.matmul(tensor1, tensor2)
print("Matrix Multiplication Result:", result)
