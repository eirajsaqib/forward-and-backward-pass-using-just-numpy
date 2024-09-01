import numpy as np
from tqdm import tqdm
inputt=np.random.rand(5,5)
kernel=np.random.rand(3,3)
kernel2=np.random.rand(2,2)


"""
inputt=np.ones([5,5])
kernel=np.ones([3,3])
kernel2=np.ones([2,2])
"""
final=np.ones([2,2])


def convolution(inputt,kernel):
  layer_out=np.zeros([inputt.shape[0]-kernel.shape[0]+1,inputt.shape[1]-kernel.shape[1]+1])
  for i in range(layer_out.shape[0]):
    for j in range(layer_out.shape[1]):
      layer_out[i,j]=np.sum(inputt[i:i+kernel.shape[0],j:j+kernel.shape[1]]*kernel)
  return layer_out

def relu(conv):
  return np.maximum(conv,0)

def loss(output,target):
  return np.mean((output-target)**2)

def forward(inputt,kernel,kernel2):
  co1=convolution(inputt,kernel)
  act_out=relu(co1)
  co2=convolution(act_out,kernel2)
  act_out2=relu(co2)
  return co1, act_out, co2, act_out2



def backward_pass(inputt,kernel,kernel2,act_out,act_out2,output,final):
  d_loss_output=2*(output-final)/output.size

  d_act_out2=d_loss_output*(act_out2>0)
  

  #go through 2 convolution

  d_kernel2=np.zeros_like(kernel2)
  d_act_out=np.zeros_like(act_out)
  for i in range(d_kernel2.shape[0]):
    for j in range (d_kernel2.shape[1]):
      d_kernel2[i, j] = np.sum(act_out[i:i + d_act_out2.shape[0], j:j + d_act_out2.shape[1]] * d_act_out2)

      d_act_out[i:i + d_act_out2.shape[0], j:j + d_act_out2.shape[1]] += d_act_out2 * kernel2[i, j]
  # Backpropagate through the first ReLU
  d_act_out1 = d_act_out * (act_out > 0)
  
  #go through 1 convolution
  d_kernel1=np.zeros_like(kernel)
  d_input=np.zeros_like(inputt)
  for i in range (d_kernel1.shape[0]):
    for j in range (d_kernel1.shape[1]):
      d_kernel1[i,j]=np.sum(inputt[i:i+d_act_out1.shape[0],j:j+d_act_out1.shape[1]]*d_act_out1)
      d_input[i:i + d_act_out1.shape[0], j:j + d_act_out1.shape[1]] += d_act_out1 * kernel[i, j]
  
  return d_kernel1,d_kernel2


epochs=1000000
lr=0.001

with tqdm (range(epochs),desc="Training process") as pbar:
  for epochs in pbar:
    co1,act_out,co2,act_out2=forward(inputt,kernel,kernel2)
    output=act_out2
    current_loss=loss(output,final)
    if epochs%100==0:
      pbar.set_postfix({'Loss': current_loss})
      d_kernel1,d_kernel2=backward_pass(inputt,kernel,kernel2,act_out,act_out2,output,final)

  # Update kernels
      kernel -= lr * d_kernel1
      kernel2 -= lr * d_kernel2

# Final output and loss after training
print("Final output after training:\n", output)
final_loss = loss(output, final)
print("Final Loss: ", final_loss)
print("Updated Kernel 1:\n", kernel)
print("Updated Kernel 2:\n", kernel2)
