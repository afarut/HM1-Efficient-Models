"""FP32; MAC=2 FLOPs. Comparisons excluded. All functions broadcast."""
import numpy as np
PARAMETERS = 1040324
PARAMETER_BYTES = 4*PARAMETERS

def flops(image_size,batch):
    s,b=np.broadcast_arrays(np.asarray(image_size,dtype=float),np.asarray(batch,dtype=float))
    # Convolutions + GAP (n-1 adds and one division) + linear MACs + biases.
    return b*(17714*s*s+313700)

def memory(image_size,batch):
    s,b=np.broadcast_arrays(np.asarray(image_size,dtype=float),np.asarray(batch,dtype=float))
    # Retained input + first conv activation + pooled output + int64 indices.
    # Excludes opaque cuDNN workspace and allocator rounding; see derivations.
    return PARAMETER_BYTES+68*b*s*s

def layer_costs(image_size,batch):
    s,b=np.broadcast_arrays(np.asarray(image_size,dtype=float),np.asarray(batch,dtype=float))
    z=np.zeros_like(s); F=[]; Q=[]
    prev=3*s*s
    for i,(ci,co,k,d) in enumerate([(3,32,7,2),(32,64,5,4),(64,128,3,8),(128,256,1,8),(256,256,3,16),(256,512,1,16)]):
        out=co*(s/d)**2; w=ci*co*k*k
        F.append(2*b*out*ci*k*k); Q.append(4*(b*(prev+out)+w))
        F.append(z); Q.append(8*b*out) # separate in-place ReLU read/write
        prev=out
        if i==0:
            out=32*(s/4)**2
            F.append(z); Q.append(4*b*(prev+out)+8*b*out)
            prev=out
    F.append(b*prev); Q.append(4*b*(prev+512))
    F.append(b*(2*512*256+256)); Q.append(4*(b*(512+256)+512*256+256))
    F.append(z); Q.append(8*b*256)
    F.append(b*(2*256*100+100)); Q.append(4*(b*(256+100)+256*100+100))
    return np.stack(F,axis=-1),np.stack(Q,axis=-1)

def bytes_moved(image_size,batch):
    return layer_costs(image_size,batch)[1].sum(axis=-1)

def latency(image_size,batch,theta):
    f,q=layer_costs(image_size,batch)
    return theta['launch_seconds']*f.shape[-1]+np.maximum(f/theta['flops_per_second'],q/theta['bytes_per_second']).sum(axis=-1)

def energy(image_size,batch,theta_energy):
    return (theta_energy['joules_per_flop']*flops(image_size,batch)
            +theta_energy['joules_per_byte']*bytes_moved(image_size,batch)
            +theta_energy['baseline_watts']*latency(image_size,batch,theta_energy['latency']))
