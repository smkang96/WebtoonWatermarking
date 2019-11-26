import torch
import numpy as np
from skimage import color

class HammingCoder():
    '''encodes and decodes hamming code
    I/O with tensors'''
    
    def __init__(self, total_bits=31, device=0):
        '''initializes encode/decode matrix'''
        self._total_bits = total_bits
        parity_bits = np.log2(total_bits+1)
        assert int(parity_bits) - parity_bits == 0, 'total_bits should be 2^n-1 form'
        self._parity_bits = int(parity_bits)
        self._data_bits = self._total_bits - self._parity_bits
        P = self._gen_P()
        self._G = torch.Tensor(np.hstack([np.eye(self._data_bits), P.T]).T).to(device)
        self._H = torch.Tensor(np.hstack([P, np.eye(self._parity_bits)])).to(device)
    
    def _gen_P(self):
        '''internal function to make parity-checking matrix and error-bit lookup table'''
        pre_p = np.zeros((self._parity_bits, self._total_bits+1)) #(5, 32)
        rem_cols = [0]
        for r_idx in range(self._parity_bits):
            block_size = 2**r_idx
            for c_idx in range(self._total_bits+1):
                pre_p[r_idx][c_idx] = int(c_idx%(2*block_size) >= block_size)
            rem_cols.append(block_size)
        P = np.delete(pre_p, rem_cols, axis=1)
        # not so pretty solution to finding error bit
        self._decode_lookup = [e for e in range(self._total_bits)
                               if e not in rem_cols] + rem_cols
        self._decode_lookup = [(e-1)%self._total_bits+1 for e in self._decode_lookup]
        self._decode_lookup = {e:i for i, e in enumerate(self._decode_lookup)}
        return P
    
    def encode(self, msg):
        '''wants to be fed floatTensors'''
        enc = torch.mv(self._G, msg)
        enc = enc%2
        return enc
    
    def decode(self, enc):
        '''wants to be fed byteTensors; returns floatTensor'''
        enc = enc.float()
        syndrome = torch.mv(self._H, enc)
        syndrome = syndrome % 2
        if torch.sum(syndrome) != 0:
            # error - performs single-bit correction
            syndrome = syndrome
            pre_error_bit = sum([v.item()*(2**i) for i, v in enumerate(syndrome)])
            error_bit = self._decode_lookup[pre_error_bit]
            if error_bit < self._data_bits:
                # worst case where error in data; otherwise on parity bit, ignore
                enc[error_bit] = 1-enc[error_bit]
        return enc[:self._data_bits]

def LAB_L2_dist(img1, img2, mode='max'):
    '''takes two rgb images and calculates the LAB L2 distance.
    Uses the CIE76 formula for perceptual distance calculation, which might
    lead to underestimations for saturated colors. In the CIE76 formula, color
    distance of 2.3 corresponds to a "just noticiable difference". 
    
    INPUT
    img1, img2: assumed to be 3-dim tensors of the form (3, *, *) 
    normalized to (-1, 1) range for convenience; please take note!
    mode: {"mean", "max"}, optional, the type of l2 dist aggregation desired.'''
    def _normalize(img):
        return ((img + 1) * 127.5).cpu().numpy().astype('uint8').transpose(1, 2, 0)
    
    img1 = _normalize(img1)
    img2 = _normalize(img2)
    lab_img1 = color.rgb2lab(img1)
    lab_img2 = color.rgb2lab(img2)
    deltaE_76 = np.sqrt(np.sum((lab_img1 - lab_img2)**2, axis=2))
    if mode == 'mean':
        return np.mean(deltaE_76)
    elif mode == 'max':
        return np.max(deltaE_76)
    else:
        raise ValueError(f'unrecognized mode {mode}')
    