# Model import
from keras.models import model_from_json

arch = open('arch_neural_distinguisher.json')
json_arch = arch.read()
nr6_speck_distinguisher = model_from_json(json_arch)
nr6_speck_distinguisher.load_weights('weights_nr6_speck.h5')
nr6_speck_distinguisher.compile(optimizer='adam',loss='mse',metrics=['acc'])

# Model test
import speck as sp
import numpy as np
import os
import math
from Crypto.Util.number import *

## logic should be INPUT : (C, C') + KEY --> OUTPUT : (Correct Key!!, Wrong Key!!)
## A model that predicts whether a ciphertext is encrypted with a given key

#print(sp.check_testvector())

## sp.make_train_data(10**2, 6) # just for debugging

delta = [0x0040, 0000]

def calc_prob(p1, p2, c1, c2, final_key):
    assert p1[0] ^ p2[0] == delta[0] and p1[1] ^ p2[1] == delta[1]

    rev_c1, rev_c2 = sp.dec_one_round(c1, final_key), sp.dec_one_round(c2, final_key)
    tmp = [None for _ in range(4)]

    tmp[0] = np.array([rev_c1[0]], dtype=np.uint16)[np.newaxis, :]
    tmp[1] = np.array([rev_c1[1]], dtype=np.uint16)[np.newaxis, :]
    tmp[2] = np.array([rev_c2[0]], dtype=np.uint16)[np.newaxis, :]
    tmp[3] = np.array([rev_c2[1]], dtype=np.uint16)[np.newaxis, :]

    X = sp.convert_to_binary(tmp)
    res = nr6_speck_distinguisher.predict(X)

    return res[0][0]

pair_num = 10

key = [bytes_to_long(os.urandom(2)) for _ in range(4)]
key = sp.expand_key(key, 7)

fk = key[-1]

def score_for_key(key, final_key):
    score = 0
    for n in range(pair_num):
        pt1 = [bytes_to_long(os.urandom(2)) for _ in range(2)]
        pt2 = [pt1[0] ^ delta[0], pt1[1] ^ delta[1]]

        ct1 = sp.encrypt(pt1, key)
        ct2 = sp.encrypt(pt2, key)

        Z = calc_prob(pt1, pt2, ct1, ct2, final_key)
        score += math.log2(Z/(1-Z))

    return score

print(score_for_key(key, fk))
print(score_for_key(key, bytes_to_long(os.urandom(2))))
