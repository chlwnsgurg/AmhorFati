from Crypto.Util.number import *
from tqdm import *

# STEP 1
# Making Elliptic Curve with complex elements
p = 2^216 * 3^137 - 1
Fp = GF(p)
K.<i> = PolynomialRing(Fp)
F = Fp.extension(i^2 + 1, "i")

E = EllipticCurve(F, [0, 6, 0, 1, 0])
# print(f"j-invariant of E : {E.j_invariant()}")

# STEP 2
# Data Parsing with 6.pdf (not data.txt)
# It satisfies the condition ord(Ps, Qs) = 2^216 and ord(Pr, Qr) = 2^137
Ps_x_re = 0x00003CCFC5E1F050030363E6920A0F7A4C6C71E63DE63A0E6475AF621995705F7C84500CB2BB61E950E19EAB8661D25C4A50ED279646CB48
Ps_x_im = 0x0001AD1C1CAE7840EDDA6D8A924520F60E573D3B9DFAC6D189941CB22326D284A8816CC4249410FE80D68047D823C97D705246F869E3EA50

Ps_y_re = 0x0001AB066B84949582E3F66688452B9255E72A017C45B148D719D9A63CDB7BE6F48C812E33B68161D5AB3A0A36906F04A6A6957E6F4FB2E0
Ps_y_im = 0x0000FD87F67EA576CE97FF65BF9F4F7688C4C752DCE9F8BD2B36AD66E04249AAF8337C01E6E4E1A844267BA1A1887B433729E1DD90C7DD2F

Qs_x_re = 0x0000C7461738340EFCF09CE388F666EB38F7F3AFD42DC0B664D9F461F31AA2EDC6B4AB71BD42F4D7C058E13F64B237EF7DDD2ABC0DEB0C6C
Qs_x_im = 0x000025DE37157F50D75D320DD0682AB4A67E471586FBC2D31AA32E6957FA2B2614C4CD40A1E27283EAAF4272AE517847197432E2D61C85F5

Qs_y_re = 0x0001D407B70B01E4AEE172EDF491F4EF32144F03F5E054CEF9FDE5A35EFA3642A11817905ED0D4F193F31124264924A5F64EFE14B6EC97E5
Qs_y_im = 0x0000E7DEC8C32F50A4E735A839DCDB89FE0763A184C525F7B7D0EBC0E84E9D83E9AC53A572A25D19E1464B509D97272AE761657B4765B3D6

Ps_x = Ps_x_re + Ps_x_im * F.gen()
Ps_y = Ps_y_re + Ps_y_im * F.gen()
Ps = E(Ps_x, Ps_y)

Qs_x = Qs_x_re + Qs_x_im * F.gen()
Qs_y = Qs_y_re + Qs_y_im * F.gen()
Qs = E(Qs_x, Qs_y)

Pr_x_re = 0x00008664865EA7D816F03B31E223C26D406A2C6CD0C3D667466056AAE85895EC37368BFC009DFAFCB3D97E639F65E9E45F46573B0637B7A9
Pr_y_re = 0x00006AE515593E73976091978DFBD70BDA0DD6BCAEEBFDD4FB1E748DDD9ED3FDCF679726C67A3B2CC12B39805B32B612E058A4280764443B

Qr_x_re = 0x00012E84D7652558E694BF84C1FBDAAF99B83B4266C32EC65B10457BCAF94C63EB063681E8B1E7398C0B241C19B9665FDB9E1406DA3D3846
Qr_y_im = 0x0000EBAAA6C731271673BEECE467FD5ED9CC29AB564BDED7BDEAA86DD1E0FDDF399EDCC9B49C829EF53C7D7A35C3A0745D73C424FB4A5FD2

Pr_x = Pr_x_re
Pr_y = Pr_y_re
Pr = E(Pr_x, Pr_y)

Qr_x = Qr_x_re
Qr_y = Qr_y_im * F.gen()
Qr = E(Qr_x, Qr_y)

assert Ps.order() == 2^216 and Qs.order() == 2^216
assert Pr.order() == 3^137 and Qr.order() == 3^137

# STEP 3 - NOT FINISHED!!
Es_A_re = 0x0000BC39A8C22AFDCAC43EFDD3AB05B45AF0A795D823CD1EC0931D386BFDE2D477DFFFBF2C8463460DE0586510E99F24323AB8E54BD0026B
Es_A_im = 0x0000045E901E3BAA12BA1A2D0A37634DEF74A6791039D723962496EB9C4C368FD50BD06BC7D7EF0B2582ADF73577537BDAA9A064C9AB0DA5
Es_A = Es_A_re + Es_A_im * F.gen()

Es = EllipticCurve(F, [0, Es_A, 0, 1, 0])

# STEP 4
data = open("./data.txt", "r").read()
data = data.split("\n")
data = [int(sigs, 16) for sigs in data]
data = [data[i] + data[i+1] * F.gen() for i in range(0, len(data), 2)]

candidate = dict()

for point_x in data:
    try:
        Q = E.lift_x(point_x)
        Q = Q * 3^137
        if Q.order() == 2^216:
            inv = inverse(3^137, 2^216)
            Q = Q * inv
            if Q in candidate.keys():
                candidate[Q] += 1
            else:
                candidate[Q] = 1
    except:
        pass

sorted_candidate = sorted(candidate.items(), key=lambda x: x[1], reverse=True)
sorted_candidate = [candidates for candidates in sorted_candidate if candidates[1] > 10]
# number of element in sorted_candidate is 2!!

S_1 = E(sorted_candidate[0][0])
S_2 = E(sorted_candidate[1][0])
assert S_1.xy()[0] == S_2.xy()[0]

# Because x-coordinate is same, we don't need to consider about the y-coordinate
# When the x-coordinate is same, the doubled x-coordinate is same also
# So in each step, the kernel that we made is always same!!
# Therfore, the isogeny is same in each steps --> the full isogeny is same!!

def doubling_isogeny(E, S):
    assert S in E
    order = S.order()

    iteration = int(log(order, 2))
    for rep in trange(iteration):
        kernel = (order // 2^(rep+1)) * S
        alpha = kernel.xy()[0]
        x_coordinate = S.xy()[0]

        a_prime = 2 * (1 - 2 * alpha^2)

        try:
            x_coordinate = x_coordinate * (alpha * x_coordinate - 1) * inverse(x_coordinate - alpha, p)
        except:
            x_coordinate = 0

        E = EllipticCurve(F, [0, a_prime, 0, 1, 0])

        S = E.lift_x(x_coordinate)

    return E

calculated_Es = doubling_isogeny(E, S_1) # same result as S_2
assert calculated_Es == Es

try:
    res = S_1 - Ps
    ns = discrete_log(res, Qs, Qs.order(), operation="+")
    S = S_1
except:
    res = S_2 - Ps
    ns = discrete_log(res, Qs, Qs.order(), operation="+")
    S = S_2

print(f"[*] The Private Key ns : {ns}")
print(f"[*] The generator of the kernel S : {S}")