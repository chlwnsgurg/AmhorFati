#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <time.h>
// round of block cipher
#define NUM_ROUND 80
// size of plaintext and key size
#define BLOCK_SIZE 512
#define P_K_SIZE 2
#define SESSION_KEY_SIZE NUM_ROUND
// basic operation
#define ROR(x, r) ((x >> r) | (x << (32 - r)))
#define ROL(x, r) ((x << r) | (x >> (32 - r)))
// example: AVX2 functions; freely remove this code and write what you want in here!
#define INLINE inline __attribute__((always_inline))
#define LOAD(x) _mm256_loadu_si256((__m256i *)x)
#define STORE(x, y) _mm256_storeu_si256((__m256i *)x, y)
#define XOR(x, y) _mm256_xor_si256(x, y)
#define OR(x, y) _mm256_or_si256(x, y)
#define AND(x, y) _mm256_and_si256(x, y)
#define SHUFFLE8(x, y) _mm256_shuffle_epi8(x, y)
#define ADD(x, y) _mm256_add_epi32(x, y)
#define SHIFT_L(x, r) _mm256_slli_epi32(x, r)
#define SHIFT_R(x, r) _mm256_srli_epi32(x, r)
int64_t cpucycles(void)
{
    unsigned int hi, lo;
    __asm__ __volatile__("rdtsc\n\t"
                         : "=a"(lo), "=d"(hi));
    return ((int64_t)lo) | (((int64_t)hi) << 32);
}
// 64-bit data
// 64-bit key
// 32-bit x 22 rounds session key
void new_key_gen(uint32_t *master_key, uint32_t *session_key)
{
    uint32_t i = 0;
    uint32_t k1, k2, tmp;
    k1 = master_key[0];
    k2 = master_key[1];
    for (i = 0; i < NUM_ROUND; i++)
    {
        k1 = ROR(k1, 8);
        k1 = k1 + k2;
        k1 = k1 ^ i;
        k2 = ROL(k2, 3);
        k2 = k1 ^ k2;
        session_key[i] = k2;
    }
}
void new_block_cipher(uint32_t *input, uint32_t *session_key, uint32_t *output)
{
    uint32_t i = 0;
    uint32_t pt1, pt2, tmp1, tmp2;
    pt1 = input[0];
    pt2 = input[1];
    for (i = 0; i < NUM_ROUND; i++)
    {
        tmp1 = ROL(pt1, 1);
        tmp2 = ROL(pt1, 8);
        tmp2 = tmp1 & tmp2;
        tmp1 = ROL(pt1, 2);
        tmp2 = tmp1 ^ tmp2;
        pt2 = pt2 ^ tmp2;
        pt2 = pt2 ^ session_key[i];
        tmp1 = pt1;
        pt1 = pt2;
        pt2 = tmp1;
    }
    output[0] = pt1;
    output[1] = pt2;
}
void new_key_gen_enc_AVX2(uint32_t *master_key, uint32_t *input_AVX, uint32_t *output_AVX)
{
    __m256i k1, k2, p1, p2, tmp, ii, add1;
    int i;
    uint32_t one[8]={1,1,1,1,1,1,1,1}, iarr[8]={-1,-1,-1,-1,-1,-1,-1,-1};
    ii = LOAD(iarr);
    add1 = LOAD(one);

    k1 = LOAD(master_key);
    k2 = LOAD(&master_key[8]);
    p1 = LOAD(input_AVX);
    p2 = LOAD(&input_AVX[8]);
    
    for (i = 0; i < NUM_ROUND; i++)
    {
        ii = ADD(ii,add1);

        k1 = XOR(ADD(OR(SHIFT_R(k1, 8), SHIFT_L(k1,24)),k2),ii);
        k2 = XOR(k1,OR(SHIFT_L(k2, 3), SHIFT_R(k2,29)));

        tmp = XOR(OR(SHIFT_L(p1, 2), SHIFT_R(p1,30)), AND(OR(SHIFT_L(p1, 1), SHIFT_R(p1,31)), OR(SHIFT_L(p1, 8), SHIFT_R(p1,24))));
        p2 = XOR(XOR(p2, tmp), k2);
        tmp = p1;
        p1 = p2;
        p2 = tmp;
    }
    STORE(output_AVX, p1);
    STORE(&output_AVX[8], p2);
}
int main()
{
    long long int kcycles, ecycles, dcycles;
    long long int cycles1, cycles2;
    int32_t i, j;
    // C implementation
    uint32_t input_C[BLOCK_SIZE][P_K_SIZE] = {0,};
    uint32_t key_C[BLOCK_SIZE][P_K_SIZE] = {0,};
    uint32_t session_key_C[BLOCK_SIZE][SESSION_KEY_SIZE] = {0,};
    uint32_t output_C[BLOCK_SIZE][P_K_SIZE] = {0,};
    // AVX implementation
    uint32_t input_AVX[BLOCK_SIZE][P_K_SIZE] = {0,};
    uint32_t key_AVX[BLOCK_SIZE][P_K_SIZE] = {0,};
    uint32_t session_key_AVX[BLOCK_SIZE][SESSION_KEY_SIZE] = {0,};
    uint32_t output_AVX[BLOCK_SIZE][P_K_SIZE] = {0,};
    // random generation for plaintext and key.
    srand(0);
    for (i = 0; i < BLOCK_SIZE; i++)
    {
        for (j = 0; j < P_K_SIZE; j++)
        {
            input_AVX[i][j] = input_C[i][j] = rand();
            key_AVX[i][j] = key_C[i][j] = rand();
        }
    }
    // execution of C implementation
    kcycles = 0;
    cycles1 = cpucycles();
    for (i = 0; i < BLOCK_SIZE; i++)
    {
        new_key_gen(key_C[i], session_key_C[i]);
        new_block_cipher(input_C[i], session_key_C[i], output_C[i]);
    }
    cycles2 = cpucycles();
    kcycles = cycles2 - cycles1;
    printf("C   implementation runs in ................. %8lld cycles", kcycles / BLOCK_SIZE);
    printf("\n");
    // KAT and Benchmark test of AVX implementation
    kcycles = 0;
    cycles1 = cpucycles();

    uint32_t rearrk[16]={0,}, rearrin[16]={0,}, rearrout[16]={0,};
    for (i = 0; i < BLOCK_SIZE; i+=8)
    {
        rearrk[0]=key_AVX[i][0]; rearrk[8]=key_AVX[i][1];
        rearrk[1]=key_AVX[i+1][0]; rearrk[9]=key_AVX[i+1][1];
        rearrk[2]=key_AVX[i+2][0]; rearrk[10]=key_AVX[i+2][1];
        rearrk[3]=key_AVX[i+3][0]; rearrk[11]=key_AVX[i+3][1];
        rearrk[4]=key_AVX[i+4][0]; rearrk[12]=key_AVX[i+4][1];
        rearrk[5]=key_AVX[i+5][0]; rearrk[13]=key_AVX[i+5][1];
        rearrk[6]=key_AVX[i+6][0]; rearrk[14]=key_AVX[i+6][1];
        rearrk[7]=key_AVX[i+7][0]; rearrk[15]=key_AVX[i+7][1];

        rearrin[0]=input_AVX[i][0]; rearrin[8]=input_AVX[i][1];
        rearrin[1]=input_AVX[i+1][0]; rearrin[9]=input_AVX[i+1][1];
        rearrin[2]=input_AVX[i+2][0]; rearrin[10]=input_AVX[i+2][1];
        rearrin[3]=input_AVX[i+3][0]; rearrin[11]=input_AVX[i+3][1];
        rearrin[4]=input_AVX[i+4][0]; rearrin[12]=input_AVX[i+4][1];
        rearrin[5]=input_AVX[i+5][0]; rearrin[13]=input_AVX[i+5][1];
        rearrin[6]=input_AVX[i+6][0]; rearrin[14]=input_AVX[i+6][1];
        rearrin[7]=input_AVX[i+7][0]; rearrin[15]=input_AVX[i+7][1];

        new_key_gen_enc_AVX2(rearrk,rearrin,rearrout);

        output_AVX[i][0] = rearrout[0]; output_AVX[i][1] = rearrout[8];
        output_AVX[i+1][0] = rearrout[1]; output_AVX[i+1][1] = rearrout[9];
        output_AVX[i+2][0] = rearrout[2]; output_AVX[i+2][1] = rearrout[10];
        output_AVX[i+3][0] = rearrout[3]; output_AVX[i+3][1] = rearrout[11];
        output_AVX[i+4][0] = rearrout[4]; output_AVX[i+4][1] = rearrout[12];
        output_AVX[i+5][0] = rearrout[5]; output_AVX[i+5][1] = rearrout[13];
        output_AVX[i+6][0] = rearrout[6]; output_AVX[i+6][1] = rearrout[14];
        output_AVX[i+7][0] = rearrout[7]; output_AVX[i+7][1] = rearrout[15];
    }
    ///////////////////////////////////////////////////////////////////////////////////////////
    for (i = 0; i < BLOCK_SIZE; i++)
    {
        for (j = 0; j < P_K_SIZE; j++)
        {
            if (output_C[i][j] != output_AVX[i][j])
            {
                printf("Test failed!!!\n");
                return 0;
            }
        }
    }
    cycles2 = cpucycles();
    kcycles = cycles2 - cycles1;
    printf("AVX implementation runs in ................. %8lld cycles", kcycles / BLOCK_SIZE);
    printf("\n");

    return 0;
}