#include <tee_internal_api.h>
//#include <tee_internal_api_extensions.h>
//#include <tee_api_defines.h>

#include <TEEencrypt_ta.h>
#include <stdio.h>
#include <string.h>
#include "aes_core.h"
#include "aes_locl.h"
#include "aes.h"
#include "e_os2.h"
#include "opensslconf.h"

#define AES_BLOCK_SIZE 16
#define MAX_LEN 100
#define KEYLEN 16
static AES_KEY expanded;
unsigned char key[KEYLEN] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

struct aes_session {
	TEE_OperationHandle op_handle;	/* RSA operation */
	TEE_ObjectHandle key_handle; /* Key handle */
};

/*
 * Called when the instance of the TA is created. This is the first call in
 * the TA.
 */
TEE_Result TA_CreateEntryPoint(void)
{
	DMSG("has been called");

	return TEE_SUCCESS;
}

/*
 * Called when the instance of the TA is destroyed if the TA has not
 * crashed or panicked. This is the last call in the TA.
 */
void TA_DestroyEntryPoint(void)
{
	DMSG("has been called");
}

/*
 * Called when a new session is opened to the TA. *sess_ctx can be updated
 * with a value to be able to identify this session in subsequent calls to the
 * TA. In this function you will normally do the global initialization for the
 * TA.
 */
TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
		TEE_Param __maybe_unused params[4],
		void __maybe_unused **sess_ctx)
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);

	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;
	struct aes_session *sess;
	sess = TEE_Malloc(sizeof(*sess), 0);
	if (!sess)
		return TEE_ERROR_OUT_OF_MEMORY;

	sess->key_handle = TEE_HANDLE_NULL;
	sess->op_handle = TEE_HANDLE_NULL;

	*sess_ctx = (void *)sess;
	DMSG("\nSession %p: newly allocated\n", *sess_ctx);

	return TEE_SUCCESS;
}

/*
 * Called when a session is closed, sess_ctx hold the value that was
 * assigned by TA_OpenSessionEntryPoint().
 */
void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
	struct aes_session *sess;

	DMSG("Session %p: release session", sess_ctx);
	sess = (struct aes_session *)sess_ctx;

	if (sess->key_handle != TEE_HANDLE_NULL)
		TEE_FreeTransientObject(sess->key_handle);
	if (sess->op_handle != TEE_HANDLE_NULL)
		TEE_FreeOperation(sess->op_handle);
	TEE_Free(sess);

	IMSG("Goodbye!\n");
}

static TEE_Result aes_init(uint32_t param_types,
			    TEE_Param params[4])
{
	//generate the expanded key
	AES_set_encrypt_key(key, KEYLEN*8, &expanded);
	
	return TEE_SUCCESS;
}


static TEE_Result aes_encrypt(uint32_t param_types,
			    TEE_Param params[4])
{
	//load plaintext
	char *in = (char *)params[0].memref.buffer;
	int in_len = 16;
	char encrypted[AES_BLOCK_SIZE] = {0};
	// unsigned char key[KEYLEN] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	
	memcpy(encrypted, in, in_len);
	
	AES_encrypt(in, encrypted, &expanded);

	memcpy(in, encrypted, in_len);
	return TEE_SUCCESS;
}

/*
 * Called when a TA is invoked. sess_ctx hold that value that was
 * assigned by TA_OpenSessionEntryPoint(). The rest of the paramters
 * comes from normal world.
 */
TEE_Result TA_InvokeCommandEntryPoint(
			void __maybe_unused *sess_ctx,
			uint32_t cmd_id,
			uint32_t param_types, 
			TEE_Param params[4])
{
	switch (cmd_id) {
	case TA_OPENSSL_CMD_ENCRYPT:
		return aes_encrypt(param_types, params);
	case TA_OPENSSL_CMD_INIT:
		return aes_init(param_types, params);
	default:
		return TEE_ERROR_BAD_PARAMETERS;
	}
}
