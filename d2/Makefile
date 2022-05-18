# include config.in
# export 

# MASTIK = "Mastik/src"
# LD_LIBRARY_PATH = $(OPENSSL)

# all: spy

# spy: 
# 	gcc cache.c set_sched.c attack.c -DMACROS -DTABLE0=$(TABLE0) -DTABLE1=$(TABLE1) -DTABLE2=$(TABLE2) -DTABLE3=$(TABLE3) -DTHRESHOLD=$(THRESHOLD) -DCPU -DCPU0=$(CPU0) -DCPU1=$(CPU1)  -L$(OPENSSL) -I$(OPENSSL)/include -Wno-deprecated-declarations -lcrypto -lpthread -std=gnu11 -o spy


# clean:
# 	rm -f *.o spy

# include config.in

INCLUDE_PATH := -I$(shell pwd)/aes 
# -L$(shell pwd)/aes -laes

targets = spy.o
# targets = elimi.o

CFLAGS ?= $(INCLUDE_PATH) 
CFLAGS += -march=armv7 


.PHONY: all
all: $(targets)
	$(CC) $(targets) $(CFLAGS) -L$(shell pwd)/aes -laes -o spy

clean:
	rm -f *.o spy

.PHONY: so
so:
	@make INCLUDE_PATH="$(INCLUDE_PATH)" -C aes/

$(targets):

