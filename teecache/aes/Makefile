target = libaes.so
source = aes.c
object = aes.o

CFLAGS = $(INCLUDE_PATH)

.PHONY: all
all: $(target)
	

$(object):

$(target):$(object)
	$(CC) $(object) -shared -fPIC -o $(target)

clean:
	rm -f *.o *.so