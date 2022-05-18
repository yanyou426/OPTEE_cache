#include <unistd.h>
#include <fcntl.h>
#include <error.h>
#include <stdio.h>



int randomize(unsigned char* buf, const size_t len){
    if(!buf) return 0;

    int fd;

    fd = open("/dev/urandom", O_RDONLY);

    if(fd <= 0){
        perror("open failed");
        return 1;
    }

    if(read(fd, (void*) buf, len) != len){
        perror("read failed");
        return 1;
    }
    
    if(close(fd)){
        perror("close failed");
        return 1;
    }

    return 0;
}

void hex_output(const unsigned char* buf, const size_t len){
    for(int i = 0; i < len; ++i){
        fprintf(stderr, "%02x", buf[i]);
    }
    fprintf(stderr, "\n");
}