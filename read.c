#include<stdio.h>
#include<stdint.h>
#include<malloc.h>
int main(){
    FILE*f=fopen("out","rb");
    fseek(f,0,SEEK_END);
    uint64_t l=ftell(f);
    fseek(f,0,SEEK_SET);
    double*__restrict v=malloc(l);
    fread(v,1,l,f);
    fclose(f);
    for(uint64_t i=0;i<l/8;i++)printf("%lu: %f\n",i,v[i]);
}
