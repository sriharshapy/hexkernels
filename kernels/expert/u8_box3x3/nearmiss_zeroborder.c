/* Near-miss: zero-pad borders instead of clamp-to-edge. */
#include <stdint.h>
void candidate_kernel(const uint8_t*in,uint8_t*o,int w,int h){
  for(int y=0;y<h;y++)for(int x=0;x<w;x++){int s=0;
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){int yy=y+dy,xx=x+dx;
      if(yy>=0&&yy<h&&xx>=0&&xx<w) s+=in[yy*w+xx];}
    o[y*w+x]=(unsigned char)(s/9);}
}
