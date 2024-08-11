#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#define N 5
using namespace std;
int a[N];
long long b[N];
float c[N];
double d[N];
void _int()
{
    puts("");
    puts("int:");

    puts("");
    memset(a, 0x7f, sizeof a); printf("%d\n", a[0]);
    memset(a, 0x80, sizeof a); printf("%d\n", a[0]);
    memset(a, 0x81, sizeof a); printf("%d\n", a[0]);

    puts("");
    memset(a, 0xbf, sizeof a); printf("%d %d\n", a[0], a[0] * 2);
    memset(a, 0xc0, sizeof a); printf("%d %d\n", a[0], a[0] * 2);
    memset(a, 0xc1, sizeof a); printf("%d %d\n", a[0], a[0] * 2);

    puts("");
    memset(a, 0x3e, sizeof a); printf("%d %d\n", a[0], a[0] * 2);
    memset(a, 0x3f, sizeof a); printf("%d %d\n", a[0], a[0] * 2);
    memset(a, 0x40, sizeof a); printf("%d %d\n", a[0], a[0] * 2);

    puts("");
    memset(a, 0x7e, sizeof a); printf("%d\n", a[0]);
    memset(a, 0x7f, sizeof a); printf("%d\n", a[0]);
    memset(a, 0x80, sizeof a); printf("%d\n", a[0]);
}
void _long()
{
    puts("");
    puts("long long:");

    puts("");
    memset(b, 0x7f, sizeof b); printf("%I64d\n", b[0]);
    memset(b, 0x80, sizeof b); printf("%I64d\n", b[0]);
    memset(b, 0x81, sizeof b); printf("%I64d\n", b[0]);

    puts("");
    memset(b, 0xbf, sizeof b); printf("%I64d %I64d\n", b[0], b[0] * 2);
    memset(b, 0xc0, sizeof b); printf("%I64d %I64d\n", b[0], b[0] * 2);
    memset(b, 0xc1, sizeof b); printf("%I64d %I64d\n", b[0], b[0] * 2);

    puts("");
    memset(b, 0x3e, sizeof b); printf("%I64d %I64d\n", b[0], b[0] * 2);
    memset(b, 0x3f, sizeof b); printf("%I64d %I64d\n", b[0], b[0] * 2);
    memset(b, 0x40, sizeof b); printf("%I64d %I64d\n", b[0], b[0] * 2);

    puts("");
    memset(b, 0x7e, sizeof b); printf("%I64d\n", b[0]);
    memset(b, 0x7f, sizeof b); printf("%I64d\n", b[0]);
    memset(b, 0x80, sizeof b); printf("%I64d\n", b[0]);
}
void _float()
{
    puts("");
    puts("float:");

    puts("");
    memset(c, 0xfd, sizeof c); printf("%f\n", c[0]);
    memset(c, 0xfe, sizeof c); printf("%f\n", c[0]);
    memset(c, 0xff, sizeof c); printf("%f\n", c[0]);

    puts("");
    memset(c, 0xcd, sizeof c); printf("%f %f\n", c[0], c[0] * 2);
    memset(c, 0xce, sizeof c); printf("%f %f\n", c[0], c[0] * 2);
    memset(c, 0xcf, sizeof c); printf("%f %f\n", c[0], c[0] * 2);

    puts("");
    memset(c, 0x4e, sizeof c); printf("%f %f\n", c[0], c[0] * 2);
    memset(c, 0x4f, sizeof c); printf("%f %f\n", c[0], c[0] * 2);
    memset(c, 0x50, sizeof c); printf("%f %f\n", c[0], c[0] * 2);

    puts("");
    memset(c, 0x7e, sizeof c); printf("%.15f\n", c[0]);
    memset(c, 0x7f, sizeof c); printf("%.15f\n", c[0]);
    memset(c, 0x80, sizeof c); printf("%.15f\n", c[0]);
}
void _double()
{
    puts("");
    puts("double:");

    puts("");
    memset(d, 0xfd, sizeof d); printf("%lf\n", d[0]);
    memset(d, 0xfe, sizeof d); printf("%lf\n", d[0]);
    memset(d, 0xff, sizeof d); printf("%lf\n", d[0]);

    puts("");
    memset(d, 0xc1, sizeof d); printf("%lf %lf\n", d[0], d[0] * 2);
    memset(d, 0xc2, sizeof d); printf("%lf %lf\n", d[0], d[0] * 2);
    memset(d, 0xc3, sizeof d); printf("%lf %lf\n", d[0], d[0] * 2);

    puts("");
    memset(d, 0x42, sizeof d); printf("%lf %lf\n", d[0], d[0] * 2);
    memset(d, 0x43, sizeof d); printf("%lf %lf\n", d[0], d[0] * 2);
    memset(d, 0x44, sizeof d); printf("%lf %lf\n", d[0], d[0] * 2);

    puts("");
    memset(d, 0x7e, sizeof d); printf("%lf\n", d[0]);
    memset(d, 0x7f, sizeof d); printf("%lf\n", d[0]);
    memset(d, 0x80, sizeof d); printf("%lf\n", d[0]);
}
int main()
{
    _int();
    _long();
    _float();
    _double();
}
