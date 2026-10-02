/* Isolated implicit null-space actions using installed SuiteSparseQR. */
#include <SuiteSparseQR_C.h>
#include <SuiteSparseQR.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc,char **argv) {
    if(argc!=3)return 2;
    FILE *file=fopen(argv[1],"rb");if(!file)return 3;
    int64_t dims[3];if(fread(dims,sizeof(int64_t),3,file)!=3)return 4;
    int64_t m=dims[0],n=dims[1],nz=dims[2];
    if(m<0||n<0||nz<0)return 5;
    std::vector<int64_t> p(n+1),i(nz);
    std::vector<double> x(nz);
    if(fread(p.data(),sizeof(int64_t),n+1,file)!=(size_t)(n+1)||
       fread(i.data(),sizeof(int64_t),nz,file)!=(size_t)nz||
       fread(x.data(),sizeof(double),nz,file)!=(size_t)nz)return 6;
    fclose(file);
    cholmod_common cc;cholmod_l_start(&cc);cc.print=0;
    cholmod_sparse a{};a.nrow=m;a.ncol=n;a.nzmax=nz;a.p=p.data();a.i=i.data();a.x=x.data();
    a.itype=CHOLMOD_LONG;a.xtype=CHOLMOD_REAL;a.dtype=CHOLMOD_DOUBLE;a.sorted=1;a.packed=1;
    auto *qr=SuiteSparseQR_C_factorize(SPQR_ORDERING_FIXED,strtod(argv[2],nullptr),&a,&cc);
    if(!qr)return 7;
    auto *internals=static_cast<SuiteSparseQR_factorization<double,int64_t>*>(qr->factors);
    int64_t rank=internals->rank;
    fwrite(&rank,sizeof(int64_t),1,stdout);
    // Export Q^T A from THIS factorization. A second QR call may use a
    // different singleton ordering, so its R cannot be paired with this Q.
    cholmod_sparse *R=SuiteSparseQR_qmult<double,int64_t>(SPQR_QTX,internals,&a,&cc);
    if(!R)return 16;
    auto *rp=static_cast<int64_t*>(R->p);auto *ri=static_cast<int64_t*>(R->i);
    auto *rx=static_cast<double*>(R->x);
    std::vector<int64_t> cp(n+1,0),ci;std::vector<double> cx;
    for(int64_t j=0;j<n;++j){
        for(int64_t k=rp[j];k<rp[j+1];++k)if(ri[k]<rank){ci.push_back(ri[k]);cx.push_back(rx[k]);}
        cp[j+1]=ci.size();
    }
    int64_t rd[3]={rank,n,(int64_t)ci.size()};
    fwrite(rd,sizeof(int64_t),3,stdout);
    fwrite(cp.data(),sizeof(int64_t),n+1,stdout);fwrite(ci.data(),sizeof(int64_t),rd[2],stdout);
    fwrite(cx.data(),sizeof(double),rd[2],stdout);
    for(int64_t j=0;j<n;++j)fwrite(&j,sizeof(int64_t),1,stdout);
    cholmod_l_free_sparse(&R,&cc);
    fflush(stdout);
    std::vector<double> b((size_t)std::max(m,n));
    int64_t command;
    while(fread(&command,sizeof(int64_t),1,stdin)==1&&command!=0) {
        if(command==7) { /* batched full Q for bounded setup-time export */
            int64_t columns;
            if(fread(&columns,sizeof(int64_t),1,stdin)!=1||columns<1||columns>256)return 18;
            std::vector<double> block(m*columns);
            if(fread(block.data(),sizeof(double),block.size(),stdin)!=block.size())return 19;
            cholmod_dense input{};input.nrow=m;input.ncol=columns;input.nzmax=m*columns;
            input.d=m;input.x=block.data();input.xtype=CHOLMOD_REAL;input.dtype=CHOLMOD_DOUBLE;
            auto *out=SuiteSparseQR_C_qmult(SPQR_QX,qr,&input,&cc);
            if(!out)return 20;
            fwrite(out->x,sizeof(double),m*columns,stdout);
            cholmod_l_free_dense(&out,&cc);fflush(stdout);continue;
        }
        int64_t size=command==3?n:(command==2?m-rank:m);
        if(command<1||command>6)return 8;
        if(fread(b.data(),sizeof(double),size,stdin)!=(size_t)size)return 9;
        cholmod_dense input{};input.ncol=1;input.xtype=CHOLMOD_REAL;input.dtype=CHOLMOD_DOUBLE;
        cholmod_dense *out=nullptr;
        if(command==1) { /* restrict to null coordinates */
            input.nrow=m;input.nzmax=m;input.d=m;input.x=b.data();
            out=SuiteSparseQR_C_qmult(SPQR_QTX,qr,&input,&cc);
            if(!out)return 10;
            fwrite(static_cast<double*>(out->x)+rank,sizeof(double),m-rank,stdout);
        } else if(command==2) { /* lift null coordinates */
            std::vector<double> full(m,0.);
            memcpy(full.data()+rank,b.data(),(m-rank)*sizeof(double));
            input.nrow=m;input.nzmax=m;input.d=m;input.x=full.data();
            out=SuiteSparseQR_C_qmult(SPQR_QX,qr,&input,&cc);
            if(!out)return 11;
            fwrite(out->x,sizeof(double),m,stdout);
        } else if(command==5||command==6) {
            input.nrow=m;input.nzmax=m;input.d=m;input.x=b.data();
            out=SuiteSparseQR_C_qmult(command==5?SPQR_QTX:SPQR_QX,qr,&input,&cc);
            if(!out)return 17;
            fwrite(out->x,sizeof(double),m,stdout);
        } else if(command==4) { /* transpose of the basic affine lift */
            input.nrow=m;input.nzmax=m;input.d=m;input.x=b.data();
            cholmod_dense *y=SuiteSparseQR_C_qmult(SPQR_QTX,qr,&input,&cc);
            if(!y)return 14;
            for(int64_t j=rank;j<m;++j)static_cast<double*>(y->x)[j]=0.;
            out=SuiteSparseQR_C_solve(SPQR_RETX_EQUALS_B,qr,y,&cc);
            cholmod_l_free_dense(&y,&cc);
            if(!out)return 15;
            fwrite(out->x,sizeof(double),n,stdout);
        } else { /* affine minimum-norm lift: R' y = E' d, u=Q y */
            input.nrow=n;input.nzmax=n;input.d=n;input.x=b.data();
            cholmod_dense *y=SuiteSparseQR_C_solve(SPQR_RTX_EQUALS_ETB,qr,&input,&cc);
            if(!y)return 12;
            out=SuiteSparseQR_C_qmult(SPQR_QX,qr,y,&cc);
            cholmod_l_free_dense(&y,&cc);
            if(!out)return 13;
            fwrite(out->x,sizeof(double),m,stdout);
        }
        cholmod_l_free_dense(&out,&cc);fflush(stdout);
    }
    SuiteSparseQR_C_free(&qr,&cc);cholmod_l_finish(&cc);
    return 0;
}
