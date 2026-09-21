C     321 wet x columns + 7 dry walls, split into 8 x tiles.
      INTEGER sNx,sNy,OLx,OLy,nSx,nSy,nPx,nPy,Nx,Ny,Nr
      PARAMETER (
     & sNx=41, sNy=1,
     & OLx=2, OLy=2,
     & nSx=8, nSy=1,
     & nPx=1, nPy=1,
     & Nx=sNx*nSx*nPx,
     & Ny=sNy*nSy*nPy,
     & Nr=161 )
      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX=OLx, MAX_OLY=OLy )
