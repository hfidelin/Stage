      SUBROUTINE integr (XX0, YY0, ZZ0,&
                   N, XOBS, YOBS, ZOBS,&
                   INABLA, ISC,&
                   IVECX, IVECY, IVECZ,&
                   FLINT, ERROR)
!
!     ON  EMPTY
      INTEGER, intent(in) :: N
      INTEGER , intent(in) :: FLINT
!
!     \nabla(1/r)   <->  flint=1 
!     1/r       <->  flint=2
!     \vec{rho}/r <-> flint=3
!
      DOUBLE PRECISION  , intent(in) :: XX0(3), YY0(3), ZZ0(3)
      DOUBLE PRECISION , intent(in) ::  XOBS, YOBS, ZOBS
!
!     ON RETURN
!
      DOUBLE PRECISION , intent(out) ::INABLA, ISC
      DOUBLE PRECISION , intent(out) :: IVECX, IVECY, IVECZ
      INTEGER     , intent(out) ::     ERROR
!
!     LOCAL VARIABLES
!
      DOUBLE PRECISION XINT(5), YINT(5), ZINT(5)
      INTEGER          J, L
      DOUBLE PRECISION XOBSPR, YOBSPR, ZOBSPR
      DOUBLE PRECISION VX, VY, VZ
      DOUBLE PRECISION V2X, V2Y, V2Z
      DOUBLE PRECISION VNX, VNY, VNZ
      DOUBLE PRECISION VTX, VTY, VTZ
      DOUBLE PRECISION DPROD, D, KOEFF
      DOUBLE PRECISION VNORM
      DOUBLE PRECISION P0, P0X, P0Y, P0Z
      DOUBLE PRECISION UX, UY, UZ
      DOUBLE PRECISION IADD, SUMADD
      DOUBLE PRECISION R0, RPLUS, RMINUS
      DOUBLE PRECISION LPLUS, LMINUS
      DOUBLE PRECISION DELTA, NUMER, DENUM
      PARAMETER       (DELTA = 1.D-8)
!
!
!

      ERROR = 1
      dprod = 0.d0
      DO J = 1, N
         XINT(J) = XX0(J)
         YINT(J) = YY0(J)
         ZINT(J) = ZZ0(J)
      ENDDO
!
      VX = XINT(2) - XINT(1)
      VY = YINT(2) - YINT(1)
      VZ = ZINT(2) - ZINT(1)
!
      V2X = XINT(3) - XINT(1)
      V2Y = YINT(3) - YINT(1)
      V2Z = ZINT(3) - ZINT(1)
!
      VNX = VY * V2Z - VZ * V2Y
      VNY = -VX * V2Z + VZ * V2X
      VNZ = VX * V2Y - VY * V2X
!
      VNORM = DSQRT(VNX * VNX + VNY * VNY + VNZ * VNZ)
!
      VNX = VNX / VNORM
      VNY = VNY / VNORM
      VNZ = VNZ / VNORM
!
      D = (XOBS - XINT(1)) * VNX +&
    (YOBS - YINT(1)) * VNY +&
    (ZOBS - ZINT(1)) * VNZ
!
      XOBSPR = XOBS - D * VNX
      YOBSPR = YOBS - D * VNY
      ZOBSPR = ZOBS - D * VNZ
!
      ISC = 0.0D0
      INABLA = 0.0D0
      IVECX = 0.0D0
      IVECY = 0.0D0
      IVECZ = 0.0D0
!
      DO J = 1, N
!
       IADD = 0
       IF  (J .EQ. N) THEN
       L = 1
       ELSE
       L = J + 1
       ENDIF
!
       VTX = XINT(L) - XINT(J)
       VTY = YINT(L) - YINT(J)
       VTZ = ZINT(L) - ZINT(J)
!
       VNORM = DSQRT(VTX * VTX + VTY * VTY + VTZ * VTZ)
       VTX = VTX / VNORM
       VTY = VTY / VNORM
       VTZ = VTZ / VNORM
!
       UX = VTY * VNZ - VTZ * VNY
       UY = -VTX * VNZ + VTZ * VNX
       UZ = VTX * VNY - VTY * VNX
!
       RMINUS = DSQRT((XOBS - XINT(J)) * (XOBS - XINT(J))&
                + (YOBS - YINT(J)) * (YOBS - YINT(J))&
                + (ZOBS - ZINT(J)) * (ZOBS - ZINT(J)))
!
       RPLUS = DSQRT((XOBS - XINT(L)) * (XOBS - XINT(L))&
               + (YOBS - YINT(L)) * (YOBS - YINT(L))&
               + (ZOBS - ZINT(L)) * (ZOBS - ZINT(L)))
!
       LMINUS = (XINT(J) - XOBSPR) * VTX +&
            (YINT(J) - YOBSPR) * VTY +&
            (ZINT(J) - ZOBSPR) * VTZ
!
       LPLUS = (XINT(L) - XOBSPR) * VTX +&
           (YINT(L) - YOBSPR) * VTY +&
           (ZINT(L) - ZOBSPR) * VTZ
!
       P0 = ((XINT(J) - XOBSPR) * (XINT(J) - XOBSPR)&
             + (YINT(J) - YOBSPR) * (YINT(J) - YOBSPR)&
             + (ZINT(J) - ZOBSPR) * (ZINT(J) - ZOBSPR)&
             - LMINUS * LMINUS)
!
!      For  correct  numerical  calculations
!       in  analise  p0 > 0  always !
!
       P0 = MAX(P0, 0.0D0)
       P0 = DSQRT(P0)
!
       R0 = DSQRT(P0 * P0 + D * D)
!
       DENUM = MAX ((RMINUS + LMINUS ), DELTA)
       NUMER = MAX ((RPLUS + LPLUS ), DELTA)
       IADD = DLOG ( NUMER / DENUM )
       INABLA = INABLA + IADD * P0 * DPROD
!
       P0X = XINT(J) - LMINUS * VTX - XOBSPR
       P0Y = YINT(J) - LMINUS * VTY - YOBSPR
       P0Z = ZINT(J) - LMINUS * VTZ - ZOBSPR
       VNORM = MAX (DSQRT (P0X * P0X + P0Y * P0Y + P0Z * P0Z ), DELTA)
       P0X = P0X / VNORM
       P0Y = P0Y / VNORM
       P0Z = P0Z / VNORM
       !print *,p0,r0,d,xx0(1:3),yy0(1:3),zz0(1:3),xobs,yobs,zobs
       !pause
!
       DPROD = P0X * UX + P0Y * UY + P0Z * UZ
!
       IF (FLINT .GT. 1) THEN
       SUMADD = P0 * IADD
       SUMADD = SUMADD - DABS(D) *&
         (DATAN2 (P0 * LPLUS , (R0 * R0 + DABS(D) * RPLUS))&
         - DATAN2 (P0 * LMINUS , (R0 * R0 + DABS(D) * RMINUS))&
         )
       SUMADD = SUMADD * DPROD
       ISC = ISC + SUMADD
!
       IF (FLINT .GT. 2) THEN
!
         KOEFF = 0.5D0 * (R0 * R0 * IADD +&
           RPLUS * LPLUS - RMINUS * LMINUS)
         IVECX = UX * KOEFF + IVECX
         IVECY = UY * KOEFF + IVECY
         IVECZ = UZ * KOEFF + IVECZ
       ENDIF
!
       ENDIF
!
      END DO
!
      RETURN
      END
      
!
!

