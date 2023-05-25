!-----------------------------------------------------------------------
! This routine solves the system (LU) y = x
!-----------------------------------------------------------------------
      SUBROUTINE prevec2( iprevec, dummy, x, y, iwork, dwork)
      INTEGER   iprevec(*), n, dummy, iwork(*)
      REAL*8    x(*),y(*),dwork(*)
      n = iprevec(1)
      call dcopy (n,x,1,y,1)
      call iluoo_solve (n, dwork(3*n+1),
     &                 iwork(n+2), iwork(2*n+3), iwork(4*n+3), y)
      return
      end

!--------------------------------------------------------
      subroutine iluoo (n, xa, ai, av, tau1, tau2, verb,
     &   work, iwork, lendwork, leniwork, 
     &   partlur, partlurout,
     &   lendworkout, leniworkout, ierr)
        implicit none
!
!  A = LU + TU + LR - S
!  A: input matrix
!  L, U: first order factors
!  T, R: second order factors (neglected after calculation)
!  S: residual matrix (neglected during the calculation)
!
!  INPUT
!  n: order of input matrix A
!
!  xa, ai, av: input matrix A in compressed sparse row format
!  normally A is scaled to have unit euclidean row and column norms prior calling iluoo_init.
!  xa(1)=1, xa(k+1)-xa(k)=number of non-zeros in row no. k, xa(n+1)=nzA;
!  ai(ka) and av(ka) are column index and value of non-zero no. ka, for ka=1, ..., nzA.
!  In a row, non-zeros of A can be sorted w.r.t. column indices or unsorted (search for "quicksort" below).
!
!  tau1: absolute threshold for entries of L and U. Elements of triangular factors greater than tau1, enter
!  L and U. Recommended values lie in interval 0.01 ... 0.1.
!  tau2: absolute threshold for entries of T and R. Elements not included in first-order factors but
!  greater than tau2, enter T and R. Recommended order of magnitude is tau1**2;
!  sometimes formula 5*tau1**2 - 0.1*tau1 may be useful.
!
!  partlur: user defined partition of available memory (work and iwork). 
!           LU occupies the room of approximately (1-partlur)*lendwork
!           R  occupies the room of approximately partlur*lendwork 
!
!  verb: zero means keep mum (save alloc failures), positive means be verbose.
!
!  OUTPUT
!  work(1:lendworkout), iwork(1:leniworkout) contain the ILU factors
!  partlurout is the optimal partition of available memory (work and iwork)
!
        integer             inf
        parameter           (inf=200 000 000)
!
! input
!
        integer             n, xa(n+1), ai(*), verb
        double precision    av(*), tau1, tau2
!
! memory
!
        integer             lendwork, leniwork
        double precision    work(lendwork)
        integer             iwork(leniwork)
        double precision    partlur, partlurout
        integer             lendworkout, leniworkout
!
! local
!
!
        integer             mv, mi
        integer             luv, ilu, iu, lui
        integer             rv, ir, ri
        integer             nnzxlu
        integer             lenlu,lenr
        integer             lenluout,lenrout
        integer             ierr
        integer             u, v
!
!---
!
 
          if (partlur .le. 0) then
            write (*,*) 'iluoo: non-positive partur'
            stop
          end if
          if (partlur .ge. 1) then
            write (*,*) 'iluoo: partlur too big'
            stop
          end if
        
          u    = 1
          v    = u    + n
          mv   = v    + n 
          luv  = mv   + n
!
          mi   = 1
          ilu  = mi   + n+1
          iu   = ilu  + n+1
          ir   = iu   + n
          lui  = ir   + n 
!
          nnzxlu = min (lendwork - 3 * n, leniwork - 4 * n - 2)
          lenlu  = (1 - partlur) * nnzxlu
          lenr   = nnzxlu - lenlu
!
          rv   = luv + lenlu
          ri   = lui + lenlu
!          
!         
!
          call  genebs (5, n, xa, ai, av, work(u), work(v))
          call  matsc (n, xa, ai, av, work(u), work(v))
          call  iluoo_init (n, xa, ai, av, tau1, tau2, inf, verb,
     &        work(mv), iwork(mi),
     &        work(luv), iwork(ilu), iwork(iu), iwork(lui),
     &        work(rv), iwork(ir), iwork(ri),
     &        lenlu, lenr, lenluout, lenrout, ierr)
          if (ierr .eq. 1) then
            if (lendwork - 3 * n .lt. leniwork - 4 * n - 2) then
                write(*,*)'iluoo: lendwork too small or partlur too big'
             else   
                write(*,*)'iluoo: leniwork too small or partlur too big'
            end if
            stop
          end if
          if (ierr .eq. 2) then
            if (lendwork - 3 * n .lt. leniwork - 4 * n - 2) then
              write(*,*)'iluoo: lendwork too small or partlur too small'
             else   
              write(*,*)'iluoo: leniwork too small or partlur too small'
            end if
            stop
          end if
          call  matisc (n, xa, ai, av, work(u), work(v))
          call  xxlusc (n, work(luv), iwork(ilu), iwork(iu), 
     &       iwork(lui), work(u), work(v))
!
          lendworkout = 3 * n  + lenluout + lenrout 
          leniworkout = 4 * n + 2 + lenluout + lenrout 
          partlurout  = lenrout * 1d0 / ( lenluout + lenrout )

!         lendworkout = lendworkout + 0.05*(lenluout + lenrout)
!         leniworkout = leniworkout + 0.05*(lenluout + lenrout)
          lendworkout = lendworkout + 1
          leniworkout = leniworkout + 1

          if (verb.eq.0) return

          write (*,*) ' iluoo:  optimal memory partition LU/R', 
     &                        ' and minimalist memory demands'
          write (*,*) '         partlur =', partlurout
          write (*,*) '        lendwork = ', lendworkout
          write (*,*) '        leniwork = ', leniworkout
          
        return
        end
!--------------------------------------------------------
      subroutine  iluoo_init (n, xa, ai, av, tau1, tau2, inf, verb,
     &  mv, mi, luv, ilu, iu, lui, rv, ir, ri,
     &  lenlu, lenr, lenluout, lenrout, ierr)
        implicit none
!
        double precision    tol_modif
        parameter           (tol_modif =  1d-12)
!
! input
!
        integer             n, xa(n+1), ai(*)
        double precision    av(*)
        double precision    tau1, tau2
        integer             inf, verb
!
! work memory
!
        double precision    mv(*)
        integer             mi(0:n)
        integer             ir(n+1), ri(*)
        double precision    rv(*)
        integer             lenlu, lenr, lenluout, lenrout
!
! output
!
        integer             ilu(n+1), iu(n), lui(*), ierr
        double precision    luv(*)
!
! local
!
        integer             k, nzra,  nzm, nzum, nzu, nzl, nmodif
!       
        ierr = 0
!        
        if (n .le. 0) then
          write (*,*) 'sflu_init: non-positive n'
          stop
        end if
!  
        do k=1,n
          mi(k) = inf
        end do  
!
        nmodif = 0
        ilu(1)=1
        ir(1)=1
!
        do k=1,n 
          nzra = xa(k+1) - xa(k)
          if (nzra .eq. 0) then
            print *, 'sflu_init: empty row at number ', k
            stop
          endif
          call dsortilu (ai(xa(k)), av(xa(k)), nzra, 2, ierr)
          call row_uncomp (n, k, nzra, ai(xa(k)), av(xa(k)), 
     &                mi, mv,  nzm, nzum, inf)
          call elim_lpart (n, k, mi, mv, inf,  tau1, tau2,
     &                luv, ilu, iu, lui, rv, ir, ri, nzm, nzum)
!          if ((ilu(k)+nzm - nzum .gt.lenlu).and.
!     &        (ir(k)+nzm -  nzum .gt.lenr))     then
!             ierr = 1
!             return
!          end if
!          if (ilu(k) + nzm - nzum .gt. lenlu) then
!             ierr = 2
!             return
!          end if
!          if (ir(k) + nzm -  nzum .gt. lenr) then
!             ierr = 3
!             return
!          end if
          call row_compr (n,  tau1, tau2, k, inf, mi, mv, 
     &        luv, ilu, iu, lui, rv, ir, ri, verb, nmodif, tol_modif,
     &        lenlu, lenr, ierr)
          if (ierr .ne. 0) return
        end do
!        
        lenluout = ilu(n+1) - 1
        lenrout  = ir(n+1) - 1
!
        nzl=0
        nzu=0
        do k=1,n
          nzl=nzl+iu(k)-ilu(k)
          nzu=nzu+ilu(k+1)-iu(k)-1
        end do
!
        if (verb.eq.0) return
        write (*,*) ' iluoo:'
        write (*,*) '     nonzeros in L = ',nzl-n
        write (*,*) '     nonzeros in U = ',nzu
!
      return
      end
!--------------------------------------------------------
      subroutine elim_lpart (n, k, mi, mv, inf, tau1, tau2,
     &  luv, ilu, iu, lui, rv, ir, ri, nzm, nzum)
        implicit none
!
! input
!
        integer           n, k, inf, mi(0:n) 
        double precision  mv(*)
        double precision  tau1, tau2
        integer           ilu(*), iu(n), lui(*)
        double precision  luv(*)
        integer           ir(*), ri(*)
        double precision  rv(*)
!
! output
!
        integer           nzm, nzum
!
! local
!
        integer          j, i, ku, kr, curr, foll
        double precision leabs, flin
!
        j = mi(0)
        do while (j .lt. k)
!         
! eliminate entry (kj) using row j of U
!
          mv(j) = mv(j) * luv(iu(j))
          leabs = abs(mv(j))
          if (leabs .le. tau2) goto 100
          curr = j
          do ku = iu(j)+1, ilu(j+1)-1
            i = lui(ku)
            if (mi(i) .ne. inf) then ! update without pondering on thresholds
              mv(i) = mv(i) - mv(j)*luv(ku)
             else ! introduce a non-zero, if threshold permits
              flin = -mv(j)*luv(ku)
              foll = curr
              do while (foll .lt. i)
                 curr = foll
                 foll = mi(curr)
              enddo ! now curr .lt. i .lt. foll
              mi(curr) = i
              mi(i) = foll
              mv(i) = flin
              nzm = nzm + 1
              if (i .gt. k) nzum = nzum+1
            endif
            curr = i
          end do  
!       
          if (leabs .le. tau1) then
            nzm = nzm - 1
            goto 100
          endif
          curr = j
          do kr = ir(j), ir(j+1)-1
            i = ri(kr)
            if (mi(i) .ne. inf) then ! update without pondering on thresholds
              mv(i) = mv(i) - mv(j)*rv(kr)
             else ! introduce a non-zero, if threshold permits
              flin = -mv(j)*rv(kr)
              foll = curr
              do while (foll .lt. i)
                 curr = foll
                 foll = mi(curr)
              enddo ! now curr .lt. i .lt. foll
              mi(curr) = i
              mi(i) = foll
              mv(i) = flin
              nzm = nzm + 1
              if (i .gt. k) nzum = nzum+1
            endif
            curr = i
          end do  
100       continue ! with the next entry
          j = mi(j)
        end do
        
      return
      end
!-------------------------------------------------------
      subroutine row_compr (n, tau1, tau2, idiag, 
     & inf, mi, mv, luv, ilu, iu, lui, rv, ir, ri, verb, nmodif,
     & tol_modif, lenlu, lenr, ierr)
        implicit none
!
! input
!
        integer          n,idiag, verb, inf, mi(0:n)
        double precision mv(*)
        double precision tau1, tau2
        double precision tol_modif
        integer          lenlu, lenr
!
! ouput
!
        double precision  luv(*)
        integer           ilu(*), iu(*), lui(*)
        double precision  rv(*)
        integer           ir(*), ri(*)
        integer           nmodif
        integer           ierr
!
! local
!
        integer           j, jn, kl, ku, kr
        double precision  ldiag, udiag, mva
!
          j = mi(0)
          ldiag = 0
          do while (j .lt. inf)
             ldiag = max(ldiag,abs(mv(j)))
             j = mi(j)
          enddo
          ldiag = 1/max(ldiag,tau2)
          j = idiag
          do while (j .lt. inf)
             mv(j) = mv(j) * ldiag
             j = mi(j)
          enddo
          j = mi(0)
          kl = ilu(idiag)
          do while (j .lt. idiag)
             if (abs(mv(j)) .gt. tau1) then
                if (kl .gt. lenlu) then
                  ierr=1
                  return
                end if
                lui(kl) = j
                luv(kl) = mv(j)
                kl = kl + 1
             endif
             jn = mi(j)
             mi(j) = inf
             j = jn
          enddo
          if (kl .gt. lenlu) then
            ierr=1
            return
          end if
          lui(kl) = idiag
          luv(kl) = ldiag
          kl = kl + 1
          iu(idiag) = kl
!          
! end of L-part
!
          if (abs(mv(j)) .gt. tol_modif) then
             udiag = 1/mv(j)
          else
             udiag = sign(1/tol_modif,mv(j))
             nmodif = nmodif + 1
          endif
          jn = mi(j)
          mi(j) = inf
          j = jn
          if (verb .ne. 0) then
             if (nmodif .eq. 1) print 1
             if (nmodif .ne. 0 .and. mod(nmodif,1000) .eq. 0) 
     &                    print 2, nmodif
          endif
          ku = iu(idiag)
          if (ku .gt. lenlu) then
            ierr=1
            return
          end if
          lui(ku) = idiag
          luv(ku) = udiag
          ku = ku + 1
          kr = ir(idiag)
          do while (j .lt. inf) ! U-part
             mva = abs(mv(j))
             if (mva .gt. tau1) then
                if (ku .gt. lenlu) then
                  ierr=1
                  return
                end if
                lui(ku) = j
                luv(ku) = mv(j)
                ku = ku + 1
             elseif (mva .gt. tau2) then
                if (kr .gt. lenr) then
                  ierr=2
                  return
                end if
                ri(kr) = j
                rv(kr) = mv(j)
                kr = kr + 1
             endif
             jn = mi(j)
             mi(j) = inf
             j = jn
          enddo
          ilu(idiag+1) = ku
          ir(idiag+1) = kr
1         format (1x, 'iluoo_init: made a zero pivot modification.')
2         format (1x, 'iluoo_init: made ', i7, 
     &                  ' zero pivot modifications.')

      return
      end
!--------------------------------------------------------
      subroutine row_uncomp (n, idiag, nz, ai, av, 
     &            mi, mv, nzm, nzum, inf)
        implicit none
!     
! input
!
        integer           n, nz, ai(*), inf, idiag
        double precision  av(*)
!
! ouput
!
        integer           nzm, nzum, mi(0:n)
        double precision  mv(*)
!
! local
!
        integer           k, ka, eol, ipred, inz
        double precision  vnz
!
        eol= inf+1
!        
        k = 0
        nzm = 0
        ipred = 0
        do ka=1,nz
           inz = ai(ka)
           vnz = av(ka)
           if (inz .le. idiag) ipred = inz
           mi(k) = inz
           k = mi(k)
           mv(k) = vnz
           nzm = nzm + 1
        enddo
        mi(k) = eol
        if (ipred .ne. idiag) then ! insert diagonal entry in structure
           mv(idiag) = 0d0
           k = mi(ipred)
           mi(ipred) = idiag
           mi(idiag) = k
           nzm = nzm + 1
        endif
        nzum = 0
        k = mi(idiag)
        do while (k .lt. inf)
           nzum = nzum + 1
           k = mi(k)
        enddo
!        
      return
      end
!--------------------------------------------------------
      subroutine iluoo_solve (n, luv, ilu, iu, lui, f)
!  INPUT
!  luv, ilu, iu, lui - structure of incomplete LU preconditioner 
!  f: right-hand side.
!  OUTPUT
!  f: solution.
!  This procedure solves the system L * U * x = f for x in-place.
!
       implicit none
!       
       integer             n, ilu(*), iu(*), lui(*)
       double precision    luv(*),f(*)
!
       integer             kl, k, j
!
        do k=1,n
          do kl=ilu(k),iu(k)-2
            j = lui(kl)
            f(k)  = f(k) - luv(kl)*f(j)
          end do
          f(k)  = f(k)*luv(iu(k)-1)
        end do
!        
        do k=n,1,-1
          do kl=iu(k)+1,ilu(k+1)-1
            j = lui(kl)
            f(k)  = f(k) - luv(kl)*f(j)
          end do
          f(k) = f(k) * luv(iu(k))
        end do
!        
      return
      end
      
! Euclidean balance scaling procedures (formerly module EBS)
      subroutine genebs (niter, n, xa, ai, av, u, v)
!      
        integer            niter, n, xa(n+1), ai(*)
        double precision   av(*)
        double precision   u(*), v(*)
!
!  generate the balance scaling
!  Keywords: Sinkhorn mapping, Sheleikhovskii method
!  G.W. Soules, The rate of convergence of Sinkhorn balancing. LAA 150: 3--40, 1991.
!
!  INPUT
!  niter: number of iterations for generation of scaling vectors (.ge. 1)
!  n: order of input matrix A
!  xa, ai, av: input matrix A in compressed sparse row format
!  xa(1)=1, xa(k+1)-xa(k)=number of non-zeros in row no. k, xa(n+1)=nzA;
!  ai(ka) and av(ka) are column index and value of non-zero no. ka, for ka=1, ..., nzA.
!  In a row, non-zeros of A can be sorted w.r.t. column indices or unsorted.
!
!  OUTPUT
!  u, v: vectors of size n such that diag(u) A diag(v) has (approx.) balanced
!  rows and columns.
!
!  USAGE
!  assume that balanced matrix is approximated by LU.
!  if Krylov iterations < nzlu / (2*n), use prevec = diag(v) inv(LU) diag(u);
!  otherwise scale LU e.g. using xxlusc.
!
        double precision   eps, subst
        parameter          (eps = 1d-25, subst = 1d0)
        integer            i, k, ka, j
        double precision   scmin
!        
        if (n .le. 0) stop 'genebs: n non-positive'
        if (niter .le. 0) stop 'genebs: niter non-positive'
!
        do k=1,n
         u(k) = 0d0
        end do
!  initial guess = e
        do k=1,n
           do ka=xa(k),xa(k+1)-1
              u(k) = u(k) + av(ka)**2
           end do
        end do
        scmin = u(1)
        do k=2,n 
          scmin = min(scmin, u(k))
        end do
        if (scmin .lt. eps) then
           write (*,*) 
     &       'genebs: (almost) zero matrix row, squared length =', scmin
           stop
        end if
        do k=1,n
           u(k) = 1d0/u(k)
        end do
!  start of iterations  
        do i=1,niter
          do k=1,n
            v(k) = 0d0
          end do
          do k=1,n
             do ka=xa(k),xa(k+1)-1
                j = ai(ka)
                v(j) = v(j) + u(k)*av(ka)**2
             enddo
          enddo
          do k=1,n
             if (v(k) .lt. eps) v(k) = subst
          enddo
          do k=1,n
             v(k) = 1d0/v(k)
          enddo
          do k=1,n
           u(k) = 0d0
          end do
          do k=1,n
             do ka=xa(k),xa(k+1)-1
                j = ai(ka)
                u(k) = u(k) + v(j)*av(ka)**2
             enddo
          enddo
          do k=1,n
             if (u(k) .lt. eps) u(k) = subst
          enddo
          do k=1,n
             u(k) = 1d0/u(k)
          enddo
        enddo
        do k=1,n
           u(k) = sqrt(u(k))
        enddo
        do k=1,n
           v(k) = sqrt(v(k))
        enddo
      return
      end 
!-------------------------------------------------------      
      subroutine matsc (n, xa, ai, av, u, v)
!      
        integer           n, xa(n+1), ai(*)
        double precision  av(*)
        double precision  u(*), v(*)
        integer k, ka, j
!
!  scale the matrix by diag(u) (from the left) and by diag(v) (from the right).
!
        do k=1,n
          do ka=xa(k),xa(k+1)-1
             j = ai(ka)
             av(ka) = u(k) * av(ka) * v(j)
          enddo
        enddo
      return
      end 
!-------------------------------------------------------      
      subroutine matisc (n, xa, ai, av, u, v)
!      
        integer           n, xa(n+1), ai(*)
        double precision  av(*)
        double precision  u(*), v(*)
        integer k, ka, j
!
!  un-scale the matrix by diag(u) (from the left) and by diag(v) (from the right)
!  (undo the effect of matsc).
!
        do k=1,n
          do ka=xa(k),xa(k+1)-1
             j = ai(ka)
             av(ka) = (av(ka) / u(k)) / v(j)
          enddo
        enddo
      return
      end
!-------------------------------------------------------      
      subroutine xxlusc (n, luv, ilu, iu, lui, u, v)
!  INPUT
!  luv, ilu, iu, lui:  structure of incomplete LU preconditioner
!  u, v: scaling vectors.
!  OUTPUT
!  xxlu: new L = inv(diag(u)) L; new U = U inv(diag(v)).
!
        integer           n
        integer           ilu(*), iu(n), lui(*)
        double precision  luv(*), u(*), v(*)
        integer           k, kl, ku
!
        do k=1,n
          do kl=ilu(k),iu(k)-2
            luv(kl) = luv(kl) / u(k)
          end do
          luv(iu(k)-1) = luv(iu(k)-1) * u (k)
        end do
!
        do k=1,n
          do ku=iu(k)+1,ilu(k+1)-1
            j = lui(ku)
            luv(ku) = luv(ku) / v(j)
          end do
          luv(iu(k)) = luv(iu(k)) * v (k)
        end do
      return
      end

