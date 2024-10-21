!--
!--	h3dd: a doulble-difference earthquake location program with
!--           spherical 3D velocity model
!--
!--	2012 SEP: based on shypo3D program by Yih-Min Wu
!--     2013 JUL: developed by Shu-Heng Lin with his master thesis
!--     2017 JUN: modified by Hsin-Hua Huang
!--

module observ
    parameter    (maxsta=3000)
    parameter    (maxobs=maxsta)
    parameter    (maxsol=4)
    character*4  stn(maxsta) 
    integer      ltds(maxsta),lnds(maxsta)
    real*8       sltm(maxsta),slnm(maxsta)
    real*8       stc(3,maxsta)
    real*8       pcor(maxsta),scor(maxsta),spcor(maxsta)
    integer      isto(maxobs),ifm(maxobs),inten(maxobs),isp(maxobs)
    real*8       secp(maxobs,2),epdis(maxobs),azimuth(maxobs),takeoff(maxobs)
    real*8       wa(maxobs),xpga(maxobs),xweio(maxobs,2),res(maxobs,2),xmls(maxobs)
    real*8       wa1(maxobs),xmls1(maxobs)
    integer      nsts,stt(2,maxsta)
    real*8       dth(maxobs,4,2)  
end module observ

module events
    integer      ltde,lnde,nobs,minp,nstations,ifixdepth
    real*8       seco,eltm,elnm,gap,epmin,fixdepth
    real*8       evc(3)
end module events

module ray
    parameter    (maxnlat=100)
    parameter    (maxnlon=100)
    parameter    (maxndep=150)
  
    real*8       bldh,bldv,lat_c(maxnlat),lon_c(maxnlon),dep_c(maxndep),ro
    real*8       vel_p(maxnlon,maxnlat,maxndep),vel_s(maxnlon,maxnlat,maxndep),elev(maxnlon,maxnlat)
    integer      nlat_c,nlon_c,ndep_c,nxyz_c,nxy_c,nx_c,ips
  
    !-- ilatdeg, ilondeg are size of map in latitude, longitude
    parameter    (ilatdeg=1000000)
    parameter    (ilondeg=1000000)
    parameter    (idepkm =1000000)
    real*8       lat1_c,lon1_c,dep1_c
    integer      ilonloc_c(ilondeg),ilatloc_c(ilatdeg),ideploc_c(idepkm)
    real*8       bx(5),by(5)                           ! LON. and LAT. RANGES OF VELOCITY MODEL
end module ray

module all
    integer      maxevt,numberob,numberob_s
    parameter    (maxevt=5000)                         ! NUMBER OF THE EARTHQUAKES
    parameter    (numberob_s=0)                        ! MATRIX SIZE (FOR SVD)
    parameter    (numberob=25000000)                   ! MATRIX SIZE (FOR LSQR)
    real*8       eqlocation(3,maxevt),eqseco(maxevt)   ! LOCATION & TIME OF EARTHQUAKE
    integer      iysto(maxevt),imsto(maxevt)           ! YEAR & MONTH 
    integer      idsto(maxevt),ihrsto(maxevt)          ! DAY & HOUR
    integer      mino(maxevt)                          ! MINUTE
    real*8       xmag(maxevt)                          ! MAGNITUDE
    integer      neq                                   ! NUMBER OF THE EQ 
    integer      eqtotal                               ! NUMBER OF THE EQ READ IN 
    integer      number                                ! NUMBER OF STATION OF THE EQ IN THE PFILE 
    integer      nit                                   ! NUMBER OF ITERATION
    integer      allnstations(maxevt)                  ! NUMBER OF STATION OF EACH EQ IN THE PFILE, AND HAVE MATADATA IN .sta FILE
    real*8       ptcal(maxevt,2000),stcal(maxevt,2000) ! P&S TRAVEL TIME OF EACH OBSERVATION
    real*8       res0(0:100),reswt(0:100)              ! UNWEIGHTED/WEIGHTED RMS OF EACH ITERATION
    real*8       rto0(0:100),rtowt(0:100)              ! RESIDUAL RATIO BETWEEN ITERATIONS
    real*8       diffrms,rmscount                      ! FOR CALCULATING WEIGHTED RMS
    real*8       orms,ormscount                        ! FOR CALCULATING UNWEIGHTED RMS
    real*8       confac                                ! CONSTRAIN FACTOR (FOR MEAN SHIFT=0 IN DOUBLE DIFFERENCE)
    real*8       wp                                    ! WEIGHTING FOR P
    real*8       ws                                    ! WEIGHTING FOR S
    real*8       wse                                   ! WEIGHTING FOR SINGLE EVENT
    real*8       wcc                                   ! WEIGHTING FOR X-CORRELATION  
    real*8       wct(maxevt,2000,2)                    ! WEIGHTING FOR CATALOG
    real*8       pri(5)                                ! A PRIORI WEIGHTING FOR CATALOG
    real*8       weiloc,weitel                         ! WEIGHTING FOR LOCAL & TELESEISMIC (SYNTETIC TEST ONLY)
    character*64 velfile                               ! VELOCITY MODEL FILE
    character*64 stafile                               ! STATION INFORMATION FILE
    integer      inv                                   ! INVERSION TYPE (1=SVD, 2=LSQR)
    integer      eqob(maxevt)                          ! OBSERVATION OF EACH EARTHQUAKE IN D-D
    real*8       discut                                ! CUT OFF DISTANCE FOR D-D
    real*8       erh_ori(maxevt)                       ! ORIGINAL ERH
    real*8       erz_ori(maxevt)                       ! ORIGINAL ERZ 
    real*8       rms_ori(maxevt)                       ! ORIGINAL RMS
    real*8       xerh_ori,xerz_ori,xrms_ori            ! TEMP VARIABLES FOR ORIGINAL ERH,ERZ,RMS
    real*8       cvm(4*maxevt,4*maxevt)                ! COVARIENCE MATRIX
    real*8       sumcvm
    real*8       var,var1,resmean                      ! VARIENCE AND MEAN RESIDUAL (ERROR ESTIMATION)
    real*8       sterr(4*maxevt)                       ! STD ERROR
    real*8       erh(maxevt),erz(maxevt)               ! ERH, ERZ
    real*8       rmscut                                ! CUTOFF RMS TO DEFINE THE TERMINATION OF ITERATION
end module all

program main
    use all
    use observ
    use events
    use ray
  
    integer*2              ic
    character*90           pfile,ofile
    character*90           card(maxsta)
    logical                alive
    integer                snglev
    integer                ierr
    integer                iskip
    character*1            chk
    real*8, allocatable :: a(:,:),s(:),v(:,:),adj(:),alldth(:,:,:,:),ddt(:)
!   real*8                 a(numberob,4*maxevt+1),s(4*maxevt),v(4*maxevt,4*maxevt),adj(4*maxevt),alldth(maxevt,maxsta,5,2)
    real*8                 geoc_to_geog,geog_to_geoc,xla,km 
    character*4            stname(maxevt,maxsta)
    integer                stnumber(maxevt,maxsta)
    character*64           line
    real*8                 ori_x,ori_y,ori_z
    real*8                 re_x,re_y
    real*8                 tccp,tccs
    real*8                 xdis

    !-- FOR LSQR
    real*8  ra(8*numberob)
    integer ja(8*numberob)
    integer na(numberob)
    common  /matinv/ra,ja,na 
    logical wantse
    integer leniw, lenrw, itnlim, nout, istop, itn
    integer,allocatable ::  iw(:)
    real*8 ,allocatable ::  rw(:) 
    real*8  vv(4*maxevt), ww(4*maxevt), se(4*maxevt), atol, btol, conlim, damp, anorm, acond, rnorm, arnorm, xnorm
 
    eigtol=0.005 ! SVD cutoff

!-- input control file	
    open(10,file="h3dd.inp",status="old")
    ncount=0
    do
      read(10,'(a)',end=200)line
      if(line(1:1)=="*")cycle
      ncount=ncount+1
      if(ncount==1) read(line,'(a90)')pfile
      if(ncount==2) read(line,'(a64)')stafile 
      if(ncount==3) read(line,'(a64)')velfile 
      if(ncount==4) read(line,*)wp,ws,wse
      if(ncount==5) read(line,*)(pri(i),i=1,5)
      if(ncount==6) read(line,*)discut
      if(ncount==7) read(line,*)inv
      if(ncount==8) read(line,*)damp
      if(ncount==9) read(line,*)rmscut
      if(ncount==10) read(line,*)nitmax
      if(ncount==11) read(line,*)confac
      if(ncount==12) read(line,*)snglev
    enddo
200 close(10)
!-- end input control file

! ifixdepth=0
! call getarg(1,pfile,ic)
! if(ic.lt.1 .or. ic.gt.64)then
!   print*,'Usage: shypo3d pfile'
! endif
! call getarg(2,card,ic)
! if(ic.gt.1 .and. ic.lt.64)then
!   if(card(1:2).eq.'-f' .or. card(1:2).eq.'-F')then
!      ifixdepth=1
!	  read(card(3:ic),*)fixdepth
!	endif
! endif

    !-- INPUT VELOCITY MODEL
    call input_vel(velfile)
    print*,"input_vel OK."

    !-- READ NUMBER OF EVENT
    open(1,file=pfile,status='old')
    eqtotal=0
    do
      read(1,'(1x,a1)',iostat=ierr)chk
      if(ierr.lt.0) exit
      if(ichar(chk).ge.ichar('1').and.ichar(chk).le.ichar('9'))eqtotal=eqtotal+1
    enddo
    close(1)
    print*,"total event:",eqtotal

    !-- ITERATION
    do nit=1,nitmax+1    !-- +1 FOR CALCULATE RESIDUAL
      print'(a10,1x,i2)'," iteration",nit

      
      !--ALLOCATE alldth
      allocate(alldth(maxevt,maxsta,5,2),stat=nerror)
      if (nerror/=0) stop "alldth is too large"   
      !--END ALLOCATE alldth

      !-- ALLOCATE a
      !-- NOTE: METRIX a IS ONLY USED WHEN inv=1 (SVD MODE)
      if (inv==1) allocate(a(numberob_s,4*maxevt+1),stat=nerror)
      if (nerror/=0) stop 'a is too large'
      !--END ALLOCATE a

      !-- ALLOCATE ddt
      !-- NOTE: ONLY USED BY LSQR
      if(inv==2)allocate(ddt(numberob),stat=nerror)
      if (nerror/=0) stop 'ddt is too large'
      !-- END ALLOCATE ddt


      !-- INPUT THE STATION COORDINATES
      call input_sta
      do neq=1,eqtotal
        print'(a13,i8)'," Earthquake#:",neq
        !-- INPUT P & S ARRIVALS FROM PFILE
        iskip=0
        call input_pfile(pfile,stname,stnumber,iskip)
        if(iskip.eq.1) cycle

        if(nit== 1)then
          !-- SAVE ORIGINAL LOCATION and TIME
          eqlocation(1:3,neq)=evc(1:3)
          eqseco(neq)=seco
        else
          evc(1:3)=eqlocation(1:3,neq)
          seco=eqseco(neq)
          dth=0.0
          res=0.0
        endif
        !-- FOR EARTHQUAKE LOCATION
        call location
        !-- SAVE THE DERIVATIVES OF EACH EQ & STATIONS
        do i=1,allnstations(neq)
          !-- for P wave
          alldth(neq,i,1,1)=dth(i,1,1) !-- dt
          alldth(neq,i,2,1)=dth(i,2,1) !-- dx
          alldth(neq,i,3,1)=dth(i,3,1) !-- dy
          alldth(neq,i,4,1)=dth(i,4,1) !-- dz
          alldth(neq,i,5,1)=res(i,1)   !-- residual
          !-- for S wave
          alldth(neq,i,1,2)=dth(i,1,2) !-- dt
          alldth(neq,i,2,2)=dth(i,2,2) !-- dx
          alldth(neq,i,3,2)=dth(i,3,2) !-- dy
          alldth(neq,i,4,2)=dth(i,4,2) !-- dz
          alldth(neq,i,5,2)=res(i,2)   !-- residual
        enddo 
      enddo

      !-- A METRIX (Ax=b)  
      if(inv==1) a=0.       !-- A METRIX (ONLY USED BY SVD)
      if(inv==2)then
        nnz=0               !-- COUNTER FOR NONZERO VALUES (ONLY USED BY LSQR)
        ra=0.               !-- NONZERO VALUES (ONLY USED BY LSQR)
        ja=0                !-- NUMBER OF COLUMN OF EACH NONZERO VALUE (ONLY USED BY LSQR)
        na=0                !-- NUMBER OF NONZERO VALUE IN EACH ROW (ONLY USED BY LSQR)
        ddt=0.              !-- RESIDUAL
      endif
      n=0                   !-- COUNTER FOR NUMBER OF ROWS IN a
      m1=0                  !-- TOTAL EVENT PAIR
      m2=0                  !-- USED EVENT PAIR
      orms=0.               !-- UNWEIGHTED RMS
      ormscount=0.          !--
      diffrms=0.            !-- WEIGHTED RMS
      rmscount=0.           !--
      eqob=0                !-- OBSERVATION FOR EACH EQ IN D-D (IF <4, IT'S AN ISOLATED EQ)
 
      do i=1,eqtotal-1
        do j=i+1,eqtotal
          m1=m1+1
          !-- CUT OFF DISTANCE
          call cal_delta(eqlocation(2,i),eqlocation(1,i),eqlocation(2,j),eqlocation(1,j),km)
          km=sqrt(km**2+(eqlocation(3,i)-eqlocation(3,j))**2)
          if(km>=discut)cycle
          m2=m2+1
          do k=1,allnstations(i)
            do l=1,allnstations(j)
              !-- FIND THE SAME STATION
              if(stname(i,k)==stname(j,l))then
                !print*,stnumber(i,k),stnumber(j,l)
                !-- PAIR-STATION DISTANCE RATIO THRESHOLD
                call cal_delta((eqlocation(2,i)+eqlocation(2,j))/2.,(eqlocation(1,i)+eqlocation(1,j))/2.,&
                               stc(2,stnumber(i,k)),stc(1,stnumber(j,l)),km)
                km=sqrt(km**2+((eqlocation(3,i)+eqlocation(3,j))/2.)**2)
                eqob(i)=eqob(i)+1
                eqob(j)=eqob(j)+1

                !-- FOR P WAVE
                !-- CHECK P WAVE EXIST OR NOT           
                if(alldth(i,k,1,1)/=0. .or. alldth(i,k,2,1)/=0. .or. alldth(i,k,3,1)/=0. .or. alldth(i,k,4,1)/=0.)then
                  if(alldth(j,l,1,1)/=0. .or. alldth(j,l,2,1)/=0. .or. alldth(j,l,3,1)/=0. .or. alldth(j,l,4,1)/=0.)then
                    if(wct(i,k,1)/=0. .and. wct(j,l,1)/=0.)then ! DELETE WT=4
                      n=n+1
            
                      if(inv==1)then
                        !-- FOR SVD
                        a(n,(i-1)*4+1:(i-1)*4+4)=alldth(i,k,1:4,1)
                        a(n,(j-1)*4+1:(j-1)*4+4)=-1.*alldth(j,l,1:4,1)
                        a(n,eqtotal*4+1)=alldth(i,k,5,1)-alldth(j,l,5,1)  
                        !-- WEIGHTING (SVD)
                        a(n,(i-1)*4+1:(i-1)*4+4)=a(n,(i-1)*4+1:(i-1)*4+4)*wp*wct(i,k,1)*wct(j,l,1) 
                        a(n,(j-1)*4+1:(j-1)*4+4)=a(n,(j-1)*4+1:(j-1)*4+4)*wp*wct(i,k,1)*wct(j,l,1) 
                        a(n,eqtotal*4+1)=a(n,eqtotal*4+1)*wp*wct(i,k,1)*wct(j,l,1)                  
                      elseif (inv==2)then
                        !-- FOR LSQR
                        ra(nnz+1:nnz+4)=alldth(i,k,1:4,1)
                        ra(nnz+5:nnz+8)=-1.*alldth(j,l,1:4,1)
                        do ii=1,4
                          ja(nnz+ii)=(i-1)*4+ii
                          ja(nnz+ii+4)=(j-1)*4+ii
                        enddo
                        na(n)=8
                        ddt(n)=alldth(i,k,5,1)-alldth(j,l,5,1)
                        !-- WEIGHTING (LSQR)
                        ra(nnz+1:nnz+8)=ra(nnz+1:nnz+8)*wp*wct(i,k,1)*wct(j,l,1)
                        ddt(n)=ddt(n)*wp*wct(i,k,1)*wct(j,l,1)
                        !-- COUNT NONEZERO
                        nnz=nnz+8
                      endif

                      !-- CALCULATE UNWEIGHTED RMS
                      orms=orms+(alldth(i,k,5,1)-alldth(j,l,5,1))**2
                      ormscount=ormscount+1.
                      !-- CALCULATE WEIGHTED RMS
                      diffrms=diffrms+((alldth(i,k,5,1)-alldth(j,l,5,1))*wp*wct(i,k,1)*wct(j,l,1))**2    
                      rmscount=rmscount+wp*wct(i,k,1)*wct(j,l,1)
                    endif 
                  endif
                endif
                !-- FOR S WAVE
                !-- CHECK S WAVE EXIST OR NOT
                if(alldth(i,k,1,2)/=0. .or. alldth(i,k,2,2)/=0. .or. alldth(i,k,3,2)/=0. .or. alldth(i,k,4,2)/=0.)then
                  if(alldth(j,l,1,2)/=0. .or. alldth(j,l,2,2)/=0. .or. alldth(j,l,3,2)/=0. .or. alldth(j,l,4,2)/=0.)then
                    if(wct(i,k,2)/=0. .and. wct(j,l,2)/=0.)then
                      n=n+1 

                      if(inv==1)then
                        !-- FOR SVD
                        a(n,(i-1)*4+1:(i-1)*4+4)=alldth(i,k,1:4,2)
                        a(n,(j-1)*4+1:(j-1)*4+4)=-1.*alldth(j,l,1:4,2)
                        a(n,eqtotal*4+1)=alldth(i,k,5,2)-alldth(j,l,5,2)
                        !-- WEIGHTING (SVD)
                        a(n,(i-1)*4+1:(i-1)*4+4)=a(n,(i-1)*4+1:(i-1)*4+4)*ws*wct(i,k,2)*wct(j,l,2)
                        a(n,(j-1)*4+1:(j-1)*4+4)=a(n,(j-1)*4+1:(j-1)*4+4)*ws*wct(i,k,2)*wct(j,l,2)
                        a(n,eqtotal*4+1)=a(n,eqtotal*4+1)*ws*wct(i,k,2)*wct(j,l,2)
                      elseif (inv==2)then
                        !-- FOR LSQR
                        ra(nnz+1:nnz+4)=alldth(i,k,1:4,2)
                        ra(nnz+5:nnz+8)=-1.*alldth(j,l,1:4,2)
                        do ii=1,4
                          ja(nnz+ii)=(i-1)*4+ii
                          ja(nnz+ii+4)=(j-1)*4+ii
                        enddo
                        na(n)=8
                        ddt(n)=alldth(i,k,5,2)-alldth(j,l,5,2)
                        !-- WEIGHTING (LSQR)
                        ra(nnz+1:nnz+8)=ra(nnz+1:nnz+8)*ws*wct(i,k,2)*wct(j,l,2)
                        ddt(n)=ddt(n)*ws*wct(i,k,2)*wct(j,l,2)
                        !-- COUNT NONZERO
                        nnz=nnz+8
                      endif

                      !-- CALCULATE UNWEIGHTED RMS
                      orms=orms+(alldth(i,k,5,2)-alldth(j,l,5,2))**2
                      ormscount=ormscount+1.
                      !-- CALCULATE WEIGHTED RMS
                      diffrms=diffrms+((alldth(i,k,5,2)-alldth(j,l,5,2))*ws*wct(i,k,2)*wct(j,l,2))**2
                      rmscount=rmscount+ws*wct(i,k,2)*wct(j,l,2)
                    endif
                  endif
                endif
              endif
            enddo
          enddo
8888    enddo
      enddo
  
      !-- NUMBER OF EVENT PAIRS FOR DOUBLE DIFFERENCE
      print'(a20,1x,i7,a1,i7)'," Event pairs for dd:",m2,"/",m1
      !-- NUMBER OF OBSERVATIONS FOR DOUBLE DIFFERENCE
      print'(a21,1x,i7)'," Observations for dd:",n 

      !-- GIVE A CONSTRAINT FOR MEAN SHIFT IS ZERO	
      do i=1,4
        n=n+1
        do j=1,eqtotal
          if (eqob(j)<=3)then !-- MOVE OUT THE CONSTRAINT FOR ISOLATED EARTHQUAKES
            if(i==1)print'(a3,1x,i5,1x,a24)',"No.",j,"event <4 stations in D-D"
            cycle
          endif
          if (inv==1)then
            !-- FOR SVD
            a(n,(j-1)*4+i)=1.*confac    !-- confac = constrain factor
          elseif(inv==2)then
            !-- FOR LSQR
            nnz=nnz+1
            ra(nnz)=1.*confac  !-- confac = constrain factor
            ja(nnz)=(j-1)*4+i
            na(n)=na(n)+1  
          endif
        enddo
        if(inv==1) a(n,eqtotal*4+1)=0.0
        if(inv==2) ddt(n)=0.0
      enddo

      !-- CALCULATE UNWEIGHTED RMS
      ormscount=ormscount+4.
      !-- CALCULATE WEIGHTED RMS
      rmscount=rmscount+4.*confac


      !-- FOR SINGLE EVENT
      if (snglev==1) then
        print*, "Joint inversion with single event method"  
        do i=1,eqtotal
!       if (eqob(i)<=3)then   !-- ONLY FOR ISOLATED EARTHQUAKES!
          do j=1,allnstations(i)
            !-- FOR P WAVE
            !-- CHECK P WAVE EXIST OR NOT
            if(alldth(i,j,1,1)/=0. .or. alldth(i,j,2,1)/=0. .or. alldth(i,j,3,1)/=0. .or. alldth(i,j,4,1)/=0.)then
              if(wct(i,j,1)/=0.)then
                n=n+1
                if(inv==1)then
                  !-- FOR SVD
                  a(n,(i-1)*4+1:(i-1)*4+4)=alldth(i,j,1:4,1)
                  a(n,eqtotal*4+1)=alldth(i,j,5,1)
                  !-- WEIGHTING (SVD)
                  a(n,(i-1)*4+1:(i-1)*4+4)=a(n,(i-1)*4+1:(i-1)*4+4)*wp*wse*wct(i,j,1)
                  a(n,eqtotal*4+1)=a(n,eqtotal*4+1)*wp*wse*wct(i,j,1)
                elseif (inv==2)then
                  !-- FOR LSQR
                  ra(nnz+1:nnz+4)=alldth(i,j,1:4,1)
                  do ii=1,4
                    ja(nnz+ii)=(i-1)*4+ii
                  enddo
                  na(n)=4
                  ddt(n)=alldth(i,j,5,1)
                  !-- WEIGHTING (LSQR)
                  ra(nnz+1:nnz+4)=ra(nnz+1:nnz+4)*wp*wse*wct(i,j,1)
                  ddt(n)=ddt(n)*wp*wse*wct(i,j,1)
                  !-- COUNT NONEZERO
                  nnz=nnz+4
                endif

                !-- CALCULATE UNWEIGHTED RMS
                orms=orms+alldth(i,j,5,1)**2
                ormscount=ormscount+1.
                !-- CALCULATE WEIGHTED RMS
                diffrms=diffrms+(alldth(i,j,5,1)*wp*wse*wct(i,j,1))**2
                rmscount=rmscount+wp*wse*wct(i,j,1)
              endif       
            endif         
            !-- FOR S WAVE
            !-- CHECK S WAVE EXIST OR NOT
            if(alldth(i,j,1,2)/=0. .or. alldth(i,j,2,2)/=0. .or. alldth(i,j,3,2)/=0. .or. alldth(i,j,4,2)/=0.)then
              if(wct(i,j,2)/=0.)then
                n=n+1 
                if(inv==1)then
                  !-- FOR SVD
                  a(n,(i-1)*4+1:(i-1)*4+4)=alldth(i,j,1:4,2)
                  a(n,eqtotal*4+1)=alldth(i,j,5,2)
                  !-- WEIGHTING (SVD)
                  a(n,(i-1)*4+1:(i-1)*4+4)=a(n,(i-1)*4+1:(i-1)*4+4)*ws*wse*wct(i,j,2)
                  a(n,eqtotal*4+1)=a(n,eqtotal*4+1)*ws*wse*wct(i,j,2)
                elseif (inv==2)then
                  !-- FOR LSQR
                  ra(nnz+1:nnz+4)=alldth(i,j,1:4,2)
                  do ii=1,4
                    ja(nnz+ii)=(i-1)*4+ii
                  enddo
                  na(n)=4
                  ddt(n)=alldth(i,j,5,2)
                  !-- WEIGHTING (LSQR)
                  ra(nnz+1:nnz+4)=ra(nnz+1:nnz+4)*ws*wse*wct(i,j,2)
                  ddt(n)=ddt(n)*ws*wse*wct(i,j,2)
                  !-- COUNT NONEZERO
                  nnz=nnz+4
                endif

                !-- CALCULATE UNWEIGHTED RMS
                orms=orms+alldth(i,j,5,2)**2
                ormscount=ormscount+1.
                !-- CALCULATE WEIGHTED RMS
                diffrms=diffrms+(alldth(i,j,5,2)*ws*wse*wct(i,j,2))**2
                rmscount=rmscount+ws*wse*wct(i,j,2)
              endif
            endif
          enddo        
!    endif  !-- ONLY FOR ISOLATED EARTHQUAKES
        enddo
      endif
  
      !-- NUMBER OF EQUATION IN A MATRIX
      print'(a29,1x,i7)'," total equations in A matrix:",n
 
      if (inv==1) then
        ! MEAN RESIDUAL & VARIENCE (ONLY FOR ERROR ESTIMATION OF SVD)
        !-- MEAN of (weighted) RESIDUAL
        resmean=0.
        do i=1,n-4
          resmean=resmean+a(i,eqtotal*4+1)
        enddo
        resmean=resmean/real(n-4)
        !-- VARIANCE
        var=0.
        var1=0.
        do i=1,n-4
          var=var+(a(i,eqtotal*4+1)-resmean)**2
          var1=var1+a(i,eqtotal*4+1)-resmean
        enddo
        var=(var-var1**2/real(n-4))
        var=var/real((n-4)-4*eqtotal)
      endif
 
      reswt(nit-1)=sqrt(diffrms/rmscount)
      res0(nit-1)=sqrt(orms/ormscount)
      if(nit>=2)then
        rtowt(nit-1)=(reswt(nit-1)-reswt(nit-2))/reswt(nit-2)*100.
        rto0(nit-1)=(res0(nit-1)-res0(nit-2))/res0(nit-2)*100.
        print'(a3,f10.4,1x,f6.2,a2,1x,a5,f10.4,1x,f6.2,a2)', &
             "RMS",res0(nit-1),rto0(nit-1)," %","RMSwt",reswt(nit-1),rtowt(nit-1)," %"
        if (abs(reswt(nit-1)-reswt(nit-2))<=rmscut .or. nit==nitmax+1)exit
      else
        print'(a3,f10.4,1x,a5,f10.4)',"RMS",res0(nit-1),"RMSwt",reswt(nit-1)
      endif

      deallocate(alldth)
  
      !---- ALLOCATE adj
      allocate(adj(4*eqtotal),stat=nerror) 
      if (nerror/=0) stop "adj is too large"


      !-- FOR SVD INVERSION
      if (inv==1)then
  
        !---- ALLOCATE s
        allocate(s(4*eqtotal),stat=nerror)
        if (nerror/=0) stop "s is too large"
  
        !---- ALLOCATE v
        allocate(v(4*eqtotal,4*eqtotal),stat=nerror)
        if (nerror/=0) stop "v is too large"

        !-- DETERMINE SINGULAR VALUE DECOMPOSITION (SVD) OF HYPOCENTER MATRIX
        call fksvd(a,s,v,n,4*eqtotal,1,.false.,.true.) 
        print*,"SVD OK."

        !-- CALCULATE ADJUSTMENTS AND APPLY TO HYPOCENTRAL PARAMETERS
        !-- DETERMINE NUMBER OF NON-ZERO SINGULAR VALUES
        nfre=0
        do i=1,4*eqtotal
          if(s(i).gt.eigtol) nfre=nfre+1
        enddo
        !-- CALCULATE ADJUSTMENTS
        do i=1,4*eqtotal
          adj(i)=0.0
          do j=1,nfre
            adj(i)=adj(i)+v(i,j)*a(j,4*eqtotal+1)/s(j)
          enddo
        enddo
        print*,"adjustments OK."

        !-- CHECK FOR POOR DEPTH CONSTRAINT
        do i=1,eqtotal
          if(s(4).le.eigtol) adj(4*i)=0.0
          adj(4*i)=adj(4*i)*0.75
        enddo

        if(nit.gt.5)then
          do i=1,4*eqtotal
            adj(i)=adj(i)*5.0/real(nit)
          enddo
        else if(nit.gt.10)then
          do i=1,4*eqtotal
            adj(i)=adj(i)*(5.0/real(nit))**2
          enddo
        endif
        if(ifixdepth.eq.1)then
          adj(4)=0.0
          evc(3)=fixdepth
        endif
        !-- ERROR ESTIMATION (HYPODD)
        !-- COVARIENCE MATRIX
        do i=1,4*eqtotal
          do j=1,i
            sumcvm= 0
            do k=1,4*eqtotal
              if(s(k).ne.0) sumcvm= sumcvm + v(i,k)*v(j,k) * (1/(s(k)*s(k)))
            enddo
            cvm(i,j)= sumcvm
            cvm(j,i)= sumcvm
          enddo
        enddo

        sterr=0.
        do i=1,4*eqtotal
          sterr(i)=sqrt(cvm(i,i))*sqrt(var)
        enddo

        do ii=1,eqtotal
          i=4*(ii-1)
          erz(ii)=sterr(i+4)
          call cal_delta(geoc_to_geog(eqlocation(2,ii)),eqlocation(1,ii)-0.5,geoc_to_geog(eqlocation(2,ii)),&
                         eqlocation(1,ii)+0.5,xdis)
          sterr(i+2)=sterr(i+2)*xdis   
          call cal_delta(geoc_to_geog(eqlocation(2,ii))-0.5,eqlocation(1,ii),geoc_to_geog(eqlocation(2,ii))+0.5,&
                         eqlocation(1,ii),xdis)
          sterr(i+3)=sterr(i+3)*xdis
          erh(ii)=sqrt(sterr(i+2)*sterr(i+2)+sterr(i+3)*sterr(i+3)+1.0e-8)
        enddo

        erh_avg=0.
        erz_avg=0.
        do i=1,eqtotal
          erh_avg=erh_avg+erh(i)
          erz_avg=erz_avg+erz(i)
        enddo
        erh_avg=erh_avg/real(eqtotal)
        erz_avg=erz_avg/real(eqtotal)

 
        deallocate(a)
        deallocate(s)
        deallocate(v)

      elseif(inv==2)then
        !-- FOR LSQR

        !--PARAMETER SETTING    
        ! damp=
        wantse=.true.
        atol=1.0e-4
        btol=1.0e-4
        conlim=1.0e7
        itnlim=4*(4*eqtotal)
        nout=46
        leniw=1
        lenrw=1
    
        !-- HYPODD PARAMETER
        ! wantse=.true.
        ! damp=
        ! atol=0.000001
        ! btol=0.000001
        ! conlim=100000.0
        ! itnlim=100*(4*eqtotal)
        ! nout=46
        ! leniw=2*(8*n)+1
        ! lenrw=8*n         
   
        allocate(iw(leniw))
        allocate(rw(lenrw))
        vv=0.
        ww=0.
        adj=0.
        se=0.
        istop = 0
        anorm = 0.0
        acond = 0.0
        rnorm = 0.0
        arnorm = 0.0
        xnorm = 0.0
        open(nout,file="lsqr.log",status="unknown")

        ! vv is not velocity vector but used in lsqr subroutine, adj is the output.
        print*,"LSQR INVERSION"
        call lsqr(n,4*eqtotal,damp,wantse,leniw,lenrw,iw,rw,ddt,vv,ww,adj,se,    &
                  atol,btol,conlim,itnlim,nout,istop,itn,anorm,acond,rnorm,arnorm,xnorm)
        print*,"LSQR OK"     

        deallocate(ddt)
        deallocate(iw)
        deallocate(rw)
      endif
    
      !-- NEW LOCATION	
      do neq=1,eqtotal
!print*,"looo",eqlocation(1,neq),eqlocation(2,neq),eqlocation(3,neq)      
        do i=1,3
          i1=(neq-1)*4+i+1
          eqlocation(i,neq)=eqlocation(i,neq)+adj(i1)
        enddo
        eqseco(neq)=eqseco(neq)+adj((neq-1)*4+1)
      enddo

      !--- YM Wu added here for CHECKING AIR AND WATER QUAKE
      do neq=1,eqtotal
        i1=0
        do i=2,nlon_c
          if(eqlocation(1,neq).gt.lon_c(i-1) .and. eqlocation(1,neq).le.lon_c(i))then
            i1=i
            exit
          endif  
        enddo
      
        i2=0
        xla=geoc_to_geog(eqlocation(2,neq))
        do i=2,nlat_c
          if(xla.gt.lat_c(i-1) .and. xla.le.lat_c(i))then
            i2=i
            exit
          endif  
        enddo
        if(i1.eq.0 .or. i2.eq.0)then
          eqlocation(3,neq)=4.0
          adj(4*neq)=0.0
        else
          dis1=dsqrt((eqlocation(1,neq)-lon_c(i1))**2   + (xla-lat_c(i2))**2   +0.0001)
          dis2=dsqrt((eqlocation(1,neq)-lon_c(i1))**2   + (xla-lat_c(i2-1))**2 +0.0001)
          dis3=dsqrt((eqlocation(1,neq)-lon_c(i1-1))**2 + (xla-lat_c(i2))**2   +0.0001)
          dis4=dsqrt((eqlocation(1,neq)-lon_c(i1-1))**2 + (xla-lat_c(i2-1))**2 +0.0001)
          xtopo=elev(i2,i1)/dis1+elev(i2-1,i1)/dis2+elev(i2,i1-1)/dis3+elev(i2-1,i1-1)/dis4
          xtopo=xtopo/(1./dis1+1./dis2+1./dis3+1./dis4)
          if(eqlocation(3,neq) .lt. -4.0  .and. ifixdepth.eq.0)eqlocation(3,neq)=-4.0
          if(eqlocation(3,neq) .lt. xtopo .and. ifixdepth.eq.0)eqlocation(3,neq)=xtopo
        endif
      enddo
      !--- END CHECKING AIR AND WATER QUAKE	
    
      deallocate(adj)

    enddo   !-- ITERATION

    !-- RESIDUAL
    open(40,file="rms.out")
    do i=0,nit-1
      write(40,'(i3,2x,2(f8.4,1x,f6.2,a2,1x))')i,res0(i),rto0(i)," %",reswt(i),rtowt(i),"%"
    enddo
  

    !-- ERROR OF INITIAL LOCATION (IN CATALOG)
    erhh=0.
    erzz=0.
    do i=1,eqtotal
      erhh= erhh+erh_ori(i)
      erzz= erzz+erz_ori(i)
    enddo
    erhh=erhh/real(eqtotal)
    erzz=erzz/real(eqtotal)
    write(40,'(a24,2f10.4)')"avg erh & erz (initial):",erhh,erzz
    write(40,'(a26,2f10.4)')"avg erh & erz (relocated):",erh_avg,erz_avg
    close(40)

    do i=1,eqtotal
      call timescal(iysto(i),imsto(i),idsto(i),ihrsto(i),mino(i),eqseco(i))
    enddo

    !-- FINAL LOCATION
    open(4,file=pfile,status='old')
    ofile=trim(pfile)//".hout"
    open(30,file=ofile)
    ofile=trim(pfile)//".dout"
    open(31,file=ofile)
    do neq=1,eqtotal
      nstations=allnstations(neq)
!      if(nstations.gt.99) nstations=99
      xla=geoc_to_geog(eqlocation(2,neq))
      write(30,100)iysto(neq),imsto(neq),idsto(neq),ihrsto(neq),mino(neq),int(eqseco(neq)*100),xla,&
                   eqlocation(1,neq),eqlocation(3,neq),xmag(neq),0.0,erh(neq),erz(neq),neq,nstations
      write(31,110)iysto(neq),imsto(neq),idsto(neq),ihrsto(neq),mino(neq),eqseco(neq),int(xla),(xla-int(xla))*60.,&
                   int(eqlocation(1,neq)),(eqlocation(1,neq)-int(eqlocation(1,neq)))*60.,eqlocation(3,neq),&
                   xmag(neq),nstations,epdis(1),0,0.,erh(neq),erz(neq),'F','3DD'
      call output_pfile(stname,stnumber,card)
      do j=1,allnstations(neq)
        write(card(j)(2:19),'(a4,f6.1,2i4)')stname(neq,j),epdis(j),nint(azimuth(j)),nint(takeoff(j))
        write(card(j)(30:34),'(f5.2)')res(j,1)
        write(card(j)(46:50),'(f5.2)')res(j,2)
        write(31,'(a)')card(j)
      enddo
    enddo
    close(30)
    close(31)
    close(4)
    print*,"DONE."

100 format(i4.4,i2.2,i2.2,1x,i2.2,i2.2,i4.4,2f10.3,f8.2,f5.1,f6.2,2f6.1,i6,i3)
110 format(1x,i4,4i2,f6.2,i2,f5.2,i3,f5.2,f6.2,f4.2,i3,f5.1,i3,f4.2,2f4.1,1x,a1,1x,a3)

end program main

subroutine input_pfile(pfile,stname,stnumber,iskip)
    !  this routine reads in the p travel times for all stations observing
    !  the current event.  first trial hypocentral parameters are read, then
    !  station names and weights and travel times.  arrivals at stations not
    !  in the station list are discarded; unnormalized weights are calculated.
    use all
    use observ
    use events
    implicit real*8 (a-h,o-z)

    !  local variables:
    character*64 pfile
    character*1 af
    real*8 dep
!   character pfile*64
    character sta*4
    real*8 xla,geog_to_geoc
    integer ierr,iskip,iptime,zzz
    character*4 stname(maxevt,maxsta)
    integer stnumber(maxevt,maxsta),stamin
    logical alive

    xpga=0.0 
    secp=0.0
    ifm=0
    epdis=0.0
    azimuth=0.0
    takeoff=0.0
    wa=0.0
    inten=0
    xweio=0.0
    res=0.0
    isp=0
    xmls=0.0
    nobs=0
!    nstations=0
 
    !-- Loading pfile time
    inquire(file=pfile,exist=alive)
    if(.not. alive)then
      stop 'Error : pfile is not found!'
    endif
    if(neq==1)open(4,file=pfile,status='old')


    ! read in pfile header
    read(4,4001)iyr,imo,iday,ihr,mino(neq),seco,ltde,eltm,lnde,elnm,dep,xmag(neq),xrms_ori,xerh_ori,xerz_ori
    4001 format(1x,i4,4i2,f6.2,i2,f5.2,i3,f5.2,f6.2,f4.1,t55,f4.2,2f4.1)

    iysto(neq)=iyr
    imsto(neq)=imo
    idsto(neq)=iday
    ihrsto(neq)=ihr
    iptime=iyr*10000+imo*100+iday

    !  store event coordinates
    evc(1)=lnde+elnm/60.0
    xla=ltde+eltm/60.0
    evc(2)=geog_to_geoc(xla)
!    if (dep==0.0) dep=dep+1.0
    evc(3)=dep
    !-- STORE ORIGINAL EVENT LOCATIONS & OCCUR TIMES
    if (nit==1 .and. neq==1) open(20,file="ini.out")
    if (nit==1) then
      erh_ori(neq)=xerh_ori
      erz_ori(neq)=xerz_ori
      rms_ori(neq)=xrms_ori
      write(20,'(4f10.4,2x,i4,4(1x,i2),1x,f6.2,2f10.4)')evc(1),geoc_to_geog(evc(2)),evc(3),&
            xmag(neq),iyr,imo,iday,ihr,mino(neq),seco,xerh_ori,xerz_ori
    endif
    if (nit==1 .and. neq==eqtotal) close(20)


    !  read in travel times to stations in groups of six
    !-- Reading station
!    minp=-1
    xminparr=999.9
    ikeysta=0
    number=0
    nstations=0
    zzz=1
    do
      read(4,'(1x,a4,f6.1,8x,a1,i3,f6.2,5x,f5.2,f6.2,5x,f5.2,f5.2,f5.2,&
           f5.2,f5.2,1x,i1,f6.1)',iostat=iresult)sta,delta,af,ipmi,parr,pwei,&
           sarr,swei,wapeak,wapeak1,xml,xml1,intensity,xpeak
!print*,"ffff",nstations,sta(1:len_trim(sta)),stn(k)(1:len_trim(stn(k))),iptime,stt(1,k),stt(2,k),sta,number,ipmi
      if(iresult < 0)exit
      if(ichar(sta(1:1)).ge.ichar("0").and.ichar(sta(1:1)).le.ichar("9"))then
        backspace(4)
        exit
      endif
!print*,"gggg",nstations,iptime,stt(1,k),stt(2,k),sta,number
      number=number+1
      if(delta.gt.0.0 .and. abs(wapeak).gt.9.9)wapeak=0.0
      if(delta.gt.0.0 .and. abs(wapeak1).gt.9.9)wapeak1=0.0
      
!print*,"hhhh",nstations,iptime,stt(1,k),stt(2,k),sta,number
!      ikey=0
!      do k=1,nsts
!        if(iptime.lt.stt(1,k).or.iptime.gt.stt(2,k)) cycle
!        if( sta(1:len_trim(sta)) .eq. stn(k)(1:len_trim(stn(k))) )then
!          ikey=k
!          exit
!        endif
!      enddo
!      if(ikey .eq. 0)cycle
!-- NEW CODE
!      if(minp .eq. -1)then
!        minp=ipmi
!        if(mino(neq) .eq. minp)then
!          seco=seco
!          mino(neq)=minp 
!          if(mino(neq) .NE. minp)then
!            if(mino(neq) .gt. minp)then                ! mino > minp
!              if(mino(neq) .gt. (minp+30))then
!              ihrsto(neq)=ihrsto(neq)+1
!              seco=60.0*(mino(neq)-60-minp)+seco
!              mino(neq)=minp
!              else
!              seco=60.*(mino(neq)-minp)+seco
!              mino(neq)=minp
!              endif
!            else
!              if(minp .gt. (mino(neq)+30))then         ! minp > mino
!              ihrsto(neq)=ihrsto(neq)-1
!              seco=60.0*(60.0+mino(neq)-minp)+seco
!              mino(neq)=minp
!              else
!              seco=60.*(mino(neq)-minp)+seco
!              mino(neq)=minp
!              endif
!            endif
!          endif
!        endif
!      endif

! ------ modified by Yu-Fang Hsu on 2020/9/28 to avoid header time error 
      ikey=0
      do k=1,nsts
        if(iptime.lt.stt(1,k) .or. iptime.gt.stt(2,k)) cycle
        if( sta(1:len_trim(sta)) .eq. stn(k)(1:len_trim(stn(k))) )then
          ikey=k
          exit
        endif
      enddo

      minp=ipmi
          if(pwei .lt. 4.5)then
            if(parr .NE. 0 .or. pwei .NE. 0)then
              if(mino(neq) .eq. minp)then
              minp=mino(neq) 
              ipmi=mino(neq) 
              else
                if(mino(neq) .gt. minp)then                ! mino > minp
                  if(mino(neq) .gt. (minp+50))then
                  ihrsto(neq)=ihrsto(neq)
                  parr=60.0*(minp+60-mino(neq))+parr
                    if(sarr .gt.0.)then 
                    sarr=60.0*(minp+60-mino(neq))+sarr
                    endif
                  minp=mino(neq)
                  ipmi=mino(neq) 
                  else
                  parr=60.*(minp-mino(neq))+parr
                    if(sarr .gt.0.)then
                    sarr=60.*(minp-mino(neq))+sarr
                    endif
                  minp=mino(neq)
                  ipmi=mino(neq) 
                  endif
                else
                  if(minp .gt. (mino(neq)+50))then         ! minp > mino
                  ihrsto(neq)=ihrsto(neq)
                  parr=60.0*(-60.0+minp-mino(neq))+parr
                    if(sarr .gt.0.)then
                    sarr=60.0*(-60.0+minp-mino(neq))+sarr
                    endif
                  minp=mino(neq)
                  ipmi=mino(neq) 
                  else
                  parr=60.*(minp-mino(neq))+parr
                    if(sarr .gt.0.)then
                    sarr=60.*(minp-mino(neq))+sarr
                    endif
                  minp=mino(neq)
                  ipmi=mino(neq) 
                  endif
                endif
              endif
            else


              ! P arrival and S arrival of S-Ptime
              if(mino(neq) .eq. minp)then
              minp=mino(neq) 
              ipmi=mino(neq) 
              else
                if(mino(neq) .gt. minp)then                ! mino > minp
                  if(mino(neq) .gt. (minp+50))then
                  sarr=60.0*(minp+60-mino(neq))+sarr
                  minp=mino(neq)
                  ipmi=mino(neq) 
                  else
                  sarr=60.*(minp-mino(neq))+sarr
                  minp=mino(neq)
                  ipmi=mino(neq) 
                  endif
                else
                  if(minp .gt. (mino(neq)+50))then         ! minp > mino
                  sarr=60.0*(-60.0+minp-mino(neq))+sarr
                  minp=mino(neq)
                  ipmi=mino(neq) 
                  else
                  sarr=60.*(minp-mino(neq))+sarr
                  minp=mino(neq)
                  ipmi=mino(neq) 
                  endif
                endif
              endif
            endif
          endif
!print*,"new",minp,seco,parr,sarr
! ------

!print*,"jjjj",nstations,iptime,stt(1,k),stt(2,k),sta,stn(k),number
!      ikey=0
!      do k=1,nsts
!        if(iptime.lt.stt(1,k) .or. iptime.gt.stt(2,k)) cycle
!        if( sta(1:len_trim(sta)) .eq. stn(k)(1:len_trim(stn(k))) )then
!          ikey=k
!          exit
!        endif
!      enddo

!print*,"dddd",ikey, sta(1:len_trim(sta)), stn(k)(1:len_trim(stn(k))),nstations,iptime,stt(1,k),stt(2,k),sta,stn(k)

      if(ikey .eq. 0)cycle
      if(parr.gt.0.0 .and. parr.lt.xminparr)then
        xminparr=parr
        ikeysta=ikey
      endif
      
!print*,"kkkk",ikey,nstations,iptime,stt(1,k),stt(2,k),sta,stn(k),number
      if(sarr.gt.0.0 .and. sarr.lt.xminparr)then
        xminparr=sarr
        ikeysta=ikey
      endif

      if(wapeak.eq.0.0 .and. xml.ne.0.0 .and. delta.gt.0.0)then
        dep=evc(3)
        wapeak=xml-pa0(delta,dep)
      endif

      if(wapeak1.eq.0.0 .and. xml1.ne.0.0 .and. delta.gt.0.0)then
        dep=evc(3)
        wapeak1=xml1-pa0(delta,dep)
      endif
  
      nstations=nstations+1 
      !-- nstations MEANS STATION WHICH HAVE STATION INFORMATION IN .sta FILE

      if(parr .gt. 0.0)then
        if(pwei.gt.4.5)then
          !-- for adding S-P time interval to join location by Ludan
          nobs=nobs+1
          wa(nstations)=wapeak
          wa1(nstations)=wapeak1
          inten(nstations)=intensity
          xpga(nstations)=xpeak
          xweio(nstations,1)=pwei
          xmls(nstations)=xml
          xmls1(nstations)=xml1
          isto(nstations)=ikey
          secp(nstations,1)=parr
          ifm(nstations)=0
          if(af.eq.'+')ifm(nstations)=1
          if(af.eq.'-')ifm(nstations)=-1
        else
          nobs=nobs+1
          wa(nstations)=wapeak
          wa1(nstations)=wapeak1
          inten(nstations)=intensity
          xpga(nstations)=xpeak
          xweio(nstations,1)=pwei
          xmls(nstations)=xml
          xmls1(nstations)=xml1
          isto(nstations)=ikey
          secp(nstations,1)=parr
          ifm(nstations)=0
          if(af.eq.'+')ifm(nstations)=1
          if(af.eq.'-')ifm(nstations)=-1
        endif
      endif
  
      if(sarr .gt. 0.0)then
        nobs=nobs+1
        wa(nstations)=wapeak
        wa1(nstations)=wapeak1
        inten(nstations)=intensity
        xpga(nstations)=xpeak
        xweio(nstations,2)=swei
        xmls(nstations)=xml
        xmls1(nstations)=xml1
        isto(nstations)=ikey
        secp(nstations,2)=sarr
      endif
  
      !-- FOR A PRIORI WEIGHTING
      if(int(pwei)==0)then
        wct(neq,nstations,1)=pri(1)
      elseif(int(pwei)==1)then
        wct(neq,nstations,1)=pri(2)
      elseif(int(pwei)==2)then
        wct(neq,nstations,1)=pri(3)
      elseif(int(pwei)==3)then
        wct(neq,nstations,1)=pri(4)
      elseif(int(pwei)==4)then
        wct(neq,nstations,1)=pri(5)
      endif
      if(int(swei)==0)then
        wct(neq,nstations,2)=pri(1)
      elseif(int(swei)==1)then
        wct(neq,nstations,2)=pri(2)
      elseif(int(swei)==2)then
        wct(neq,nstations,2)=pri(3)
      elseif(int(swei)==3)then
        wct(neq,nstations,2)=pri(4)
      elseif(int(swei)==4)then
        wct(neq,nstations,2)=pri(5)
      endif

      !-- SAVE THE STATION NAME OF EACH EARTHQUAKE
      stname(neq,nstations)=sta
      stnumber(neq,nstations)=ikey
!print*,"uuuu",nstations,stt(1,k),stt(2,k),sta,ikey,number    

    enddo
  
    allnstations(neq)=nstations
    if(neq==eqtotal)close(4)
 
    if(nobs .lt. 4) then
      print*,'Phases less than 4 for doing location, SKIP!'
      iskip=1
    endif

    if(ltde.eq.0 .and. lnde.eq.0)then
      if(ikeysta.gt.0)then
        evc(1)=stc(1,ikeysta)+0.01
        evc(2)=stc(2,ikeysta)+0.01
        evc(3)=10.0
      else
        evc(1)=stc(1,1)+0.01
        evc(2)=stc(2,1)+0.01
        evc(3)=10.0
      endif
    endif
end subroutine input_pfile

subroutine input_sta
    use all
    use observ
    use ray
    implicit real*8 (a-h,o-z)

    logical alive
    integer it1,it2,l,m
    real*8 xlo,xla,xelev,geog_to_geoc
        
    !  this routine reads in the station list, sets up the
    !  coordinate system, and calculates the stations' cartesian coordinates.
    !  subroutines required: setorg; disto;
    !  common block variables:
    pcor=0.0
    scor=0.0
    spcor=0.0

    inquire(file=stafile,exist=alive)
    if(alive)then
      open(1,file=stafile,status='old')
    else
      stop 'Error: station file not exists under current directory!'
    endif

    nsts=0
    j=0
    !-- read in station list
    do 
      j=j+1
      if(j > maxsta)exit
!      read(1,'(a4,i3,f5.2,i4,f5.2,f7.1,3f5.2,2(1x,i8)i)',iostat=iresult)stn(j),ltds(j),sltm(j), &
!           lnds(j),slnm(j),xelev,pcor(j),scor(j),spcor(j),it1,it2
      read(1,*,iostat=iresult)stn(j),xlo,xla,xelev,it1,it2

      if(iresult < 0)exit
      call locpt(xlo,xla,bx,by,5,l,m)
      if(l .gt. 0)then      
        !-- STORE STATION COORDINATES
!        if (lnds(j)>=0) then
!          stc(1,j)=lnds(j)+slnm(j)/60.0
!        else
!          stc(1,j)=lnds(j)-slnm(j)/60.0
!        endif
!        if (ltds(j)>=0) then
!          xla=ltds(j)+sltm(j)/60.0
!        else
!          xla=ltds(j)-sltm(j)/60.0
!        endif
        stc(1,j)=xlo 
        stc(2,j)=geog_to_geoc(xla)
        stc(3,j)=-xelev*1.0e-3
        stt(1,j)=it1
        stt(2,j)=it2
        nsts=j
      else
        j=j-1
      endif
    enddo
    close(1)
end subroutine input_sta

subroutine output_pfile(stname,stnumber,card)
    use all
    use ray
    use observ
    use events
    implicit real*8 (a-h,o-z)
    real*8 xe,ye,ze,xr,yr,zr
    real*8 w(3,16385)
    integer npoints,i,stmi
    integer stnumber(maxevt,maxsta)
    character*4 stname(maxevt,maxsta)
    real*8 ptime,stime,dpi,epid,azi
    character*90 card(maxsta)
    dpi=asin(1.)/90.

    !-- skip the pfile header
    read(4,*)

    !-- Reading phase information
    xe=eqlocation(1,neq)
    ye=eqlocation(2,neq)
    ze=eqlocation(3,neq)
    i=1
    do
      read(4,'(a)',iostat=iresult)card(i)
      if (iresult < 0) exit
      if(ichar(card(i)(2:2)).ge.ichar("0").and.ichar(card(i)(2:2)).le.ichar("9"))then
        backspace(4)
        exit
      endif
      imatch=0
      do j=1,allnstations(neq)
         if (trim(card(i)(2:6)).eq.stname(neq,j)) then
            imatch=1
            exit
         endif
      enddo
      if (imatch.eq.0) cycle
      ns=stnumber(neq,i)
      xr=stc(1,ns)
      yr=stc(2,ns)
      zr=stc(3,ns)
      call distaz(90.-ye,xe,90.-yr,xr,epid,azi)
      epid=epid*dpi*ro  !-- Epicentral distance
      epdis(i)=epid                
      azimuth(i)=nint(azi)  
      read(card(i)(22:50),'(i2,f6.2,5x,f5.2,f6.2)')stmi,secp(i,1),xweio(i,1),secp(i,2)                
      if(secp(i,1) .gt. 0.0 .and. xweio(i,1).lt. 4.5 )then
        !-- For P arrival cases
        ips=1  !-- 1 for P velocity & 2 for S velocity
        call pbr(ye,xe,ze,yr,xr,zr,w,npoints,ptime)
        tkofag=angle(w,1,2)
        takeoff(i)=nint(tkofag)                     
	res(i,1)=(secp(i,1)+real(stmi*60))-(eqseco(neq)+real(mino(neq)*60))-ptime-pcor(ns)
!print*,"rres",res(i,1)
        if (res(i,1).gt.86000.) res(i,1)=res(i,1)-86400.
!print*,'1 ',stname(neq,i),res(i,1),ptime,stmi,mino(neq),secp(i,1),eqseco(neq)
        if (res(i,1).gt.3500.) res(i,1)=res(i,1)-3600.
!print*,'1 ',stname(neq,i),res(i,1),ptime,stmi,mino(neq),secp(i,1),eqseco(neq)
      else if(secp(i,1) .gt. 0.0 .and. xweio(i,1).gt. 4.5 )then
        !-- For S-P time differences case
        !-- For P wave
        ips=1  !-- 1 for P velocity & 2 for S velocity
        call pbr(ye,xe,ze,yr,xr,zr,w,npoints,ptime)
        tkofag=angle(w,1,2)
        takeoff(i)=nint(tkofag)

        !-- for S wave
        ips=2  !-- 1 for P velocity & 2 for S velocity
        call pbr(ye,xe,ze,yr,xr,zr,w,npoints,stime)
        tkofag=angle(w,1,2)

        secp(i,2)=0.0
        res(i,2)=0.0
        xweio(i,2)=0.0
      else
        secp(i,1)=0.0
        res(i,1)=0.0
        xweio(i,1)=0.0
      endif

      if(secp(i,2) .gt. 0.0 )then
        !-- For S arrival cases	    	
        ips=2  !-- 1 for P velocity & 2 for S velocity
        call pbr(ye,xe,ze,yr,xr,zr,w,npoints,stime) 
        tkofag=angle(w,1,2)
	res(i,2)=(secp(i,2)+real(stmi*60))-(eqseco(neq)+real(mino(neq)*60))-stime-scor(ns)
        if (res(i,2).gt.86000.) res(i,2)=res(i,2)-86400.
        if (res(i,2).gt.3500.) res(i,2)=res(i,2)-3600.
!print*,'2 ',stname(neq,i),res(i,2),stime,stmi,mino(neq),secp(i,2),eqseco(neq)
      else
	secp(i,2)=0.0
	res(i,2)=0.0
	xweio(i,2)=0.0
      endif 
      i=i+1 
    enddo
  
    if(neq==eqtotal)close(4)

end subroutine output_pfile

real*8 function geog_to_geoc(xla)
    implicit none
    real*8 xla,RAD_PER_DEG,B2A_SQ
    RAD_PER_DEG=0.0174532925199432955
    B2A_SQ=0.993305521
    geog_to_geoc = atan(B2A_SQ*tan(RAD_PER_DEG*xla)) / RAD_PER_DEG
    return
end function geog_to_geoc

real*8 function geoc_to_geog(xla)
    implicit none
    real*8 xla,RAD_PER_DEG,B2A_SQ
    RAD_PER_DEG=0.0174532925199432955
    B2A_SQ=0.993305521
    geoc_to_geog = atan(tan(RAD_PER_DEG*xla)/B2A_SQ) / RAD_PER_DEG
    return
end function geoc_to_geog 

subroutine input_vel(velfile)
    use ray
    implicit real*8 (a-h,o-z)
    logical alive
    character*64 velfile

    ! input the number of gridpoints in x, y and z directions
    inquire(file=velfile,exist=alive)
    if(alive)then
      open(1,file=velfile,status='old')
    else
      stop 'Error: velocity model does not exist under current directory!'
    endif

    read(1,*)bldh,bldv,nlon_c,nlat_c,ndep_c
    read(1,*)(lon_c(i),i=1,nlon_c)
    read(1,*)(lat_c(i),i=1,nlat_c)
    read(1,*)(dep_c(i),i=1,ndep_c)
    !-- READ P VELOCITY MODEL
    do k=1,ndep_c
      do j=1,nlat_c
        read(1,*)(vel_p(i,j,k),i=1,nlon_c)
      enddo
    enddo 
 
    !-- READ S VELOCITY MODEL
    do k=1,ndep_c
      do j=1,nlat_c
        read(1,*)(vel_s(i,j,k),i=1,nlon_c)
      enddo
    enddo

    call bldmap
    nxyz_c=nlon_c*nlat_c*ndep_c
    nxy_c=nlon_c*nlat_c
    nx_c=nlon_c
    nxyz2_c=(nlon_c-2)*(nlat_c-2)*(ndep_c-2)
    nxy2_c=(nlon_c-2)*(nlat_c-2)
    nx2_c=nlon_c-2
    ave=0.0
    do i=1,nlat_c
      ave=ave+lat_c(i)
    enddo
    ave=ave/real(nlat_c)
    ro=earthr(ave)

! set boundary of local model
    bx(1)=lon_c(1)
    bx(3)=lon_c(nlon_c)
    by(1)=lat_c(1)
    by(3)=lat_c(nlat_c)
    bx(2)=bx(1)
    by(2)=by(3)
    bx(4)=bx(3)
    by(4)=by(1)
    bx(5)=bx(1)
    by(5)=by(1)
end subroutine input_vel

subroutine bldmap
    use ray
    implicit real*8 (a-h,o-z)
    real*8 lon_now,lat_now,dep_now

    !-- for crustal velocity
    
    lon1_c=bldh-lon_c(1)
    ilonmax=(1e-10)+(lon_c(nlon_c)+lon1_c)/bldh
    lat1_c=bldh-lat_c(1)
    ilatmax=(1e-10)+(lat_c(nlat_c)+lat1_c)/bldh
    dep1_c=bldv-dep_c(1)
    idepmax=(1e-10)+(dep_c(ndep_c)+dep1_c)/bldv
    if ((ilonmax.gt.ilondeg).or.(ilatmax.gt.ilatdeg).or.(idepmax.gt.idepkm)) then
      print*,"Error, model dimension out of range!"
      stop
    endif
    ilon=1
    do i=1,ilonmax
      ilon1=ilon+1
      lon_now=float(i)*bldh-lon1_c
      if (lon_now.ge.lon_c(ilon1)) ilon=ilon1
      ilonloc_c(i)=ilon
    enddo

    do i=ilonmax+1,ilondeg
      ilonloc_c(i)=0
    enddo
    ilat=1

    do i=1,ilatmax
      ilat1=ilat+1
      lat_now=float(i)*bldh-lat1_c
      if (lat_now.ge.lat_c(ilat1)) ilat=ilat1
      ilatloc_c(i)=ilat
    enddo
    do i=ilatmax+1,ilatdeg
      ilatloc_c(i)=0
    enddo
    idep=1
    do i=1,idepmax
      idep1=idep+1
      dep_now=float(i)*bldv-dep1_c
      if (dep_now.ge.dep_c(idep1)) idep=idep1
      ideploc_c(i)=idep
    enddo
    do i=idepmax+1,idepkm
      ideploc_c(i)=0
    enddo
end subroutine bldmap

subroutine intmap_3d(lon,lat,dep,ip,jp,kp)
    use ray
    implicit real*8 (a-h,o-z)
    real*8 lon,lat,dep
    ip=int(1e-10+(lon+lon1_c)/bldh)
    jp=int(1e-10+(lat+lat1_c)/bldh)
    kp=int(1e-10+(dep+dep1_c)/bldv)
!print*,"oooo",stn,lon,lat,dep,lon1_c,lat1_c,dep1_c,ip,jp,kp
    if ((ip.le.0).or.(jp.le.0).or.(kp.le.0)) then
      print*,"Error,lon,lat,dep out of range!"
      print*,"lon=",lon,"lat=",lat,"dep=",dep
      print*,"ip,jp,kp",ip,jp,kp
      stop
    endif 

    ip=ilonloc_c(ip)
    jp=ilatloc_c(jp)
    kp=ideploc_c(kp)
    if ((ip.eq.0).or.(jp.eq.0).or.(kp.eq.0)) then
      print*,"Error,crust lon,lat out of range!"
      print*,"lon=",lon,"lat=",lat,"dep=",dep
      print*,"ip,jp,kp",ip,jp,kp
      stop
    endif
    return
end subroutine intmap_3d

function velocity(r,pa,ra)
    use ray
    implicit real*8 (a-h,o-z)
    real*8 lat,lon,dep,shiftlo
    real*8 r,pa,ra,r2d
    real*8 v,velocity
    common /coord/ shiftlo
    r2d = 90./asin(1.)
    lat=geoc_to_geog(90.0-pa*r2d)
    lon=ra*r2d+shiftlo
    dep=ro-r
    call vel_3d(lon,lat,dep,v)
    velocity=v
    return
end function velocity

subroutine vel_3d(lon,lat,dep,v)
    use ray
    implicit real*8 (a-h,o-z)
    real*8 lon,lat,dep,v
    real*8 lonf,lonf1,latf,latf1,depf,depf1
    real*8 wv(2,2,2)
    common /weight/ wv,ip,jp,kp
    if (lon.lt.lon_c(1)) lon=lon_c(1)+1e-7
    if (lon.gt.lon_c(nlon_c)) lon=lon_c(nlon_c)-1e-7
    if (lat.lt.lat_c(1)) lat=lat_c(1)+1e-7
    if (lat.gt.lat_c(nlat_c)) lat=lat_c(nlat_c)-1e-7
    if (dep.lt.dep_c(1)) dep=dep_c(1)+1e-7
    if (dep.gt.dep_c(ndep_c)) dep=dep_c(ndep_c)-1e-7
    call intmap_3d(lon,lat,dep,ip,jp,kp)
    ip1=ip+1
    jp1=jp+1
    kp1=kp+1
    if((ip1.gt.nlon_c).or.(jp1.gt.nlat_c).or.(kp1.gt.ndep_c))then
      print*,"Error, ip1,jp1,kp1 out of range!"
      print*,"ip1=",ip1,"jp1=",jp1,"kp1=",kp1
      stop
    endif
    lonf=(lon-lon_c(ip))/(lon_c(ip1)-lon_c(ip))
    latf=(lat-lat_c(jp))/(lat_c(jp1)-lat_c(jp))
    depf=(dep-dep_c(kp))/(dep_c(kp1)-dep_c(kp))
    lonf1=1.0-lonf
    latf1=1.0-latf
    depf1=1.0-depf
    wv(1,1,1)=lonf1*latf1*depf1
    wv(2,1,1)=lonf*latf1*depf1
    wv(1,2,1)=lonf1*latf*depf1
    wv(2,2,1)=lonf*latf*depf1
    wv(1,1,2)=lonf1*latf1*depf
    wv(2,1,2)=lonf*latf1*depf
    wv(1,2,2)=lonf1*latf*depf
    wv(2,2,2)=lonf*latf*depf
    if(ips.eq.2)then
      v= wv(1,1,1)*vel_s(ip,jp,kp)+wv(2,1,1)*vel_s(ip1,jp,kp)   &
      +wv(1,2,1)*vel_s(ip,jp1,kp) +wv(2,2,1)*vel_s(ip1,jp1,kp)  &
      +wv(1,1,2)*vel_s(ip,jp,kp1) +wv(2,1,2)*vel_s(ip1,jp,kp1)  &
      +wv(1,2,2)*vel_s(ip,jp1,kp1)+wv(2,2,2)*vel_s(ip1,jp1,kp1)
    else
      v= wv(1,1,1)*vel_p(ip,jp,kp)+wv(2,1,1)*vel_p(ip1,jp,kp)   &
      +wv(1,2,1)*vel_p(ip,jp1,kp) +wv(2,2,1)*vel_p(ip1,jp1,kp)  &
      +wv(1,1,2)*vel_p(ip,jp,kp1) +wv(2,1,2)*vel_p(ip1,jp,kp1)  &
      +wv(1,2,2)*vel_p(ip,jp1,kp1)+wv(2,2,2)*vel_p(ip1,jp1,kp1)
    endif
    return
end subroutine vel_3d

subroutine pbr(evla,evlo,evdp,stla,stlo,stel,w,np,tk)
  use ray
  implicit real*8(a-h,o-z)
  parameter (msg=16384)
  real*8 w(3,msg+1)
  real*8 r(msg+1), a(msg+1), b(msg+1)
  integer ni,i
  real*8 shiftlo
  real*8 aas,bbs,hs,aar,bbr,hr
  real*8 xfac,flim,mins
  real*8 dpi,r2d
  real*8 velocity,rtim
  real*8 tk
  real*8 as,ar
  real*8 bre,bso,dlo
  real*8 ad,rs,rr
  real*8 x1,y1,z1,x2,y2,z2,x3,y3,z3,dx,dy,dz
  real*8 r1,a1,b1,r2,a2,b2,r3,a3,b3
  real*8 x,y,z,acosa,sina,cosa,to,tp
  real*8 dn,ddn,dr,da,db
  real*8 dseg,ddseg
  real*8 v1,v2,v3
  real*8 upz,dwz
  real*8 vr1,vr2,vr,vb1,vb2,vb,va1,va2,va
  real*8 pr,pa,pb
  real*8 vrd,rvr,rva,rvb,rvs
  real*8 cc,rcur,rdr,rda,rdb,rpr,ap,bp
  real*8 adV,bdV,rdV
  real*8 RNULL
      
  common /coord/ shiftlo
  data    RNULL /0.0e10/
! right now force the receiver at elevation of 0
!      hr=0.0
!      write(*,*)aas, bbs, hs, aar, bbr, hr
!                                                                       
!       parameters for calculation                                      
!         xfac   = enhancement factor (see um and thurber, 1987)        
!         nloop  = number of bending iterations
!         n1, n2 = min & max of ray segments
!         mins  = min. length of segment (km)
!       initialization                                                  

  aas=evla
  bbs=evlo
  hs=evdp
  aar=stla
  bbr=stlo
  hr=stel

  ni     = msg+1
  xfac   = 1.5   !-- 1.2 to 1.5, James 1.5, Hypo3d 1.2
  n1     = 2
  n2     = msg
 ! n2     = 128
  nloop  = 6400
 ! nloop  = 3200
  flim   = 1.e-4/100.
  mins   = 2.0
  dpi = asin(1.)/ 90.
  r2d = 90./asin(1.)

!-- Check coordinates
  if(aas.LT.-90.OR.aas.GT.90.)then
    write(*,*)'Latitude of source is out of range'
    stop
  endif
  if(aar.LT.-90.OR.aar.GT.90.)then
    write(*,*)'Latitude of station is out of range'
    stop
  endif
  if(bbs.LT.-180.OR.bbs.GT.180.)then
!    print*,bbs,evlo
    write(*,*)'Longitude of source is out of range'
    stop
  endif
  if(bbr.LT.-180.OR.bbr.GT.180.)then
    write(*,*)'Longitude of station is out of range'
    stop
  endif

!-- longitude and latitude range from 0 to 180. 
!-- This program does not work with angles
!-- greater than 180.       

!-- Pass from latitude to colatitude

  as = (90.00-aas) * dpi
  ar = (90.00-aar) * dpi

  if(bbr.LT.0.0)then
    bre=360.+bbr
  else
    bre=bbr
  endif

  if(bbs.LT.0.0)then
    bso=360.+bbs
  else
    bso=bbs
  endif
  dlo=abs(bso-bre)

  if(dlo.LT.180.)then
    shiftlo=0.0e10
    if(bso.LT.bre)then
      shiftlo=bso-(180.-dlo)/2.
      bbs=(180.-dlo)/2.
      bbr=bbs+dlo
    else
      shiftlo=bre-(180.-dlo)/2.
      bbr=(180.-dlo)/2.
      bbs=bbr+dlo
    endif
  else
    dlo=360.0000-dlo
    shiftlo=0.0e10
    if(bso.LT.bre)then
      shiftlo=bso-(dlo+(180.-dlo)/2.)
      bbs=(180.-dlo)/2.+dlo 
      bbr=bbs-dlo
    else    
      shiftlo=bre-(dlo+(180.-dlo)/2.)
      bbr=(180.-dlo)/2.+dlo
      bbs=bbr-dlo
    endif
  endif

  bs = bbs * dpi
  br = bbr * dpi  
  ad = (as + ar) / 2.                                               
  rs = ro - hs
  rr = ro - hr                                                      
!
! *** initial straight ray ***                                           
!     ni : number of ray segments
  ni = n1
  x1 = rs*sin(as)*cos(bs)                                       
  y1 = rs*sin(as)*sin(bs)                                       
  z1 = rs*cos(as)                                               
  x2 = rr*sin(ar)*cos(br)                                       
  y2 = rr*sin(ar)*sin(br)                                       
  z2 = rr*cos(ar)       
  dx = x2-x1
  dy = y2-y1
  dz = z2-z1
  dlen=sqrt(dx*dx+dy*dy+dz*dz+1.0e-8)
  if (ni.lt.2) ni=2
  dx = (x2-x1) / ni
  dy = (y2-y1) / ni
  dz = (z2-z1) / ni
  do j=1,ni+1
    x = x1 + dx*(j-1)
    y = y1 + dy*(j-1)
    z = z1 + dz*(j-1)                                             
    r(j) = sqrt(x**2 + y**2 + z**2+1.0e-8)                         
    acosa=z/r(j)
    if(acosa.LT.-1.)acosa=-1.
    if(acosa.GT.1)acosa=1.
    a(j) = acos(acosa)                                   
    acosa=x/r(j)/sin(a(j))
    if(acosa.LT.-1.)acosa=-1.
    if(acosa.GT.1)acosa=1.
    b(j) = acos(acosa)
    if(y.LT.0.00000)b(j)=360.00000*dpi-b(j)
  enddo
  to = rtim(ni+1,r,a,b)
  tp = to
  do i=1,ni+1
    w(1,i) = r(i)
    w(2,i) = a(i)
    w(3,i) = b(i)
  enddo
! *** number of points loop ***
  loops = 0
  do while(ni .le. n2)
! *** interation loop ***                                             
    do l=1,nloop                                              
      loops = loops + 1
      do kk=2,ni
        !-- see um & thurber (1987) p.974.
        if(mod(kk,2) .eq. 0) then
          k = kk/2 + 1
        else
          k = ni+1 - (kk-1)/2
        endif        
        r1 = r(k-1)                                 
        a1 = a(k-1)                                 
        b1 = b(k-1)                                 
        x1 = r1*sin(a1)*cos(b1)                           
        y1 = r1*sin(a1)*sin(b1)                           
        z1 = r1*cos(a1)                                   
        r3 = r(k+1)                                 
        a3 = a(k+1)                                 
        b3 = b(k+1)                                 
        x3 = r3*sin(a3)*cos(b3)                           
        y3 = r3*sin(a3)*sin(b3)                           
        z3 = r3*cos(a3)                                   
        dx = x3 - x1                                      
        dy = y3 - y1                                      
        dz = z3 - z1                                      
        x2 = x1 + dx/2                                    
        y2 = y1 + dy/2                                    
        z2 = z1 + dz/2                                    
        r2 = sqrt(x2**2 + y2**2 + z2**2+1.0e-8)                 
        acosa=z2/r2
        if(acosa.LT.-1.)acosa=-1.
        if(acosa.GT.1)acosa=1.
        a2 = acos(acosa)                                  
        sina = sin(a2)                                    
        cosa = cos(a2)                                    
        acosa=x2/r2/sina
        if(acosa.LT.-1.)acosa=-1.
        if(acosa.GT.1)acosa=1.
        b2 = acos(acosa)                          
        if(y.LT.0.00000)b2=360.00000*dpi-b2
        dn = dx**2 + dy**2 + dz**2                       
        ddn = sqrt(dn+1.0e-8)                                    
        dr = (r3-r1) / ddn                                
        da = (a3-a1) / ddn                                
        db = (b3-b1) / ddn
        !--  Begin find the gradients and velocities
        !-- first find the length of segment
        dseg=sqrt((dx/2)**2+(dy/2)**2+(dz/2)**2+1.0e-8)
        ddseg=dseg/2.
        !   Now ddseg will be a distance to find dV
        !   along the coordinates 
        !   Determine velocity at 3 points       
        v1 = velocity(r1,a1,b1)                            
        v2 = velocity(r2,a2,b2)                            
        v3 = velocity(r3,a3,b3)                           
        !-- Begin to determine coordinates
        !-- of pints surroundibg point a2,b2,r2
        !-- at the distance ddseg
        upz = r2+ddseg
        dwz = r2-ddseg
        if(upz.gt.(ro+10.0))then  !--- I guess it should be ro+10.0
          upz=ro+10.0
          dwz=upz-dseg
        endif

        if(dwz.le.0.)then
          dwz=0.00000001
          !-- set to ro, mistake?
          upz=ro
        endif
        !-- The following if-endif is just for P & S, thus comment out for SKS & PKP !!!
        !-- This gives the lowermost mantle Vp in the outer core
    
!     xxxlat=geoc_to_geog(90.0-a2*r2d)
!  		xxxlon=b2*r2d+shiftlo
!  		xxxdep=ro-r2
!    print*,l,kk,xxxlat,xxxlon,xxxdep
!    print*,ro-upz,ro-dwz
    
        vr1 = velocity(upz,a2,b2)
        vr2 = velocity(dwz,a2,b2)         
        vr=(vr1-vr2)/dseg
        call  km2deg(a2,b2,r2,ddseg,RNULL,adV,bdV,rdV) 
        vb2 = velocity(rdV,adV,bdV)
        call  km2deg(a2,b2,r2,-1.*ddseg,RNULL,adV,bdV,rdV)
        vb1 = velocity(rdV,adV,bdV)
        vb=-1.*(vb1-vb2)/dseg
        call  km2deg(a2,b2,r2,RNULL,ddseg,adV,bdV,rdV)
        va2 = velocity(rdV,adV,bdV)
        call  km2deg(a2,b2,r2,RNULL,-1.*ddseg,adV,bdV,rdV)
        va1 = velocity(rdV,adV,bdV)
        va=-1.*(va1-va2)/dseg
        !-- spherical
        !-- velocity gradient
        !-- va = va / r2
        !-- vb = vb / r2 / sina
        !-- (tangential vector) = (slowness vector) / s
        pr = dr
        pa = r2 * da
        pb = r2 * sina * db
        vrd = pr*vr + pa*va + pb*vb
        rvr = vr - vrd*pr
        rva = va - vrd*pa
        rvb = vb - vrd*pb
        rvs = sqrt(rvr*rvr + rva*rva + rvb*rvb+1.0e-8)               
        if(rvs .eq. 0.) then                              
          r(k) = r2                                   
          a(k) = a2                                   
          b(k) = b2                                   
        else                                        
          rvr = rvr / rvs                                 
          rva = rva / rvs                                 
          rvb = rvb / rvs                                 
          cc   = (1./v1+1./v3)/2.                          
          rcur = vr*rvr + va*rva + vb*rvb 
!          PRINT*, rcur
          !   Tut esli rcur < 0.0 proishodit hernia
          !   poetomu postavlen abs. Ne yasno mozhno li eto delat
          !   ili net no rabotaet. Obichno oshibka poyavliaetsia
          !   ochen redko v nekotorih tochkah
          ! v etom sluchae abs prosto ne daet oshibki y posledniaya iteraciya
          !  uzhe ne imeet rcur negativnim y podgoniaet normalno reshenie
          !  ( mozhet bit)
          if(rcur.LE.0.0)then
            !write(*,*)'Negative'
            rcur=abs(rcur)
          endif                  
          rcur = (cc*v2+1.) / (4.*cc*rcur)
          rcur = -rcur + sqrt(rcur**2+dn/(8.*cc*v2)+1.0e-8) 
          rdr = rvr * rcur
          rda = rva * rcur
          rdb = rvb * rcur
          rpr  = r2 + rdr
          ap  = a2 + rda/r2
          bp  = b2 + rdb/(r2*sina)
          r(k) = (rpr-r(k))*xfac + r(k)
          !	if r(k)>6371 then force it to the surface.
        if (r(k).gt.(ro+10.0)) r(k)=ro+10.0
          a(k) = (ap-a(k))*xfac + a(k)
          b(k) = (bp-b(k))*xfac + b(k)
        endif
      enddo
      idstn=ni
      do j=1,ni+1
        w(1,j) = r(j)
        w(2,j) = a(j)
        w(3,j) = b(j)
      enddo
      ni=idstn
      tk = rtim(ni+1,r,a,b)
      if(abs(to-tk) .le. to*flim)  go to 310                        
      to = tk                                                       
    enddo
    310    continue   
    to=tk
    !-- skip increasing of segment number if minimum length
    !-- of segment is exceed or maximum number of segments
    !-- was reached
    if(dseg.lt.mins.or.ni.ge.n2) then 
      igood=1
      go to 66666
    endif
    !-- double the number of points.
    ni = ni * 2
    do i=1,ni/2+1
      r(i*2-1) = w(1,i)
      a(i*2-1) = w(2,i)
      b(i*2-1) = w(3,i)
    enddo
    do k=2,ni,2
      r1 = r(k-1)
      a1 = a(k-1)                                 
      b1 = b(k-1)                                 
      x1 = r1*sin(a1)*cos(b1)                           
      y1 = r1*sin(a1)*sin(b1)                           
      z1 = r1*cos(a1)                                   
      r3 = r(k+1)                                 
      a3 = a(k+1)                                 
      b3 = b(k+1)                                 
      x3 = r3*sin(a3)*cos(b3)                           
      y3 = r3*sin(a3)*sin(b3)                           
      z3 = r3*cos(a3)                                   
      dx = x3 - x1                                      
      dy = y3 - y1                                      
      dz = z3 - z1                                      
      x2 = x1 + dx/2                                    
      y2 = y1 + dy/2                                    
      z2 = z1 + dz/2                                    
      r2 = sqrt(x2**2 + y2**2 + z2**2+1.0e-8)                  
      acosa=z2/r2
      if(acosa.LT.-1.)acosa=-1.
      if(acosa.GT.1)acosa=1.
      a2 = acos(acosa)
      sina = sin(a2)                                    
      acosa=x2/r2/sina
      if(acosa.LT.-1.)acosa=-1.
      if(acosa.GT.1)acosa=1.
      b2 = acos(acosa)
      if(y.LT.0.00000)b2=360.00000*dpi-b2
      r(k) = r2
      a(k) = a2
      b(k) = b2
    enddo
    tk = rtim(ni+1,r,a,b)
    !-- here i change tp and put to
    if(abs(to-tk) .le. to*flim) then
      igood=1
      go to 99999
    endif
    to = tk 
  enddo                                                                  
  99999 continue
  !-- write(*,*)"poslednii",ni
  idstn=ni
  do i=1,ni+1
    w(1,i) = r(i)
    w(2,i) = a(i)
    w(3,i) = b(i)
  enddo
  ni=idstn
  66666 continue
  !-- Return coordinates to the origin
  idstn=ni
  do k=1,ni+1
    w(1,k) = ro-w(1,k)
    w(2,k) = w(2,k)*r2d
    w(2,k) = geoc_to_geog(90.0-w(2,k))
    w(3,k) = w(3,k)*r2d+shiftlo
    if(w(3,k).lt.0.)w(3,k)=360.+w(3,k)
  enddo 
  ni=idstn
  np=ni+1
  !-- convert ray point to cartesion coord.
!print*,"YYYY",pa,r2,da,a1,a3,ra
  return

end subroutine pbr
  
subroutine km2deg(ala,alo,adp,dx,dy,bla,blo,bdp)
  implicit real*8(a-h,o-z)
  real*8 ala,alo,adp,dx,dy,bla,blo,bdp
  real*8 dpi,dps
  !-- This subroutine calculate position of new point
  !-- in polar coordinates basing on the coordinates
  !-- of main point in radians ( la is colatitude) and dx and dy in kilometers
  dpi = asin(1.)/ 90.
  dps=adp*SIN(ala)
  blo=alo+atan2(dx,dps)
  bla=ala+atan2(dy,adp)
  if(bla.gt.(180.*dpi))then
    bla=360.*dpi-bla
    blo=blo+180.*dpi
  endif
  if(bla.lt.0.)then
    bla=abs(bla)
    blo=blo+180.*dpi
  endif
  if(blo.lt.0.)blo=360.*dpi+blo
  if(blo.gt.(360.*dpi))blo=blo-(360.*dpi)
  bdp=sqrt(adp**2+dx**2+dy**2+1.0e-8)
  return
end subroutine km2deg

function rtim(m, r, a, b)                      
  implicit real*8(a-h,o-z)
  real*8 x1,y1,z1,x2,y2,z2,dl
  real*8 rv2,sm,rv1,rtim
  parameter (msg = 16384)
  real*8 r(msg+1), a(msg+1), b(msg+1)
  integer m
  if(m.GT.(msg+1))write(*,*)'*'
  rtim = 0.
  rv1 = 1./velocity(r(1),a(1),b(1))
  do j=1,m-1
    x1 = r(j)*sin(a(j))*cos(b(j))
    y1 = r(j)*sin(a(j))*sin(b(j))
    z1 = r(j)*cos(a(j))                                               
    x2 = r(j+1)*sin(a(j+1))*cos(b(j+1))
    y2 = r(j+1)*sin(a(j+1))*sin(b(j+1))
    z2 = r(j+1)*cos(a(j+1))
    dl = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
    rv2 = 1./velocity(r(j+1),a(j+1),b(j+1))
    sm = (rv1 + rv2) / 2.         
    rtim = rtim + sqrt(dl+1.0e-8)*sm
    rv1 = rv2
  enddo                                                        
end function rtim  

real*8 function earthr(xlat)
  !  this routine establishes the short distance conversion factors
  !  given the origin of coordinates
  !  the rotation angle is converted to radians also
  !  common block variables:

  !  local variables:
  double precision dlt1,dxlt,drad,drlt,xlat
  data re/6378.163/, ell/298.26/
  drad=1.7453292d-2
  drlt=9.9330647d-1
  dxlt=dble(xlat*60.0)
  !  conversion factor for latitude
  dlt1=datan(drlt*dtan(dxlt*drad/60.d0))
  earthr=re*(1.0-sngl(dsin(dlt1)**2)/ell)
end function earthr

real*8 function pa0(dst,depth)
  implicit real*8 (a-h,o-z)
  real*8 dst,depth
  real   dist
  dist=dst*dst+depth*depth
  dist=sqrt(dist+1.0e-6)
  if(depth.gt.35.0) then
    pa0=-0.0033*dist-0.833*alog10(dist)-1.01
  else if( dist .le. 80.0)then
    pa0=-0.00716*dist-alog10(dist)-0.39
  else
    pa0=-0.00261*dist-0.833*alog10(dist)-1.07
  endif
  pa0=-pa0
end function pa0

subroutine location
!  this routine locates the given event in the initial velocity
!  structure using approximate ray tracing
!  common block variables:
  use all
  use observ
  use ray
  use events
  implicit real*8 (a-h,o-z)

!  real*8 sterr(maxsol)
!  character pfile*48,qty,por
!  real*8 s(maxsol),v(maxsol,maxsol),adj(maxsol)
!  real*8 a(maxobs,maxobs)
  real*8 xe,ye,ze,xr,yr,zr
  real*8 w(3,16385)
  integer npoints,nitloc
  real*8 ptime,stime,dpi,colat,re,xdis,epid,azi,v0
!print*,"tttt",ptime,stime,dpi,colat,re,xdis,epid,azi,v0
!  real*8 rmscut    

  !        nitloc - max of iterations for hypocenter location.
  !        eigtol - SVD cutoff in hypocentral adjustments. If smallest
  !                 eigenvalue in geiger's matrix is < eigtol, the depth
  !                 is not adjusted, and a message is printed.
  !        rmscut - value for rms residual below which hypocentral adjustments
  !                 are terminated. ()
  !  10 .005 0.02    ***  nitmxp eigtol rmscut  ***
  nitloc=15
  eigtol=0.005
! rmscut=0.02
  dpi=asin(1.)/90.
  rmswt=0.0

!  EVENT COORDINATES
!  do nit=1,nitloc     ! THIS LOOP IS FOR ITERATION (ORIGINAL)
    xe=evc(1)
    ye=evc(2)
    ze=evc(3)
    colat=90.0-ye
    re=ro-ze
    hypoerror=0.0
    dis_ave=0.0
    ikk=0
!--  LOOP OVER ALL OBSERVATIONS OF THIS EVENT
    do no=1,allnstations(neq) 
      ns=isto(no)
      xr=stc(1,ns)
      yr=stc(2,ns)
      zr=stc(3,ns)
      call distaz(colat,xe,90.-yr,xr,epid,azi)
      epid=epid*dpi*ro  !-- Epicentral distance
      epdis(no)=epid                     
      azimuth(no)=nint(azi)                     

      if(secp(no,1) .gt. 0.0 .and. xweio(no,1).lt. 4.5 )then
        !-- For P arrival cases
        ips=1  !-- 1 for P velocity & 2 for S velocity
        call pbr(ye,xe,ze,yr,xr,zr,w,npoints,ptime)
        call cal_delta(w(2,1),w(3,1),w(2,2),w(3,2),xdis)
        !tkofag=dacos( (w(1,2)-w(1,1)) /dsqrt(xdis**2+(w(1,2)-w(1,1))**2) )/dpi
        tkofag=angle(w,1,2)
        call vel_3d(w(3,1),w(2,1),w(1,1),v0)
        p=(re*dsin(tkofag*dpi))/v0
        takeoff(no)=nint(tkofag)                     

        dth(no,2,1)=(-1.0)*(p*dsin(azi*dpi)*dsin(colat*dpi))*dpi
        dth(no,3,1)=(-1.0)*(p*dcos(azi*dpi))*dpi
        dth(no,4,1)=(-1.0)*dcos(tkofag*dpi)/v0
        dth(no,1,1)=1.0       
        res(no,1)=secp(no,1)-ptime-seco-pcor(ns)
        ptcal(neq,no)=ptime
        dis_ave=dis_ave+epid
        ikk=ikk+1
      else if(secp(no,1) .gt. 0.0 .and. xweio(no,1).gt. 4.5 )then
        !-- For S-P time differences case
        !-- For P wave
        ips=1  !-- 1 for P velocity & 2 for S velocity
        call pbr(ye,xe,ze,yr,xr,zr,w,npoints,ptime)
        call cal_delta(w(2,1),w(3,1),w(2,2),w(3,2),xdis)
        !tkofag=acos( (w(1,2)-w(1,1)) /sqrt(xdis**2+(w(1,2)-w(1,1))**2) )/dpi
        tkofag=angle(w,1,2)
        call vel_3d(w(3,1),w(2,1),w(1,1),v0)
        p=(re*sin(tkofag*dpi))/v0
        takeoff(no)=nint(tkofag)

        ddx=(-1.)*(p*sin(azi*dpi)*sin(colat*dpi))*dpi
        ddy=(-1.)*(p*cos(azi*dpi))*dpi
        ddz=(-1)*cos(tkofag*dpi)/v0

    !-- for S wave
        ips=2  !-- 1 for P velocity & 2 for S velocity
        call pbr(ye,xe,ze,yr,xr,zr,w,npoints,stime)
        !tkofag=acos( (w(1,2)-w(1,1)) /sqrt(xdis**2+(w(1,2)-w(1,1))**2) )/dpi
        tkofag=angle(w,1,2)
        call vel_3d(w(3,1),w(2,1),w(1,1),v0)
        p=(re*sin(tkofag*dpi))/v0

        dth(no,2,1)=(-1.0)*(p*sin(azi*dpi)*sin(colat*dpi))*dpi -ddx
        dth(no,3,1)=(-1.0)*(p*cos(azi*dpi))*dpi -ddy
        dth(no,4,1)=(-1.0)*cos(tkofag*dpi)/v0 -ddz
        dth(no,1,1)=0.0
        res(no,1)=secp(no,1)-(stime-ptime)-spcor(ns)
        
        do i=1,4
          dth(no,i,2)=0.0
        enddo
        secp(no,2)=0.0
        res(no,2)=0.0
        xweio(no,2)=0.0
        dis_ave=dis_ave+epid
        ikk=ikk+1
      endif

      if(secp(no,2) .gt. 0.0 )then
        !-- For S arrival cases	    	
        ips=2  !-- 1 for P velocity & 2 for S velocity
        call pbr(ye,xe,ze,yr,xr,zr,w,npoints,stime) 
        !tkofag=dacos( (w(1,2)-w(1,1)) /dsqrt(xdis**2+(w(1,2)-w(1,1))**2) )/dpi
        tkofag=angle(w,1,2)
        call vel_3d(w(3,1),w(2,1),w(1,1),v0)        
        p=(re*dsin(tkofag*dpi))/v0

        dth(no,2,2)=(-1.0)*(p*dsin(azi*dpi)*dsin(colat*dpi))*dpi
        dth(no,3,2)=(-1.0)*(p*dcos(azi*dpi))*dpi
        dth(no,4,2)=(-1.0)*dcos(tkofag*dpi)/v0
        dth(no,1,2)=1.0
	res(no,2)=secp(no,2)-stime-seco-scor(ns)
	stcal(neq,no)=stime
	if(secp(no,1).eq.0.0)then
	  epdis(no)=epid
	  azimuth(no)=nint(azi)
	  takeoff(no)=nint(tkofag)
	endif
	dis_ave=dis_ave+epid
	ikk=ikk+1
      else
	do i=1,4
	  dth(no,i,2)=0.0
	enddo
	secp(no,2)=0.0
	res(no,2)=0.0
	xweio(no,2)=0.0
  endif  
!print*,"here",xe,ye,ze
  enddo
  !pause 
  dis_ave=dis_ave/real(ikk)

end subroutine location

!subroutine wthyp_hypo71(a,rmswt,rms,nnobs,hypoerror,nit,dis_ave)
!  !  routine to apply weighting to hypocenter matrix + residuals
!  use observ
!  use events
!  implicit real*8 (a-h,o-z)
!
!  real*8 w(maxobs,2),dis_ave
!  real*8 a(maxobs,maxobs)
!  integer nit
!
!  pi=3.14159
!  !  residual weighting
!  !  calculate initial rms residual and normalization factor
!  wsum=0.0
!  nwr=0
!  do i=1,nstations
!    is=isto(i)
!	do j=1,2
!	  if(secp(i,j) .gt. 0.0)then
!        w(i,j)=weighting(int(xweio(i,j)),epdis(i),res(i,j),nit,dis_ave)
!	    if(w(i,j).eq.0.0)cycle
!	    nwr=nwr+1
!        wsum=wsum+w(i,j)*w(i,j)
!	  endif
!	enddo
!  enddo
!  wfac=sqrt(nwr/wsum)
!
!  do i=1,nstations
!	do j=1,2
!      w(i,j)=w(i,j)*wfac
!	enddo
!  enddo
!
!  !  calculate residual weights
!  nwr=0
!  wnorm=0.0
!  hypoerror=0.0
!  wsum=0.0
!  do i=1,nstations
!    do j=1,2
!      if(w(i,j).eq.0.0)cycle
!      wnorm=wnorm+w(i,j)
!      wsum=wsum+w(i,j)*w(i,j)
!      nwr=nwr+1
!	  hypoerror=hypoerror+res(i,j)*res(i,j)*w(i,j)*w(i,j)
!	enddo
!  enddo
!  wfac=nwr/wnorm
!  hypoerror=hypoerror*nwr/wsum
!
!  wnorm=0.0
!  rmswt=0.0
!  !  normalize weights, apply to hypocenter matrix, and place into
!  !  a-matrix for inversion
!  avwt=0.0
!  sum2=0.0
!  k=0
!  do i=1,nstations
!    do j=1,2
!	  if(w(i,j) .le. 0.)cycle
!	  k=k+1 
!      w(i,j)=w(i,j)*wfac
!      wnorm=wnorm+w(i,j)*w(i,j)
!      rmswt=rmswt+(res(i,j)*w(i,j))**2
!      do jj=1,4
!        a(k,jj)=dth(i,jj,j)*w(i,j)
!	  enddo
!      a(k,5)=res(i,j)*w(i,j)
!	  avwt=avwt+w(i,j)
!	  sum2=sum2+res(i,j)*res(i,j)*w(i,j)
!	enddo
!  enddo
!  nnobs=k
!  rmswt=sqrt(rmswt/wnorm)
!  avwt=avwt/real(nnobs)
!  sum2=sum2/avwt
!  if(nnobs.gt.4)then
!	rms=sum2/real(nnobs-4)
!  else
!	rms=sum2
!  endif
!  rms=sqrt(rms+1.0e-8)
!end	subroutine wthyp_hypo71


!-This subroutine is to change Lon. Lat. to Km unit
subroutine cal_delta(elat,elon,slat,slon,delta)
  implicit real*8 (a-h,o-z)
  real*8 elat,elon,slat,slon,delta
  real*8 avlat,a,b,dlat,dlon,dx,dy
    
  avlat=0.5*(elat+slat)
  a=1.840708+avlat*(.0015269+avlat*(-.00034+avlat*(1.02337e-6)))
  b=1.843404+avlat*(-6.93799e-5+avlat*(8.79993e-6+avlat*(-6.47527e-8)))
  dlat=slat-elat
  dlon=slon-elon
  dx=a*dlon*60.
  dy=b*dlat*60.
  delta=sqrt(dx*dx+dy*dy)
endsubroutine

!
!  calculate distance, azimuth between source (c0,l0) and receiver (c1,l1).
!  colatitude:[0,180] (geographic), longitude [0,360].
!  distance <=180.
!  azimuth [0,360), measured at source location clockwise from north.
subroutine distaz(c0,l0,c1,l1,dist,az)
  implicit real*8 (a-h,o-z)
  real*8 colat0,lon0,colat1,lon1,del,azimuth
  real*8 cosdel,cosaz,sinaz,tmp
  real*8 c0,l0,c1,l1
  real*8 eps,RAD_PER_DEG
  eps=1.0e-8
  RAD_PER_DEG=0.0174532925199432955

  colat0=c0*RAD_PER_DEG
  colat1=c1*RAD_PER_DEG
  lon0=l0*RAD_PER_DEG
  lon1=l1*RAD_PER_DEG

  !--/*calculate distance*/
  cosdel=cos(colat0)*cos(colat1)+sin(colat0)*sin(colat1)*cos(lon1-lon0)
  if(cosdel > 1.0) cosdel=cosdel-eps
  if(cosdel <-1.0) cosdel=cosdel+eps
  del=acos(cosdel)
  dist=del/RAD_PER_DEG

  !-- /*calculate azimuth*/
  tmp=sin(colat0)*sin(del)
  if (tmp <=eps) then 
    !-- /*special case: source at pole or del=0 or 180*/
    if(c0<=eps) az=180.0
    if(c0 >=(180.-eps)) az=0.0
    if(dist<=eps) az=-999.0
    if(dist >= (180.-eps)) az=-999.0
    return
  endif
  cosaz=(cos(colat1)-cos(colat0)*cos(del))/tmp
  sinaz=sin(colat1)*sin(lon1-lon0)/sin(del)
  azimuth=atan2(sinaz,cosaz)/RAD_PER_DEG
  if(azimuth <0.) azimuth=azimuth+360.
  az =azimuth
end subroutine distaz

!subroutine fmp
!  use observ
!  use events
!  implicit real*8 (a-h,o-z)
!
!  real*8 xgap(maxobs)
!
!  epmin=999999.
!  do i=1,nstations
!	if(epdis(i) .lt. epmin)epmin=epdis(i)
!	xgap(i)=azimuth(i)
!  enddo
!
!  do i=1,nstations-1
!	do j=i+1,nstations
!	  if(xgap(i) .gt. xgap(j))then
!	    gap=xgap(i)
!	    xgap(i)=xgap(j)
!	    xgap(j)=gap
!	  endif
!	enddo
!  enddo
!
!  gap=0.0
!  do i=1,nstations-1
!	if( (xgap(i+1)-xgap(i)) .gt. gap)gap=xgap(i+1)-xgap(i)
!  enddo
!  xtmp=360.-xgap(nstations)+xgap(1)
!  if(xtmp .gt. gap)gap=xtmp
!end subroutine fmp

!
!----- subroutine for calculate location quality -------------------
!      The same as HYPO71 ---
!subroutine qual(quald,rms,erh,erz,nr,gap,dmin,depth)
!  implicit real*8 (a-h,o-z)
!  character*1 quald,q(4)
!  integer*1 iqs,iqd
!  q(1)='A'
!  q(2)='B'
!  q(3)='C'
!  q(4)='D'
!
!  iqs=4
!  if(rms.lt.0.5  .and. erh.le.5.0)iqs=3
!  if(rms.lt.0.3  .and. erh.le.2.5 .and. erz.le.5.)iqs=2
!  if(rms.lt.0.15 .and. erh.le.1.0 .and. erz.le.2.)iqs=1
!  iqd=4
!  if(nr.ge.6 .and. gap.le.180.0 .and. dmin.le.50.)iqd=3
!  d1=depth
!  if(d1.lt.5.0)d1=5.0
!  d2=2.*d1
!  if(nr.ge.6 .and. gap.le.135.0 .and. dmin.le.d2)iqd=2
!  if(nr.ge.6 .and. gap.le.90.0  .and. dmin.le.d1)iqd=1
!  iqd=(iqs+iqd+1)/2
!  if(iqd.gt.4)iqd=4
!  quald=q(iqd)
!end	subroutine qual
!
!   Adjustment the time format source code comes from C.H. Chang
!   A little modified for remove level by Yih-Min Wu
!   Date : 03/01/1999 at CWB Seismolgy Center
!
subroutine timescal(year,month,day,hour,minut,sec)
  implicit real*8 (a-h,o-z)

  integer :: year,month,day,hour,minut
  real*8    :: sec
  integer :: n_day,juln

  !-- Preocessing day
  do
    n_day=365
    if(mod(year,  4) == 0) n_day=366
    if(mod(year,100) == 0) n_day=365
    if(mod(year,400) == 0) n_day=366

    call julian(year,month,day,juln,1)
    if(juln.le.n_day .and. juln.gt.0)exit
    if(juln.gt.n_day) then
      juln=juln-n_day
      year=year+1
    else
      year=year-1
      n_day=365
      if(mod(year,  4) == 0) n_day=366
      if(mod(year,100) == 0) n_day=365
      if(mod(year,400) == 0) n_day=366
      juln=juln+n_day
    endif
    call julian(year,month,day,juln,-1)
  enddo

  !-- processing sec
  do
    if(sec.lt.60.0 .and. sec.ge.0.0)exit
    if(sec.ge.60.0) then
!    if (sec-60.>=-0.005)then
      sec=sec-60.0
      minut=minut+1
    else
      sec=sec+60.0
      minut=minut-1
    endif
  enddo

  !-- processing minute
  do
    if(minut.lt.60 .and. minut.ge.0)exit
    if(minut.ge.60) then
      minut=minut-60
      hour =hour +1
    else
      minut=minut+60
      hour =hour -1
    endif
  enddo

  !-- processing hour
  do
    if(hour.lt.24 .and. hour.ge.0)exit
    if(hour.ge.24) then
      hour=hour-24
      juln=juln+1
    else
      hour=hour+24
      juln=juln-1
    endif
  enddo

  if(juln .gt. n_day) then
    juln=juln-n_day
    year=year+1
  endif
  if(juln .le. 0) then
    year =year-1
    n_day=365
    if(mod(year,  4) == 0) n_day=366
    if(mod(year,100) == 0) n_day=365
    if(mod(year,400) == 0) n_day=366
    juln=juln+n_day
  endif
  call julian(year,month,day,juln,-1)
end subroutine timescal

!ccccc
!-----j_contl  1 year,month,day to year,julian
!             -1 year,julian    to year,month,day
subroutine julian(n_year,n_month,n_day,n_jday,j_contl)
  implicit real*8 (a-h,o-z)

  integer :: n_year,n_month,n_day,n_jday,j_contl,i
  integer :: month(12)=(/31,28,31,30,31,30,31,31,30,31,30,31/)
  if(mod(n_year,  1) .eq. 0) month(2)=28
  if(mod(n_year,  4) .eq. 0) month(2)=29
  if(mod(n_year,100) .eq. 0) month(2)=28
  if(mod(n_year,400) .eq. 0) month(2)=29
  if(j_contl.eq. 1) then
    n_jday=0
    do i=1,n_month-1
       n_jday=month(i)+n_jday
    enddo
    n_jday=n_day+n_jday
  endif
  if(j_contl.eq.-1) then
    n_month=1
    n_day  =n_jday
    do i=1,12
      if(n_day.le.month(i)) exit
      if(n_day.gt.month(i)) then
        n_month=n_month+1
        n_day  =n_day-month(i)
      endif
    enddo
  endif
end subroutine julian

!--------- function for calculate weight ---------------------------
real*8 function weighting(jj,dist,res,nit,dis_ave)
  implicit real*8 (a-h,o-z)
  integer nit
  real*8 dis_ave

  data xnear,xfar/30.0,300.0/
  data tres/1.0/
  if(dis_ave.gt.500.0)then
    xnear=200.0
    xfar=2000.0
  endif
  j=mod(jj,5)
  wt = 1.0
  if(j.ge.0.and.j.le.4)then
    wt=wt*float(4-j)/4.
  else
    wt=0.0
  endif
  if(nit.gt.1)then
    if(dist.gt.xnear)then
      wt=wt*(xfar-xnear)/(9.*dist+xfar-10.*xnear)
    endif
    wt=wt*(tres/(tres+abs(res)))**2
  endif
  weighting=wt
end function weighting
!--------------------------------------------------------------------

!subroutine  fksvd (a, s, v, mmax, nmax, m, n, p, withu, withv)
subroutine  fksvd (a, s, v, m, n, p, withu, withv)
  use all
  implicit real*8 (a-h,o-z)

  parameter (maxobs=numberob_s)
  parameter (maxsol=maxevt*4)
! parameter (maxobsol=) !-- maxobs*maxsol
    
  integer    m, n, p
  real*8     r, w, cs, sn, tol, f, x, eps, g, y
  real*8     eta, h, q, z
  integer    i, j, k, l, l1, n1, np
  logical    withu, withv
  real*8 a(maxobs,4*maxevt+1)
  real*8 s(4*maxevt), v(4*maxevt,4*maxevt)
  real*8 as1(maxobs),as2(maxobs)

  !     ------------------------------------------------------------------
  !
  !     this is a translation of a cdc 6600 fortran program to ibm 360
  !     fortran iv.  this subroutine uses short precision arithmetic.
  !     a long precision version is available under the name 'dsvd'.
  !
  !     this subroutine replaces earlier subroutines with the same name,
  !    689   6       &   &0&  s of a complex ar&thmetic program, published
  !     as algorithm 358.  this current program is faster, more accurate
  !     and less obscure in describing its capabilities.
  !
  !     original programmer=  r. c. singleton
  !     360 version by=       j. g. lewis
  !     last revision of this subroutine=  4 december 1973
  !
  !     ------------------------------------------------------------------
  !
  !     additional subroutine needed=  rotate
  !
  !     ------------------------------------------------------------------
  !
  !
  !     this subroutine computes the singular value decomposition
  !     of a real m*n matrix a, i.e. it computes matrices u, s, and v
  !     such that
  !
  !                  a = u * s * vt ,
  !     where
  !              u is an m*n matrix and ut*u = i, (ut=transpose
  !                                                    of u),
  !              v is an n*n matrix and vt*v = i, (vt=transpose
  !                                                    of v),
  !        and   s is an n*n diagonal matrix.
  !
  !     description of parameters=
  !
  !     a = real array. a contains the matrix to be decomposed.
  !         the original data are lost.  if withv=.true., then
  !         the matrix u is computed and stored in the array a.
  !
  !     mmax = integer variable.  the number of rows in the
  !            array a.
  !
  !     nmax = integer variable.  the number of rows in the
  !            array v.
  !
  !     m,n = integer variables.  the number of rows and columns
  !           in the matrix stored in a.  (ng=mg=100.  if it is
  !           necessary to solve a larger problem, then the
  !           amount of storage allocated to the array t must
  !           be increased accordingly.)  if mlt n , then either
  !           transpose the matrix a or add rows of zeros to
  !           increase m to n.
  !
  !     p = integer variable.  if p'0, then columns n+1, . . . ,
  !         n+p of a are assumed to contain the columns of an m*p
  !         matrix b.  this matrix is multiplied by ut, and upon
  !         exit, a contains in these same columns the n*p matrix
  !         ut*b. (p'=0)
  !
  !     withu, withv = logical variables.  if withu=.true., then
  !         the matrix u is computed and stored in the array a.
  !         if withv=.true., then the matrix v is computed and
  !         stored in the array v.
  !
  !     s = real array.  s(1), . . . , s(n) contain the diagonal
  !         elements of the matrix s ordered so than s(i)>=s(i+1),
  !         i=1, . . . , n-1.
  !
  !     v = real array.  v contains the matrix v.  if withu
  !         and withv are not both =.true., then the actual
  !         parameter corresponding to a and v may be the same.
  !
  !     this subroutine is a real version of a fortran subroutine
  !     by businger and golub, algorithm 358=  singular value
  !     decomposition of a complex matrix, comm. acm, v. 12,
  !     no. 10, pp. 564-565 (oct. 1969).
  !     with revisions by rc singleton, may 1972.
  !     ------------------------------------------------------------------
  !     machine constants eta and tol:    0.119209E-06    0.117549E-37
  
  !real*8  t(maxobs*maxsol)
  real*8,allocatable :: t(:)
  
  allocate(t(maxobs*maxsol),stat=nerror)
  if (nerror/=0)then
    print*,"T ERROR WHEN ALLOCATE!"
    stop   
  endif 
  
  !    eclipse constants
  data eta,tol/1.2e-7,1.2e-38/
  !     eta (16**-6) and tol (16**-59) are machine dependent constants
  !     for ibm 360/370 computers (short form arithmetic).
  !     eta is the machine epsilon (relative accuracy)]
  !     tol is the smallest representable real divided by eta.
  nmax=maxsol
  mmax=maxobs

  np = n + p
  n1 = n + 1

  !     householder reduction to bidiagonal form
  g = 0.0
  eps = 0.0
  l = 1
  do 
    t(l) = g
    k = l
    l = l + 1
    ! elimination of a(i,k), i=k+1, . . . , m
    s(k) = 0.0
    z = 0.0
    do i = k,m
      z = z + a(i,k)**2
    enddo
	if(z.ge.tol)then
      g = sqrt(z)
      f = a(k,k)
      if (f.ge.0.0) g = - g
      s(k) = g
      h = g * (f - g)
      a(k,k) = f - g
	  if(k.ne.np)then
        do j = l,np
          f = 0
          do i = k,m
            f = f + a(i,k)*a(i,j)
          enddo
          f = f/h
          do i = k,m
            a(i,j) = a(i,j) + f*a(i,k)
          enddo
        enddo
	  endif
	endif

    ! elimination of a(k,j), j=k+2, . . . , n
    eps = amax1(eps,abs(s(k)) + abs(t(k)))
    if (k.eq.n) exit
    g = 0.0
    z = 0.0
    do j = l,n
      z = z + a(k,j)**2
    enddo
    if (z.lt.tol) cycle
    g = sqrt(z)
    f = a(k,l)
    if (f.ge.0.0) g = - g
    h = g * (f - g)
    a(k,l) = f - g
    do j = l,n
      t(j) = a(k,j)/h
    enddo
    do i = l,m
      f = 0
      do j = l,n
        f = f + a(k,j)*a(i,j)
      enddo
      if (abs(f).lt.1.0e-25) f=0.
      do j = l,n
        a(i,j) = a(i,j) + f*t(j)
      enddo
	enddo
  enddo  
  ! tolerance for negligible elements
  eps = eps*eta

  ! accumulation of transformations
  if(withv)then
	do k=n,1,-1
      if (k.ne.n .and. t(l).ne.0.0)then
        h = a(k,l)*t(l)
        do j = l,n
          q = 0
          do i = l,n
            q = q + a(k,i)*v(i,j)
          enddo
          q = q/h
          do i = l,n
            v(i,j) = v(i,j) + q*a(k,i)
	      enddo
        enddo
	  endif

      do j=1,n
        v(k,j) = 0.0
      enddo
      v(k,k)=1.0
      l = k
	enddo
  endif

  if(withu)then
    g = s(n)
    if (g.ne.0.0) g = 1.0/g
	do k=n,1,-1
	  if(k .ne. n)then
        do j = l,n
          a(k,j) = 0
        enddo
        g = s(k)
        if(g.ne.0.0)then
          h = a(k,k)*g
          do j = l,n
            q = 0
            do i = l,m
              q = q + a(i,k)*a(i,j)
            enddo
            q = q/h
            do i = k,m
              a(i,j) = a(i,j) + q*a(i,k)
            enddo
          enddo
          g = 1.0/g
		endif
	  endif

      do j = k,m
        a(j,k) = a(j,k)*g
      enddo
      a(k,k) = a(k,k) + 1.0
      l = k
	enddo
  endif

  ! qr diagonalization
  ! test for split
  do k=n,1,-1

    do
	  l = k
	  do 
        if(abs(t(l)).le.eps) goto 290
        l=l-1
        if(abs(s(l)).le.eps)exit
      enddo

	  ! cancellation
      cs = 0.0
      sn = 1.0
      l1 = l
      l = l + 1
      do i = l,k
        f = sn*t(i)
        t(i) = cs*t(i)
        if (abs(f).le.eps)exit
        h = s(i)
        w = sqrt(f*f + h*h)
        s(i) = w
        cs = h/w
        sn = - f/w
        do ia=1,m
          as1(ia) = sngl(a(ia,l1))
          as2(ia) = sngl(a(ia,i))
        enddo
        if(withu) call hyrot(mmax,as1, as2, cs, sn, m)
        do ia=1,m
          a(ia,l1)=dble(as1(ia))
          a(ia,i)=dble(as2(ia))
	    enddo
        if(np.eq.n)cycle
        do j = n1,np
          q = a(l1,j)
          r = a(i,j)
          a(l1,j) = q*cs + r*sn
          a(i,j) = r*cs - q*sn
        enddo
      enddo

      ! test for convergence
      290 w = s(k)

      if(l.eq.k)exit

      ! origin shift
      x = s(l)
      y = s(k-1)
      g = t(k-1)
      h = t(k)
      f = ((y - w)*(y + w) + (g - h)*(g + h))/(2.0*h*y)
      g = sqrt(f*f + 1.0)
      if (f.lt.0.0) g = - g
      f = ((x - w)*(x + w) + (y/(f + g) - h)*h)/x

      ! qr step
      cs = 1.0
      sn = 1.0
      l1 = l + 1
      do i = l1,k
        g = t(i)
        y = s(i)
        h = sn*g
        g = cs*g
        w = sqrt(h*h + f*f)
        t(i-1) = w
        cs = f/w
        sn = h/w
        f = x*cs + g*sn
        g = g*cs - x*sn
        h = y*sn
        y = y*cs
        if(withv) call hyrot(mmax,v(1,i-1),v(1,i),cs,sn,n)
        w = sqrt(h*h + f*f)
        s(i-1) = w
        cs = f/w
        sn = h/w
        f = cs*g + sn*y
        x = cs*y - sn*g
        do ia=1,m
          as1(ia) = sngl(a(ia,i-1))
          as2(ia) = sngl(a(ia,i))
        enddo
        if(withu) call hyrot(mmax,as1, as2, cs, sn, m)
        do ia=1,m
          a(ia,i-1)=dble(as1(ia))
          a(ia,i)=dble(as2(ia))
	    enddo
        if (n.ne.np)then
          do j = n1,np
            q = a(i-1,j)
            r = a(i,j)
            a(i-1,j) = q*cs + r*sn
            a(i,j) = r*cs - q*sn
          enddo
		endif
      enddo

      t(l) = 0.0
      t(k) = f
      s(k) = x
	enddo

    ! convergence
    if (w.lt.0.0)then
      s(k) = - w
      if(withv)then
        do j = 1,n
          v(j,k) = - v(j,k)
        enddo
	  endif
	endif

  enddo

  ! sort singular values
  do k = 1,n
    g = -1.0
    do i = k,n
      if (s(i).lt.g)cycle
      g = s(i)
      j = i
    enddo
	if (j .eq. k)cycle
    s(j) = s(k)
    s(k) = g
	if(withv)then
      do i = 1,n
        q = v(i,j)
        v(i,j) = v(i,k)
        v(i,k) = q
	  enddo
	endif
    if(withu)then
	  do i = 1,m
        q = a(i,j)
        a(i,j) = a(i,k)
        a(i,k) = q
	  enddo
	endif
    if (n.eq.np)cycle
    do i = n1,np
      q = a(j,i)
      a(j,i) = a(k,i)
      a(k,i) = q
	enddo
  enddo

  deallocate(t)

end	subroutine fksvd

subroutine hyrot(nmax,x, y, cs, sn, n)
  implicit real*8 (a-h,o-z)
  integer n
  real*8    x(nmax), y(nmax), cs, sn
  real*8    xx
  integer j
  do j = 1, n
    xx = x(j)
    x(j) = xx*cs + y(j)*sn
    y(j) = y(j)*cs - xx*sn
  enddo
end subroutine hyrot

SUBROUTINE locpt (x0, y0, x, y, n, l, m)
!-----------------------------------------------------------------------
! GIVEN A POLYGONAL LINE CONNECTING THE VERTICES (X(I),Y(I)) (I = 1,...,N)
! TAKEN IN THIS ORDER.  IT IS ASSUMED THAT THE POLYGONAL PATH IS A LOOP,
! WHERE (X(N),Y(N)) = (X(1),Y(1)) OR THERE IS AN ARC FROM (X(N),Y(N)) TO
! (X(1),Y(1)).  N.B. The polygon may cross itself any number of times.

! (X0,Y0) IS AN ARBITRARY POINT AND L AND M ARE VARIABLES.
! On output, L AND M ARE ASSIGNED THE FOLLOWING VALUES ...

!    L = -1   IF (X0,Y0) IS OUTSIDE THE POLYGONAL PATH
!    L =  0   IF (X0,Y0) LIES ON THE POLYGONAL PATH
!    L =  1   IF (X0,Y0) IS INSIDE THE POLYGONAL PATH

! M = 0 IF (X0,Y0) IS ON OR OUTSIDE THE PATH.  IF (X0,Y0) IS INSIDE THE
! PATH THEN M IS THE WINDING NUMBER OF THE PATH AROUND THE POINT (X0,Y0).

! Fortran 66 version by A.H. Morris
! Converted to ELF90 compatibility by Alan Miller, 15 February 1997

!-----------------------

IMPLICIT NONE
INTEGER :: n
REAL*8  :: x0, y0, x(n), y(n)
INTEGER :: l, m

!     Local variables
INTEGER :: i, n0
REAL*8  :: angle, eps, pi, pi2, sum, theta, theta1, thetai, tol, u, v

!     ****** EPS IS A MACHINE DEPENDENT CONSTANT. EPS IS THE
!            SMALLEST NUMBER SUCH THAT 1.0 + EPS > 1.0

eps = EPSILON(1.0)

!-----------------------------------------------------------------------
n0 = n
IF (x(1) == x(n) .AND. y(1) == y(n)) n0 = n - 1
pi = ATAN2(0.0, -1.0)
pi2 = 2.0*pi
tol = 4.0*eps*pi
l = -1
m = 0

u = x(1) - x0
v = y(1) - y0
IF (u == 0.0 .AND. v == 0.0) GO TO 20
IF (n0 < 2) RETURN
theta1 = ATAN2(v, u)

sum = 0.0
theta = theta1
DO i = 2, n0
  u = x(i) - x0
  v = y(i) - y0
  IF (u == 0.0 .AND. v == 0.0) GO TO 20
  thetai = ATAN2(v, u)
  
  angle = ABS(thetai - theta)
  IF (ABS(angle - pi) < tol) GO TO 20
  IF (angle > pi) angle = angle - pi2
  IF (theta > thetai) angle = -angle
  sum = sum + angle
  theta = thetai
END DO

angle = ABS(theta1 - theta)
IF (ABS(angle - pi) < tol) GO TO 20
IF (angle > pi) angle = angle - pi2
IF (theta > theta1) angle = -angle
sum = sum + angle

!     SUM = 2*PI*M WHERE M IS THE WINDING NUMBER

m = ABS(sum)/pi2 + 0.2
IF (m == 0) RETURN
l = 1
IF (sum < 0.0) m = -m
RETURN

!     (X0, Y0) IS ON THE BOUNDARY OF THE PATH

20 l = 0
RETURN
END SUBROUTINE locpt
 
function angle(w,pt1,pt2)
    use ray
    implicit real*8 (a-h,o-z)
    integer pt1,pt2
    parameter (msg=16384)
    real*8 w(3,msg+1),dpi,r2d
    real*8 a1,a2,a3,b1,b2,b3,al,bl,dot
    real*8 lon1,lat1,dep1,lon2,lat2,dep2
    real*8 lon11,lon22,cola1,cola2
    real*8 dlo,shift,geog_to_geoc
    real*8 x1,y1,z1,x2,y2,z2,r1,r2
    dpi=asin(1.0)/90.0
    r2d=90./asin(1.)
        
! point 1
    lon1=w(3,pt1)
    lat1=geog_to_geoc(w(2,pt1))
    dep1=w(1,pt1)
! point 2
    lon2=w(3,pt2)
    lat2=geog_to_geoc(w(2,pt2))
    dep2=w(1,pt2)
        
    cola1=90.-lat1
    cola2=90.-lat2
!   print*,"pt1:",lon1,lat1,dep1," pt2:",lon2,lat2,dep2
        
    if (lon1.lt.0.0) then
       lon11=360.+lon1
    else
       lon11=lon1
    endif
    if (lon2.lt.0.0) then
       lon22=360.+lon2
    else
       lon22=lon2
    endif
    dlo=abs(lon22-lon11)
!   print*,"lon11,lon22:",lon11,lon22," dlo:",dlo
    if (dlo.lt.180.) then
       shift=0.0e10
       if (lon22.lt.lon11) then
          shift=lon22-(180.-dlo)/2.
          lon2=(180.-dlo)/2.
          lon1=lon2+dlo
       else
          shift=lon11-(180.-dlo)/2.
          lon1=(180.-dlo)/2.
          lon2=lon1+dlo
       endif
    else
       dlo=360.0-dlo
       shift=0.0e10
       if (lon22.lt.lon11) then
          shift=lon22-(dlo+(180.-dlo)/2.)
          lon2=(180.-dlo)/2.+dlo
          lon1=lon2-dlo
       else
          shift=lon11-(dlo+(180.-dlo)/2.)
          lon1=(180.-dlo)/2.+dlo
          lon2=lon1-dlo
       endif
    endif
!   print*,"pt1:",lon1,lat1,dep1," pt2:",lon2,lat2,dep2
        
    r1=ro-dep1
    r2=ro-dep2
!   print*,r1,r2
    x1 = r1*sin(cola1*dpi)*cos(lon1*dpi)
    y1 = r1*sin(cola1*dpi)*sin(lon1*dpi)
    z1 = r1*cos(cola1*dpi)
    x2 = r2*sin(cola2*dpi)*cos(lon2*dpi)
    y2 = r2*sin(cola2*dpi)*sin(lon2*dpi)
    z2 = r2*cos(cola2*dpi)
!   print*,"x1,y1,z1:",x1,y1,z1
!   print*,"x2,y2,z2:",x2,y2,z2
        
    a1=x2-x1
    a2=y2-y1
    a3=z2-z1
    b1=0.-x1
    b2=0.-y1
    b3=0.-z1
!   print*,"a1,a2,a3:",a1,a2,a3
!   print*,"b1,b2,b3:",b1,b2,b3
    dot=(a1*b1)+(a2*b2)+(a3*b3)
    al=sqrt(a1**2+a2**2+a3**2)
    bl=sqrt(b1**2+b2**2+b3**2)
    dot=(a1*b1)+(a2*b2)+(a3*b3)
    al=sqrt(a1**2+a2**2+a3**2)
    bl=sqrt(b1**2+b2**2+b3**2)
!   print*,"dot:",dot," al,bl:",al,bl
    angle=acos(dot/(al*bl))/dpi
    return
end function
