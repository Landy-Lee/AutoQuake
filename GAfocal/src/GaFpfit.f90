program cwbgafpl
  implicit none
  integer nsta,igap,nreading,nup
  character fn*24,pfile*24,phead*72,card(1000)*83
  integer iazi(1000),itkof(1000),ipor(1000)
  integer iresult,i,j,la,lo
  real xla,xlo,depth,xml,Rup
  character P1
  integer yy,y2,mm,dd,hh,mi,ss,nused

  real Q,sol_str(100),sol_dip(100),sol_rake(100)
  real str_sdv(100),dip_sdv(100),rake_sdv(100),avsdv
  integer nsol

  print*,'Input the phase file'
  read(*,'(a)')fn

  open(31,file='results.txt',status='unknown')

  nused=0
  iresult=1
  open(15,file=fn,status='old')
  do
!	    read(15,'(a72,1x,a12,i4)',iostat=iresult)phead,pfile,nsta
!		if(iresult.lt.0 .or. nsta.eq.0)exit
!		do i=1,nsta
!		  read(15,'(a83)')card(i)
!		enddo
!		read(phead,'(19x,i2,f5.2,i3,f5.2,f6.2,f4.2)')la,xla,lo,xlo,depth,xml
!		
!		xla=la+xla/60.0
!		xlo=lo+xlo/60.0

        if (iresult.lt.0) exit  ! check if the last iteration reached the end of the file
        read(15,'(a72)',iostat=iresult)phead
        if (iresult.lt.0) exit
        nsta=1
        do
           read(15,'(a83)',iostat=iresult)card(nsta)
           if (iresult.lt.0) exit
           if (ichar(card(nsta)(2:2)).ge.ichar("0").and.&
               ichar(card(nsta)(2:2)).le.ichar("9")) then
              backspace(15)
              exit
           endif
           nsta=nsta+1
        enddo
        nsta=nsta-1
        read(phead,'(1x,i4,4i2,i3,3x,i2,f5.2,i3,f5.2,f6.2,f4.2)')yy,mm,dd,hh,&
             mi,ss,la,xla,lo,xlo,depth,xml
        if (yy/100.eq.20) y2=yy-2000
        if (yy/100.eq.19) y2=yy-1900
        write(pfile,'(6i2.2,a2,i2.2)')y2,mm,dd,hh,mi,ss,".P",y2
        xla=la+xla/60.0
        xlo=lo+xlo/60.0

        nreading=0
        nup=0
        do i=1,nsta
          read(card(i)(12:20),'(2i4,a1)')la,lo,p1
          if(lo.gt.90)then
            lo=180-lo
            if(la.lt.180)then
              la=la+180
            else
              la=la-180
            endif
          endif
          if(p1.eq.'+')then
            nreading=nreading+1
            iazi(nreading)=la
            itkof(nreading)=lo
            ipor(nreading)=1
            nup=nup+1
          else if(p1.eq.'-')then
            nreading=nreading+1
            iazi(nreading)=la
            itkof(nreading)=lo
            ipor(nreading)=-1
          endif
        enddo

        if(nreading.le.10)cycle
        if(nup.eq.nreading .or. nup.eq.0)cycle
        Rup=1.0*nup/real(nreading)

        !-- Find Gap
        do i=1,nreading-1
          do j=i,nreading
            if(iazi(i).gt.iazi(j))then
              igap=iazi(i)
              iazi(i)=iazi(j)
              iazi(j)=igap
            endif
          enddo
        enddo
        igap=iazi(1)+360-iazi(nreading)
        do i=2,nreading
          if( (iazi(i)-iazi(i-1)) .gt. igap ) igap=iazi(i)-iazi(i-1)
        enddo
        if(igap.gt.180)cycle
        nused=nused+1

        !-- writing pfile
        open(1,file='pfiles/'//pfile,status='unknown')
        write(1,'(a)')phead
        do i=1,nsta
          write(1,'(a)')card(i)
        enddo
        close(1)

        print*,'Pfile : ',pfile

        call maingafpfit(pfile,nsol,q,sol_str,sol_dip,sol_rake,str_sdv,dip_sdv,rake_sdv)

        !if(q.lt.0.1)cycle
        if(nsol.gt.2)cycle
        if(nsol.eq.2 .and. q.lt.1.0)cycle

        avsdv=(str_sdv(1)+dip_sdv(1)+rake_sdv(1))/3.0
        if(avsdv.lt.45.0)then
          read(phead,'(1x,i4,4i2,i3)')yy,mm,dd,hh,mi,ss
          write(31,40)yy,mm,dd,hh,mi,ss,xlo,xla,depth,xml,nint(sol_str(1)),nint(str_sdv(1)),nint(sol_dip(1)),&
                      nint(dip_sdv(1)),nint(sol_rake(1)),nint(rake_sdv(1)),q,nreading,pfile
40        format(i4,'/',i2.2,'/',i2.2,1x,i2.2,':',i2.2,':',i2.2,1x,f7.3,1x,f6.3,1x,f6.1,1x,f3.1,3(i5,'+-',i3),1x,f6.2,1x,i3,1x,a16)
        endif
  enddo
  close(15)
  close(31)
end program cwbgafpl


subroutine maingafpfit(pfile,nsol,q,sol_str,sol_dip,sol_rake,str_sdv,dip_sdv,rake_sdv)

  implicit none
  character*64 fn,ouf
  real Q,sol_str(100),sol_dip(100),sol_rake(100)
  real str_sdv(100),dip_sdv(100),rake_sdv(100),fscore
  integer nsol,i
  character*4 c4
  character pfile*24,cmd*64

  fn='pfiles/'//pfile

  call GaFpfit(fn,nsol,q,sol_str,sol_dip,sol_rake,str_sdv,dip_sdv,rake_sdv,fscore)
  print*,'# of solutions:',nsol

!  if(q.lt.0.001 .or. nsol.gt.1)return 

  cmd='./src/focalplot.sh XXXXXXXXXXXX.PXX  180   90   45'
  write(cmd(20:50),'(a16,3i5)')trim(pfile),nint(sol_str(1)),nint(sol_dip(1)),nint(sol_rake(1))
  call system(cmd)
!  c4='.001'
!  do i=1,nsol
!	write(c4(2:4),'(i3.3)')i
!	ouf='psplots\'//pfile//c4
!	call pfocal(fn,ouf,sol_str(i),sol_dip(i),sol_rake(i),str_sdv(i),dip_sdv(i),rake_sdv(i),q,fscore)
!  enddo

end subroutine maingafpfit

subroutine GaFpfit(fn,nsol,q,sol_str,sol_dip,sol_rake,str_sdv,dip_sdv,rake_sdv,fscore)

   implicit none

   character*64 fn

   real azi(1000),tk(1000),po(1000),f_str(1000),f_dip(1000),f_rake(1000)
   real mutation,reproduction,fscore
   integer nbits,npopulation,npo,nresults
   real Q,sol_str(100),sol_dip(100),sol_rake(100)
   real str_sdv(100),dip_sdv(100),rake_sdv(100)
   integer nsol

   npopulation=800	   !-- Population number
   nbits=3			   !-- n bit for mutation
   reproduction=0.036  !-- ratio for reproduction
   mutation=0.72	   !-- ratio for mutation

   call read_pfile(fn,npo,azi,tk,po)

   call ga_find_fpl(azi,tk,po,npo,f_str,f_dip,f_rake,nresults,fscore,npopulation,reproduction,mutation,nbits)

   call calculating_quality(npo,azi,po,f_str,f_dip,f_rake,nresults,fscore,Q,nsol,sol_str,sol_dip,sol_rake,str_sdv,dip_sdv,rake_sdv)
    
end subroutine GaFpfit

subroutine calculating_quality(npo,azi,po,f_str,f_dip,f_rake,nresults,fscore,Q,nsol,nsol_str,nsol_dip,nsol_rake,&
                               nstr_sdv,ndip_sdv,nrake_sdv)
  implicit none
  real azi(1000),po(1000),f_str(1000),f_dip(1000),f_rake(1000),ff_str(1000),ff_dip(1000),ff_rake(1000)
  real xazi(1000),xpo(1000)
  integer npo,nresults,i,j,k,nnsol(1000)
  real Q,Q_gap,Q_npo,Q_score,Q_por
  real fscore,gap
  real*8 dipx(2),strkx(2),rakex(2)
  real*8 tpazimuth(2),tpplunge(2)
  integer nsol
  real sol_str(100),sol_dip(100),sol_rake(100),str,dip,rake
  real str_sdv(100),dip_sdv(100),rake_sdv(100),xtmp,nsol_str(100),nsol_dip(100),nsol_rake(100)
  real nstr_sdv(100),ndip_sdv(100),nrake_sdv(100),xdeg
  integer ihit(9,9,9),id_sol(100),k1,k2,i1,i2

  xazi=azi
  xpo=po
  xdeg=90.0

  !-- Find Gap
  do i=1,npo-1
    do j=i,npo
	  if(xazi(i).gt.xazi(j))then
	    gap=xazi(i)
		xazi(i)=xazi(j)
		xazi(j)=gap
	  endif
	enddo
  enddo

  gap=xazi(1)+360.0-xazi(npo)
  do i=2,npo
    if( (xazi(i)-xazi(i-1)) .gt. gap ) gap=xazi(i)-xazi(i-1)
  enddo
  
  if(gap.ge.180.0)then
    Q_gap=0.0
  else
    Q_gap=(180.0-gap)/90.0
  endif

  if(npo.lt.10)then
    Q_npo=0.0
  else if(npo.lt.50)then
    Q_npo=(npo-10.0)/20.0
  else
    Q_npo=2.0
  endif

  print*,fscore
  if(fscore .lt. 0.4)then      ! default=0.7
    Q_score=0.0
  else
    Q_score=(fscore-0.4)/0.15  ! default=0.7
  endif

  j=0
  do i=1,npo
    if(po(i).gt.0.5)j=j+1
  enddo
  Q_por=( 0.5-abs(j*1.0/real(npo)-0.5) )/0.25

  q=Q_gap*Q_npo*Q_score*Q_por
  if(q.lt.0.001)then
    q=0.0
    return
  endif

  !-- Find the results
  do i=1,nresults
    if(f_str(i).gt.360.0)f_str(i)=f_str(i)-360.0
    if(f_str(i).lt.0.0)  f_str(i)=360.0+f_str(i)
	call newdbcouple(f_str(i),f_dip(i),f_rake(i),strkx,dipx,rakex,tpazimuth,tpplunge)

	do j=1,2
	  if(strkx(j).gt.360.0) strkx(j)=strkx(j)-360.0
	  if(strkx(j).lt.0.0)  strkx(j)=strkx(j)+360.0
	  if(rakex(j).lt.0.0) rakex(j)=360.0+rakex(j)
	enddo

	if(strkx(1).lt.strkx(2))then
	  str=strkx(1)
	  dip=dipx(1)
	  rake=rakex(1)
	else
	  str=strkx(2)
	  dip=dipx(2)
	  rake=rakex(2)
	endif

	ff_str(i)=str
	ff_dip(i)=dip
	ff_rake(i)=rake
	nnsol(i)=1
  enddo

  do i=1,nresults-1
    if(nnsol(i).eq.0)cycle
    do j=i+1,nresults
	  if(abs(ff_str(i)-ff_str(j)).lt.0.2 .and. abs(ff_dip(i)-ff_dip(j)).lt.0.2 .and. abs(ff_rake(i)-ff_rake(j)).lt.0.2	)then
		nnsol(j)=0
	  endif
    enddo
  enddo

  j=0
  ihit=0
  do i=1,nresults
    if(nnsol(i).eq.0)cycle
	j=j+1
	f_str(j)=ff_str(i)
	f_dip(j)=ff_dip(i)
	f_rake(j)=ff_rake(i)
	nnsol(j)=0
	do k=1,9
	do k1=1,9
	do k2=1,9
	  if(f_str(j).ge. (40.*(k-1.0)) .and. f_str(j).le. (40.*k) .and. f_dip(j).ge. (10.*(k1-1.0)) .and. f_dip(j).le. (10.*k1) &
	     .and. f_rake(j).ge. (40.*(k2-1.0)) .and. f_rake(j).le. (40.*k2) )then
	    ihit(k,k1,k2)=ihit(k,k1,k2)+1
	  endif
	enddo
	enddo
	enddo
  enddo
  nresults=j
  ff_str=f_str
  ff_dip=f_dip
  ff_rake=f_rake

  j=0
  do i=1,9
  do i1=1,9
  do i2=1,9
    if(ihit(i,i1,i2).gt.j)then
	  j=ihit(i,i1,i2)
	  k=i
	  k1=i1
	  k2=i2
	endif
  enddo
  enddo
  enddo

  str=k*40.-20.0
  dip=k1*10.-5.0
  rake=k2*40.-20.0


  xtmp=99999.0
  do i=1,nresults
    if( (abs(f_str(i)-str)+abs(f_dip(i)-dip)+abs(f_rake(i)-rake)) .lt. xtmp)then
	  xtmp=abs(f_str(i)-str)+abs(f_dip(i)-dip)+abs(f_rake(i)-rake)
	  j=i
	endif
  enddo
  nsol=1
  sol_str(nsol)=f_str(j)
  sol_dip(nsol)=f_dip(j)
  sol_rake(nsol)=f_rake(j)
  nnsol=0

  do i=1,nresults
	nnsol(i)=0
	call newdbcouple(f_str(i),f_dip(i),f_rake(i),strkx,dipx,rakex,tpazimuth,tpplunge)

	do j=1,2
	  if(strkx(j).gt.360.0) strkx(j)=strkx(j)-360.0
	  if(strkx(j).lt.0.0)  strkx(j)=strkx(j)+360.0
	  if(rakex(j).lt.0.0) rakex(j)=360.0+rakex(j)
	  if(rakex(j).gt.360.0) rakex(j)=rakex(j)-360.0
	enddo

	!-- error due to do not deal with the strike
	do j=1,nsol

	  do k=1,2
 	    str=strkx(k)
	    dip=dipx(k)
	    rake=rakex(k)
	    if(abs(str-sol_str(j))<xdeg .and. abs(dip-sol_dip(j))<xdeg .and. abs(rake-sol_rake(j))<xdeg )then
		  nnsol(i)=j
		  f_str(i)=str
		  f_dip(i)=dip
		  f_rake(i)=rake
	    else if(abs(str-sol_str(j))<xdeg .and. abs(dip-sol_dip(j))<xdeg .and. (360.-abs(rake-sol_rake(j)))<xdeg )then
		  nnsol(i)=j
		  f_str(i)=str
		  f_dip(i)=dip
		  if(sol_rake(j).lt.xdeg)then
		    f_rake(i)=rake-360.0
		  else
		    f_rake(i)=360.0+rake
		  endif
	    else if( (360.-abs(str-sol_str(j)))<xdeg .and. abs(dip-sol_dip(j))<xdeg .and. abs(rake-sol_rake(j))<xdeg )then
		  nnsol(i)=j
		  f_dip(i)=dip
		  f_rake(i)=rake
		  if(sol_str(j).lt.xdeg)then
		    f_str(i)=str-360.0
		  else
		    f_str(i)=360.0+str
		  endif
	    else if( (360.-abs(str-sol_str(j)))<xdeg .and. abs(dip-sol_dip(j))<xdeg .and. (360.-abs(rake-sol_rake(j)))<xdeg )then
		  nnsol(i)=j
		  f_dip(i)=dip
		  if(sol_str(j).lt.xdeg)then
		    f_str(i)=str-360.0
		  else
		    f_str(i)=360.0+str
		  endif
		  if(sol_rake(j).lt.xdeg)then
		    f_rake(i)=rake-360.0
		  else
		    f_rake(i)=360.0+rake
		  endif
	    endif
	    if(nnsol(i).ne.0)exit

 	    str=strkx(k)
	    dip=dipx(k)
	    rake=rakex(k)

	    if(str.lt.180.0)then
		  str=str+180.0
	    else
		  str=str-180.0
	    endif
	    if(dip.gt.45.0)then
	      dip=180.0-dip
		  rake=-1.0*rake
	    else
	      dip=-1.0*dip
	      rake=180.0+1.0*rake
	    endif
	    if(rake.lt.0.0)rake=360.+rake
	    if(rake.gt.360.0)rake=rake-360.0

	    if(abs(str-sol_str(j))<xdeg .and. abs(dip-sol_dip(j))<xdeg .and. abs(rake-sol_rake(j))<xdeg )then
		  nnsol(i)=j
		  f_str(i)=str
		  f_dip(i)=dip
		  f_rake(i)=rake
	    else if(abs(str-sol_str(j))<xdeg .and. abs(dip-sol_dip(j))<xdeg .and. (360.-abs(rake-sol_rake(j)))<xdeg )then
		  nnsol(i)=j
		  f_str(i)=str
		  f_dip(i)=dip
		  if(sol_rake(j).lt.xdeg)then
		    f_rake(i)=rake-360.0
		  else
		    f_rake(i)=360.0+rake
		  endif
	    else if( (360.-abs(str-sol_str(j)))<xdeg .and. abs(dip-sol_dip(j))<xdeg .and. abs(rake-sol_rake(j))<xdeg )then
		  nnsol(i)=j
		  f_dip(i)=dip
		  f_rake(i)=rake
		  if(sol_str(j).lt.xdeg)then
		    f_str(i)=str-360.0
		  else
		    f_str(i)=360.0+str
		  endif
	    else if( (360.-abs(str-sol_str(j)))<xdeg .and. abs(dip-sol_dip(j))<xdeg .and. (360.-abs(rake-sol_rake(j)))<xdeg )then
		  nnsol(i)=j
		  f_dip(i)=dip
		  if(sol_str(j).lt.xdeg)then
		    f_str(i)=str-360.0
		  else
		    f_str(i)=360.0+str
		  endif
		  if(sol_rake(j).lt.xdeg)then
		    f_rake(i)=rake-360.0
		  else
		    f_rake(i)=360.0+rake
		  endif
	    endif
	    if(nnsol(i).ne.0)exit
	  enddo
	  if(nnsol(i).ne.0)exit
	enddo

	if(nnsol(i).eq.0)then
	  nsol=nsol+1
	  sol_str(nsol)=f_str(i)
	  sol_dip(nsol)=f_dip(i)
	  sol_rake(nsol)=f_rake(i)
	  nnsol(i)=nsol
	endif
	if(nsol.eq.100)exit
  enddo

  do i=1,nsol
	k=0
	str=0.0
	dip=0.0
	rake=0.0
	id_sol(i)=0
    do j=1,nresults
	  if(nnsol(j).eq.i)then
	    k=k+1
		str=str+f_str(j)
		dip=dip+f_dip(j)
		rake=rake+f_rake(j)	
	  endif	  
	enddo

	if(k.gt.0)then
	  str=str/real(k)
	  dip=dip/real(k)
	  rake=rake/real(k)
	else
	  id_sol(i)=100
	  cycle
	endif

	xtmp=999999999.9
	do j=1,nresults
	  if(nnsol(j).eq.i)then
		if( abs(str-f_str(j))+abs(dip-f_dip(j))+abs(rake-f_rake(j)) .lt. xtmp )then
		  xtmp=abs(str-f_str(j))+abs(dip-f_dip(j))+abs(rake-f_rake(j))
		  sol_str(i)=f_str(j)
		  sol_dip(i)=f_dip(j)
		  sol_rake(i)=f_rake(j)
		endif
      endif
	enddo
	
	str_sdv(i)=0.0
	dip_sdv(i)=0.0
	rake_sdv(i)=0.0

	xtmp=0.0
	do j=1,nresults
	  if(nnsol(j).eq.i)then
	    xtmp=xtmp+1.0
		str_sdv(i)=str_sdv(i)+(f_str(j)-str)*(f_str(j)-str)
		dip_sdv(i)=dip_sdv(i)+(f_dip(j)-dip)*(f_dip(j)-dip)
		rake_sdv(i)=rake_sdv(i)+(f_rake(j)-rake)*(f_rake(j)-rake)
      endif
	enddo

	if(xtmp.gt.1.5)then
	  str_sdv(i)=2.*sqrt(str_sdv(i)/xtmp)
	  dip_sdv(i)=2.*sqrt(dip_sdv(i)/xtmp)
	  rake_sdv(i)=2.*sqrt(rake_sdv(i)/xtmp)
	else
	  str_sdv(i)=1.4
	  dip_sdv(i)=1.4
	  rake_sdv(i)=1.4
	endif
	
	if( str_sdv(i).lt.1.0) str_sdv(i)=1.0
	if( dip_sdv(i).lt.1.0) dip_sdv(i)=1.0
	if(rake_sdv(i).lt.1.0)rake_sdv(i)=1.0

	if(sol_dip(i).gt.90.0)then
	  sol_str(i)=sol_str(i)+180.0
	  sol_dip(i)=180.0-sol_dip(i)
	  sol_rake(i)=360.-sol_rake(i)
	else if(sol_dip(i).lt.0.0)then
	  sol_str(i)=sol_str(i)+180.0
	  sol_dip(i)=-1.0*sol_dip(i)
	  sol_rake(i)=360.-sol_rake(i)  
	endif
	if(sol_str(i).gt.360.0)sol_str(i)=sol_str(i)-360.0
	if(sol_rake(i).gt.180.0)sol_rake(i)=sol_rake(i)-360.0
	if(sol_rake(i).lt.-180.0)sol_rake(i)=sol_rake(i)+360.0
  enddo
  !-- passing results

  j=0
  do i=1,nsol
    if(id_sol(i).eq.100)then
	  cycle
	else
	  j=j+1
	  nsol_str(j)=sol_str(i)
	  nsol_dip(j)=sol_dip(i)
	  nsol_rake(j)=sol_rake(i)
	  nstr_sdv(j)=str_sdv(i)
	  ndip_sdv(j)=dip_sdv(i)
	  nrake_sdv(j)=rake_sdv(i)
	endif
  enddo
  nsol=j

  Q=Q_gap*Q_npo*Q_score*Q_por/real(nsol)
end subroutine calculating_quality

subroutine read_pfile(pfile,npo,azi,tk,po)
  character*64 pfile
  character ud
  real azi(1000),tk(1000),po(1000)
  integer npo,ichange

  ichange=0

  open(8,file=pfile,status='old')
  read(8,*)

  npo=0
  do 
    read(8,'(11x,i4,i4,a1)',iostat=iret)iazi,itka,ud
	if(iret < 0)exit
	if(ud.eq.' ')cycle
    if(itka.gt.90)then
      iazi=180+iazi
      itka=180-itka                  
      if(iazi.gt.360) iazi=iazi-360
    endif
	if(ichange .eq. 1)then
      if(ud.eq.'+') xud=-1.0
      if(ud.eq.'-') xud= 1.0
      if(ud.eq.' ') xud= 0.0
	else
      if(ud.eq.'+') xud= 1.0
      if(ud.eq.'-') xud=-1.0
      if(ud.eq.' ') xud= 0.0
	endif
	if(xud.eq.0.0)cycle
	npo=npo+1
	azi(npo)=iazi
	tk(npo)=itka
	po(npo)=xud
  enddo
  close(8)
end subroutine read_pfile


!***** Calculating two planes strike,dip, rake and P&T 
subroutine newdbcouple(strk,dip,rake,strkx,dipx,rakex,tpazimuth,tpplunge)
  real*8 pi,degree,x,y,z
  real*8 dip1,strk1,rake1,fp11,fp12,fp13
  real*8 dip2,strk2,rake2,fp21,fp22,fp23
  real*8 dipt,strkt,tax1,tax2,tax3
  real*8 dipp,strkp,pax1,pax2,pax3
  real*8 val
  real*8 dipx(2),strkx(2),rakex(2)
  real*8 tpazimuth(2),tpplunge(2)
  real*4 strk,dip,rake
  pi=1.0
  pi=4.0*datan(pi)
  degree=pi/180.0
  dipx(1) =dip
  strkx(1)=strk
  rakex(1)=rake
  dip1 = dip*degree
  strk1=strk*degree
  rake1=rake*degree
  fp11= dsin(dip1)*dcos(strk1)
  fp12=-dsin(dip1)*dsin(strk1)
  fp13= dcos(dip1)
  fp21=dcos(rake1)*dsin(strk1)-dcos(dip1)*dsin(rake1)*dcos(strk1)
  fp22=dcos(rake1)*dcos(strk1)+dcos(dip1)*dsin(rake1)*dsin(strk1)
  fp23=dsin(rake1)*dsin(dip1)
  tax1=fp11+fp21
  tax2=fp12+fp22
  tax3=fp13+fp23
  val =tax1*tax1+tax2*tax2+tax3*tax3
  val =dsqrt(val)
  tax1=tax1/val
  tax2=tax2/val
  tax3=tax3/val
  pax1=fp11-fp21
  pax2=fp12-fp22
  pax3=fp13-fp23
  val =pax1*pax1+pax2*pax2+pax3*pax3
  val =dsqrt(val)
  pax1=pax1/val
  pax2=pax2/val
  pax3=pax3/val
  x=fp21
  y=fp22
  z=fp23
  if(fp23.lt.0.0)then
    x=-x
    y=-y
    z=-z
  endif
  dip2=dacos(z)
  if(z.lt.1.0)then
    x=x/dsin(dip2)
    y=y/dsin(dip2)
  endif
  z=-1.0*y
  y=x
  x=z
  if(y.gt. 1.0) y= 1.0
  if(y.lt.-1.0) y=-1.0
  val=dacos(y)
  val=dabs(val)
  if(x.ge.0.0) strk2= val
  if(x.lt.0.0) strk2=-val
  if(fp23.lt.0.0)then
    x=-x
    y=-y
  endif
  val=x*fp11+y*fp12
  if(val.gt. 1.0)val= 1.0
  if(val.lt.-1.0)val=-1.0
  val=dacos(val)
  val=dabs(val)
  if(fp23.ge.0.0)rake2= val
  if(fp23.lt.0.0)rake2=-val
  dip2 = dip2/degree
  strk2=strk2/degree
  rake2=rake2/degree
  if(strk2.lt.  0.)strk2=strk2+360.0
  if(strk2.ge.360.)strk2=strk2-360.0
  dipx(2) =dip2
  strkx(2)=strk2
  rakex(2)=rake2
  x=tax1
  y=tax2
  z=tax3
  if(tax3.gt.0.0)then
    x=-x
    y=-y
    z=-z
  endif
  val =dabs(z)
  dipt=dasin(val)
  if(z.ne.0.0)then
    x=x/dcos(dipt)
    y=y/dcos(dipt)
  endif
  if(y.gt. 1.0) y= 1.0
  if(y.lt.-1.0) y=-1.0
  val=dacos(y)
  val=dabs(val)
  if(x.ge.0.0) strkt= val
  if(x.lt.0.0) strkt=-val
  dipt = dipt/degree
  strkt=strkt/degree
  if(strkt.lt.  0.) strkt=strkt+360.0
  if(strkt.ge.360.) strkt=strkt-360.0
  tpplunge(1) =dipt
  tpazimuth(1)=strkt
  x=pax1
  y=pax2
  z=pax3
  if(pax3.gt.0.0)then
    x=-x
    y=-y
    z=-z
  endif
  val =dabs(z)
  dipp=dasin(val)
  if(z.ne.0.0)then
    x=x/dcos(dipp)
    y=y/dcos(dipp)
  endif
  if( y.gt. 1.0)y= 1.0
  if( y.lt.-1.0)y=-1.0
  val=dacos(y)
  val=dabs(val)
  if(x.ge.0.0)strkp= val
  if(x.lt.0.0)strkp=-val
  dipp = dipp/degree
  strkp=strkp/degree
  if(strkp.lt.  0.) strkp=strkp+360.0
  if(strkp.ge.360.) strkp=strkp-360.0
  tpplunge(2) =dipp
  tpazimuth(2)=strkp
end subroutine newdbcouple


subroutine ga_find_fpl(azi,tk,po,npo,f_str,f_dip,f_rake,nresults,fscore,npopulation,reproduction,mutation,nbits)
  
  implicit none

  real azi(1000),tk(1000),po(1000),score,reproduction,mutation
  integer npo,npopulation

  !-- GA analysis
  integer*1 PP(10000,25),GA(10000,25),GACODE(25),G1(25),G2(25),G3(25),G4(25)
  real PP_SCORE(10000),str,dip,rake,f_str(1000),f_dip(1000),f_rake(1000),fscore
  integer NPP,nit,i,j,k,jj,kk,nbits,k1
  real xx,xsum
  integer nresults

  call random_seed()
  
  npp=npopulation

  !-- producing present generation
  do i=1,npp
    call random_number(xx)
	str=xx*360.0
    call random_number(xx)
	dip=xx*90.0
    call random_number(xx)
	rake=xx*360.0
	call ga_encode(str,dip,rake,gacode)
	pp(i,:)=gacode
	call cal_fpl_score(str,dip,rake,npo,azi,tk,po,score)
	pp_score(i)=score 
  enddo

  !--call mean_sdv(pp_score,npp,xmean,sdv)

  nit=20

  do i=1,nit

	call sorting_score(pp,pp_score,npp)

	xsum=0.0
	do j=1,npp/4
	  xsum=xsum+pp_score(j)
	enddo
	xsum=xsum*4./real(npp)
	if(xsum.gt.0.999)exit

	!-- Reproduction prefect or with in top (reproduction*100) %
	j=0
	do 
	  j=j+1
	  if(j.le.nint(npp*reproduction))then
	    ga(j,:)=pp(j,:)
	  else
		exit
	  endif
	enddo
	j=j-1

	do k1=1,nint(npp*mutation)
	  !-- for crossover && mutation
      call random_number(xx)
	  k=xx*npp/2+1
	  g1=pp(k,:)

	  do
        call random_number(xx)
	    kk=xx*npp/2+1
		if(kk.ne.k)exit
	  enddo
	  g2=pp(kk,:)

	  call crossover(g1,g2,g3,g4)

	  g1=g3

	  do kk=1,nbits
        call random_number(xx)
	    jj=xx*25+1
	    if(jj.ge.1 .and. jj.le.25)then
	      if(g1(jj).eq.0)then
	        g1(jj)=1  
	      else
	        g1(jj)=0
	      endif
	    endif
	  enddo
	  j=j+1
	  ga(j,:)=g1
	enddo

	!-- crossover from top 50% of the presents
	do
      call random_number(xx)
	  k=xx*npp/2+1
	  g1=pp(k,:)

	  do
        call random_number(xx)
	    kk=xx*npp/2+1
		if(kk.ne.k)exit
	  enddo
	  g2=pp(kk,:)

	  call crossover(g1,g2,g3,g4)

	  j=j+1
	  ga(j,:)=g3
	  if(j.eq.npp)exit
	  j=j+1
	  ga(j,:)=g4
	  if(j.eq.npp)exit
	enddo
	pp=ga
	do j=1,npp
	  gacode=pp(j,:)
	  call ga_decode(str,dip,rake,gacode)
	  call cal_fpl_score(str,dip,rake,npo,azi,tk,po,score)
	  pp_score(j)=score
	enddo
    !--call mean_sdv(pp_score,npp,xmean,sdv)
  enddo

  call sorting_score(pp,pp_score,npp)

  fscore=pp_score(1)
  nresults=0
  do i=1,npp
    if(pp_score(i).eq.fscore)Then
	  nresults=nresults+1
	  gacode=pp(i,:)
	  call ga_decode(str,dip,rake,gacode)
	  f_str(nresults)=str
	  f_dip(nresults)=dip
	  f_rake(nresults)=rake
	endif
  enddo 

  !print*,'Nresults is',nresults

end subroutine ga_find_fpl


subroutine crossover(g1,g2,g3,g4)
  integer*1 G1(25),G2(25),G3(25),G4(25)
  do i=1,5
    g3(i)=g2(i)
	g4(i)=g1(i)
  enddo

  do i=6,9
    g3(i)=g1(i)
	g4(i)=g2(i)
  enddo
  
  do i=10,13
    g3(i)=g2(i)
	g4(i)=g1(i)
  enddo
  
  do i=14,16
    g3(i)=g1(i)
	g4(i)=g2(i)
  enddo
  
  do i=17,21
    g3(i)=g2(i)
	g4(i)=g1(i)
  enddo
  
  do i=22,25
    g3(i)=g1(i)
	g4(i)=g2(i)
  enddo

end subroutine crossover
 
subroutine sorting_score(pp,pp_score,npp)
  integer*1 PP(10000,25),G(25)
  real PP_SCORE(10000),PS

  do i=1,npp-1
    do j=i+1,npp
	  if(pp_score(i).lt.pp_score(j))then
	    ps=pp_score(i)
		g=pp(i,:)
		pp(i,:)=pp(j,:)
		pp_score(i)=pp_score(j)
		pp(j,:)=g
		pp_score(j)=ps
	  endif
	enddo
  enddo

end subroutine sorting_score

subroutine mean_sdv(x,n,xmean,sdv)
  integer n
  real x(n),xmean,sdv

  xmean=0.0
  do i=1,n
    xmean=xmean+x(i)
  enddo
  xmean=xmean/real(n)

  sdv=0.0
  do i=1,n
    sdv=sdv+(x(i)-xmean)*(x(i)-xmean)
  enddo
  sdv=sqrt(sdv/real(n))

end subroutine mean_sdv

subroutine ga_encode(str,dip,rake,gacode)
  integer*1 gacode(25)
  real str,dip,rake
  integer istr,idip,irake
  istr=str/360.0*512.0
  idip=dip/90.0*128.0
  irake=rake/360.0*512.0
  do i=1,9
    gacode(i)=mod(istr,2)
	istr=istr/2
  enddo
  do i=10,16
    gacode(i)=mod(idip,2)
	idip=idip/2
  enddo
  do i=17,25
    gacode(i)=mod(irake,2)
	irake=irake/2
  enddo
end subroutine ga_encode

subroutine ga_decode(str,dip,rake,gacode)
  integer*1 gacode(25)
  real str,dip,rake
  integer istr,idip,irake
  istr=0
  do i=1,9
	istr=istr+gacode(i)*2**(i-1)
  enddo
  idip=0
  do i=10,16
	idip=idip+gacode(i)*2**(i-10)
  enddo
  irake=0
  do i=17,25
	irake=irake+gacode(i)*2**(i-17)
  enddo
  str=istr*360./512.0
  dip=idip*90.0/128.0
  rake=irake*360.0/512.0
end subroutine ga_decode

subroutine cal_fpl_score(str,dip,rake,npo,azi,tk,po,score)
  real azi(1000),tk(1000),po(1000),score
  real str,dip,rake
  integer npo
    
  PI=ACOS(-1.D0)
  
  CDR=PI/180.
  cstr=str*cdr
  cdip=dip*cdr
  crake=rake*cdr

  n=0
  do i=1,npo
    call PAMP(CDIP,CSTR,CRAKE,AZI(I),TK(I),AMP)
	if( (amp*po(i)).gt.0.0 )then
	  n=n+1
	else
	  n=n-1
	endif
  enddo
  score=1.0*n/real(npo)
end subroutine cal_fpl_score

SUBROUTINE PAMP(DIP,STR,SLP,AZIMUTH,TAKEOFF,AMP)
  PI=ACOS(-1.D0)
  CDR=PI/180.
  SNSLP=SIN(SLP)
  CSSLP=COS(SLP)
  SNDIP=SIN(DIP)
  CSDIP=COS(DIP)
  SN2DP=2.*SNDIP*CSDIP
  CS2DP=CSDIP*CSDIP-SNDIP*SNDIP
! TAKEOFF ANGLE
  REM=TAKEOFF*CDR
  SNEM=SIN(REM)
  CSEM=COS(REM)
  SNEM2=SNEM*SNEM
  CSEM2=CSEM*CSEM
  SN2EM=2.*SNEM*CSEM
! AZIMUTH
  AZM=AZIMUTH*CDR-STR
  SNAZ=SIN(AZM)
  CSAZ=COS(AZM)
  SN2AZ=2.*SNAZ*CSAZ
  SNAZ2=SNAZ*SNAZ
! P-WAVE RADIATION
  AMP=CSSLP*SNDIP*SNEM2*SN2AZ
  AMP=AMP-CSSLP*CSDIP*SN2EM*CSAZ
  TEMP=CSEM2-SNEM2*SNAZ2
  AMP=AMP+SNSLP*SN2DP*TEMP
  AMP=AMP+SNSLP*CS2DP*SN2EM*SNAZ
END SUBROUTINE PAMP
