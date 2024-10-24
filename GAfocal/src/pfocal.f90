subroutine pfocal(pfile,ouf,strk,dip,rake,strk_sdv,dip_sdv,rake_sdv,q,fscore)

  integer pgopen
  real*8 dipx,strkx,rakex,tpplunge,tpazimuth
  real dip,strk,rake,strk_sdv,dip_sdv,rake_sdv,q,fscore
  character*64 pfile,ouf

  common/dcouple/dipx(2),strkx(2),rakex(2)
  common/ptpola/tpazimuth(2),tpplunge(2)
  !print*,'Input Pfile name'
  !read(*,'(a)')pfile
  !print*,'Input Strike(-360 to 360), Dip(0 to 90), Rake(-180 to 180)'
  !print*,'Remark: Rake=90 is pure thrust, Rake=-90 is pure normal fault!'
  !read(*,*)strk,dip,rake
  !print*,'1-Changing polarities'
  !read(*,*)ichange
  ichange=0
   

  istat=PGOPEN(ouf(1:len_trim(ouf))//'.ps/vcps')
  call pgsubp(1,2)
  call dbcouple(strk,dip,rake)
  call pgslw(2)
  call pgenv(-12.0,12.0,-12.0,12.0,1,-1,1.0) !just,iaxis
  call pgscf(3)
  ri=10.0
  call plot_circle(0.0,0.0,ri)
  call pattn(ri)
  call plot_text(ri,strk_sdv,dip_sdv,rake_sdv,q,fscore)
  call plot_pfile(pfile,ri,ichange)
  call plot_PT_axis(ri)

  call pgend
end subroutine pfocal


subroutine plot_pfile(pfile,ri,ichange)
  character*64 pfile,card
  real ri
  character ud
  real*8 sq2,deg
  open(8,file=pfile,status='old')
  read(8,'(1x,i2)')iy2000
  rewind(8)
  if(iy2000<50)then
    read(8,4001)iyr,imo,iday,ihr,min,sec,lat,xlat,lon,xlon,dep,xmag
    4001 format(1x,i4,4i2,f6.2,i2,f5.2,i3,f5.2,f6.2,f4.1)
  else
    read(8,4002)iyr,imo,iday,ihr,min,sec,lat,xlat,lon,xlon,dep,xmag
    4002 format(1x,i2,4i2,f6.2,i2,f5.2,i3,f5.2,f6.2,f4.1)
    iyr=iyr+1900
  endif
  xlat=lat+xlat/60.
  xlon=lon+xlon/60.
  card='1234/67/90 23:56:89.1 M4.6'
  write(card,10)iyr,imo,iday,ihr,min,sec,xmag
  10 format(i4.4,'/',i2.2,'/',i2.2,', ',i2.2,':',i2.2,':',f5.2,', ','M',f3.1)
  call pgsch(0.75)
  call pgtext(-11.5,11.0,trim(card))

  write(card,11)xlat,xlon
  11 format('Lat.:',f7.3,', Long.:',f8.3)
  call pgtext(-11.5,10.2,trim(card))

  write(card,12)dep
  12 format('Depth:',f6.1,' km')
  call pgtext(-11.5,9.4,trim(card))

  r=ri
  dx=ri/50.
  sq2=2.0
  sq2=dsqrt(sq2)
  deg=1.0
  deg=4.0*datan(deg)/180.0
  call pgsch(1.0)
  call pgslw(2)
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

    rotaz=(iazi-90.0)*deg
    strk=iazi*deg
    tang=itka*deg
    tang2=tang/2.0
    stan=sin(tang2)
    rx=r*stan*sq2
    px=rx*sin(strk)
    py=rx*cos(strk)
	if(xud.lt.0.0) call plot_circle(px,py,dx)
	if(xud.gt.0.0) call pgcirc(px,py,dx)
  enddo
  close(8)
end subroutine plot_pfile

subroutine plot_PT_axis(ri)
  implicit real*8 (a-h,o-y)
  real px,py,ri,dx
  common/ptpola/ strk(2),dip(2)
  call pgsch(1.2)
  call pgslw(2)
  r=ri
  dx=ri/40.
  sq2=2.0
  sq2=dsqrt(sq2)
  deg=1.0
  deg=4.0*datan(deg)/180.0
  call pgsch(1.0)
  do i=1,2
    str=strk(i)*deg
    di1=dip(i)*deg
	xtmp=1.0
    thiti=2.0*datan(xtmp)-di1
    sii=dsin(thiti/2.0)
    rx=r*sq2*sii
    px=rx*dsin(str)
    py=rx*dcos(str)
    if(i.eq.1)then
	  !-- for T-axix
	  call pgtext(px-dx,py-dx,'T')
	else 
	  !-- for P-axis
	  call pgtext(px-dx,py-dx,'P')
	endif
  enddo
end	subroutine plot_PT_axis

subroutine plot_text(ri,strk_sdv,dip_sdv,rake_sdv,q,fscore)
  real*8 dipx,strkx,rakex,tpplunge,tpazimuth
  character*120 card
  integer idip,istr,irake
  real strk_sdv,dip_sdv,rake_sdv,q,fscore

  common/dcouple/dipx(2),strkx(2),rakex(2)
  common/ptpola/tpazimuth(2),tpplunge(2)

  call pgsch(1.0)
  d=ri/20.
  call pgmove(-d,0.)
  call pgdraw(d,0.)
  call pgmove(0.,-d)
  call pgdraw(0.,d)
  d=d/2.
  call pgmove(0.,ri-d)
  call pgdraw(0.,ri+d)
  call pgtext(-d,ri+1.5*d,'N')
  call pgmove(0.,-ri-d)
  call pgdraw(0.,-ri+d)
  call pgtext(-d,-ri-3.5*d,'S')
  call pgmove(ri-d,0.)
  call pgdraw(ri+d,0.)
  call pgtext(ri+1.5*d,-d,'E')
  call pgmove(-ri-d,0.)
  call pgdraw(-ri+d,0.)
  call pgtext(-ri-3.5*d,-d,'W')

  !--
  call pgsch(0.75)
  card='Strike:IIII, Dip:III, Rake:IIII.'
  istr=dnint(strkx(1))
  write(card(8:11),'(i4)')istr
  idip=dnint(dipx(1))
  write(card(18:20),'(i3)')idip
  irake=dnint(rakex(1))
  write(card(28:31),'(i4)')irake
  call pgtext(-11.5,-10.5,'A. '//card)
  
  istr=dnint(strkx(2))
  write(card(8:11),'(i4)')istr
  idip=dnint(dipx(2))
  write(card(18:20),'(i3)')idip
  irake=dnint(rakex(2))
  write(card(28:31),'(i4)')irake
  call pgtext(-11.5,-11.2,'B. '//card)

  Card='P-axis. Azimuth:iiii, Plunge:iii.'
  istr=dnint(tpazimuth(2))
  write(card(17:20),'(i4)')istr
  idip=dnint(tpplunge(2))
  write(card(30:32),'(i3)')idip
  call pgtext(1.3,-10.5,card)
  Card='T-axis. Azimuth:iiii, Plunge:iii.'
  istr=dnint(tpazimuth(1))
  write(card(17:20),'(i4)')istr
  idip=dnint(tpplunge(1))
  write(card(30:32),'(i3)')idip
  call pgtext(1.4,-11.2,card)

  dx=ri/50.
  call plot_circle(-11.0,7.8,dx)
  call pgtext(-10.7,7.8-dx,'Down')
  call pgcirc(-11.0,8.6,dx)
  call pgtext(-10.7,8.6-dx,'Up')

  call pgtext(3.5,11.0,'Fault Plane A Uncertainty')
  write(card,20)nint(strk_sdv),nint(dip_sdv),nint(rake_sdv)
20 format('Strike+-',i3,', Dip+-',i3,', Rake+-',i3,'.')
  call pgtext(0.8,10.2,card)
  write(card,22)q
22 format('Quality Index = ',f5.2)
  call pgtext(4.8,9.4,card) 
  xmisfit=(1.-fscore)/2.
  write(card,23)xmisfit*100.0
23 format('Misfit = ',f5.2,'%')
  call pgtext(6.5,8.6,card) 
end subroutine plot_text

subroutine plot_circle(xc,yc,ri)
  real xc,yc,ri
  real*8 pi,dpi
  pi=1.0
  pi=4.0*datan(pi)
  dpi=pi*2./360.
  
  do i=0,360
    ra=i*dpi
    px=xc+ri*cos(ra)
	py=yc+ri*sin(ra)
	if(i.eq.0)then
	  call pgmove(px,py)
	else
	  call pgdraw(px,py)
	endif
  enddo
 
end subroutine plot_circle

subroutine pattn(ri)
  implicit real*8 (a-h,o-y)
  real*4 px,py,ri
  common/dcouple/ dip(2),strk(2),slip(2)
  r=ri
  x=1.0
  deg=4.0*datan(x)/180.0
  sq2=2.0
  sq2=dsqrt(sq2)
  do i=1,2
    str=strk(i)*deg
	if(dip(i).gt.89.9999)then
      di=dip(i)*deg-0.0001
	else if(dip(i).lt. 0.0001)then
      di=dip(i)*deg+0.0001
	else
      di=dip(i)*deg
	endif
    zxo=r*dcos(str)
    zyo=r*dsin(str)
	px=zyo
	py=zxo
    call pgmove(px,py)
    do j=90,-90,-1
      k=j+90
      a=j
      theta=a*deg
      ta1=dtan(di)
      si1=dsin(theta)
      co1=dcos(theta)
      aa1=co1*ta1
      aa2=aa1**2.0+1.0
      aa3=dsqrt(aa2)
      coi=aa1/aa3
      thiti=dacos(coi)
      sii=dsin(thiti/2.0)
      x=r*sii*co1*sq2
      y=r*sii*si1*sq2
      zxx=x*dcos(str)+y*dsin(str)
      zyy=-1.0*x*dsin(str)+y*dcos(str)
	  px=zxx
	  py=zyy
      call pgdraw(px,py)
    enddo
  enddo
end	subroutine pattn


!***** Calculating two planes strike,dip, rake and P&T 
subroutine dbcouple(strk,dip,rake)
  real*8 pi,degree,x,y,z
  real*8 dip1,strk1,rake1,fp11,fp12,fp13
  real*8 dip2,strk2,rake2,fp21,fp22,fp23
  real*8 dipt,strkt,tax1,tax2,tax3
  real*8 dipp,strkp,pax1,pax2,pax3
  real*8 val
  real*8 dipx,strkx,rakex,tpplunge,tpazimuth
  common/dcouple/dipx(2),strkx(2),rakex(2)
  common/ptpola/tpazimuth(2),tpplunge(2)
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
end subroutine dbcouple

