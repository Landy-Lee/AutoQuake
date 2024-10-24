#!/bin/bash
echo $1
rm -f psplots/${1}.ps
gmtset BASEMAP_TYPE=plain
gmtset PLOT_DEGREE_FORMAT=+
psbasemap -JS0/-90/6i -R0/360/-90/0 -B30/30 -P -K -V -Y3.0i > psplots/${1}.ps
echo "0 1.2 22 0 0 MC $1" | pstext -JX6i -R-1/1/-1/1 -G0 -N -K -O -V >> psplots/${1}.ps
#grep $1 results.txt | gawk '{print 0,0,$5,$7,$9,$11,$6,0,0}' | psmeca -J -R -Sa6i -G150/230/255 -L4/0 -N -M -K -O -V >> psplots/${1}.ps
psmeca -J -R -Sa6i -G150/230/255 -L4/0 -N -M -K -O -V >> psplots/${1}.ps <<EOF
0 0 0 $2 $3 $4 5 0 0
EOF
gawk 'NR>1 && substr($0,20,1)=="+" {if(int(substr($0,16,4))>90) print 180+substr($0,12,4),90-substr($0,16,4); else print substr($0,12,4),substr($0,16,4)-90}' pfiles/$1 | psxy -JS0/-90/6i -R0/360/-90/0 -S+0.4 -W8/0 -N -K -O -V >> psplots/${1}.ps
gawk 'NR>1 && substr($0,20,1)=="-" {if(int(substr($0,16,4))>90) print 180+substr($0,12,4),90-substr($0,16,4); else print substr($0,12,4),substr($0,16,4)-90}' pfiles/$1 | psxy -J -R -Sc0.3 -W6/0 -N -O -V >> psplots/${1}.ps

#gs psplots/${1}.ps
