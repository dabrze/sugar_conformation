#!/bin/bash
echo "(mg33)"
echo "Correct RMSD for bonds and angles; phenix.refine; .geo and _final.geo files" 
echo "(refine with 'write_final_geo_file=True')" 
#echo "input file --->" $1 "file from phenix.refine (mg33)" 
echo ""
if [ -z "$2" ] 
	then 
	da=4.0
else 
	da=$2 
fi
if [ -z "$3" ] 
	then 
	db=0.02 
else 
	db=$3 
fi
if [ -e $1 ]
	then
#    echo "ok"
	echo "input file --->" $1 ".geo file from phenix.refine (mg33)" 
	echo ""
##echo $da
awk -v da="$da"  '
function abs(value)
{
  return (value<0?-value:value);
}
function sgn(x)
{ 
if (x>0) return "-"; 
else if (x<0) return "+" ; 
else return " "
}

BEGIN { na=0;da2s=0; aaa=0 }

{

# if ($1=="Disulphide") {n=$5; getline; getline;
# for (i = 1; i <= n; ++i) {getline;getline;getline;getline;getline;} ; getline; print $0}

# Angles

if ($1=="angle" && substr($2,1,4)=="pdb=")
	{
		NF-=1;
		if ((NF-2)==5){a1=$3"^"$4" "$5""$7" "$6; a111=$2}
		else
		{a1=$3" "$4""$6" "$5; a111=$2};
#		
		getline; 
		NF-=1; 
		if ((NF-1)==5){a2=$2"^"$3" "$4""$6" "$5;a222=$1}
		else
		{a2=$2" "$3""$5" "$4;a222=$1};
#
		getline; 
		NF-=1; 
		if ((NF-1)==5){a3=$2"^"$3" "$4""$6" "$5; a333=$1}
		else
		{a3=$2" "$3" "$4""$5; a333=$1};
#
		getline;
		getline; 
		NF-=3; 
##		if ($3==0) {$3="0.00"}
##		if (abs($3)>da && (substr(a1,1,1)!="H" && substr(a2,1,1)!="H" && substr(a3,1,1)!="H" ))
##		{printf "%-15s %-15s %-12s %8.3f %8.3f %s %6.3f \n",a1,a2,a3,$1,$2,sgn($3),abs($3);na+=1;da2=$3*$3;da2s+=da2}

		if (substr(a1,1,1)!="H" && substr(a2,1,1)!="H" && substr(a3,1,1)!="H" && substr(a111,1,6)!= "pdb=\"H" && substr(a222,1,6)!="pdb=\"H" && substr(a333,1,6)!="pdb=\"H" )
		{na+=1;da2=$3*$3;da2s+=da2}

#		else 
#		{print a1, a2, a3;aaa+=1 }
	} 
}END {
print "na=",na; 
#print "aaa=",aaa; 
printf "rmsd angles=%8.3f \n\n",(da2s/na)**0.5}' $1 
### | awk '{print $NF,$0}' | sort -n | cut -f2- -d' ' 

awk -v db="$db" '
function abs(value)
{
  return (value<0?-value:value);
}
function sgn(x)
{ 
if (x>0) return "+"; 
else if (x<0) return "-" ; 
else return " "
}

BEGIN { nb=0;db2s=0; bbb=0 }

{

# Bonds

if ($1=="bond" && substr($2,1,4)=="pdb=")
	{
		NF-=1;
		if ((NF-2)==5){b1=$3"^"$4" "$5""$7" "$6; b111=$2}
		else
		{b1=$3" "$4""$6" "$5;b111=$2};
#		
		getline; 
		NF-=1; 
		if ((NF-1)==5){b2=$2"^"$3" "$4""$6" "$5; b222=$1}
		else
		{b2=$2" "$3""$5" "$4; b222=$1};
#		
		getline; 
		getline; 
		NF-=3; 
##		if ($3==0) {$3="0.00"}
##		print b1,"\t",b2,"\t",$1,$2,"\t",sgn($3),abs($3)
##		if (abs($3)>db && (substr(b1,1,1)!="H" && substr(b2,1,1)!="H"))
##		{printf "%-15s %-15s %8.3f %8.3f %s %6.3f\n",b1,b2,$1,$2,sgn($3),abs($3); nb+=1;db2=$3*$3;db2s+=db2}

		if (substr(b1,1,1)!="H" && substr(b2,1,1)!="H" &&  substr(b111,1,6)!= "pdb=\"H" && substr(b222,1,6)!="pdb=\"H"  )
		{nb+=1;db2=$3*$3;db2s+=db2}

#		else 
#		{print b1, b2, b111, b222; bbb+=1}
	} 

} END {
print "nb=",nb;  
#print "bbb=",bbb;
printf "rmsd bonds= %8.3f \n\n",(db2s/nb)**0.5}' $1 

###| awk '{print $NF,$0}' | sort -n | cut -f2- -d' ' 

else
    echo "Input file $1 does not exist !"
echo
echo "usage: read-geo phenix-output-file.geo"
echo

fi


