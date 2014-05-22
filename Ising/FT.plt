reset

#set terminal postscript enhanced
#set output "FT.ps"

#set logscale x

set xrange [-1:1]
#set yrange [.65:.75]

bJ = 4.
set xrange [-1:1]
plot -0.5*bJ*x**2+ log(2*cosh(bJ*x))

H(x) = -x*log(x) - (1-x)*log(1-x)
replot 0.5*bJ*x**2+ H((x+1)/2)