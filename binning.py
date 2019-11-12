#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:59:52 2019

@author: Heitor
"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

     


def Resize(source, length):
    step = float(len(source) - 1) / (length - 1)
    for i in range(length):
        key = i * step
        low = source[int(math.floor(key))]
        high = source[int(math.ceil(key))]
        ratio = key % 1
        yield (1 - ratio) * low + ratio * high

 
    
def save(wl,fl,NAME):
    FNAME=NAME+'.B'
    file = open(FNAME, 'w')
    file.write(NAME[12:25]+'\n')
    file.write('# Wavelength(A) Flux \n')
    n=0   
    for i in fl:
        flu  = float(fl[n])
        wave = float(wl[n])
        file.write('%7.5f %10.7f\n'%(wave,flu))     
        n=n+1
    print(NAME+' saved as ' + FNAME)
   
    
#-----------------------------------------

sns.set_style("white")
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.5})

#-----------------------------------------

star='G'

met='m1'


 
line_list= 'linelistCUBES.txt'

Z,lamb = np.genfromtxt(line_list, unpack=True, dtype=str)

print('\n' +str(len(Z))+ ' lines. \n')


#-----------------------------------------
#spectrum 

#CUBES

INPUT_spec11= 'StellarGrid/flux_'+star+'_'+met+'_m05.norm.nulbad.0.150'
INPUT_spec12= 'StellarGrid/flux_'+star+'_'+met+'_0.norm.nulbad.0.150'
INPUT_spec13= 'StellarGrid/flux_'+star+'_'+met+'_p05.norm.nulbad.0.150'
INPUT_spec14= 'StellarGrid/flux_'+star+'_'+met+'_p10.norm.nulbad.0.150'

wl11,flux11 = np.genfromtxt(INPUT_spec11, skip_header=2, unpack=True)
wl12,flux12 = np.genfromtxt(INPUT_spec12, skip_header=2, unpack=True)
wl13,flux13 = np.genfromtxt(INPUT_spec13, skip_header=2, unpack=True)
wl14,flux14 = np.genfromtxt(INPUT_spec14, skip_header=2, unpack=True)


#UVES

INPUT_spec21= 'StellarGrid/flux_'+star+'_'+met+'_m05.norm.nulbad.0.070'
INPUT_spec22= 'StellarGrid/flux_'+star+'_'+met+'_0.norm.nulbad.0.070'
INPUT_spec23= 'StellarGrid/flux_'+star+'_'+met+'_p05.norm.nulbad.0.070'
INPUT_spec24= 'StellarGrid/flux_'+star+'_'+met+'_p10.norm.nulbad.0.070'

wl21,flux21 = np.genfromtxt(INPUT_spec21, skip_header=2, unpack=True)
wl22,flux22 = np.genfromtxt(INPUT_spec22, skip_header=2, unpack=True)
wl23,flux23 = np.genfromtxt(INPUT_spec23, skip_header=2, unpack=True)
wl24,flux24 = np.genfromtxt(INPUT_spec24, skip_header=2, unpack=True)




#wlfl11 = numpy.delete(wlfl11, (14990), axis=1)
#-----------------------------------------

#REBIN


#CUBES
pace=0.06
x= int((wl11[len(wl11)-1]-wl11[0] )/pace)

wl11b=list(Resize(wl11, x))
flux11b=list(Resize(flux11, x))


x = int((wl12[len(wl12)-1]-wl12[0] )/pace)

wl12b=list(Resize(wl12, x))
flux12b=list(Resize(flux12, x))


x= int((wl13[len(wl13)-1]-wl13[0] )/pace)

wl13b=list(Resize(wl13, x))
flux13b=list(Resize(flux13, x))

x= int((wl14[len(wl14)-1]-wl13[0] )/pace)

wl14b=list(Resize(wl14, x))
flux14b=list(Resize(flux14, x))

#--------------

#UVES
pace=0.03
x= int((wl21[len(wl21)-1]-wl21[0] )/pace)

wl21b=list(Resize(wl21, x))
flux21b=list(Resize(flux21, x))


x= int((wl22[len(wl22)-1]-wl22[0] )/pace)

wl22b=list(Resize(wl22, x))
flux22b=list(Resize(flux22, x))

x= int((wl23[len(wl23)-1]-wl23[0] )/pace)

wl23b=list(Resize(wl23, x))
flux23b=list(Resize(flux23, x))


x= int((wl24[len(wl24)-1]-wl24[0] )/pace)

wl24b=list(Resize(wl24, x))
flux24b=list(Resize(flux24, x))



#-----------------------------------------

plt.plot(wl11,flux11, 'k')
plt.plot(wl11b, flux11b, 'red')


#-----------------------------------------

#CUBES SAVE
save(wl11b,flux11b,INPUT_spec11)
save(wl12b,flux12b,INPUT_spec12)
save(wl13b,flux13b,INPUT_spec13)
save(wl14b,flux14b,INPUT_spec14)


#UVES SAVE
save(wl21b,flux21b,INPUT_spec21)
save(wl22b,flux22b,INPUT_spec22)
save(wl23b,flux23b,INPUT_spec23)
save(wl24b,flux24b,INPUT_spec24)































#-----------------------------------------

