#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:50:16 2019

@author: Heitor
"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from astropy.modeling import models
from specutils import Spectrum1D, SpectralRegion
import matplotlib.pyplot as plt #doctest:+SKIP
from specutils.analysis import snr
from specutils.analysis import snr_derived
from specutils.analysis import gaussian_sigma_width, gaussian_fwhm, fwhm, fwzi
import random


#-----------------------------------------


def snr_spec(flux,wl,n):
    
    sample = len(wl)
    noise = n * np.asarray(random.sample(range(0,len(wl)),sample))/len(wl)
    unc = StdDevUncertainty(noise)
    
    
    fluxn = [[] for i in range(len(wl))]
    i=0
    for inc in unc:
        fluxn[i]=flux[i]+noise[i]
        i=i+1
    
    spec1d = Spectrum1D(spectral_axis=wl*u.AA, flux=fluxn*u.Jy ,  uncertainty= unc )

    #ax = plt.subplots()[1]
    #ax.plot(spec1d.spectral_axis, spec1d.flux)
    #ax.set_xlim([3520,3550])

    sn1 = snr(spec1d, SpectralRegion(3070*u.AA, 3090*u.AA))
    sn  = snr_derived(spec1d,SpectralRegion(3070*u.AA, 3090*u.AA))
    
    #print('SNR1: '+ str(snr(spec1d)), SpectralRegion(3500*u.AA, 3550*u.AA))
    print('SNR: '+ str(sn1))
    #print('SNR: '+ str(sn))
    #print('FWHM:'+str(fwhm(spec1d)))
    
    #0.042 = snr 50
    #
    
    try:    
        return fluxn
    except:
        raise Exception('Check S/N function')
        
        
        
   
#-----------------------------------------

sns.set_style("white")
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.5})

#-----------------------------------------


line_list= 'linelistCUBES.txt'

Z,lamb = np.genfromtxt(line_list, unpack=True, dtype=str)

print('\n' +str(len(Z))+ ' lines. \n')


#-----------------------------------------
#spectrum 

INPUT_spec11= 'StellarGrid/flux_G_m3_m05.norm.nulbad.0.150.B'
INPUT_spec12= 'StellarGrid/flux_G_m3_0.norm.nulbad.0.150.B'
INPUT_spec13= 'StellarGrid/flux_G_m3_05.norm.nulbad.0.150.B'


wl11,flux11 = np.genfromtxt(INPUT_spec11, skip_header=2, unpack=True)
wl12,flux12 = np.genfromtxt(INPUT_spec12, skip_header=2, unpack=True)
wl13,flux13 = np.genfromtxt(INPUT_spec13, skip_header=2, unpack=True)

INPUT_spec21= 'StellarGrid/flux_G_m3_m05.norm.nulbad.0.070.B'
INPUT_spec22= 'StellarGrid/flux_G_m3_0.norm.nulbad.0.070.B'
INPUT_spec23= 'StellarGrid/flux_G_m3_05.norm.nulbad.0.070.B'


wl21,flux21 = np.genfromtxt(INPUT_spec21, skip_header=2, unpack=True)
wl22,flux22 = np.genfromtxt(INPUT_spec22, skip_header=2, unpack=True)
wl23,flux23 = np.genfromtxt(INPUT_spec23, skip_header=2, unpack=True)



OUTPUT=['fig1.pdf']

#-----------------------------------------



f1 = plt.figure(figsize=(12,7))


ax1 = f1.add_subplot(211)

snr1=0.06 #snr 50
ax1.plot(wl11,snr_spec(flux11,wl11,snr1),linewidth=2.5, label='-0.5',color='purple')
ax1.plot(wl12,snr_spec(flux12,wl12,snr1),linewidth=2.5, label='0.0',color='black')
ax1.plot(wl13,snr_spec(flux13,wl13,snr1),linewidth=2.5, label='+0.5',color='blue')

#ax1.legend(loc=2)
ax1.set_title('R=20,000')
#ax1.set_xlabel('Wavelength ( $\AA$ )')
ax1.set_ylabel('Arbritrary Flux')

#ax1.set_xlim([lamb_1-1,lamb_1+1])
#ax1.set_ylim([0.8,1.02])

n=0
for i in Z:
    ax1.text(float(lamb[n]), 0.2, i , fontsize=12)
    ax1.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
    n=n+1
    
#------------------
    
    
ax2 = f1.add_subplot(212, sharex=ax1)

snr2=0.06 #snr 50
ax2.plot(wl21,snr_spec(flux21,wl21,snr2),linewidth=2.5, label='-0.5',color='purple')
ax2.plot(wl22,snr_spec(flux22,wl22,snr2),linewidth=2.5, label='0.0',color='black')
ax2.plot(wl23,snr_spec(flux23,wl23,snr2),linewidth=2.5, label='+0.5',color='blue')

#ax1.legend(loc=2)
ax2.set_title('R=40,000')
ax2.set_xlabel('Wavelength ( $\AA$ )')
ax2.set_ylabel('Arbritrary Flux')

#ax1.set_xlim([lamb_1-1,lamb_1+1])
#ax1.set_ylim([0.8,1.02])

n=0
for i in Z:
    ax2.text(float(lamb[n]), 0.2, i , fontsize=12)
    ax2.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
    n=n+1



plt.tight_layout() 
plt.show()

#f1.savefig(OUTPUT[0])


print('\n')

#-----------------------------------------
#-----------------------------------------



f2 = plt.figure(figsize=(12,10))


ax11 = f2.add_subplot(221)

snr1=0.02 #snr 100
ax11.plot(wl11,snr_spec(flux11,wl11,snr1),linewidth=2.5, label='-0.5',color='purple')
ax11.plot(wl12,snr_spec(flux12,wl12,snr1),linewidth=2.5, label='0.0',color='black')
ax11.plot(wl13,snr_spec(flux13,wl13,snr1),linewidth=2.5, label='+0.5',color='blue')

n=0
for i in Z:
    ax11.text(float(lamb[n]), 0.3, i , fontsize=12)
    ax11.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
    n=n+1

#ax1.legend(loc=2)
ax11.set_title('R=20,000 S/N=50')
#ax1.set_xlabel('Wavelength ( $\AA$ )')
ax11.set_ylabel('Arbritrary Flux')
#ax1.set_xlim([lamb_1-1,lamb_1+1])
#ax1.set_ylim([0.8,1.02])

#------------------

ax12 = f2.add_subplot(223, sharex=ax11)

snr2=0.02 #snr 100
ax12.plot(wl21,snr_spec(flux21,wl21,snr2),linewidth=2.5, label='-0.5',color='purple')
ax12.plot(wl22,snr_spec(flux22,wl22,snr2),linewidth=2.5, label='0.0',color='black')
ax12.plot(wl23,snr_spec(flux23,wl23,snr2),linewidth=2.5, label='+0.5',color='blue')

#ax1.legend(loc=2)
ax12.set_title('R=40,000 S/N=50')
ax12.set_xlabel('Wavelength ( $\AA$ )')
ax12.set_ylabel('Arbritrary Flux')

#ax1.set_xlim([lamb_1-1,lamb_1+1])
#ax1.set_ylim([0.8,1.02])

n=0
for i in Z:
    ax12.text(float(lamb[n]), 0.3, i , fontsize=12)
    ax12.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
    n=n+1

#-----------------------##########################


ax21 = f2.add_subplot(222, sharex=ax11)

snr1=0.01 #snr 100
ax21.plot(wl11,snr_spec(flux11,wl11,snr1),linewidth=2.5, label='-0.5',color='purple')
ax21.plot(wl12,snr_spec(flux12,wl12,snr1),linewidth=2.5, label='0.0',color='black')
ax21.plot(wl13,snr_spec(flux13,wl13,snr1),linewidth=2.5, label='+0.5',color='blue')

n=0
for i in Z:
    ax21.text(float(lamb[n]), 0.3, i , fontsize=12)
    ax21.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
    n=n+1

#ax1.legend(loc=2)
ax21.set_title('R=20,000 S/N=100')
#ax1.set_xlabel('Wavelength ( $\AA$ )')
#ax21.set_ylabel('Arbritrary Flux')
#ax1.set_xlim([lamb_1-1,lamb_1+1])
#ax1.set_ylim([0.8,1.02])

#------------------

ax22 = f2.add_subplot(224, sharex=ax11)

snr2=0.01 #snr 100
ax22.plot(wl21,snr_spec(flux21,wl21,snr2),linewidth=2.5, label='-0.5',color='purple')
ax22.plot(wl22,snr_spec(flux22,wl22,snr2),linewidth=2.5, label='0.0',color='black')
ax22.plot(wl23,snr_spec(flux23,wl23,snr2),linewidth=2.5, label='+0.5',color='blue')

#ax1.legend(loc=2)
ax22.set_title('R=40,000 S/N=100')
ax22.set_xlabel('Wavelength ( $\AA$ )')
#ax22.set_ylabel('Arbritrary Flux')

#ax1.set_xlim([lamb_1-1,lamb_1+1])
#ax1.set_ylim([0.8,1.02])

n=0
for i in Z:
    ax22.text(float(lamb[n]), 0.3, i , fontsize=12)
    ax22.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
    n=n+1



#plt.clf()










#-----------------------------------------
