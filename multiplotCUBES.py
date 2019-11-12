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
    #sn  = snr_derived(spec1d,SpectralRegion(3070*u.AA, 3090*u.AA))    
    #print('SNR1: '+ str(snr(spec1d)), SpectralRegion(3500*u.AA, 3550*u.AA))
    print('S/N: '+ str(sn1))
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

star='G'

met='m3'


 
line_list= 'linelistCUBES.txt'

Z,lamb = np.genfromtxt(line_list, unpack=True, dtype=str)

print('\n' +str(len(Z))+ ' lines. \n')


#-----------------------------------------
#spectrum 

INPUT_spec11= 'StellarGrid/flux_'+star+'_'+met+'_m05.norm.nulbad.0.150.B'
INPUT_spec12= 'StellarGrid/flux_'+star+'_'+met+'_0.norm.nulbad.0.150.B'
INPUT_spec13= 'StellarGrid/flux_'+star+'_'+met+'_p05.norm.nulbad.0.150.B'
INPUT_spec14= 'StellarGrid/flux_'+star+'_'+met+'_p10.norm.nulbad.0.150.B'



wl11,flux11 = np.genfromtxt(INPUT_spec11, skip_header=2, unpack=True)
wl12,flux12 = np.genfromtxt(INPUT_spec12, skip_header=2, unpack=True)
wl13,flux13 = np.genfromtxt(INPUT_spec13, skip_header=2, unpack=True)
wl14,flux14 = np.genfromtxt(INPUT_spec14, skip_header=2, unpack=True)


INPUT_spec21= 'StellarGrid/flux_'+star+'_'+met+'_m05.norm.nulbad.0.070.B'
INPUT_spec22= 'StellarGrid/flux_'+star+'_'+met+'_0.norm.nulbad.0.070.B'
INPUT_spec23= 'StellarGrid/flux_'+star+'_'+met+'_p05.norm.nulbad.0.070.B'
INPUT_spec24= 'StellarGrid/flux_'+star+'_'+met+'_p10.norm.nulbad.0.070.B'



wl21,flux21 = np.genfromtxt(INPUT_spec21, skip_header=2, unpack=True)
wl22,flux22 = np.genfromtxt(INPUT_spec22, skip_header=2, unpack=True)
wl23,flux23 = np.genfromtxt(INPUT_spec23, skip_header=2, unpack=True)
wl24,flux24 = np.genfromtxt(INPUT_spec24, skip_header=2, unpack=True)


print(' --- \n')

#NOISE
snr1=0.12

fl11n1 = snr_spec(flux11,wl11,snr1)
fl12n1 = snr_spec(flux12,wl12,snr1)
fl13n1 = snr_spec(flux13,wl13,snr1)
fl14n1 = snr_spec(flux14,wl14,snr1)


print('\n')

fl21n1 = snr_spec(flux21,wl21,snr1)
fl22n1 = snr_spec(flux22,wl22,snr1)
fl23n1 = snr_spec(flux23,wl23,snr1)
fl24n1 = snr_spec(flux24,wl24,snr1)


print(' --- \n')

snr2=0.07

fl11n2 = snr_spec(flux11,wl11,snr2)
fl12n2 = snr_spec(flux12,wl12,snr2)
fl13n2 = snr_spec(flux13,wl13,snr2)
fl14n2 = snr_spec(flux14,wl14,snr2)


print('\n')

fl21n2 = snr_spec(flux21,wl21,snr2)
fl22n2 = snr_spec(flux22,wl22,snr2)
fl23n2 = snr_spec(flux23,wl23,snr2)
fl24n2 = snr_spec(flux24,wl24,snr2)

print(' --- \n')

snr3=0.02

fl11n3 = snr_spec(flux11,wl11,snr3)
fl12n3 = snr_spec(flux12,wl12,snr3)
fl13n3 = snr_spec(flux13,wl13,snr3)
fl14n3 = snr_spec(flux14,wl14,snr3)


print('\n')

fl21n3 = snr_spec(flux21,wl21,snr3)
fl22n3 = snr_spec(flux22,wl22,snr3)
fl23n3 = snr_spec(flux23,wl23,snr3)
fl24n3 = snr_spec(flux24,wl24,snr3)



print(' --- \n')

#-----------------------------------------
#-----------------------------------------

#lamb2 = ['3130.42', '3131.07']

c=0

for LAMB in lamb:
    
    print('\n'+str(Z[c])+' '+LAMB+'\n')
    

    f2 = plt.figure(figsize=(16,10))


    ax11 = f2.add_subplot(231)

    snr1=0.12 #snr 50
    
    ax11.plot(wl11,fl11n1,linewidth=2.5, label='-0.5',color='purple')
    ax11.plot(wl12,fl12n1,linewidth=2.5, label='0.0',color='black')
    ax11.plot(wl13,fl13n1,linewidth=2.5, label='+0.5',color='blue')
    ax11.plot(wl14,fl14n1,linewidth=2.5, label='+1.0',color='gray')

    n=0
    for i in Z:
        ax11.text(float(lamb[n]), 0.2, i , fontsize=12)
        ax11.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
        n=n+1

    #ax1.legend(loc=2)
    ax11.set_title('R=20,000 S/N=50')
    #ax1.set_xlabel('Wavelength ( $\AA$ )')
    ax11.set_ylabel('Arbritrary Flux')
    ax11.set_xlim([float(LAMB)-1.2,float(LAMB)+1.2])
    ax11.set_ylim([0.1,1.15])

#------------------


    ax12 = f2.add_subplot(234, sharex=ax11)

    #snr2=0.02 #snr 50
    ax12.plot(wl21,fl21n1,linewidth=2.5, label='-0.5',color='purple')
    ax12.plot(wl22,fl22n1,linewidth=2.5, label='0.0',color='black')
    ax12.plot(wl23,fl23n1,linewidth=2.5, label='+0.5',color='blue')
    ax12.plot(wl24,fl24n1,linewidth=2.5, label='+1.0',color='gray')


    #ax1.legend(loc=2)
    ax12.set_title('R=40,000 S/N=50')
    ax12.set_xlabel('Wavelength ( $\AA$ )')
    ax12.set_ylabel('Arbritrary Flux')

    ax12.set_xlim([float(LAMB)-1.2,float(LAMB)+1.2])
    ax12.set_ylim([0.0,1.15])

    n=0
    for i in Z:
        ax12.text(float(lamb[n]), 0.2, i , fontsize=12)
        ax12.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
        n=n+1

#-----------------------##########################
    

    ax21 = f2.add_subplot(232, sharex=ax11)

    snr2=0.07 #snr 100
    
    ax21.plot(wl11,fl11n2,linewidth=2.5, label='-0.5',color='purple')
    ax21.plot(wl12,fl12n2,linewidth=2.5, label='0.0',color='black')
    ax21.plot(wl13,fl13n2,linewidth=2.5, label='+0.5',color='blue')
    ax21.plot(wl14,fl14n2,linewidth=2.5, label='+1.0',color='gray')

    n=0
    for i in Z:
        ax21.text(float(lamb[n]), 0.2, i , fontsize=12)
        ax21.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
        n=n+1

    #ax1.legend(loc=2)
    ax21.set_title('R=20,000 S/N=100')
    #ax1.set_xlabel('Wavelength ( $\AA$ )')
    #ax21.set_ylabel('Arbritrary Flux')
    ax21.set_xlim([float(LAMB)-1.2,float(LAMB)+1.2])
    ax21.set_ylim([0.1,1.15])

#------------------

    ax22 = f2.add_subplot(235, sharex=ax11)

    #snr2=0.01 #snr 100
    ax22.plot(wl21,fl21n2,linewidth=2.5, label='-0.5',color='purple')
    ax22.plot(wl22,fl22n2,linewidth=2.5, label='0.0',color='black')
    ax22.plot(wl23,fl23n2,linewidth=2.5, label='+0.5',color='blue')
    ax22.plot(wl24,fl24n2,linewidth=2.5, label='+1.0',color='gray')
    
    n=0
    for i in Z:
        ax22.text(float(lamb[n]), 0.2, i , fontsize=12)
        ax22.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')    

    #ax1.legend(loc=2)
    ax22.set_title('R=40,000 S/N=100')
    ax22.set_xlabel('Wavelength ( $\AA$ )')
    #ax22.set_ylabel('Arbritrary Flux')

    ax22.set_xlim([float(LAMB)-1.2,float(LAMB)+1.2])
    ax22.set_ylim([0.0,1.15])


#-----------------------##########################
    

    ax31 = f2.add_subplot(233, sharex=ax11)

    snr3=0.02 #snr 100
    
    ax31.plot(wl11,fl11n3,linewidth=2.5, label='-0.5',color='purple')
    ax31.plot(wl12,fl12n3,linewidth=2.5, label='0.0',color='black')
    ax31.plot(wl13,fl13n3,linewidth=2.5, label='+0.5',color='blue')
    ax31.plot(wl14,fl14n3,linewidth=2.5, label='+1.0',color='gray')

    n=0
    for i in Z:
        ax31.text(float(lamb[n]), 0.2, i , fontsize=12)
        ax31.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
        n=n+1

    #ax1.legend(loc=2)
    ax31.set_title('R=20,000 S/N=200')
    #ax1.set_xlabel('Wavelength ( $\AA$ )')
    #ax21.set_ylabel('Arbritrary Flux')
    ax31.set_xlim([float(LAMB)-1.2,float(LAMB)+1.2])
    ax31.set_ylim([0.1,1.15])

#------------------

    ax32 = f2.add_subplot(236, sharex=ax11)

    #snr2=0.01 #snr 100
    ax32.plot(wl21,fl21n3,linewidth=2.5, label='-0.5',color='purple')
    ax32.plot(wl22,fl22n3,linewidth=2.5, label='0.0',color='black')
    ax32.plot(wl23,fl23n3,linewidth=2.5, label='+0.5',color='blue')
    ax32.plot(wl24,fl24n3,linewidth=2.5, label='+1.0',color='gray')

    #ax1.legend(loc=2)
    ax32.set_title('R=40,000 S/N=200')
    ax32.set_xlabel('Wavelength ( $\AA$ )')
    #ax22.set_ylabel('Arbritrary Flux')

    ax32.set_xlim([float(LAMB)-1.2,float(LAMB)+1.2])
    ax32.set_ylim([0.0,1.15])
    
    
    n=0
    for i in Z:
        ax32.text(float(lamb[n]), 0.2, i , fontsize=12)
        ax32.axvline(x= float(lamb[n]), linewidth=0.6, color='k', ls='--')
    
        n=n+1
    
    f2.suptitle(star+' '+met+' '+ str(Z[c])+' '+str(LAMB))
        
    #for test
    #plt.savefig('test'+str(c)+'.pdf')

    plt.savefig('./figs/'+star+'/'+star+met+'/plot_'+star+'_'+met+'_'+str(Z[c])+str(LAMB)+'.pdf')

    plt.clf()
    c=c+1
    

print('\n DONE... ')






#-----------------------------------------
