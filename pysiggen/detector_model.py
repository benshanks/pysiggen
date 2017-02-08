#!/usr/local/bin/python

#import sys
import numpy as np
import copy, math
from scipy import  signal, interpolate, ndimage

from _pysiggen import Siggen

#Does all the interfacing with siggen for you, stores/loads lookup tables, and does electronics shaping

class Detector:
  def __init__(self, siggen_config_file=None, temperature=0, timeStep=None, numSteps=None, maxWfOutputLength=None, setup_dict=None, t0_padding=0):

    if setup_dict is not None:
        self.__setstate__(setup_dict)
    elif siggen_config_file is not None:
        self.conf_file = siggen_config_file

        if timeStep is None or numSteps is None:
          self.siggenInst = Siggen(siggen_config_file)
        else:
          self.siggenInst =  Siggen(siggen_config_file, timeStep, numSteps)

        self.time_step_size = self.siggenInst.GetCalculationTimeStep()
        self.num_steps = np.int(self.siggenInst.GetOutputLength())
        self.calc_length = np.int(self.siggenInst.GetCalculationLength())

        if maxWfOutputLength is None:
            self.wf_output_length = self.num_steps
        else:
            self.wf_output_length = np.int(maxWfOutputLength)

        # print "Time step size is %d" % self.time_step_size
        # print "There will be %d steps calculated" % self.calc_length
        # print "There will be %d steps in output" % self.wf_output_length

        (self.detector_radius, self.detector_length) = self.siggenInst.GetDimensions()
        (self.detector_radius, self.detector_length) = np.floor( [self.detector_radius*10, self.detector_length*10] )/10.
        self.taper_length = self.siggenInst.GetTaperLength()
        # print "radius is %f, length is %f" % (self.detector_radius, self.detector_length)


        # print "Using model-based velocity numbers..."
        self.siggenInst.set_velocity_type(1)

        #stuff for field interp
        self.wp_function = None
        self.efld_r_function = None
        self.efld_z_function = None
        self.rr = None
        self.zz = None
        self.wp_pp = None
        self.pcRadList = None
        self.pcLenList = None
        self.gradList = None

        self.trapping_rc = None
        self.t0_padding = t0_padding

        #stuff for waveform interpolation
        #round here to fix floating point accuracy problem
        data_to_siggen_size_ratio = np.around(10. / self.time_step_size,3)
        if not data_to_siggen_size_ratio.is_integer():
          print "Error: siggen step size must evenly divide into 10 ns digitization period (ratio is %f)" % data_to_siggen_size_ratio
          exit(0)
        elif data_to_siggen_size_ratio < 10:
          round_places = 0
        elif data_to_siggen_size_ratio < 100:
          round_places = 1
        elif data_to_siggen_size_ratio < 1000:
          round_places = 2
        else:
          print "Error: Ben was too lazy to code in support for resolution this high"
          exit(0)
        self.data_to_siggen_size_ratio = np.int(data_to_siggen_size_ratio)

        #Holders for wf simulation
        self.raw_siggen_data = np.zeros( self.num_steps, dtype=np.dtype('f4'), order="C" )
        self.padded_siggen_data = np.zeros( self.num_steps + self.t0_padding, dtype=np.dtype('f4'), order="C" )
        self.raw_charge_data = np.zeros( self.calc_length, dtype=np.dtype('f4'), order="C" )
        self.processed_siggen_data = np.zeros( self.wf_output_length, dtype=np.dtype('f4'), order="C" )

###########################################################################################################################
  def LoadFields(self, fieldFileName):
    self.fieldFileName = fieldFileName

    with np.load(fieldFileName) as data:
      data = np.load(fieldFileName)
      wpArray  = data['wpArray']
      efld_rArray = data['efld_rArray']
      efld_zArray = data['efld_zArray']
      gradList = data['gradList']
      pcRadList = data['pcRadList']
      pcLenList = data['pcLenList']

    self.gradList = gradList
    self.pcRadList = pcRadList
    self.pcLenList = pcLenList

    r_space = np.arange(0, wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4'))
    z_space = np.arange(0, wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4'))

#    self.wp_functions = np.empty((wpArray.shape[0],wpArray.shape[1]), dtype=np.object)
#    self.efld_r_functions = np.empty((wpArray.shape[0],wpArray.shape[1]), dtype=np.object)
#    self.efld_z_functions = np.empty((wpArray.shape[0],wpArray.shape[1]), dtype=np.object)
    self.wpArray = wpArray
    self.efld_rArray = efld_rArray
    self.efld_zArray = efld_zArray
##
#    for r in range(wpArray.shape[0]):
#      for z in range(wpArray.shape[1]):
#        self.wp_functions[r,z] = interpolate.RectBivariateSpline(pcRadList, pcLenList, wpArray[r,z,:,:], kx=1, ky=1)
#        self.efld_r_functions[r,z] = interpolate.RegularGridInterpolator((gradList, pcRadList, pcLenList), efld_rArray[r,z,:,:,:])
#        self.efld_z_functions[r,z] = interpolate.RegularGridInterpolator((gradList, pcRadList, pcLenList), efld_zArray[r,z,:,:,:])
#
    self.wp_function = interpolate.RegularGridInterpolator((r_space, z_space, pcRadList, pcLenList), wpArray, )
    self.efld_r_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList, pcLenList), efld_rArray, )
    self.efld_z_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList, pcLenList), efld_zArray,)

    (self.rr, self.zz) = np.meshgrid(r_space, z_space)
###########################################################################################################################
  def LoadFieldsGrad(self, fieldFileName, pcLen=None, pcRad=None):
    self.fieldFileName = fieldFileName

    with np.load(fieldFileName) as data:
      data = np.load(fieldFileName)
      wpArray  = data['wpArray']
      efld_rArray = data['efld_rArray']
      efld_zArray = data['efld_zArray']
      gradList = data['gradList']
      if pcLen is  None:
          pcLen = data['pcLen']
      if pcRad is  None:
          pcRad = data['pcRad']

    self.gradList = gradList
    self.pcLen = pcLen
    self.pcRad = pcRad

    if 'gradMultList' in data:
        self.gradMultList = data['gradMultList']
    else:
        self.gradMultList = [1]

    self.siggenInst.ReadEFieldsFromArray(efld_rArray, efld_zArray, wpArray)

    # r_space = np.arange(0, wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4'))
    # z_space = np.arange(0, wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4'))

    self.wpArray = wpArray
    self.efld_rArray = efld_rArray
    self.efld_zArray = efld_zArray

    # self.efld_r_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, ), efld_rArray, )
    # self.efld_z_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList,), efld_zArray,)
    #
    # (self.rr, self.zz) = np.meshgrid(r_space, z_space)

  def SetFields(self, pcRad, pcLen, impurityGrad, method="full"):
    if method=="nearest":
#      print "WARNING: DOING A CHEAP FIELD SET"
      return self.SetFieldsByNearest(pcRad, pcLen, impurityGrad)
    else:
      return self.SetFieldsFullInterp(pcRad, pcLen, impurityGrad)

  def SetFieldsGradInterp(self, impurityGrad):

    self.impurityGrad = impurityGrad
    gradIdx = (np.abs(self.gradList-impurityGrad)).argmin()
    self.SetFieldsGradIdx(gradIdx)
    #     rr = self.rr
    #     zz = self.zz
    #     efld_r_function = self.efld_r_function
    #     efld_z_function = self.efld_z_function
    #
    #     gradgrad = np.ones_like(rr) * impurityGrad
    #
    #     points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), ], dtype=np.dtype('f4') ).T
    #
    #     new_ef_r = np.array(efld_r_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    #     new_ef_z = np.array(efld_z_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    #
    # #    grad_idx = find_nearest_idx(self.gradList, impurityGrad)
    # #    new_ef_r = np.copy(self.efld_rArray[:,:,grad_idx][:,:,0])
    # #    new_ef_z = np.copy(self.efld_zArray[:,:,grad_idx][:,:,0])
    #
    #     self.siggenInst.SetFields(new_ef_r, new_ef_z, self.wpArray)


  def SetFieldsGradIdx(self, gradIdx):
      self.siggenInst.SetActiveEfld(gradIdx,0)

  def SetFieldsGradMultIdx(self, gradIdx, multIdx):
      self.siggenInst.SetActiveEfld(gradIdx,multIdx)

  def SetFieldsFullInterp(self, pcRad, pcLen, impurityGrad):
    self.pcRad = pcRad
    self.pcLen = pcLen
    self.impurityGrad = impurityGrad

    rr = self.rr
    zz = self.zz
    wp_function = self.wp_function
    efld_r_function = self.efld_r_function
    efld_z_function = self.efld_z_function

    radrad = np.ones_like(rr) * pcRad
    lenlen = np.ones_like(rr) * pcLen
    gradgrad = np.ones_like(rr) * impurityGrad

    points_wp =  np.array([rr.flatten() , zz.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T
    points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T

    new_wp = np.array(wp_function( points_wp ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    new_ef_r = np.array(efld_r_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    new_ef_z = np.array(efld_z_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")

    self.siggenInst.SetPointContact( pcRad, pcLen )
    self.siggenInst.SetFields(new_ef_r, new_ef_z, new_wp)

  def SetFieldsByNearest(self, pcRad, pcLen, impurityGrad):
    self.pcRad = pcRad
    self.pcLen = pcLen
    self.impurityGrad = impurityGrad

    grad_idx = find_nearest_idx(self.gradList, impurityGrad)
    rad_idx = find_nearest_idx(self.pcRadList, pcRad)
    len_idx = find_nearest_idx(self.pcLenList, pcLen)

    new_wp = np.copy(self.wpArray[:,:,rad_idx, len_idx][:,:,0])
    new_ef_r = np.copy(self.efld_rArray[:,:,grad_idx,rad_idx, len_idx][:,:,0])
    new_ef_z = np.copy(self.efld_zArray[:,:,grad_idx,rad_idx, len_idx][:,:,0])

    wp_function = self.wp_function
    efld_r_function = self.efld_r_function
    efld_z_function = self.efld_z_function

#    grad_idx = np.searchsorted(self.gradList, impurityGrad, side="left")
#    rad_idx = np.searchsorted(self.pcRadList, pcRad, side="left")
#    len_idx = np.searchsorted(self.pcLenList, pcLen, side="left")

##    wpArray = self.wpArray[:,:,rad_idx, len_idx]
##    wpArrayNext = self.wpArray[:,:,rad_idx-1, len_idx-1]
#    wpArray = self.efld_rArray[:,:,grad_idx, rad_idx, len_idx]
#    wpArrayNext = self.efld_rArray[:,:,grad_idx-1,rad_idx-1, len_idx-1]
#    wpArray[np.where(wpArray==0)] = np.nan
#    div= np.divide(np.subtract(wpArray, wpArrayNext), wpArray)
#    import matplotlib.pyplot as plt
#
#    div_true = np.zeros_like(div)
#    div_true[np.where(div > 0.01)] = 1
#
#    plt.imshow(div_true.T, origin='lower')
#    plt.colorbar()
#    plt.show()
#    exit(0)

    #do the interp for the closest... 5mm?
#    min_distance_r = 20
#    min_distance_z = 30
#    r_space = np.around(np.arange(min_distance_r, self.wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4')), 1)
#    z_space = np.around(np.arange(min_distance_z, self.wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4')),1)
#    rr, zz = np.meshgrid(r_space, z_space)
#    radrad = np.ones_like(rr) * pcRad
#    lenlen = np.ones_like(rr) * pcLen
#    gradgrad = np.ones_like(rr) * impurityGrad
#    points_wp =  np.array([rr.flatten() , zz.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T
#    new_wp[min_distance_r*10:len(r_space)+min_distance_r*10,  min_distance_z*10:len(z_space)+min_distance_z*10] =  wp_function( points_wp ).reshape(rr.shape).T

    min_distance_r = 0
    min_distance_z = 0
    r_space = np.around(np.arange(min_distance_r, self.wpArray.shape[0]/10 , 0.1, dtype=np.dtype('f4')),1)
    z_space = np.around(np.arange(min_distance_z, self.wpArray.shape[1]/10 , 0.1, dtype=np.dtype('f4')),1)
    rr, zz = np.meshgrid(r_space, z_space)
    radrad = np.ones_like(rr) * pcRad
    lenlen = np.ones_like(rr) * pcLen
    gradgrad = np.ones_like(rr) * impurityGrad
    points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T
    new_ef_r[min_distance_r*10:len(r_space)+min_distance_r*10,  min_distance_z*10:len(z_space)+min_distance_z*10] = efld_r_function( points_ef ).reshape(rr.shape).T

    min_distance_r = 0#1.5
    min_distance_z = 0#1.5
    r_space = np.around(np.arange(min_distance_r, self.wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4')),1)
    z_space = np.around(np.arange(min_distance_z, self.wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4')),1)
    rr, zz = np.meshgrid(r_space, z_space)
    radrad = np.ones_like(rr) * pcRad
    lenlen = np.ones_like(rr) * pcLen
    gradgrad = np.ones_like(rr) * impurityGrad
    points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T
    new_ef_z[np.int(min_distance_r*10):len(r_space)+np.int(min_distance_r*10),  np.int(min_distance_z*10):len(z_space)+np.int(min_distance_z*10)] =  efld_z_function( points_ef ).reshape(rr.shape).T

#    import matplotlib.pyplot as plt
##    plt.imshow(np.abs(new_wp.T - self.wpArray[:,:,rad_idx, len_idx][:,:,0].T) , origin='lower'
#    plt.imshow(np.sqrt(np.add(new_ef_r.T**2, new_ef_z.T**2)) , origin='lower')
#    plt.colorbar()
#
#    plt.figure()
#    plt.imshow(np.abs(new_ef_z.T - self.efld_zArray[:,:,grad_idx,rad_idx, len_idx][:,:,0].T)  , origin='lower')
#    plt.colorbar()
#
#    plt.figure()
#    plt.imshow(np.abs(new_ef_r.T - self.efld_rArray[:,:,grad_idx,rad_idx, len_idx][:,:,0].T) , origin='lower')
#    plt.colorbar()
#
#    plt.show()
#    exit()

    self.siggenInst.SetPointContact( pcRad, pcLen )
    self.siggenInst.SetFields(new_ef_r, new_ef_z, new_wp)

###########################################################################################################################
  def ReinitializeDetector(self):
    self.LoadFieldsGrad(self.fieldFileName, pcLen=self.pcLen, pcRad=self.pcRad, )

#    self.SetTemperature(self.temperature)
#    self.SetFields(self.pcRad, self.pcLen, self.impurityGrad)
    # self.SetFieldsGradInterp( self.impurityGrad)
###########################################################################################################################
  def SetTemperature(self, h_temp, e_temp=0):
    self.temperature = h_temp

    if e_temp == 0:
      e_temp = h_temp

    self.siggenInst.SetTemperature(h_temp, e_temp)
###########################################################################################################################
  def SetTransferFunction(self, b, c, d, RC1_in_us, RC2_in_us, rc1_frac, isDirect=False, isOld=False, num_gain  = 1.):
    #the (a + b)/(1 + 2c + d**2) sets the gain of the system
    #we don't really care about the gain, so just set b, and keep the sum a+b
    #at some arbitrary constant (100 here), and divide out the total gain later

    if isOld:
        a= 1
    else:
        a = num_gain - b

    if not isDirect:
        c = 2*c
        d = d**2

    self.num = [a, b, 0.]
    self.den = [1., c, d]
    self.dc_gain = (a+b) / (1 + c + d)

    RC1= 1E-6 * (RC1_in_us)
    self.rc1_for_tf = np.exp(-1./1E8/RC1)

    RC2 = 1E-6 * (RC2_in_us)
    self.rc2_for_tf = np.exp(-1./1E8/RC2)

    self.rc1_frac = rc1_frac

    #rc integration for gretina low-pass filter (-3dB at 50 MHz)
    rc_int = 2 * 49.9 * 33E-12
    self.rc_int_exp = np.exp(-1./1E8/rc_int)
    self.rc_int_gain = 1./ (1-self.rc_int_exp)


  def SetTransferFunctionByTF(self, num, den):
    #should already be discrete params
    (self.num, self.den) = (num, den)
###########################################################################################################################
  def IsInDetector(self, r, phi, z):
    taper_length = self.taper_length
    if r > np.floor(self.detector_radius*10.)/10. or z > np.floor(self.detector_length*10.)/10.:
      return 0
    elif r <0 or z <=0:
      return 0
    elif z < taper_length and r > (self.detector_radius - taper_length + z):
      return 0
    elif phi <0 or phi > np.pi/4:
      return 0
    elif r**2/self.pcRad**2 + z**2/self.pcLen**2 < 1:
      return 0
    else:
      return 1
###########################################################################################################################
  def GetSimWaveform(self, r,phi,z,scale, switchpoint,  numSamples, smoothing=None):
    sig_wf = self.GetRawSiggenWaveform(r, phi, z)
    if sig_wf is None:
      return None
    #smoothing for charge cloud size effects
    if smoothing is not None:
      ndimage.filters.gaussian_filter1d(sig_wf, smoothing, output=sig_wf)
    sim_wf = self.ProcessWaveform(sig_wf, numSamples, scale, switchpoint)
    return sim_wf
########################################################################################################
  def GetRawSiggenWaveform(self, r,phi,z, energy=1):

    x = r * np.sin(phi)
    y = r * np.cos(phi)
    self.raw_siggen_data.fill(0.)

    calcFlag = self.siggenInst.GetSignal(x, y, z, self.raw_siggen_data);
    if calcFlag == -1:
#      print "Holes out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
      return None
    if np.amax(self.raw_siggen_data) == 0:
      print "found zero wf at r=%0.2f, phi=%0.2f, z=%0.2f (calcflag is %d)" % (r, phi, z, calcFlag)
      return None
    return self.raw_siggen_data

###########################################################################################################################
  def MakeRawSiggenWaveform(self, r,phi,z, charge, output_array=None):
    #Has CALCULATION, not OUTPUT, step length (ie, usually 1ns instead of 10ns binning)

    if output_array is None:
        output_array = self.raw_charge_data
    else:
        if len(output_array) != self.calc_length:
            print "output array must be length %d (the current siggen calc length setting)" % self.calc_length
            exit(0)

    x = r * np.sin(phi)
    y = r * np.cos(phi)
    output_array.fill(0.)

    calcFlag = self.siggenInst.MakeSignal(x, y, z, output_array, charge);
    if calcFlag == -1:
      return None

    return output_array
###########################################################################################################################
  def MakeSimWaveform(self, r,phi,z,energy, switchpoint,  numSamples, h_smoothing = None, alignPoint="t0", trapType="holesOnly", doMaxInterp=True):

    self.raw_siggen_data.fill(0.)
    ratio = np.int(self.calc_length / self.num_steps)
    if ratio != 1:
        print "Hardcoded values are set up which can't handle this ratio of calc signal length to num steps."
        print "(which is to say, everything on the calc side should be in 1 ns steps)"
        exit(0)

    hole_wf = self.MakeRawSiggenWaveform(r, phi, z, 1)
    if hole_wf is None:
      return None

    #this is a comical mess of memory management
    self.raw_siggen_data.fill(0)
    self.raw_siggen_data[:] = hole_wf[:]

    electron_wf = self.MakeRawSiggenWaveform(r, phi, z, -1)
    if electron_wf is  None:
      return None

    return self.TurnChargesIntoSignal(electron_wf, self.raw_siggen_data, energy, switchpoint,  numSamples, h_smoothing, alignPoint, trapType, doMaxInterp)
###########################################################################################################################
  def TurnChargesIntoSignal(self, electron_wf, hole_wf, energy, switchpoint,  numSamples, h_smoothing = None, alignPoint="t0", trapType="holesOnly", doMaxInterp=True):
      #this is for parallel computing, to allow hole and electron wfs to be separately calculated

    if hole_wf is None or electron_wf is None:
      return None

    self.padded_siggen_data.fill(0.)
    self.padded_siggen_data[self.t0_padding:] += hole_wf[:]

    #charge trapping (for holes only)
    if trapType == "holesOnly" and self.trapping_rc is not None:
        self.ApplyChargeTrapping(self.padded_siggen_data)

    #add in the electron component
    self.padded_siggen_data[self.t0_padding:] += electron_wf[:]

    #charge trapping (holes and electrons)
    if trapType == "fullSignal" and self.trapping_rc is not None:
        self.ApplyChargeTrapping(self.padded_siggen_data)

    #scale wf for energy
    self.padded_siggen_data *= energy

    #gaussian smoothing
    if h_smoothing is not None:
      ndimage.filters.gaussian_filter1d(self.padded_siggen_data, h_smoothing, output=self.padded_siggen_data)

    if alignPoint == "t0":
        sim_wf = self.ProcessWaveform(self.padded_siggen_data, switchpoint, numSamples)
    elif alignPoint == "max":
        sim_wf = self.ProcessWaveformByMax( self.padded_siggen_data, switchpoint, numSamples, doMaxInterp=doMaxInterp)
    return sim_wf
###########################################################################################################################
  def ApplyChargeTrapping(self, wf):
    trapping_rc = self.trapping_rc * 1E-6
    trapping_rc_exp = np.exp(-1./1E9/trapping_rc)
    charges_collected_idx = np.argmax(wf) + 1
    wf[:charges_collected_idx]= signal.lfilter([1., -1], [1., -trapping_rc_exp], wf[:charges_collected_idx])
    wf[charges_collected_idx:] = wf[charges_collected_idx-1]
########################################################################################################
  def ProcessWaveformByMax(self, siggen_wf, align_point, outputLength, doMaxInterp=True):
    siggen_len = self.num_steps + self.t0_padding
    siggen_len_output = siggen_len/self.data_to_siggen_size_ratio

    temp_wf = np.zeros( self.wf_output_length+2)
    temp_wf[0:siggen_len_output] = siggen_wf[::self.data_to_siggen_size_ratio]
    temp_wf[siggen_len_output::] =  temp_wf[siggen_len_output-1]

    # filter for the transfer function
    temp_wf= signal.lfilter(self.num, self.den, temp_wf)
    temp_wf /= self.dc_gain

    #filter for the exponential decay
    rc2_num_term = self.rc1_for_tf*self.rc1_frac - self.rc1_for_tf - self.rc2_for_tf*self.rc1_frac
    temp_wf= signal.lfilter([1., -1], [1., -self.rc1_for_tf], temp_wf)
    temp_wf= signal.lfilter([1., rc2_num_term], [1., -self.rc2_for_tf], temp_wf)

    #filter for low-pass filter on gretina card
    temp_wf= signal.lfilter([1,0], [1,-self.rc_int_exp], temp_wf)
    temp_wf /= self.rc_int_gain

    smax = np.amax(temp_wf)
    if smax == 0:
      return None

    #find the max of the filtered wf... which sucks because now its 10ns sampled.  lets do a spline interp
    sim_max_idx = np.argmax(temp_wf)

    if doMaxInterp:
        interp_length = 2
        if sim_max_idx <= interp_length or sim_max_idx >= (len(temp_wf) - interp_length):
            return None

        signal_peak_fn = interpolate.interp1d( np.arange(sim_max_idx-interp_length, sim_max_idx+interp_length+1), temp_wf[sim_max_idx-interp_length:sim_max_idx+interp_length+1], kind='cubic', assume_sorted=True, copy=False)
        interp_idxs = np.linspace(sim_max_idx-1,sim_max_idx+1, 101)
        interp = signal_peak_fn(interp_idxs)
        interp_sim_max_idx = interp_idxs[np.argmax(interp)] #this is the wf max, to nearest hundredth of a sample

        siggen_offset = interp_sim_max_idx - sim_max_idx #how far in front of the max you should be sampling the siggen waveforms
    else:
        siggen_offset = 0

    max_idx = sim_max_idx
    if siggen_offset < 0:
        max_idx -=1
        siggen_offset = 1 + siggen_offset

    align_point_ceil = np.int( np.ceil(align_point) )

    #TODO: is this right?
    start_idx = align_point_ceil - max_idx

    num_samples_to_fill = outputLength - start_idx

    siggen_interp_fn = interpolate.interp1d(np.arange(len(temp_wf)), temp_wf, kind="linear", copy="False", assume_sorted="True")

    offset = align_point_ceil - align_point
    sampled_idxs = np.arange(num_samples_to_fill) + offset + siggen_offset

    if sampled_idxs[0] > 1:
        sampled_idxs = np.insert(sampled_idxs,  0, sampled_idxs[0]-1)
        start_idx -=1
        num_samples_to_fill +=1
    if start_idx <0:
        return None


    self.processed_siggen_data.fill(0.)

    coarse_vals =   siggen_interp_fn(sampled_idxs)

    if doMaxInterp:
        fine_idxs = np.argwhere(np.logical_and(sampled_idxs > sim_max_idx-interp_length, sampled_idxs < sim_max_idx + interp_length))
        fine_vals = signal_peak_fn(sampled_idxs[fine_idxs])
        coarse_vals[fine_idxs] = fine_vals

    try:
        self.processed_siggen_data[start_idx:start_idx+num_samples_to_fill] = coarse_vals
    except ValueError:
        print len(self.processed_siggen_data)
        print start_idx
        print num_samples_to_fill
        print sampled_idxs
        exit(0)

    return self.processed_siggen_data[:outputLength]


########################################################################################################
  def ProcessWaveform(self, siggen_wf,  switchpoint, outputLength):
    '''Use interpolation instead of rounding'''

    siggen_len = self.num_steps + self.t0_padding
    siggen_len_output = siggen_len/self.data_to_siggen_size_ratio

    #resample the siggen wf to the 10ns digitized data frequency w/ interpolaiton
    switchpoint_ceil= np.int( np.ceil(switchpoint) )

    # print "siggen len output is %d" % siggen_len_output

    pad_space = outputLength - siggen_len_output
    # print "padspace minus switch %d" % (pad_space - switchpoint_ceil)

    if pad_space - switchpoint_ceil  < 0 :
        num_samples_to_fill = siggen_len_output - (switchpoint_ceil-pad_space)
    else:
        num_samples_to_fill = siggen_len_output - 1

    siggen_interp_fn = interpolate.interp1d(np.arange(siggen_len ), siggen_wf, kind="linear", copy="False", assume_sorted="True")
    siggen_start_idx = (switchpoint_ceil - switchpoint) * self.data_to_siggen_size_ratio
    sampled_idxs = np.arange(num_samples_to_fill)*self.data_to_siggen_size_ratio + siggen_start_idx

    # print "switchpoint ceil is %d" % switchpoint_ceil
    # print "siggen_start_idx is %d" % siggen_start_idx
    # print  sampled_idxs

    self.processed_siggen_data.fill(0.)

    try:
      self.processed_siggen_data[switchpoint_ceil:switchpoint_ceil+len(sampled_idxs)] = siggen_interp_fn(sampled_idxs)
      self.processed_siggen_data[switchpoint_ceil+len(sampled_idxs)::] = self.processed_siggen_data[switchpoint_ceil+len(sampled_idxs)-1]
    except ValueError:
      print "Something goofy happened here during interp"
      print "siggen len output is %d (calculated is %d)" % (siggen_len_output, siggen_wf.size)
      print "desired output length is %d" % outputLength
      print "switchpoint is %d" % switchpoint
      print "siggen start idx is %d" % siggen_start_idx
      print "num samples to fill is %d" % num_samples_to_fill
      print sampled_idxs
      exit(0)
    #   return None

    #filter for the damped oscillation
    self.processed_siggen_data[switchpoint_ceil-1:outputLength]= signal.lfilter(self.num, self.den, self.processed_siggen_data[switchpoint_ceil-1:outputLength])

    #filter for the exponential decay
    rc2_num_term = self.rc1_for_tf*self.rc1_frac - self.rc1_for_tf - self.rc2_for_tf*self.rc1_frac
    self.processed_siggen_data[switchpoint_ceil-1:outputLength]= signal.lfilter([1., -1], [1., -self.rc1_for_tf], self.processed_siggen_data[switchpoint_ceil-1:outputLength])
    self.processed_siggen_data[switchpoint_ceil-1:outputLength]= signal.lfilter([1., rc2_num_term], [1., -self.rc2_for_tf], self.processed_siggen_data[switchpoint_ceil-1:outputLength])
    self.processed_siggen_data /= self.dc_gain

    smax = np.amax(self.processed_siggen_data[:outputLength])
    if smax == 0:
      return None

    return self.processed_siggen_data[:outputLength]
########################################################################################################
  #For pickling a detector object
  def __getstate__(self):
    # Copy the object's state from self.__dict__ which contains
    # all our instance attributes. Always use the dict.copy()
    # method to avoid modifying the original state.

    #manually do a deep copy of the velo data
    self.siggenSetup = self.siggenInst.GetSafeConfiguration()

    state = self.__dict__.copy()
    # Remove the unpicklable entries.
    del state['rr']
    del state['zz']
    del state['raw_siggen_data']
    del state['efld_r_function']
    del state['efld_z_function']
    del state['wp_function']
    del state['pcRadList']
    del state['gradList']
    del state['pcLenList']
    del state['siggenInst']

    return state

  def __setstate__(self, state):
    # Restore instance attributes
    self.__dict__.update(state)
    # Restore the previously opened file's state. To do so, we need to
    # reopen it and read from it until the line count is restored.

    self.siggenInst =  Siggen(savedConfig=self.siggenSetup)
    self.siggenInst.set_velocity_type(1)
    self.raw_siggen_data = np.zeros( self.num_steps, dtype=np.dtype('f4'), order="C" )
    self.raw_charge_data = np.zeros( self.calc_length, dtype=np.dtype('f4'), order="C" )
    self.processed_siggen_data = np.zeros( self.wf_output_length, dtype=np.dtype('f4'), order="C" )

    self.wp_function = None
    self.efld_r_function = None
    self.efld_z_function = None
    self.rr = None
    self.zz = None
    self.wp_pp = None
    self.pcRadList = None
    self.pcLenList = None
    self.gradList = None

    # self.LoadFields(self.fieldFileName)


  def ReflectPoint(self, r,z):
    #algorithm shamelessly ripped from answer on http://stackoverflow.com/questions/3306838/algorithm-for-reflecting-a-point-across-a-line
    m = self.detector_length / self.detector_radius
    d = (r + m*z)/(1+m*m)
    new_r = 2*d-r
    new_z = 2*d*m - z

    return (new_r, new_z)

def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return [idx-1]
    else:
        return [idx]
#  def __del__(self):
#    del self.wp_pp
#    del self.siggenInst
