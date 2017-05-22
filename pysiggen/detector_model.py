#!/usr/local/bin/python

#import sys
import numpy as np
import copy, math
import ctypes
from scipy import  signal, interpolate, ndimage
import numbers

from ._pysiggen import Siggen

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
        self.top_bullet_radius = self.siggenInst.GetTopBulletRadius()

        #Just be real conservative, say anything <3 is in the PC (currently not fitting anything that close to the PC anyway)
        (self.pcLen, self.pcRad) = 3,3#self.siggenInst.GetPointContactDimensions()

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
        self.gradMultList = None
        self.impAvgList = None


        self.trapping_rc = None
        self.rc_int_exp = None #antialiasing rc
        self.t0_padding = t0_padding

        #stuff for waveform interpolation
        #round here to fix floating point accuracy problem
        data_to_siggen_size_ratio = np.around(10. / self.time_step_size,3)

        self.data_to_siggen_size_ratio = np.int(data_to_siggen_size_ratio)

        #Holders for wf simulation
        self.raw_siggen_data = np.zeros( self.num_steps, dtype=np.dtype('f4'), order="C" )
        self.padded_siggen_data = np.zeros( self.num_steps + self.t0_padding, dtype=np.dtype('f4'), order="C" )
        self.raw_charge_data = np.zeros( self.calc_length, dtype=np.dtype('f4'), order="C" )
        self.processed_siggen_data = np.zeros( self.wf_output_length, dtype=np.dtype('f4'), order="C" )

        self.siggen_interp_fn = None
        self.signal_peak_fn = None
        self.temp_wf = np.zeros( self.wf_output_length+2, dtype=np.dtype('f4'), order="C" )
        self.temp_wf_sig = np.zeros( (self.wf_output_length+2)*self.data_to_siggen_size_ratio, dtype=np.dtype('f4'), order="C" )

###########################################################################################################################
  def LoadFieldsGrad(self, fieldFileName,):
    self.fieldFileName = fieldFileName

    with np.load(fieldFileName) as data:
      data = np.load(fieldFileName)
      wpArray  = data['wpArray']
      efld_rArray = data['efld_rArray']
      efld_zArray = data['efld_zArray']
      gradList = data['gradList']

    self.gradList = gradList

    if 'measured_imp' in data:
        self.measured_impurity = data['measured_imp']
    if 'measured_grad' in data:
        self.measured_imp_grad = data['measured_grad']

    self.impAvgList = None
    self.pcRadList = None
    self.pcLenList = None
    if 'impAvgList' in data:
        self.impAvgList = data['impAvgList']
    if 'pcRadList' in data:
        self.pcRadList = data['pcRadList']
    if 'pcLenList' in data:
        self.pcLenList = data['pcLenList']

    self.wpArray = wpArray
    self.efld_rArray = efld_rArray
    self.efld_zArray = efld_zArray

    # import matplotlib.pyplot as plt
    # for i in range(len(gradList)):
    #     for j in range(len(self.impAvgList)):
    #         print "%d, %d max %f" % (i,j, np.amax(np.abs(self.efld_rArray[:,:,i,j])))
    #         # plt.imshow(self.efld_rArray[:,:,i,j])
    #         # plt.show()

    self.siggenInst.SetActiveWpot(self.wpArray)
    self.siggenInst.SetActiveEfld(self.efld_rArray, self.efld_zArray)

    imp_grad_step, avg_grad_step = 0., 0.
    if len(gradList) > 1: imp_grad_step = gradList[1] - gradList[0]
    if len(self.impAvgList) > 1:  avg_grad_step = self.impAvgList[1] - self.impAvgList[0]

    self.siggenInst.SetGradParams(imp_grad_step, gradList[0], avg_grad_step, self.impAvgList[0], len(gradList), len(self.impAvgList))

    if self.pcLenList is not None and self.pcRadList is not None:
        rad_step, len_step = 0., 0.
        if len(self.pcRadList) > 1: rad_step = self.pcRadList[1] - self.pcRadList[0]
        if len(self.pcLenList) > 1: len_step = self.pcLenList[1] - self.pcLenList[0]
        self.siggenInst.SetPointContactParams(rad_step, self.pcRadList[0], len_step, self.pcLenList[0], len(self.pcRadList), len(self.pcLenList))

    # print self.efld_rArray[30*10,30*10,0,0]
    # print self.efld_zArray[30*10,30*10,0,0]
    # # exit(0)
    # import matplotlib.pyplot as plt
    # plt.imshow(self.efld_rArray[:,:,0,0])
    # plt.figure()
    # plt.imshow(self.efld_zArray[:,:,0,0])
    # plt.show()

  def SetPointContact(self, pcrad, pclen):
      if pcrad < self.pcRadList[0] or pcrad > self.pcRadList[-1]:
          print( "pc rad {0} is out of range [{1},{2}]".format(pcrad, self.pcRadList[0], self.pcRadList[-1]) )
          exit(0)
      if pclen < self.pcLenList[0] or pclen > self.pcLenList[-1]:
          print( "pclen {0} is out of range [{1},{2}]".format(pclen, self.pcLenList[0], self.pcLenList[-1]) )
          exit(0)

      self.siggenInst.SetPointContact(pcrad, pclen)


  def SetGrads(self, imp_grad, avg_imp):
      if imp_grad < self.gradList[0] or imp_grad > self.gradList[-1]:
          print( "impurity gradient {0} is out of range [{1},{2}]".format(imp_grad, self.gradList[0], self.gradList[-1]) )
          exit(0)
      if avg_imp < self.impAvgList[0] or avg_imp > self.impAvgList[-1]:
          print( "avg impurity {0} is out of range [{1},{2}]".format(avg_imp, self.impAvgList[0], self.impAvgList[-1]) )
          exit(0)

      self.siggenInst.SetGrads(imp_grad, avg_imp)

  def SetFieldsGradMultIdx(self, gradIdx, multIdx):
      self.siggenInst.SetActiveEfld(gradIdx,multIdx)


###########################################################################################################################
  def ReinitializeDetector(self):
    self.LoadFieldsGrad(self.fieldFileName,  )

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
  def SetTransferFunction(self, b, c, d, RC1_in_us, RC2_in_us, rc1_frac, isDirect=False, num_gain  = 1., digPeriod = 1E8):
    #the (a + b)/(1 + 2c + d**2) sets the gain of the system
    #we don't really care about the gain, so just set b, and keep the sum a+b
    #at some arbitrary constant (100 here), and divide out the total gain later

    a = num_gain - b

    if not isDirect:
        c = 2*c
        d = d**2

    self.num = [a, b, 0.]
    self.den = [1., c, d]
    self.dc_gain = (a+b) / (1 + c + d)

    RC1= 1E-6 * (RC1_in_us)
    self.rc1_for_tf = np.exp(-1./digPeriod/RC1)

    RC2 = 1E-6 * (RC2_in_us)
    self.rc2_for_tf = np.exp(-1./digPeriod/RC2)

    self.rc1_frac = rc1_frac

  def SetAntialiasingRC(self, rc_int_in_ns):
    #rc_int is in ns

    #rc integration for gretina low-pass filter (-3dB at 50 MHz)
    # rc_int = 2 * 49.9 * 33E-12
    rc_int = rc_int_in_ns*1E-9
    self.rc_int_exp = np.exp(-1./1E8/rc_int)
    self.rc_int_gain = 1./ (1-self.rc_int_exp)

  def SetTransferFunctionByTF(self, num, den):
    #should already be discrete params
    (self.num, self.den) = (num, den)

  def SetTransferFunctionPhi(self, phi, omega, d, RC1_in_us, RC2_in_us, rc1_frac, digPeriod  = 1E8, num0=0):
      c = -d * np.cos(omega)
      b_ov_a = c - np.tan(phi) * np.sqrt(d**2-c**2)

      den_gain = (1 + 2*c + d**2)

      a = (1 - num0 )/(1+b_ov_a)
      b = a * b_ov_a

      self.num = [a, b, num0]
      self.den = [1., 2*c, d**2]
      self.dc_gain = (a+b+num0) / (1 + 2*c + d**2)

      RC1= 1E-6 * (RC1_in_us)
      self.rc1_for_tf = np.exp(-1./digPeriod/RC1)

      RC2 = 1E-6 * (RC2_in_us)
      self.rc2_for_tf = np.exp(-1./digPeriod/RC2)

      self.rc1_frac = rc1_frac


  def SetTransferFunctionGain(self, phi, omega, gain, RC1_in_us, RC2_in_us, rc1_frac, digPeriod  = 1E8):

    g =  1 - 1./gain

    cd = - np.cos(omega)
    dc = 1./cd
    dc2 = (dc)**2

    c = (-2 + np.sqrt(4 - 4*dc2*g  )) / (2*dc2)
    d = dc * c

    ba = c - np.tan(phi)*np.sqrt(d**2-c**2)

    self.num = [1,ba,0]
    self.den =  [1.0, 2*c, d**2]
    self.dc_gain = np.sum(self.num)/np.sum(self.den)

    RC1= 1E-6 * (RC1_in_us)
    self.rc1_for_tf = np.exp(-1./digPeriod/RC1)

    RC2 = 1E-6 * (RC2_in_us)
    self.rc2_for_tf = np.exp(-1./digPeriod/RC2)

    self.rc1_frac = rc1_frac


###########################################################################################################################
  def IsInDetector(self, r, phi, z):
    taper_length = self.taper_length
    if r > np.floor(self.detector_radius*10.)/10. or z > np.floor(self.detector_length*10.)/10.:
      return 0
    if r <0 or z <=0:
      return 0
    if z < taper_length and r > (self.detector_radius - taper_length + z):
      return 0
    if phi <0 or phi > np.pi/4:
      return 0
    if r**2/self.pcRad**2 + z**2/self.pcLen**2 < 1:
      return 0
    if (z > (self.detector_length - self.top_bullet_radius) ) and (r > (self.detector_radius - self.top_bullet_radius) ):
      if np.sqrt( (z-self.detector_length  + self.top_bullet_radius)**2 + (r-self.detector_radius  + self.top_bullet_radius)**2) >= self.top_bullet_radius:
          return 0

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

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    self.raw_siggen_data.fill(0.)

    calcFlag = self.siggenInst.GetSignal(x, y, z, self.raw_siggen_data);
    if calcFlag == -1:
    #   print ("Holes out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z))
      return None
    if not np.any(self.raw_siggen_data):
      print( "found zero wf at r={0}, phi={1}, z={2} (calcflag is {3})".format(r, phi, z, calcFlag) )
      return None

    return self.raw_siggen_data

###########################################################################################################################
  def MakeRawSiggenWaveform(self, r,phi,z, charge, output_array=None):
    #Has CALCULATION, not OUTPUT, step length (ie, usually 1ns instead of 10ns binning)

    if output_array is None:
        output_array = self.raw_charge_data
    else:
        if len(output_array) != self.calc_length:
            print( "output array must be length {0} (the current siggen calc length setting)".format(self.calc_length) )
            exit(0)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    output_array.fill(0.)

    calcFlag = self.siggenInst.MakeSignal(x, y, z, output_array, charge);
    if calcFlag == -1:
      return None

    return output_array
###########################################################################################################################
  def MakeSimWaveform(self, r,phi,z,energy, switchpoint,  numSamples, h_smoothing = None, h_smoothing2 =None,
                            alignPoint="t0", trapType="holesOnly", doMaxInterp=True, interpType="linear"):

    self.raw_siggen_data.fill(0.)
    ratio = np.int(self.calc_length / self.num_steps)
    if ratio != 1:
        print( "Hardcoded values are set up which can't handle this ratio of calc signal length to num steps." )
        print( "(which is to say, everything on the calc side should be in 1 ns steps)" )
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

    return self.TurnChargesIntoSignal(electron_wf, self.raw_siggen_data, energy, switchpoint,  numSamples, h_smoothing, h_smoothing2, alignPoint, trapType, doMaxInterp, interpType)
###########################################################################################################################
  def TurnChargesIntoSignal(self, electron_wf, hole_wf, energy, switchpoint,  numSamples, h_smoothing = None, h_smoothing2=None,
                            alignPoint="t0", trapType="holesOnly", doMaxInterp=True, interpType="linear"):
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
    if h_smoothing2 is not None:
      sig = h_smoothing2[0]
      p = h_smoothing2[1]
      window = signal.general_gaussian(np.int(np.ceil(sig)*4), sig=sig, p=p, )
      window /= np.sum(window)

      pad = len(window)
      wf_pad = np.pad(self.padded_siggen_data, (pad,pad), 'constant', constant_values=(0, self.padded_siggen_data[-1]))
      wf_pad= signal.convolve(wf_pad, window, 'same')
      self.padded_siggen_data = wf_pad[pad:-pad]

    if alignPoint == "t0":
        sim_wf = self.ProcessWaveform(self.padded_siggen_data, switchpoint, numSamples)
    elif alignPoint == "max":
        sim_wf = self.ProcessWaveformByMax( self.padded_siggen_data, switchpoint, numSamples, doMaxInterp=doMaxInterp)
    elif isinstance(alignPoint, numbers.Number):
        sim_wf = self.ProcessWaveformByTimePointFine( self.padded_siggen_data, switchpoint, alignPoint, numSamples, interpType=interpType)
    return sim_wf
###########################################################################################################################
  def ApplyChargeTrapping(self, wf):

    period = 1E8 * self.data_to_siggen_size_ratio

    trapping_rc = self.trapping_rc * 1E-6
    trapping_rc_exp = np.exp(-1./period/trapping_rc)
    charges_collected_idx = np.argmax(wf) + 1
    wf[:charges_collected_idx]= signal.lfilter([1., -1], [1., -trapping_rc_exp], wf[:charges_collected_idx])
    wf[charges_collected_idx:] = wf[charges_collected_idx-1]


  def ProcessWaveformByTimePointFine(self, siggen_wf, align_point, align_percent, outputLength, interpType="linear"):
    # print("FINE!")
    # siggen_len = self.num_steps + self.t0_padding
    # siggen_len_output = np.int(siggen_len/self.data_to_siggen_size_ratio)
    temp_wf_sig = self.temp_wf_sig
    temp_wf_sig[0:len(siggen_wf)] = siggen_wf
    temp_wf_sig[len(siggen_wf):] = siggen_wf[-1]

    # filter for the transfer function
    temp_wf_sig= signal.lfilter(self.num, self.den, temp_wf_sig)
    temp_wf_sig /= self.dc_gain

    #filter for the exponential decay
    rc2_num_term = self.rc1_for_tf*self.rc1_frac - self.rc1_for_tf - self.rc2_for_tf*self.rc1_frac
    temp_wf_sig= signal.lfilter([1., -1], [1., -self.rc1_for_tf], temp_wf_sig)
    temp_wf_sig= signal.lfilter([1., rc2_num_term], [1., -self.rc2_for_tf], temp_wf_sig)

    smax = np.amax(temp_wf_sig)
    if smax == 0:
      return None

    #now downsample it
    temp_wf = self.temp_wf
    temp_wf[:] = temp_wf_sig[::self.data_to_siggen_size_ratio]

    #linear interpolation to find the alignPointIdx: find the "real" alignpoint in the simualted array
    alignarr = np.copy(temp_wf)/smax
    first_idx = np.searchsorted(alignarr, align_percent, side='left') - 1

    if first_idx+1 == len(alignarr) or first_idx <0:
        return None

    siggen_offset = (align_percent - alignarr[first_idx]) * (1) / (alignarr[first_idx+1] - alignarr[first_idx])

    #
    align_point_ceil = np.int( np.ceil(align_point) )
    start_idx = align_point_ceil - first_idx

    if start_idx <0:
        return None

    self.siggen_interp_fn = interpolate.interp1d(np.arange(len(temp_wf)), temp_wf, kind=interpType, copy="False", assume_sorted="True")

    num_samples_to_fill = outputLength - start_idx
    offset = align_point_ceil - align_point
    sampled_idxs = np.arange(num_samples_to_fill) + offset + siggen_offset

    self.processed_siggen_data.fill(0.)
    coarse_vals =   self.siggen_interp_fn(sampled_idxs)

    try:
        self.processed_siggen_data[start_idx:start_idx+num_samples_to_fill] = coarse_vals
    except ValueError:
        print( len(self.processed_siggen_data) )
        print( start_idx)
        print( num_samples_to_fill)
        print( sampled_idxs)
        exit(0)

    return self.processed_siggen_data[:outputLength]

  def ProcessWaveformByTimePoint(self, siggen_wf, align_point, align_percent, outputLength, interpType="linear"):
    siggen_len = self.num_steps + self.t0_padding
    siggen_len_output = np.int(siggen_len/self.data_to_siggen_size_ratio)

    temp_wf = self.temp_wf
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
    if self.rc_int_exp is not None:
        temp_wf= signal.lfilter([1,0], [1,-self.rc_int_exp], temp_wf)
        temp_wf /= self.rc_int_gain

    smax = np.amax(temp_wf)
    if smax == 0:
      return None

    #linear interpolation to find the alignPointIdx: find the "real" alignpoint in the simualted array
    alignarr = np.copy(temp_wf)/smax
    first_idx = np.searchsorted(alignarr, align_percent, side='left') - 1

    if first_idx+1 == len(alignarr) or first_idx <0:
        return None

    siggen_offset = (align_percent - alignarr[first_idx]) * (1) / (alignarr[first_idx+1] - alignarr[first_idx])

    #
    align_point_ceil = np.int( np.ceil(align_point) )
    start_idx = align_point_ceil - first_idx

    if start_idx <0:
        return None

    self.siggen_interp_fn = interpolate.interp1d(np.arange(len(temp_wf)), temp_wf, kind=interpType, copy="False", assume_sorted="True")

    num_samples_to_fill = outputLength - start_idx
    offset = align_point_ceil - align_point
    sampled_idxs = np.arange(num_samples_to_fill) + offset + siggen_offset

    self.processed_siggen_data.fill(0.)
    coarse_vals =   self.siggen_interp_fn(sampled_idxs)

    try:
        self.processed_siggen_data[start_idx:start_idx+num_samples_to_fill] = coarse_vals
    except ValueError:
        print( len(self.processed_siggen_data) )
        print( start_idx)
        print( num_samples_to_fill)
        print( sampled_idxs)
        exit(0)

    return self.processed_siggen_data[:outputLength]


########################################################################################################
  def ProcessWaveformByMax(self, siggen_wf, align_point, outputLength, doMaxInterp=True):
    siggen_len = self.num_steps + self.t0_padding
    siggen_len_output = np.int(siggen_len/self.data_to_siggen_size_ratio)

    temp_wf = self.temp_wf
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
    if self.rc_int_exp is not None:
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

        self.signal_peak_fn = interpolate.interp1d( np.arange(sim_max_idx-interp_length, sim_max_idx+interp_length+1), temp_wf[sim_max_idx-interp_length:sim_max_idx+interp_length+1], kind='cubic', assume_sorted=True, copy=False)
        interp_idxs = np.linspace(sim_max_idx-1,sim_max_idx+1, 101)
        interp = self.signal_peak_fn(interp_idxs)
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

    self.siggen_interp_fn = interpolate.interp1d(np.arange(len(temp_wf)), temp_wf, kind="linear", copy="False", assume_sorted="True")

    offset = align_point_ceil - align_point
    sampled_idxs = np.arange(num_samples_to_fill) + offset + siggen_offset

    if sampled_idxs[0] > 1:
        sampled_idxs = np.insert(sampled_idxs,  0, sampled_idxs[0]-1)
        start_idx -=1
        num_samples_to_fill +=1
    if start_idx <0:
        return None

    self.processed_siggen_data.fill(0.)
    coarse_vals =   self.siggen_interp_fn(sampled_idxs)

    if doMaxInterp:
        fine_idxs = np.argwhere(np.logical_and(sampled_idxs > sim_max_idx-interp_length, sampled_idxs < sim_max_idx + interp_length))
        fine_vals = self.signal_peak_fn(sampled_idxs[fine_idxs])
        coarse_vals[fine_idxs] = fine_vals

    try:
        self.processed_siggen_data[start_idx:start_idx+num_samples_to_fill] = coarse_vals
    except ValueError:
        print( len(self.processed_siggen_data) )
        print( start_idx)
        print( num_samples_to_fill)
        print( sampled_idxs)
        exit(0)

    return self.processed_siggen_data[:outputLength]


########################################################################################################
  def ProcessWaveform(self, siggen_wf,  switchpoint, outputLength):
    '''Use interpolation instead of rounding'''

    self.processed_siggen_data.fill(0.)

    siggen_len = self.num_steps + self.t0_padding
    siggen_len_output = siggen_len/self.data_to_siggen_size_ratio

    if switchpoint == 0:
        print("Not currently working to time-align at t=0 :(")
        exit(0)
    else:
        #resample the siggen wf to the 10ns digitized data frequency w/ interpolaiton
        switchpoint_ceil= np.int( np.ceil(switchpoint) )

        # print( "siggen len output is %d" % siggen_len_output

        pad_space = outputLength - siggen_len_output
        # print "padspace minus switch %d" % (pad_space - switchpoint_ceil)

        if pad_space - switchpoint_ceil  < 0 :
            num_samples_to_fill = siggen_len_output - (switchpoint_ceil-pad_space)
        else:
            num_samples_to_fill = siggen_len_output - 1

        self.siggen_interp_fn = interpolate.interp1d(np.arange(siggen_len ), siggen_wf, kind="linear", copy="False", assume_sorted="True")
        siggen_start_idx = (switchpoint_ceil - switchpoint) * self.data_to_siggen_size_ratio
        sampled_idxs = np.arange(num_samples_to_fill)*self.data_to_siggen_size_ratio + siggen_start_idx

        # print "switchpoint ceil is %d" % switchpoint_ceil
        # print "siggen_start_idx is %d" % siggen_start_idx
        # print  sampled_idxs

        try:
          self.processed_siggen_data[switchpoint_ceil:switchpoint_ceil+len(sampled_idxs)] = self.siggen_interp_fn(sampled_idxs)
          self.processed_siggen_data[switchpoint_ceil+len(sampled_idxs)::] = self.processed_siggen_data[switchpoint_ceil+len(sampled_idxs)-1]
        except ValueError:
          print("Something goofy happened here during interp")
          print("siggen len output is {0} (calculated is {1})".format(siggen_len_output, siggen_wf.size) )
          print("desired output length is {0}".format(outputLength))
          print( "switchpoint is {0}".format(switchpoint))
          print( "siggen start idx is {0}".format(siggen_start_idx))
          print( "num samples to fill is {0}".format(num_samples_to_fill))
          print( sampled_idxs)
          exit(0)
        #   return None

    #filter for the damped oscillation
    self.processed_siggen_data[switchpoint_ceil-1:outputLength]= signal.lfilter(self.num, self.den, self.processed_siggen_data[switchpoint_ceil-1:outputLength])

    #filter for low-pass filter on gretina card
    if self.rc_int_exp is not None:
        self.processed_siggen_data[switchpoint_ceil-1:outputLength]= signal.lfilter([1,0], [1,-self.rc_int_exp], self.processed_siggen_data[switchpoint_ceil-1:outputLength])
        self.processed_siggen_data[switchpoint_ceil-1:outputLength] /= self.rc_int_gain

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

def getPointer(floatfloat):
  return (floatfloat.__array_interface__['data'][0] + np.arange(floatfloat.shape[0])*floatfloat.strides[0]).astype(np.intp)

#  def __del__(self):
#    del self.wp_pp
#    del self.siggenInst
