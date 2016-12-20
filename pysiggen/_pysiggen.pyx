# file: cqueue.pxd

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, memset

import numpy as np
import cython
cimport numpy as np
cimport csiggen

cdef extern from "siggen_helpers.c":
    int read_velocity_table(csiggen.velocity_lookup* v_lookup, csiggen.MJD_Siggen_Setup *setup)
    int temperature_modify_velocity_table(float e_temp, float h_temp, csiggen.velocity_lookup* v_lookup_saved,  csiggen.velocity_lookup* modified_v_lookup, csiggen.MJD_Siggen_Setup *setup)

cdef class Siggen:

  cdef csiggen.MJD_Siggen_Setup fSiggenData
  cdef csiggen.velocity_lookup* fVelocityFileData #as read straight out of the drift velo file
  cdef csiggen.velocity_lookup* fVelocityTempData #temperature-adjuisted values for use in siggen

  cdef csiggen.cyl_pt *** efld_ptr_arr

  cdef csiggen.cyl_pt** pEfld;
  cdef float** pWpot;

  #helper arrays for siggen calcs
  cdef float* sum
  cdef float* tmp

#  cdef csiggen.point* pDpath_e
#  cdef csiggen.point* pDpath_h

  def __init__(self, conffilename="", timeStepLength=-1., numTimeSteps=-1, savedConfig=None):

    if savedConfig is not None:
      self.SetConfiguration(savedConfig)
      self.reinit_from_saved_state()

      self.set_velocity_type(1)
      csiggen.set_hole_params(66333., 0.744, 181., 107270., 0.580, 100., &self.fSiggenData)

    else:

      csiggen.read_config(conffilename, &self.fSiggenData);

      self.fSiggenData.rmin  = 0;
      self.fSiggenData.rmax  = self.fSiggenData.xtal_radius;
      self.fSiggenData.rstep = self.fSiggenData.xtal_grid;
      self.fSiggenData.zmin  = 0;
      self.fSiggenData.zmax  = self.fSiggenData.xtal_length;
      self.fSiggenData.zstep = self.fSiggenData.xtal_grid;

      self.fSiggenData.rlen = np.int((self.fSiggenData.rmax - self.fSiggenData.rmin)/self.fSiggenData.rstep) + 1;
      self.fSiggenData.zlen = np.int((self.fSiggenData.zmax - self.fSiggenData.zmin)/self.fSiggenData.zstep) + 1;

      csiggen.field_setup(&self.fSiggenData);

      if timeStepLength ==-1. or numTimeSteps ==-1:
        self.fSiggenData.ntsteps_out = self.fSiggenData.time_steps_calc / np.int(self.fSiggenData.step_time_out/self.fSiggenData.step_time_calc);
      else:
        self.set_time_step_length(timeStepLength)
#        self.set_calc_time_step_length(timeStepLength)
        self.set_time_step_number(numTimeSteps)

    self.fSiggenData.dpath_e = <csiggen.point *> PyMem_Malloc(self.fSiggenData.time_steps_calc*sizeof(csiggen.point));
    self.fSiggenData.dpath_h = <csiggen.point *> PyMem_Malloc(self.fSiggenData.time_steps_calc*sizeof(csiggen.point));

    self.fSiggenData.v_params = <csiggen.velocity_params *> PyMem_Malloc(sizeof(csiggen.velocity_params));
#    self.fSiggenData.v_params.h_100_mu0 =66333.
#    self.fSiggenData.v_params.h_100_beta = 0.744
#    self.fSiggenData.v_params.h_100_e0 = 181
#    self.fSiggenData.v_params.h_111_mu0 = 107270
#    self.fSiggenData.v_params.h_111_beta = 0.580
#    self.fSiggenData.v_params.h_111_e0 = 100

    self.sum = <float *> PyMem_Malloc(self.fSiggenData.time_steps_calc*sizeof(float));
    self.tmp = <float *> PyMem_Malloc(self.fSiggenData.time_steps_calc*sizeof(float));

    self.ReadVelocityTable()
    self.SetTemperature(self.fSiggenData.xtal_temp)

    #default params are reggiani
    csiggen.set_hole_params(66333., 0.744, 181., 107270., 0.580, 100., &self.fSiggenData)


  def __dealloc__(self):
    if self.fSiggenData.dpath_e is not NULL:
      PyMem_Free(self.fSiggenData.dpath_e)
    if self.fSiggenData.dpath_h is not NULL:
      PyMem_Free(self.fSiggenData.dpath_h)
    if self.fSiggenData.v_params is not NULL:
      PyMem_Free(self.fSiggenData.v_params)
    if self.fVelocityFileData is not NULL:
      PyMem_Free(self.fVelocityFileData)
    if self.fVelocityTempData is not NULL:
      PyMem_Free(self.fVelocityTempData)
    if self.pWpot is not NULL:
      PyMem_Free(self.pWpot)
    if self.pEfld is not NULL:
      PyMem_Free(self.pEfld)
    if self.sum is not NULL:
      PyMem_Free(self.sum)
    if self.tmp is not NULL:
      PyMem_Free(self.tmp)

    # if self.

  cdef reinit_from_saved_state(self):
    #init the WP and E-fld
    cdef csiggen.cyl_pt **efld;
    efld = <csiggen.cyl_pt **> PyMem_Malloc(self.fSiggenData.rlen*sizeof(csiggen.cyl_pt*));
    for i in range(self.fSiggenData.rlen):
      efld[i] = <csiggen.cyl_pt *> PyMem_Malloc(self.fSiggenData.zlen*sizeof(csiggen.cyl_pt));
    self.fSiggenData.efld = efld;

    cdef float** wpot;
    wpot = <float **> malloc(self.fSiggenData.rlen*sizeof(float*));
    for i in range(self.fSiggenData.rlen):
      wpot[i] = <float *> malloc(self.fSiggenData.zlen*sizeof(float));
    self.fSiggenData.wpot = wpot;

    self.pEfld = efld;
    self.pWpot = wpot;

  cdef c_get_signal(self, float x, float y, float z, float* signal):
    print "Getting signal at (%0.2f, %0.2f, %0.2f)" % (x,y,z)
    cdef csiggen.point pt
    pt.x = x
    pt.y = y
    pt.z = z

    return csiggen.get_signal( pt, signal, &self.fSiggenData)

  def GetSignal(self, float x, float y, float z, np.ndarray[float, ndim=1, mode="c"] input not None):
    return self.c_get_signal(x,y,z, &input[0])

  @cython.boundscheck(False)
  @cython.wraparound(False)
  cdef c_make_signal(self, float x, float y, float z, float* signal, float charge):
    cdef csiggen.point pt
    pt.x = x
    pt.y = y
    pt.z = z

    #memset(self.fSiggenData.dpath_e, 0, self.fSiggenData.time_steps_calc*sizeof(csiggen.point));
    #memset(self.fSiggenData.dpath_h, 0, self.fSiggenData.time_steps_calc*sizeof(csiggen.point));

    flag = csiggen.make_signal( pt, signal, charge, &self.fSiggenData)
    for j in range(1, self.fSiggenData.time_steps_calc):
      signal[j] += signal[j-1]

    return flag


  def MakeSignal(self, float x, float y, float z, np.ndarray[float, ndim=1, mode="c"] input not None, float charge):
    return  self.c_make_signal(x,y,z, &input[0], charge)


  def ChargeCloudCorrect(self, np.ndarray[float, ndim=1, mode="c"] input not None, charge_cloud_size):
    self.c_charge_cloud_correction(&input[0], charge_cloud_size)


  @cython.boundscheck(False)
  @cython.wraparound(False)
  cdef c_charge_cloud_correction(self, float* signal, float charge_cloud_size):
      tsteps = self.fSiggenData.time_steps_calc
      sum = self.sum
      tmp = self.tmp

      dt = np.int( (1.5 + charge_cloud_size / (self.fSiggenData.step_time_calc * self.fSiggenData.initial_vel)) )
      if (self.fSiggenData.initial_vel < 0.00001): dt = 0

      if dt > 1:
        w = (np.float( dt)) / 2.355
        l = dt/10

        if (l < 1): l = 1;

        for j in range(tsteps):
          sum[j] = 1.0;
          tmp[j] = signal[j];

        for k in np.arange(l, 2*dt, l):
          x = k/w;
          y = np.exp(-x*x);
          for j in np.arange(tsteps - k):
            sum[j] += y;
            tmp[j] += signal[j+k] * y;
            sum[j+k] += y;
            tmp[j+k] += signal[j] * y;

          for j in range(tsteps):
            signal[j] = tmp[j]/sum[j];

  def GetCalculationLength(self):
    return self.fSiggenData.time_steps_calc

  cpdef set_time_step_length(self, float timeStepLength):
    if timeStepLength < self.fSiggenData.step_time_calc:
      print "Also reducing time step calc to %f" % timeStepLength
      self.fSiggenData.step_time_calc = timeStepLength;
    self.fSiggenData.step_time_out = timeStepLength

  cpdef set_calc_time_step_length(self, float timeStepOutLength):
      self.fSiggenData.step_time_calc = timeStepOutLength;

  cpdef set_time_step_number(self, int waveformLength):
    self.fSiggenData.ntsteps_out = waveformLength;
    self.fSiggenData.time_steps_calc = waveformLength * np.int(self.fSiggenData.step_time_out/self.fSiggenData.step_time_calc);

  cpdef set_velocity_type(self, int veloType):
      self.fSiggenData.velocity_type = veloType;

  cpdef set_hole_params(self, h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0):
#      print "setting hole params"
#      print "velo type is %d" % self.fSiggenData.velocity_type
      csiggen.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0, &self.fSiggenData)


  cdef c_read_velocity_table(self):
    #read in the drift velocity table
    if self.fVelocityFileData is NULL:
      self.fVelocityFileData = <csiggen.velocity_lookup* > malloc(21*sizeof(csiggen.velocity_lookup))
    read_velocity_table(self.fVelocityFileData, &self.fSiggenData)

    #and copy the current version of it to an unsaved array w/ a hard copy
    if self.fVelocityTempData is NULL:
#      print "about to malloc the velocity adjustment with length %d" % self.fSiggenData.v_lookup_len
      self.fVelocityTempData = <csiggen.velocity_lookup* > PyMem_Malloc(self.fSiggenData.v_lookup_len*sizeof(csiggen.velocity_lookup))
#    else:
#      print "about to realloc the velocity adjustment with length %d" % self.fSiggenData.v_lookup_len
#      self.fVelocityTempData = <csiggen.velocity_lookup* > PyMem_Realloc(self.fVelocityTempData, self.fSiggenData.v_lookup_len*sizeof(self.fVelocityTempData))

#    print "about to copy the velocity adjustment table with length %d" % self.fSiggenData.v_lookup_len
    for i in range(self.fSiggenData.v_lookup_len):
#      print "copying index %d..." % i
#      print self.fVelocityFileData[i].e

      self.fVelocityTempData[i].e =     self.fVelocityFileData[i].e
      self.fVelocityTempData[i].e100 =  self.fVelocityFileData[i].e100
      self.fVelocityTempData[i].e110 =  self.fVelocityFileData[i].e110
      self.fVelocityTempData[i].e111 =  self.fVelocityFileData[i].e111
      self.fVelocityTempData[i].h100 =  self.fVelocityFileData[i].h100
      self.fVelocityTempData[i].h110 =  self.fVelocityFileData[i].h110
      self.fVelocityTempData[i].h111 =  self.fVelocityFileData[i].h111
      self.fVelocityTempData[i].ea =    self.fVelocityFileData[i].ea
      self.fVelocityTempData[i].eb =    self.fVelocityFileData[i].eb
      self.fVelocityTempData[i].ec =    self.fVelocityFileData[i].ec
      self.fVelocityTempData[i].ebp =   self.fVelocityFileData[i].ebp
      self.fVelocityTempData[i].ecp =   self.fVelocityFileData[i].ecp
      self.fVelocityTempData[i].ha =    self.fVelocityFileData[i].ha
      self.fVelocityTempData[i].hb =    self.fVelocityFileData[i].hb
      self.fVelocityTempData[i].hc =    self.fVelocityFileData[i].hc
      self.fVelocityTempData[i].hbp =   self.fVelocityFileData[i].hbp
      self.fVelocityTempData[i].hcp =   self.fVelocityFileData[i].hcp
      self.fVelocityTempData[i].hcorr = self.fVelocityFileData[i].hcorr
      self.fVelocityTempData[i].ecorr = self.fVelocityFileData[i].ecorr

  cpdef ReadVelocityTable(self):
    self.c_read_velocity_table()

  def ReadCorrectedVelocity(self):
    for i in range(self.fSiggenData.v_lookup_len):

      if i >3: continue

      print "Row number %d" % i
      print self.fVelocityTempData[i].e
      print self.fVelocityTempData[i].e100
      print self.fVelocityTempData[i].e110
      print self.fVelocityTempData[i].e111
      print self.fVelocityTempData[i].h100
      print self.fVelocityTempData[i].h110
      print self.fVelocityTempData[i].h111
      print self.fVelocityTempData[i].ea
      print self.fVelocityTempData[i].eb
      print self.fVelocityTempData[i].ec
      print self.fVelocityTempData[i].ebp
      print self.fVelocityTempData[i].ecp
      print self.fVelocityTempData[i].ha
      print self.fVelocityTempData[i].hb
      print self.fVelocityTempData[i].hc
      print self.fVelocityTempData[i].hbp
      print self.fVelocityTempData[i].hcp
      print self.fVelocityTempData[i].hcorr
      print self.fVelocityTempData[i].ecorr

  def SetTemperature(self, h_temp, e_temp=0):
    self.fSiggenData.xtal_temp = h_temp
    if e_temp == 0:
      e_temp = h_temp
    temperature_modify_velocity_table(e_temp, h_temp, self.fVelocityFileData, self.fVelocityTempData, &self.fSiggenData)
    self.fSiggenData.v_lookup = self.fVelocityTempData

  def SetPointContact(self, pcRad, pcLen):
    self.fSiggenData.pc_radius = pcRad
    self.fSiggenData.pc_length = pcLen

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def ReadEFieldsFromArray(self, efld_rArray, efld_zArray, wpot_array):
      number_eflds = efld_rArray.shape[2]
    #   self.efld_ptr_arr = np.empty(number_eflds, dtype=np.obj)
      self.malloc_efld_array(number_eflds)
      self.malloc_wp()

      for  (i) in range(self.fSiggenData.rlen):
          for (j) in range(self.fSiggenData.zlen):
            self.pWpot[i][j] = wpot_array[i,j]
      self.fSiggenData.wpot = self.pWpot

      for grad_idx in range(number_eflds):
        new_ef_r = efld_rArray[:,:,grad_idx]
        new_ef_z = efld_zArray[:,:,grad_idx]
        for  (i) in range(self.fSiggenData.rlen):
            for (j) in range(self.fSiggenData.zlen):
              self.efld_ptr_arr[grad_idx][i][j].r = new_ef_r[i,j]
              self.efld_ptr_arr[grad_idx][i][j].phi = 0.;
              self.efld_ptr_arr[grad_idx][i][j].z = new_ef_z[i,j]


  def SetActiveEfld(self, grad_idx):
    self.fSiggenData.efld = self.efld_ptr_arr[grad_idx]

  cdef malloc_efld_array(self, number_eflds):
    # cdef csiggen.cyl_pt *** efld_ptr_arr
    self.efld_ptr_arr = <csiggen.cyl_pt ***> PyMem_Malloc(number_eflds*sizeof(csiggen.cyl_pt**));

    cdef csiggen.cyl_pt ** efld
    for i in range(number_eflds):
        self.efld_ptr_arr[i] = <csiggen.cyl_pt **> PyMem_Malloc(self.fSiggenData.rlen*sizeof(csiggen.cyl_pt*));
        for j in range(self.fSiggenData.rlen):
            self.efld_ptr_arr[i][j] = <csiggen.cyl_pt *> PyMem_Malloc(self.fSiggenData.zlen*sizeof(csiggen.cyl_pt));

  cdef malloc_wp(self):
    cdef float ** wpot
    wpot =  <float **> PyMem_Malloc(self.fSiggenData.rlen*sizeof(float*));
    for j in range(self.fSiggenData.rlen):
        wpot[j] = <float *> PyMem_Malloc(self.fSiggenData.zlen*sizeof(float));
        # memset(wpot, 0, self.fSiggenData.zlen*sizeof(float));
    self.pWpot = wpot

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def SetFields(self, new_ef_r, new_ef_z, new_wp):
    for  (i) in range(self.fSiggenData.rlen):
      for (j) in range(self.fSiggenData.zlen):
        self.fSiggenData.efld[i][j].r = new_ef_r[i,j]
        self.fSiggenData.efld[i][j].phi = 0.;
        self.fSiggenData.efld[i][j].z = new_ef_z[i,j]
        self.fSiggenData.wpot[i][j] = new_wp[i,j]

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def ReadFields(self):
    efld_r = np.empty((self.fSiggenData.rlen, self.fSiggenData.zlen))
    efld_phi = np.empty((self.fSiggenData.rlen, self.fSiggenData.zlen))
    efld_z = np.empty((self.fSiggenData.rlen, self.fSiggenData.zlen))
    wp = np.empty((self.fSiggenData.rlen, self.fSiggenData.zlen))
    for  (i) in range(self.fSiggenData.rlen):
      for (j) in range(self.fSiggenData.zlen):
         if self.fSiggenData.efld[i][j].phi != 0:
          print "oh man this is a bummer"
         efld_r[i,j] = self.fSiggenData.efld[i][j].r
         efld_phi[i,j] = self.fSiggenData.efld[i][j].phi
         efld_z[i,j] = self.fSiggenData.efld[i][j].z
         wp[i,j] = self.fSiggenData.wpot[i][j]
    return (efld_r, efld_phi, efld_z, wp)

  def GetTemperature(self):
    return self.fSiggenData.xtal_temp
  def GetDimensions(self):
    return ( self.fSiggenData.xtal_radius, self.fSiggenData.xtal_length)

  def FindDriftVelocity(self, float x, float y, float z):
    self.c_find_drift_velocity( x, y, z)

  cdef c_find_drift_velocity(self, float x, float y, float z):
    cdef csiggen.point pt
    pt.x = x
    pt.y = y
    pt.z = z

    cdef csiggen.vector v
    csiggen.drift_velocity( pt, -1., &v, &self.fSiggenData)
    print "x: %f" % v.x
    print "y: %f" % v.y
    print "z: %f" % v.z


#  @cython.boundscheck(False)
#  @cython.wraparound(False)
#  def SetElectricField(self, np.ndarray[float, ndim=2, mode="c"] efldR not None, np.ndarray[float, ndim=2, mode="c"] efldZ not None):
#    for  i in range(self.fSiggenData.rlen):
#      for j in range(self.fSiggenData.zlen):
#        self.fSiggenData.efld[i][j].r = efldR[i][j];
#        self.fSiggenData.efld[i][j].phi = 0.;
#        self.fSiggenData.efld[i][j].z = efldZ[i][j];
#
#  def SetWeightingPotential(self, np.ndarray[float, ndim=2, mode="c"] newWpot not None):
#    self.c_set_weighting_potential( &newWpot[0] )
#
#  cdef c_set_weighting_potential(self, float** newWpot):
#  #Try to set without copying to make it speedy
#  #NB: May be a terrible idea.  The "Free"-ing will DEFINITELY go poorly.
#  #Live fast, die young YOLO
#    self.fSiggenData.wpot = newWpot;


  def GetSafeConfiguration(self):

    siggenConfig = {}
    siggenConfig["verbosity"]  = self.fSiggenData.verbosity;              # 0 = terse, 1 = normal, 2 = chatty/verbose
    siggenConfig["velocity_type"]  = self.fSiggenData.velocity_type;


    # geometry
    siggenConfig["xtal_length"]  = self.fSiggenData.xtal_length;          # z length
    siggenConfig["xtal_radius"]  = self.fSiggenData.xtal_radius;          # radius
    siggenConfig["top_bullet_radius"]  = self.fSiggenData.top_bullet_radius;    # bulletization radius at top of crystal
    siggenConfig["bottom_bullet_radius"]  = self.fSiggenData.bottom_bullet_radius; # bulletization radius at bottom of BEGe crystal
    siggenConfig["pc_length"]  = self.fSiggenData.pc_length;            # point contact length
    siggenConfig["pc_radius"]  = self.fSiggenData.pc_radius;            # point contact radius
    siggenConfig["taper_length"]  = self.fSiggenData.taper_length;         # size of 45-degree taper at bottom of ORTEC-type crystal
    siggenConfig["wrap_around_radius"]  = self.fSiggenData.wrap_around_radius;   # wrap-around radius for BEGes. Set to zero for ORTEC
    siggenConfig["ditch_depth"]  = self.fSiggenData.ditch_depth;          # depth of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    siggenConfig["ditch_thickness"]  = self.fSiggenData.ditch_thickness;      # width of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    siggenConfig["Li_thickness"]  = self.fSiggenData.Li_thickness;         # depth of full-charge-collection boundary for Li contact

    # electric fields & weighing potentials
    siggenConfig["xtal_grid"]  = self.fSiggenData.xtal_grid;            # grid size in mm for field files (either 0.5 or 0.1 mm)
    siggenConfig["impurity_z0"]  = self.fSiggenData.impurity_z0;          # net impurity concentration at Z=0, in 1e10 e/cm3
    siggenConfig["impurity_gradient"]  = self.fSiggenData.impurity_gradient;    # net impurity gradient, in 1e10 e/cm4
    siggenConfig["impurity_quadratic"]  = self.fSiggenData.impurity_quadratic;   # net impurity difference from linear, at z=L/2, in 1e10 e/cm3
    siggenConfig["impurity_surface"]  = self.fSiggenData.impurity_surface;     # surface impurity of passivation layer, in 1e10 e/cm2
    siggenConfig["impurity_radial_add"]  = self.fSiggenData.impurity_radial_add;  # additive radial impurity at outside radius, in 1e10 e/cm3
    siggenConfig["impurity_radial_mult"]  = self.fSiggenData.impurity_radial_mult; # multiplicative radial impurity at outside radius (neutral=1.0)
    siggenConfig["impurity_rpower"]  = self.fSiggenData.impurity_rpower;      # power for radial impurity increase with radius
    siggenConfig["xtal_HV"]  = self.fSiggenData.xtal_HV;              # detector bias for fieldgen, in Volts
    siggenConfig["max_iterations"]  = self.fSiggenData.max_iterations ;       # maximum number of iterations to use in mjd_fieldgen
    siggenConfig["write_field"]  = self.fSiggenData.write_field;          # set to 1 to write V and E to output file, 0 otherwise
    siggenConfig["write_WP"]  = self.fSiggenData.write_WP;             # set to 1 to calculate WP and write it to output file, 0 otherwise
    siggenConfig["bulletize_PC"]  = self.fSiggenData.bulletize_PC;         # set to 1 for inside of point contact hemispherical, 0 for cylindrical

    siggenConfig["drift_name"] = str(self.fSiggenData.drift_name);

    # signal calculation
    siggenConfig["xtal_temp"]  = self.fSiggenData.xtal_temp;            # crystal temperature in Kelvin
    siggenConfig["preamp_tau"]  = self.fSiggenData.preamp_tau;           # integration time constant for preamplifier, in ns
    siggenConfig["time_steps_calc"]  = self.fSiggenData.time_steps_calc;      # number of time steps used in calculations
    siggenConfig["step_time_calc"]  = self.fSiggenData.step_time_calc;       # length of time step used for calculation, in ns
    siggenConfig["step_time_out"]  = self.fSiggenData.step_time_out;        # length of time step for output signal, in ns
    #    nonzero values in the next few lines significantly slow down the code
    siggenConfig["charge_cloud_size"]  = self.fSiggenData.charge_cloud_size;    # initial FWHM of charge cloud, in mm; set to zero for point charges
    siggenConfig["use_diffusion"]  = self.fSiggenData.use_diffusion;        # set to 0/1 for ignore/add diffusion as the charges drift
    siggenConfig["energy"]  = self.fSiggenData.energy;               # set to energy > 0 to use charge cloud self-repulsion, in keV

    siggenConfig["coord_type"]  = self.fSiggenData.coord_type;           # set to CART or CYL for input point coordinate system
    siggenConfig["ntsteps_out"]  = self.fSiggenData.ntsteps_out;          # number of time steps in output signal

    # data for fields.c
    siggenConfig["rmin"]  = self.fSiggenData.rmin;
    siggenConfig["rmax"]  = self.fSiggenData.rmax;
    siggenConfig["rstep"]  = self.fSiggenData.rstep;
    siggenConfig["zmin"]  = self.fSiggenData.zmin;
    siggenConfig["zmax"]  = self.fSiggenData.zmax;
    siggenConfig["zstep"]  = self.fSiggenData.zstep;

    siggenConfig["rlen"]  = self.fSiggenData.rlen;
    siggenConfig["zlen"]  = self.fSiggenData.zlen;           # dimensions of efld and wpot arrays
    siggenConfig["v_lookup_len"]  = self.fSiggenData.v_lookup_len;

    # data for calc_signal.c
    siggenConfig["initial_vel"]  = self.fSiggenData.initial_vel;
    siggenConfig["final_vel"]  = self.fSiggenData.final_vel;  # initial and final drift velocities for charges collected to PC
    siggenConfig["dv_dE"]  = self.fSiggenData.dv_dE;     # derivative of drift velocity with field ((mm/ns) / (V/cm))
    siggenConfig["v_over_E"]  = self.fSiggenData.v_over_E;  # ratio of drift velocity to field ((mm/ns) / (V/cm))
    siggenConfig["final_charge_size"]  = self.fSiggenData.final_charge_size;     # in mm

    return siggenConfig;

  def SetConfiguration(self, siggenConfig):

    self.fSiggenData.verbosity = siggenConfig["verbosity"];              # 0 = terse, 1 = normal, 2 = chatty/verbose
    self.fSiggenData.velocity_type = siggenConfig["velocity_type"];
    # geometry
    self.fSiggenData.xtal_length = siggenConfig["xtal_length"];          # z length
    self.fSiggenData.xtal_radius = siggenConfig["xtal_radius"];          # radius
    self.fSiggenData.top_bullet_radius = siggenConfig["top_bullet_radius"];    # bulletization radius at top of crystal
    self.fSiggenData.bottom_bullet_radius = siggenConfig["bottom_bullet_radius"]; # bulletization radius at bottom of BEGe crystal
    self.fSiggenData.pc_length = siggenConfig["pc_length"];            # point contact length
    self.fSiggenData.pc_radius = siggenConfig["pc_radius"];            # point contact radius
    self.fSiggenData.taper_length = siggenConfig["taper_length"];         # size of 45-degree taper at bottom of ORTEC-type crystal
    self.fSiggenData.wrap_around_radius = siggenConfig["wrap_around_radius"];   # wrap-around radius for BEGes. Set to zero for ORTEC
    self.fSiggenData.ditch_depth = siggenConfig["ditch_depth"];          # depth of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    self.fSiggenData.ditch_thickness = siggenConfig["ditch_thickness"];      # width of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    self.fSiggenData.Li_thickness = siggenConfig["Li_thickness"];         # depth of full-charge-collection boundary for Li contact

    # electric fields & weighing potentials
    self.fSiggenData.xtal_grid = siggenConfig["xtal_grid"];            # grid size in mm for field files (either 0.5 or 0.1 mm)
    self.fSiggenData.impurity_z0 = siggenConfig["impurity_z0"];          # net impurity concentration at Z=0, in 1e10 e/cm3
    self.fSiggenData.impurity_gradient = siggenConfig["impurity_gradient"];    # net impurity gradient, in 1e10 e/cm4
    self.fSiggenData.impurity_quadratic = siggenConfig["impurity_quadratic"];   # net impurity difference from linear, at z=L/2, in 1e10 e/cm3
    self.fSiggenData.impurity_surface = siggenConfig["impurity_surface"];     # surface impurity of passivation layer, in 1e10 e/cm2
    self.fSiggenData.impurity_radial_add = siggenConfig["impurity_radial_add"];  # additive radial impurity at outside radius, in 1e10 e/cm3
    self.fSiggenData.impurity_radial_mult = siggenConfig["impurity_radial_mult"]; # multiplicative radial impurity at outside radius (neutral=1.0)
    self.fSiggenData.impurity_rpower = siggenConfig["impurity_rpower"];      # power for radial impurity increase with radius
    self.fSiggenData.xtal_HV = siggenConfig["xtal_HV"];              # detector bias for fieldgen, in Volts
    self.fSiggenData.max_iterations = siggenConfig["max_iterations"];       # maximum number of iterations to use in mjd_fieldgen
    self.fSiggenData.write_field = siggenConfig["write_field"];          # set to 1 to write V and E to output file, 0 otherwise
    self.fSiggenData.write_WP = siggenConfig["write_WP"];             # set to 1 to calculate WP and write it to output file, 0 otherwise
    self.fSiggenData.bulletize_PC = siggenConfig["bulletize_PC"];         # set to 1 for inside of point contact hemispherical, 0 for cylindrical

    strcpy(self.fSiggenData.drift_name,  siggenConfig["drift_name"]);

    # signal calculation
    self.fSiggenData.xtal_temp = siggenConfig["xtal_temp"];            # crystal temperature in Kelvin
    self.fSiggenData.preamp_tau = siggenConfig["preamp_tau"];           # integration time constant for preamplifier, in ns
    self.fSiggenData.time_steps_calc = siggenConfig["time_steps_calc"];      # number of time steps used in calculations
    self.fSiggenData.step_time_calc = siggenConfig["step_time_calc"];       # length of time step used for calculation, in ns
    self.fSiggenData.step_time_out = siggenConfig["step_time_out"];        # length of time step for output signal, in ns
    #    nonzero values in the next few lines significantly slow down the code
    self.fSiggenData.charge_cloud_size = siggenConfig["charge_cloud_size"];    # initial FWHM of charge cloud, in mm"]; set to zero for point charges
    self.fSiggenData.use_diffusion = siggenConfig["use_diffusion"];        # set to 0/1 for ignore/add diffusion as the charges drift
    self.fSiggenData.energy = siggenConfig["energy"];               # set to energy > 0 to use charge cloud self-repulsion, in keV

    self.fSiggenData.coord_type = siggenConfig["coord_type"];           # set to CART or CYL for input point coordinate system
    self.fSiggenData.ntsteps_out = siggenConfig["ntsteps_out"];          # number of time steps in output signal

    # data for fields.c
    self.fSiggenData.rmin = siggenConfig["rmin"];
    self.fSiggenData.rmax = siggenConfig["rmax"];
    self.fSiggenData.rstep = siggenConfig["rstep"];
    self.fSiggenData.zmin = siggenConfig["zmin"];
    self.fSiggenData.zmax = siggenConfig["zmax"];
    self.fSiggenData.zstep = siggenConfig["zstep"];

    self.fSiggenData.rlen = siggenConfig["rlen"];
    self.fSiggenData.zlen = siggenConfig["zlen"];           # dimensions of efld and wpot arrays
    self.fSiggenData.v_lookup_len = siggenConfig["v_lookup_len"];

    # data for calc_signal.c
    self.fSiggenData.initial_vel = siggenConfig["initial_vel"];
    self.fSiggenData.final_vel = siggenConfig["final_vel"];  # initial and final drift velocities for charges collected to PC
    self.fSiggenData.dv_dE = siggenConfig["dv_dE"];     # derivative of drift velocity with field ((mm/ns) / (V/cm))
    self.fSiggenData.v_over_E = siggenConfig["v_over_E"];  # ratio of drift velocity to field ((mm/ns) / (V/cm))
    self.fSiggenData.final_charge_size = siggenConfig["final_charge_size"];     # in mm



cdef public int drift_velocity_python(csiggen.point pt, csiggen.cyl_pt e, csiggen.point field_norm, float q, csiggen.vector *velo, csiggen.MJD_Siggen_Setup *setup):

    absfield = np.sqrt( e.r**2 + e.z**2)
    phi_0 = np.arctan2(field_norm.y,field_norm.x)
    theta_0 = np.arccos(field_norm.z  )

    if q>0:
      #holes holes holes

#      theta_0 = 2*np.pi - theta_0
#      if field.z < 0 and theta_0 < np.pi/2:
#        theta_0 = np.pi + theta_0

#      if phi_0<0: phi_0 = 2*np.pi + phi_0

      (v_r, v_theta, v_phi) = find_hole_velo(absfield, theta_0, phi_0)

      r = np.sqrt( pt.x**2 + pt.y**2 + pt.z**2 )
      phi = np.arctan2(pt.y,pt.x)
      theta = np.arccos(pt.z / r )

      r = absfield
      theta = theta_0
      phi = phi_0

#      print "calc spherical: (%f, %f, %f)" % (v_r, v_theta, v_phi)

      v_x = np.sin(theta)*np.cos(phi) * v_r + np.cos(theta)*np.cos(phi) * v_theta - np.sin(theta)*np.sin(phi) * v_phi
      v_y = np.sin(theta)*np.sin(phi) * v_r + np.cos(theta)*np.sin(phi) * v_theta + np.sin(theta)*np.cos(phi) * v_phi
      v_z = np.cos(theta) * v_r - np.sin(theta) * v_theta

      velo.x = np.around(v_x,5)
      velo.y = np.around(v_y,5)
      velo.z = np.around(v_z,5)

#      beta = theta_0#phi #+ np.pi/4 + np.pi/4
#      alpha = phi_0
#
#      R_y = np.matrix( [ [np.cos(beta), 0, np.sin(beta)], [0, 1., 0], [-np.sin(beta), 0, np.cos(beta)]]  )
#      R_z = np.matrix( [[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0,0,1.] ] )
#      R_j = np.dot(R_z, R_y)
#
#      v_prime = np.array([v_r, v_theta, v_phi])
#      v = np.array(np.dot(R_j, v_prime))

#      v_x = np.sin(theta)*np.cos(phi) * v[0][0] + r*np.cos(theta)*np.cos(phi) * v[0][1] - r *np.sin(theta)*np.sin(phi) * v[0][2]
#      v_y = np.sin(theta)*np.sin(phi) * v[0][0] + r*np.cos(theta)*np.sin(phi) * v[0][1] + r *np.sin(theta)*np.cos(phi) * v[0][2]
#      v_z = np.cos(theta) * v[0][0] - r *np.sin(theta) * v[0][2]


#      if pt.z == 15:
#        print "calculating position: (%f, %f, %f)" % (pt.x , pt.y , pt.z)
#        print "  e: (%f, %f, %f)" % (e.r , e.phi , e.z)
#        print "  absfield: %f" % absfield
#        print "  field norm: (%f, %f, %f)" % (field_norm.x , field_norm.y , field_norm.z)
#        print "  phi_0: %0.2f pi" % (phi_0 / np.pi)
#        print "  theta_0: %0.2f pi" % (theta_0 / np.pi)
#
#        print "  phi: %0.2f pi" % (phi / np.pi)
#        print "  theta: %0.2f pi" % (theta / np.pi)
#
#        print "  v_x: %f or %f or %f" % (v_r, v_x, v[0][0])
#        print "  v_y: %f or %f or %f" % (v_theta, v_y, v[0][1])
#        print "  v_z: %f or %f or %f" % (v_phi, v_z, v[0][2])

#      print v.shape
#      print v[0][0]
#      print v[0][1]
#      print v[0][1]
#      print v[0][2]

#      velo.x = v_x
#      velo.y = v_y
#      velo.z = v_z

#      velo.x = v[0][0]
#      velo.y = v[0][1]
#      velo.z = v[0][2]

      return 0

    else: #electrons
      (v_x, v_y, v_z) = find_electron_velo(absfield, theta_0, phi_0)
      velo.x = v_x
      velo.y = v_y
      velo.z = v_z

      return 0


def find_hole_velo(field, theta, phi ):

    #these are the reggiani numbers
    v_100 = drift_velo_model(field, 66333., 0.744, 181.)
    v_111 = drift_velo_model(field, 107270., 0.580, 100.)

    if v_100 == 0: return 0.
    v_rel = v_111 / v_100

    k_0 = 9.2652 - 26.3467*v_rel + 29.6137*v_rel**2 -12.3689 * v_rel**3

    lambda_k0 = -0.01322 * k_0 + 0.41145*k_0**2 - 0.23657 * k_0**3 + 0.04077*k_0**4
    omega_k0 = 0.006550*k_0 - 0.19946*k_0**2 + 0.09859*k_0**3 - 0.01559*k_0**4

    v_r = v_100 * (1- lambda_k0*(np.sin(theta)**4*np.sin(2*phi)**2 + np.sin(2*theta)**2 ) )
    v_theta = v_100 * omega_k0 * (2*np.sin(theta)**3*np.cos(theta)*np.sin(2*phi)**2 + np.sin(4*theta) )
    v_phi = v_100 * omega_k0 * np.sin(theta)**3*np.sin(4*phi)

    return (v_r, v_theta, v_phi)


def find_electron_velo(field_0, theta, phi):

  field = np.array([field_0*np.sin(theta)*np.cos(phi), field_0*np.sin(theta)*np.sin(phi), field_0*np.cos(theta)])

  eta_0 = 0.496
  b = 0.0296
  e_ref = 1200
#find_drift_velocity_bruyneel(field, 38609, 0.805, 511., -171)

#  eta_0 = 0.422
#  b = 0.201
#  e_ref = 1200.
  eta = eta_0 + b * np.log( field/e_ref )

  m_l = 1.64
  m_t = 0.0819
  gamma_0 = 2.888

  j_0 = np.diag(np.array([m_t**-1,m_l**-1,m_t**-1]))

  E_star = np.empty((4,3))
  inv_nu = np.empty(4)

  alphas = np.empty((4,3,3))

  for i in range(1,5):
    beta = np.arccos(np.sqrt(2./3))
    alpha = (i-1) * np.pi / 2. + np.pi/4


    R_x = np.matrix( [ [1., 0, 0], [0, np.cos(beta), np.sin(beta)], [0, -np.sin(beta), np.cos(beta)]]  )
    R_z = np.matrix( [[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0,0,1.] ] )

    R_j = np.dot(R_x, R_z)
#    R_j = R_j.round(10)

#    print R_x
#    print R_z

    alpha_i = np.dot(R_j.T, np.dot(j_0, R_j))

    alphas[i-1, :,:] = alpha_i

#    print R_j
    alpha_i = alpha_i.round(10)
#    print alpha_i
#    print np.sqrt(alpha_i)

    from scipy import linalg

    E_star[i-1,:] = np.dot(linalg.sqrtm(alpha_i), field)
#    print E_star[i-1,:]

    e_star_i_norm = np.linalg.norm(E_star[i-1,:])

    eta_star_i = eta_0 + b * np.log( e_star_i_norm/e_ref )

#    print eta_star_i

    inv_nu[i-1] = np.power(e_star_i_norm, eta_star_i)**-1

  sum_nu = np.sum(inv_nu)

  v_d = np.empty((4,3))

  for i in range(1,5):
    E_star_i = E_star[i-1,:]
    e_star_i_norm = np.linalg.norm(E_star[i-1,:])

    v_100_i = drift_velo_model(e_star_i_norm/gamma_0, 38609, 0.805, 511., -171) #
    #v_100_i = drift_velo_model(e_star_i_norm/gamma_0, 40180., 0.72, 493., 589)
#    print v_100_i
    mu_star_i = v_100_i / gamma_0 / e_star_i_norm

    n_i = inv_nu[i-1] / sum_nu

#    print E_star_i
#    print e_star_i_norm

    v_d[i-1] = n_i * mu_star_i * np.dot(alphas[i-1, :,:], field)

  v_d = -1*np.around(np.sum(v_d, axis=0),5)
  return v_d


def drift_velo_model(E, mu_0, beta, E_0, mu_n = 0):

  v = (mu_0 * E) / np.power(1+(E/E_0)**beta, 1./beta) - mu_n*E

  return v * 10 * 1E-9
