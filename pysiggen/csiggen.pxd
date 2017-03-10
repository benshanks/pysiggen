# file: cqueue.pxd

cdef extern from "point.h":
  cdef struct point:
    float x
    float y
    float z

  ctypedef point point
  ctypedef point vector
  ctypedef point int_pt

cdef extern from "cyl_point.h":
  cdef struct cyl_pt:
    float r
    float phi
    float z

  ctypedef cyl_pt cyl_pt

cdef extern from "mjd_siggen.h":
  cdef struct velocity_lookup:
    float e;
    float e100;
    float e110;
    float e111;
    float h100;
    float h110;
    float h111;
    float ea;
    float eb;
    float ec;
    float ebp;
    float ecp;
    float ha;
    float hb;
    float hc;
    float hbp;
    float hcp;
    float hcorr;
    float ecorr;

  cdef struct velocity_params:
    float h_100_mu0;
    float h_100_beta;
    float h_100_e0;
    float h_111_mu0;
    float h_111_beta;
    float h_111_e0;
    float k0_0
    float k0_1
    float k0_2
    float k0_3

  ctypedef velocity_params velocity_params


  ctypedef struct MJD_Siggen_Setup:
    int verbosity;              # 0 = terse, 1 = normal, 2 = chatty/verbose
    int velocity_type;          # 0 = david, 1 = ben

    # geometry
    float xtal_length;          # z length
    float xtal_radius;          # radius
    float top_bullet_radius;    # bulletization radius at top of crystal
    float bottom_bullet_radius; # bulletization radius at bottom of BEGe crystal
    float pc_length;            # point contact length
    float pc_radius;            # point contact radius
    float taper_length;         # size of 45-degree taper at bottom of ORTEC-type crystal
    float wrap_around_radius;   # wrap-around radius for BEGes. Set to zero for ORTEC
    float ditch_depth;          # depth of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    float ditch_thickness;      # width of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    float Li_thickness;         # depth of full-charge-collection boundary for Li contact

    # electric fields & weighing potentials
    float xtal_grid;            # grid size in mm for field files (either 0.5 or 0.1 mm)
    float impurity_z0;          # net impurity concentration at Z=0, in 1e10 e/cm3
    float impurity_gradient;    # net impurity gradient, in 1e10 e/cm4
    float impurity_quadratic;   # net impurity difference from linear, at z=L/2, in 1e10 e/cm3
    float impurity_surface;     # surface impurity of passivation layer, in 1e10 e/cm2
    float impurity_radial_add;  # additive radial impurity at outside radius, in 1e10 e/cm3
    float impurity_radial_mult; # multiplicative radial impurity at outside radius (neutral=1.0)
    float impurity_rpower;      # power for radial impurity increase with radius
    float xtal_HV;              # detector bias for fieldgen, in Volts
    int   max_iterations;       # maximum number of iterations to use in mjd_fieldgen
    int   write_field;          # set to 1 to write V and E to output file, 0 otherwise
    int   write_WP;             # set to 1 to calculate WP and write it to output file, 0 otherwise
    int   bulletize_PC;         # set to 1 for inside of point contact hemispherical, 0 for cylindrical

    # file names
    char drift_name[256];       # drift velocity lookup table
    char field_name[256];       # potential/efield file name
    char wp_name[256];          # weighting potential file name

    # signal calculation
    float xtal_temp;            # crystal temperature in Kelvin
    float preamp_tau;           # integration time constant for preamplifier, in ns
    int   time_steps_calc;      # number of time steps used in calculations
    float step_time_calc;       # length of time step used for calculation, in ns
    float step_time_out;        # length of time step for output signal, in ns
    #    nonzero values in the next few lines significantly slow down the code
    float charge_cloud_size;    # initial FWHM of charge cloud, in mm; set to zero for point charges
    int   use_diffusion;        # set to 0/1 for ignore/add diffusion as the charges drift
    float energy;               # set to energy > 0 to use charge cloud self-repulsion, in keV

    int   coord_type;           # set to CART or CYL for input point coordinate system
    int   ntsteps_out;          # number of time steps in output signal

    # data for fields.c
    float rmin, rmax, rstep;
    float zmin, zmax, zstep;
    int   rlen, zlen;           # dimensions of efld and wpot arrays
    int   v_lookup_len;
    velocity_lookup* v_lookup;
    velocity_params* v_params;
    cyl_pt** efld;
    float** wpot;

    # data for calc_signal.c
    point *dpath_e
    point *dpath_h;      # electron and hole drift paths
    float initial_vel, final_vel;  # initial and final drift velocities for charges collected to PC
    float dv_dE;     # derivative of drift velocity with field ((mm/ns) / (V/cm))
    float v_over_E;  # ratio of drift velocity to field ((mm/ns) / (V/cm))
    double final_charge_size;     # in mm

  int read_config(char *config_file_name, MJD_Siggen_Setup *setup);

cdef extern from "calc_signal.h":
  ctypedef struct Signal:
      pass

  int signal_calc_init(char *config_file_name, MJD_Siggen_Setup *setup);
  int get_signal(point pt, float *signal, MJD_Siggen_Setup *setup)
  int make_signal(point pt, float *signal, float q, MJD_Siggen_Setup *setup)
  int signal_calc_finalize(MJD_Siggen_Setup *setup);
  int rc_integrate(float *s_in, float *s_out, float tau, int time_steps);
  int drift_path_e(point **path, MJD_Siggen_Setup *setup);
  int drift_path_h(point **path, MJD_Siggen_Setup *setup);
  void tell(const char *format, ...);
  void error(const char *format, ...);

cdef extern from "fields.h":
  int field_setup(MJD_Siggen_Setup *setup);
  int fields_finalize(MJD_Siggen_Setup *setup);
  int wpotential(point pt, float *wp, MJD_Siggen_Setup *setup);
  int drift_velocity(point pt, float q, vector *velocity, MJD_Siggen_Setup *setup);
  int read_fields(MJD_Siggen_Setup *setup);
  void set_temp(float temp, MJD_Siggen_Setup *setup);
  void set_hole_params(float h_100_mu0, float h_100_beta, float h_100_e0, float h_111_mu0, float h_111_beta, float h_111_e0, MJD_Siggen_Setup *setup);
  void set_k0_params(float k0_0, float k0_1, float k0_2, float k0_3, MJD_Siggen_Setup *setup);
