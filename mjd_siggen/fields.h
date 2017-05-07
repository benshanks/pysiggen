/* fields.h -- based on m3d2s.f by I-Yang Lee
 * Karin Lagergren
 *
 * This module handles the electric field and weighting potential and
 * calculates drift velocities
 */

#ifndef _FIELDS_H
#define _FIELDS_H

/* calculate anisotropic drift velocities? (vel. depends on angle between
   el. field and crystal axis; otherwise the velocity will always be
   in the direction of the el. field
*/
#define DRIFT_VEL_ANISOTROPY 1

#include "point.h"
#include "mjd_siggen.h"

/* field_setup
   given a field directory file, read electic field and weighting
   potential tables from files listed in directory
   returns 0 for success
*/
int field_setup(MJD_Siggen_Setup *setup);

/* free malloc()'ed memory and do other cleanup*/
int fields_finalize(MJD_Siggen_Setup *setup);

/* wpotential
   gives (interpolated or extrapolated ) weighting potential
   at point pt. These values are stored in wp.
   returns 0 for success, 1 on failure.
*/
int wpotential(point pt, float *wp, MJD_Siggen_Setup *setup);

/* drift_velocity
   calculates drift velocity for charge q at point pt
   returns 0 on success, 1 if successful but extrapolation was needed,
   and -1 for failure
*/
int drift_velocity(point pt, float q, vector *velocity, MJD_Siggen_Setup *setup);
int drift_velocity_ben(point pt, float q, vector *velocity, MJD_Siggen_Setup *setup);

int read_fields(MJD_Siggen_Setup *setup);

/*set detector temperature. 77F (no correction) is the default
   MIN_TEMP & MAX_TEMP defines allowed range*/
void set_temp(float temp, MJD_Siggen_Setup *setup);

void set_hole_params(float h_100_mu0, float h_100_beta, float h_100_e0, float h_111_mu0, float h_111_beta, float h_111_e0, MJD_Siggen_Setup *setup);
void set_k0_params(float k0_0, float k0_1, float k0_2, float k0_3, MJD_Siggen_Setup *setup);

float get_wpot_by_index(int row, int col,int pcrad, int pclen, MJD_Siggen_Setup* setup );
float get_efld_r_by_index(int row, int col, int grad, int imp, int pcrad, int pclen,MJD_Siggen_Setup* setup );
float get_efld_z_by_index(int row, int col, int grad, int imp, int pcrad, int pclen,MJD_Siggen_Setup* setup );


#endif /*#ifndef _FIELDS_H*/
